from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from m3l2.app.config import get_settings
from m3l2.app.db import (
    ExecutionRecord,
    ForecastCache,
    ModelRegistry,
    RegisteredSite,
    SessionLocal,
    SiteProfile,
    SiteSnapshot,
    SiteStatusSnapshot,
    create_tables,
    utc_now,
)
from m3l2.app.schemas import IngestRunRequest, PredictRequest, PredictionResponse
from m3l2.auth.router import router as auth_router
from m3l2.broker_mock.router import router as mock_broker_router
from m3l2.inference.forecast_refresh import refresh_forecasts
from m3l2.inference.predict import predict as run_predict, predict_many
from m3l2.ingestion.jobs import run_ingestion
from m3l2.ingestion.site_adapter import SiteAdapterValidationError, normalise_site_profile, normalise_site_status
from m3l2.site_adapter.auth import SitePrincipal, current_principal
from m3l2.site_adapter.control_plane import router as site_adapter_router
from m3l2.site_adapter.mock_l3 import router as mock_l3_router
from m3l2.training.registry import get_active_model, get_model_by_version, list_models, serialise_model
from m3l2.training.train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
scheduler: BackgroundScheduler | None = None


def _schema_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _parse_optional_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _serialise_dt(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _serialise_row(row: Any) -> dict[str, Any]:
    return {column.name: _serialise_dt(getattr(row, column.name)) for column in row.__table__.columns}


def _submission_errors(errors: list[dict[str, Any]], index: int | None = None) -> list[dict[str, Any]]:
    prefix: list[Any] = ["body"] if index is None else ["body", index]
    return [{**error, "loc": prefix + list(error.get("loc", []))} for error in errors]


def _warnings(index: int, profile_or_status: dict[str, Any]) -> dict[str, Any] | None:
    fields = profile_or_status.pop("_warnings", [])
    if not fields:
        return None
    return {"index": index, "fields": fields}


PROFILE_SUBMISSION_DESCRIPTION = """
Submit one static site profile object or a batch of profile objects. The API maps canonical fields and supported
aliases into the static profile schema before persistence, rejects conflicting aliases, and preserves unrecognised
fields in `extensions`.

Required: `site_id`.
Optional: `ri_type` (`network`, `cloud`, `grid`; omitted values persist as `unknown`), `location`,
`compute_capacity` (CPU cores/vCPUs or adapter-native compute units, >= 0), `gpu_capacity` (GPUs, >= 0),
`storage_capacity` (GB, >= 0), `network_topology`, `link_capacities` object,
`supported_workload_types` string list, `energy_capabilities` object, and `static_pue_baseline` (PUE, >= 1).

Aliases: generic/IoT `site`, `site_name`, `pue`; IoT `facility`, `total_nodes`, `node_count`, `topology`;
OpenStack `cloud_name`, `name`, `region_name`, `availability_zone`, `total_vcpus`, `vcpus_total`,
`cpu_capacity`, `total_gpus`, `gpus_total`, `total_disk_gb`, `disk_gb_total`.

Validation: invalid types, enum values, ranges, missing identity, and alias conflicts return HTTP 422 with
field-specific `detail` entries. Unknown fields are returned in submission warnings and stored in `extensions`;
extensions are not used by training or prediction until explicitly mapped.
"""


STATUS_SUBMISSION_DESCRIPTION = """
Submit one dynamic site status object or a batch of status objects. The API maps canonical fields and supported
aliases into the dynamic metrics schema before persistence, rejects conflicting aliases, and preserves unrecognised
fields in `extensions`.

Required: `site_id`, `timestamp` (explicit ISO-8601 timestamp).
Optional: `ri_type` (`network`, `cloud`, `grid`; omitted values persist as `unknown`), `operational_status`
(`UP`, `DOWN`, `DEGRADED`, `MAINTENANCE`), `maintenance_flag`, `scheduled_maintenance` object,
`node_availability` and `link_availability` ratios (0..1), `stability_score` (0..1), `packet_loss` (% 0..100),
`network_jitter` (ms, >= 0), `network_utilization` (% 0..100), `available_bandwidth` (Mbps, >= 0),
`cpu_util_avg`/`gpu_util_avg` (% 0..100), free CPU/GPU capacity (>= 0), queue/job counts (integers >= 0),
`provisioning_delay_s` (seconds, >= 0), `load_index` (0..1), `energy_consumed` (Wh, >= 0), `pue_estimate` (>= 1),
`carbon_intensity` (gCO2/kWh, >= 0), `energy_per_task_proxy` (>= 0), `update_frequency` (seconds, > 0),
`data_confidence` and `coverage_ratio` (0..1), and `stale_flag`.

Aliases: timestamp `ts`, IoT `bucket_15m`, OpenStack/IoT `updated_at`; status `state`, `status`; maintenance
`maintenance`, OpenStack `planned_maintenance`; CPU/GPU utilisation `cpu_utilization`, `cpu_utilisation`,
`cpu_util`, `cpu_util_percent`, `gpu_utilization`, `gpu_utilisation`, `gpu_util`, `gpu_util_percent`; stability
`stability`, `stability_index`; staleness `stale`, `is_stale`; network/energy aliases `jitter_ms`,
`packet_loss_percent`, `available_bandwidth_mbps`, `energy_wh`, `ci_gco2_kwh`, `pue`.
IoT availability may use explicit ratios or counts: `alive_nodes`/`active_nodes` with `total_nodes`/`node_count`,
and `active_links` with `total_links`. OpenStack utilisation may be derived from `total_vcpus`/`vcpus_total`
and `free_vcpus`/`vcpus_free`, or from `total_gpus`/`gpus_total` and `free_gpus`/`gpus_free`.

Validation: invalid types, enum values, ranges, missing timestamp/identity, impossible counts, inconsistent
maintenance state, and conflicting explicit-vs-derived values return HTTP 422 with field-specific `detail` entries.
Unknown fields are returned in submission warnings and stored in `extensions`; extensions are not used by training
or prediction until explicitly mapped.
"""


PROFILE_EXAMPLES = {
    "generic_profile": {
        "summary": "Generic static profile",
        "value": {
            "site_id": "SLICES-GR-UTH",
            "ri_type": "grid",
            "location": "UTH",
            "compute_capacity": 128,
            "gpu_capacity": 0,
            "storage_capacity": 2048,
            "network_topology": "Mesh",
            "link_capacities": {"core_mbps": 10000},
            "supported_workload_types": ["batch", "stream", "ml"],
            "energy_capabilities": {"metering": True},
            "static_pue_baseline": 1.2,
            "local_owner": "UTH",
        },
    },
    "iot_profile_aliases": {
        "summary": "IoT adapter aliases with independent RI type",
        "value": {
            "site": "SLICES-GR-UTH",
            "ri_type": "network",
            "facility": "Volos lab",
            "node_count": 32,
            "topology": "Mesh",
            "pue": 1.35,
            "sensor_generation": "v2",
        },
    },
    "openstack_profile_aliases": {
        "summary": "OpenStack capacity aliases",
        "value": {
            "cloud_name": "OPENSTACK-DEMO",
            "ri_type": "cloud",
            "region_name": "eu-west",
            "total_vcpus": 256,
            "total_gpus": 8,
            "total_disk_gb": 50000,
        },
    },
}


STATUS_EXAMPLES = {
    "uth_dynamic_metrics": {
        "summary": "Canonical UTH status metrics",
        "value": {
            "site_id": "SLICES-GR-UTH",
            "ri_type": "grid",
            "timestamp": "2026-09-08T07:00:00Z",
            "operational_status": "DEGRADED",
            "maintenance_flag": False,
            "node_availability": 0.95,
            "link_availability": 0.98,
            "stability_score": 0.99,
            "packet_loss": 0.1,
            "network_jitter": 2.0,
            "network_utilization": 42.0,
            "available_bandwidth": 1000.0,
            "cpu_util_avg": 55.0,
            "queue_length": 3,
            "remaining_jobs": 7,
            "load_index": 0.61,
            "energy_consumed": 123.4,
            "pue_estimate": 1.2,
            "carbon_intensity": 250.0,
            "update_frequency": 3600,
            "data_confidence": 0.9,
            "coverage_ratio": 0.95,
            "stale_flag": False,
            "local_note": "retained as extension",
        },
    },
    "iot_counts": {
        "summary": "IoT counts derive availability ratios",
        "value": {
            "site": "SLICES-GR-UTH",
            "ri_type": "network",
            "ts": "2026-09-08T07:00:00Z",
            "alive_nodes": 0,
            "total_nodes": 32,
            "active_links": 7,
            "total_links": 10,
            "cpu_utilization": 0,
            "stability": 1.0,
            "stale": False,
        },
    },
    "openstack_utilisation": {
        "summary": "OpenStack utilisation derived from free and total capacity",
        "value": {
            "cloud_name": "OPENSTACK-DEMO",
            "ri_type": "cloud",
            "updated_at": "2026-09-08T07:00:00Z",
            "total_vcpus": 256,
            "free_vcpus": 120,
            "total_gpus": 8,
            "free_gpus": 2,
            "pending_vms": 4,
            "vm_provisioning_delay_s": 180,
        },
    },
}


SUBMISSION_VALIDATION_RESPONSE = {
    "description": "Submission validation failed with field-specific errors",
    "content": {
        "application/json": {
            "example": {
                "detail": [
                    {
                        "loc": ["body", 0, "timestamp"],
                        "msg": "timestamp is required",
                        "type": "value_error.missing",
                    },
                    {
                        "loc": ["body", 0, "node_availability"],
                        "msg": "node_availability must be less than or equal to 1",
                        "type": "value_error.range",
                    },
                ]
            }
        }
    },
}


def get_db() -> Session:
    with SessionLocal() as session:
        yield session


def _scheduled_cycle() -> None:
    logger.info("Starting scheduled M3L2 ingestion and training cycle")
    try:
        ingestion_summary = run_ingestion()
        training_summary = train_model(force=False)
        logger.info("Scheduled M3L2 cycle completed: ingestion=%s training=%s", ingestion_summary, training_summary)
    except Exception:
        logger.exception("Scheduled M3L2 cycle failed")


def _scheduled_forecast_refresh() -> None:
    logger.info("Starting scheduled M3L2 forecast refresh")
    try:
        summary = refresh_forecasts(force=True)
        logger.info("Scheduled M3L2 forecast refresh completed: %s", summary)
    except Exception:
        logger.exception("Scheduled M3L2 forecast refresh failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler
    create_tables()
    settings = get_settings()
    if settings.enable_scheduler and (scheduler is None or not scheduler.running):
        scheduler = BackgroundScheduler(timezone="UTC")
        scheduler.add_job(
            _scheduled_cycle,
            "interval",
            hours=settings.train_interval_hours,
            id="m3l2_ingest_train",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        scheduler.add_job(
            _scheduled_forecast_refresh,
            "interval",
            minutes=settings.forecast_refresh_minutes,
            id="m3l2_forecast_refresh",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        scheduler.start()
        logger.info(
            "Started M3L2 scheduler with %sh training interval and %sm forecast refresh interval",
            settings.train_interval_hours,
            settings.forecast_refresh_minutes,
        )
    yield
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Stopped M3L2 scheduler")


app = FastAPI(
    title="GreenDIGIT M3L2 MVP API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "M3L2 L2DB, Site Adapter Control Plane, prediction API, and mock broker flow.\n\n"
        "**Authentication**\n\n"
        "- Open `/auth/login` to obtain a 24-hour JWT using email and password.\n"
        "- The first login registers a password only if the email is listed in `allowed_emails.txt`.\n"
        "- Use the token as `Authorization: Bearer <token>` on protected L2 endpoints.\n"
        "- JSON token clients can call `POST /auth/token` or `GET /auth/token`.\n\n"
        "**Built-in mock L3 Site Adapter**\n\n"
        "- No separate container is required; it runs inside this FastAPI service.\n"
        "- Register a test site with `adapter_base_url` set to "
        "`http://127.0.0.1:8000/mock-l3/sites/{site_id}` so L2 pull calls loop back to the mock.\n"
        "- The mock exposes `/capabilities`, `/availability`, `/usage`, and `/efficiency` for snapshot validation.\n"
        "- For public testing through Nginx, browse `/mock-l3/sites/{site_id}/capabilities`; for registered "
        "adapter callbacks use the internal `127.0.0.1:8000` URL."
    ),
    openapi_tags=[
        {
            "name": "l2-prediction",
            "description": "EUR-facing L2 prediction endpoint for inferred site availability, resources, and feasibility.",
        },
        {
            "name": "m3l2-ops",
            "description": "Operational health, ingestion, training, model registry, and cache metrics.",
        },
        {
            "name": "training-data",
            "description": "Compatibility endpoints for loading training-compatible site profile and status data.",
        },
        {
            "name": "Auth",
            "description": (
                "EIMPS-style login endpoints. Obtain a 24-hour JWT from `/auth/login` or `/auth/token`, "
                "then use `Authorization: Bearer <token>` on protected L2 endpoints."
            ),
        },
        {
            "name": "l2-site-adapter",
            "description": (
                "L2 Site Adapter Control Plane: register sites, push/pull snapshots, read latest L2DB data, "
                "and proxy workload submissions to registered L3 adapters."
            ),
        },
        {
            "name": "Mock L3 Site Adapter",
            "description": (
                "Built-in mock L3 adapter for validation. It runs in the same API container. "
                "Use `http://127.0.0.1:8000/mock-l3/sites/{site_id}` as a registered site's `adapter_base_url`."
            ),
        },
        {
            "name": "mock-broker",
            "description": "Mock broker flow that predicts, selects a site, and submits a workload through L2.",
        },
    ],
    swagger_ui_parameters={"persistAuthorization": True},
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(auth_router)
app.include_router(site_adapter_router)
app.include_router(mock_l3_router)
app.include_router(mock_broker_router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/auth/login")


@app.get("/health", tags=["m3l2-ops"])
def health(session: Session = Depends(get_db)) -> dict[str, Any]:
    active = get_active_model(session)
    return {"status": "ok", "db": "ok", "active_model_version": active.version if active else None}


@app.post("/ingest/run", tags=["m3l2-ops"])
def ingest_run(request: IngestRunRequest | None = None) -> dict[str, Any]:
    payload = _schema_dump(request) if request else {}
    return run_ingestion(**payload)


@app.post("/train", tags=["m3l2-ops"])
def train() -> dict[str, Any]:
    return train_model(force=True)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    include_in_schema=False,
    responses={503: {"description": "No active model is available"}},
)
def predict(request: PredictRequest):
    return _predict_response(request)


@app.post(
    "/l2/predict",
    response_model=PredictionResponse,
    tags=["l2-prediction"],
    responses={
        401: {"description": "Bearer token is required"},
        404: {"description": "Candidate sites could not be resolved"},
        503: {"description": "No active model is available"},
    },
)
def l2_predict(
    request: PredictRequest,
    principal: SitePrincipal = Depends(current_principal),
):
    return _predict_response(request)


def _predict_response(request: PredictRequest):
    try:
        result = run_predict(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result.get("status") == "no_active_model":
        return JSONResponse(status_code=503, content=result)
    if result.get("status") == "candidate_sites_not_found":
        return JSONResponse(status_code=404, content=result)
    return result


@app.post("/predict/batch", include_in_schema=False)
def predict_batch(requests: list[PredictRequest]):
    try:
        responses = predict_many(requests)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    status_code = 200
    for result in responses:
        if result.get("status") == "no_active_model":
            status_code = 503
        elif result.get("status") == "candidate_sites_not_found" and status_code == 200:
            status_code = 404
    if status_code != 200:
        return JSONResponse(status_code=status_code, content=responses)
    return responses


@app.get("/models", tags=["m3l2-ops"])
def models(session: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [serialise_model(row) for row in list_models(session)]


@app.get("/models/{version}", tags=["m3l2-ops"])
def model(version: str, session: Session = Depends(get_db)) -> dict[str, Any]:
    row = get_model_by_version(session, version)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Model version not found: {version}")
    return serialise_model(row)


@app.get("/metrics", tags=["m3l2-ops"])
def metrics(session: Session = Depends(get_db)) -> dict[str, Any]:
    active = get_active_model(session)
    latest_ingested_at = session.execute(select(ExecutionRecord.ingested_at).order_by(desc(ExecutionRecord.ingested_at))).scalars().first()
    return {
        "execution_records_count": session.scalar(select(func.count()).select_from(ExecutionRecord)),
        "site_profiles_count": session.scalar(select(func.count()).select_from(SiteProfile)),
        "site_status_snapshots_count": session.scalar(select(func.count()).select_from(SiteStatusSnapshot)),
        "models_count": session.scalar(select(func.count()).select_from(ModelRegistry)),
        "registered_sites_count": session.scalar(select(func.count()).select_from(RegisteredSite)),
        "site_snapshots_count": session.scalar(select(func.count()).select_from(SiteSnapshot)),
        "active_model_version": active.version if active else None,
        "latest_ingested_at": latest_ingested_at.isoformat() if latest_ingested_at else None,
        "forecast_cache_count": session.scalar(select(func.count()).select_from(ForecastCache)),
    }


@app.post(
    "/site-profiles",
    tags=["training-data"],
    description=PROFILE_SUBMISSION_DESCRIPTION,
    responses={422: SUBMISSION_VALIDATION_RESPONSE},
)
def upsert_site_profiles(
    payload: dict[str, Any] | list[dict[str, Any]] = Body(..., openapi_examples=PROFILE_EXAMPLES),
    adapter_type: Literal["generic", "iot", "openstack"] = Query(
        "generic",
        description="Selects the input alias adapter. It does not classify `ri_type`.",
    ),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    items = payload if isinstance(payload, list) else [payload]
    profiles = []
    warnings = []
    errors = []
    for index, item in enumerate(items):
        try:
            profile = normalise_site_profile(item, adapter_type=adapter_type)
        except SiteAdapterValidationError as exc:
            errors.extend(_submission_errors(exc.errors, index if isinstance(payload, list) else None))
            continue
        warning = _warnings(index, profile)
        if warning:
            warnings.append(warning)
        profiles.append(profile)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    upserted = 0
    for profile in profiles:
        existing = session.execute(select(SiteProfile).where(SiteProfile.site_id == profile["site_id"])).scalar_one_or_none()
        profile["updated_at"] = utc_now()
        if existing is None:
            session.add(SiteProfile(**profile))
        else:
            for key, value in profile.items():
                setattr(existing, key, value)
        upserted += 1
    session.commit()
    return {"upserted": upserted, "adapter_type": adapter_type, "warnings": warnings}


@app.get("/site-profiles", tags=["training-data"])
def list_site_profiles(session: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [_serialise_row(row) for row in session.execute(select(SiteProfile).order_by(SiteProfile.site_id)).scalars()]


@app.post(
    "/site-status",
    tags=["training-data"],
    description=STATUS_SUBMISSION_DESCRIPTION,
    responses={422: SUBMISSION_VALIDATION_RESPONSE},
)
def ingest_site_status(
    payload: dict[str, Any] | list[dict[str, Any]] = Body(..., openapi_examples=STATUS_EXAMPLES),
    adapter_type: Literal["generic", "iot", "openstack"] = Query(
        "generic",
        description="Selects the input alias adapter. It does not classify `ri_type`.",
    ),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    items = payload if isinstance(payload, list) else [payload]
    statuses = []
    warnings = []
    errors = []
    for index, item in enumerate(items):
        try:
            status = normalise_site_status(item, adapter_type=adapter_type)
        except SiteAdapterValidationError as exc:
            errors.extend(_submission_errors(exc.errors, index if isinstance(payload, list) else None))
            continue
        warning = _warnings(index, status)
        if warning:
            warnings.append(warning)
        statuses.append(status)
    if errors:
        raise HTTPException(status_code=422, detail=errors)

    inserted = 0
    for status in statuses:
        session.add(SiteStatusSnapshot(**status, ingested_at=utc_now()))
        inserted += 1
    session.commit()
    return {"inserted": inserted, "adapter_type": adapter_type, "warnings": warnings}


@app.get("/site-status/latest", tags=["training-data"])
def latest_site_status(site_id: str | None = None, session: Session = Depends(get_db)) -> list[dict[str, Any]]:
    sites = [site_id] if site_id else [
        site for site in session.execute(select(SiteStatusSnapshot.site_id).distinct()).scalars().all() if site
    ]
    rows = []
    for site in sites:
        row = session.execute(
            select(SiteStatusSnapshot)
            .where(SiteStatusSnapshot.site_id == site)
            .order_by(desc(SiteStatusSnapshot.timestamp))
        ).scalars().first()
        if row:
            rows.append(_serialise_row(row))
    return rows


@app.delete("/control/execution-records", tags=["m3l2-ops"])
def delete_execution_records(
    source: str | None = None,
    site_id: str | None = None,
    start_ts: str | None = None,
    end_ts: str | None = None,
    dry_run: bool = True,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    query = select(ExecutionRecord)
    if site_id:
        query = query.where(ExecutionRecord.site_id == site_id)
    start = _parse_optional_ts(start_ts)
    end = _parse_optional_ts(end_ts)
    if start:
        query = query.where(ExecutionRecord.start_ts >= start)
    if end:
        query = query.where(ExecutionRecord.start_ts < end)

    matched = []
    for row in session.execute(query).scalars():
        if source and (row.raw_json or {}).get("source_file") != source:
            continue
        matched.append(row)

    if not dry_run:
        for row in matched:
            session.delete(row)
        session.commit()

    return {"matched": len(matched), "deleted": 0 if dry_run else len(matched), "dry_run": dry_run}


@app.delete("/control/site-status", tags=["m3l2-ops"])
def delete_site_status(
    source: str | None = None,
    site_id: str | None = None,
    start_ts: str | None = None,
    end_ts: str | None = None,
    dry_run: bool = True,
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    query = select(SiteStatusSnapshot)
    if site_id:
        query = query.where(SiteStatusSnapshot.site_id == site_id)
    start = _parse_optional_ts(start_ts)
    end = _parse_optional_ts(end_ts)
    if start:
        query = query.where(SiteStatusSnapshot.timestamp >= start)
    if end:
        query = query.where(SiteStatusSnapshot.timestamp < end)

    matched = []
    for row in session.execute(query).scalars():
        if source and (row.raw_json or {}).get("source_file") != source:
            continue
        matched.append(row)
    if not dry_run:
        for row in matched:
            session.delete(row)
        session.commit()
    return {"matched": len(matched), "deleted": 0 if dry_run else len(matched), "dry_run": dry_run}
