from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
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
)
from m3l2.app.operator_config import config_payload, training_readiness, update_config
from m3l2.app.schemas import IngestRunRequest, PredictRequest, PredictionResponse
from m3l2.auth.router import router as auth_router
from m3l2.inference.forecast_refresh import refresh_forecasts
from m3l2.inference.predict import predict as run_predict, predict_many
from m3l2.ingestion.jobs import run_ingestion
from m3l2.site_adapter.auth import SitePrincipal, current_principal, require_roles
from m3l2.site_adapter.control_plane import router as site_adapter_router
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


def require_same_site_for_ops(principal: SitePrincipal, site_id: str) -> None:
    if principal.site_id != site_id:
        raise HTTPException(status_code=403, detail="JWT site_id does not match requested site_id")


def get_db() -> Session:
    with SessionLocal() as session:
        yield session


def _service_effective_config() -> dict[str, Any]:
    with SessionLocal() as session:
        return config_payload(session)["effective"]


def configure_scheduler_jobs() -> dict[str, Any]:
    global scheduler
    cfg = _service_effective_config()
    if scheduler is None:
        scheduler = BackgroundScheduler(timezone="UTC")

    scheduler.remove_all_jobs()
    if not cfg["automatic_training"]:
        return {"status": "disabled", "automatic_training": False}

    scheduler.add_job(
        _scheduled_cycle,
        "interval",
        hours=int(cfg["training_frequency_hours"]),
        id="m3l2_train",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    scheduler.add_job(
        _scheduled_forecast_refresh,
        "interval",
        minutes=int(cfg["forecast_refresh_minutes"]),
        id="m3l2_forecast_refresh",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    if not scheduler.running:
        scheduler.start()
    return {
        "status": "scheduled",
        "automatic_training": True,
        "training_frequency_hours": cfg["training_frequency_hours"],
        "forecast_refresh_minutes": cfg["forecast_refresh_minutes"],
        "jobs": sorted(job.id for job in scheduler.get_jobs()),
    }


def _scheduled_cycle() -> None:
    logger.info("Starting scheduled M3L2 training cycle")
    try:
        training_summary = train_model(force=False)
        logger.info("Scheduled M3L2 training cycle completed: %s", training_summary)
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
    if settings.enable_scheduler:
        logger.info("M3L2 scheduler configured: %s", configure_scheduler_jobs())
    yield
    if scheduler:
        if scheduler.running:
            scheduler.shutdown(wait=False)
        scheduler = None
        logger.info("Stopped M3L2 scheduler")


app = FastAPI(
    title="GreenDIGIT M3L2 MVP API",
    version="0.1.0",
    lifespan=lifespan,
    description=(
        "M3L2 L2DB, Site Adapter Control Plane, and prediction API.\n\n"
        "**Authentication**\n\n"
        "- Open `/auth/login` to obtain a 24-hour JWT using email and password.\n"
        "- The first login registers a password only if the email is listed in `allowed_emails.txt`.\n"
        "- Use the token as `Authorization: Bearer <token>` on protected L2 endpoints.\n"
        "- JSON token clients can call `POST /auth/token` or `GET /auth/token`.\n\n"
        "**Site telemetry**\n\n"
        "- Authorised publishers submit L2 site snapshots directly to `/l2/sites/{site_id}/snapshots`.\n"
        "- The first accepted snapshot automatically creates the operational site record."
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
            "name": "Auth",
            "description": (
                "EIMPS-style login endpoints. Obtain a 24-hour JWT from `/auth/login` or `/auth/token`, "
                "then use `Authorization: Bearer <token>` on protected L2 endpoints."
            ),
        },
        {
            "name": "l2-site-adapter",
            "description": (
                "L2 Site Adapter Control Plane: push snapshots, read latest L2DB data, "
                "and inspect stored site telemetry sections."
            ),
        },
    ],
    swagger_ui_parameters={"persistAuthorization": True},
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(auth_router)
app.include_router(site_adapter_router)


@app.get("/", include_in_schema=False)
def root() -> HTMLResponse:
    return HTMLResponse(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GreenDIGIT M3L2</title>
    <link rel="stylesheet" href="/static/auth.css">
</head>
<body>
    <main class="auth-shell">
        <section class="auth-panel">
            <img src="/static/cropped-GD_logo.png" alt="GreenDIGIT" class="auth-logo">
            <h1>GreenDIGIT M3L2 API</h1>
            <h2>Operator access</h2>
            <div class="button-row">
                <a class="button-link" href="/auth/login">Login/token</a>
                <a class="button-link secondary" href="/docs">Open API Docs</a>
            </div>
            <div class="info">
                <p>Site telemetry endpoints accept authorised submissions independently from EIMPS.</p>
            </div>
            <footer class="grant-footer">
                <p>This work is funded from the European Union's Horizon Europe research and innovation programme through the <a href="https://greendigit-project.eu/" target="_blank" rel="noopener">GreenDIGIT project</a>, under Grant Agreement No. <a href="https://cordis.europa.eu/project/id/101131207" target="_blank" rel="noopener">101131207</a>.</p>
                <img src="/static/EN-Funded-by-the-EU-POS-2.png" alt="Funded by the European Union">
            </footer>
        </section>
    </main>
</body>
</html>"""
    )


@app.get("/health", tags=["m3l2-ops"])
def health(session: Session = Depends(get_db)) -> dict[str, Any]:
    active = get_active_model(session)
    return {"status": "ok", "db": "ok", "active_model_version": active.version if active else None}


@app.post("/ingest/run", tags=["m3l2-ops"])
def ingest_run(request: IngestRunRequest | None = None) -> dict[str, Any]:
    payload = _schema_dump(request) if request else {}
    return run_ingestion(**payload)


@app.post("/train", tags=["m3l2-ops"])
def train(principal: SitePrincipal = Depends(require_roles("site_admin"))) -> dict[str, Any]:
    return train_model(force=True, triggered_by_email=principal.email)


@app.get("/ops/config/ui", response_class=HTMLResponse, include_in_schema=False)
def operator_config_page() -> HTMLResponse:
    return HTMLResponse(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M3L2 Operator Configuration</title>
    <link rel="stylesheet" href="/static/auth.css">
    <style>
        .config-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
        .config-grid label { display: grid; gap: 6px; color: #243042; font-size: 0.92rem; }
        .config-grid input, .config-grid select { width: 100%; box-sizing: border-box; }
        .inline-check { display: flex; align-items: center; gap: 8px; margin: 10px 0; color: #4b5563; }
        .inline-check input { width: auto; }
        .button-row { display: flex; gap: 10px; flex-wrap: wrap; }
        .button-row button { flex: 1 1 140px; }
        pre { white-space: pre-wrap; word-break: break-word; background: #111827; color: #e5e7eb; padding: 12px; border-radius: 8px; font-size: 0.82rem; }
        .status-line { min-height: 1.4rem; color: #374151; }
        .site-picker { display: grid; gap: 6px; color: #243042; font-size: 0.92rem; margin-bottom: 14px; }
        .logout-button { background: #5f6b63; }
    </style>
</head>
<body>
    <main class="auth-shell">
        <section class="auth-panel auth-panel-wide">
            <img src="/static/cropped-GD_logo.png" alt="GreenDIGIT" class="auth-logo">
            <h1>M3L2 Operator Configuration</h1>
            <dl class="token-meta">
                <div><dt>Email</dt><dd id="user-email">-</dd></div>
                <div><dt>Current site</dt><dd id="user-site">-</dd></div>
                <div><dt>Role</dt><dd id="user-role">-</dd></div>
            </dl>
            <form id="config-form">
                <label class="site-picker">Configuration scope
                    <select id="site-id"></select>
                </label>
                <label class="inline-check"><input type="checkbox" disabled> Connect to EIMPS <span>Not yet available</span></label>
                <div class="config-grid">
                    <label>Automatic training <select id="automatic_training"><option value="true">Enabled</option><option value="false">Manual</option></select></label>
                    <label>Training frequency hours <input id="training_frequency_hours" type="number" min="1" max="168"></label>
                    <label>Training window hours <input id="training_window_hours" type="number" min="1"></label>
                    <label>Minimum usable records <input id="min_usable_records" type="number" min="1"></label>
                    <label>Forecast horizon hours <input id="forecast_horizon_hours" type="number" min="1"></label>
                    <label>Forecast step minutes <input id="forecast_step_minutes" type="number" min="1"></label>
                    <label>Forecast refresh minutes <input id="forecast_refresh_minutes" type="number" min="1"></label>
                    <label>Aggregation interval minutes <input id="aggregation_interval_minutes" type="number" min="1"></label>
                    <label>Submission cadence minutes <input id="submission_cadence_minutes" type="number" min="1"></label>
                    <label>Staleness limit minutes <input id="staleness_limit_minutes" type="number" min="1"></label>
                    <label>Minimum coverage ratio <input id="minimum_coverage_ratio" type="number" min="0" max="1" step="0.01"></label>
                    <label>Model <select id="model_name"><option value="hist_gradient_boosting_mvp">hist_gradient_boosting_mvp</option><option value="xgb" disabled>xgb - Not yet available</option><option value="lstm" disabled>lstm - Not yet available</option></select></label>
                </div>
                <div class="button-row">
                    <button type="button" id="load">Load</button>
                    <button type="submit">Save</button>
                    <button type="button" id="train">Train now</button>
                    <button type="button" id="logout" class="logout-button">Logout</button>
                </div>
            </form>
            <p id="status" class="status-line"></p>
            <h2>Effective Configuration</h2>
            <pre id="effective">{}</pre>
            <h2>Training Readiness</h2>
            <pre id="readiness">{}</pre>
        </section>
    </main>
    <script>
        const fields = ["training_frequency_hours", "training_window_hours", "min_usable_records", "forecast_horizon_hours", "forecast_step_minutes", "forecast_refresh_minutes", "aggregation_interval_minutes", "submission_cadence_minutes", "staleness_limit_minutes", "minimum_coverage_ratio", "model_name", "automatic_training"];
        const status = document.getElementById("status");
        const loginUrl = "/auth/login?next=/ops/config/ui&role=site_admin";
        let principal = null;
        const token = () => localStorage.getItem("m3l2_token") || "";
        const siteId = () => document.getElementById("site-id").value.trim();
        const auth = () => ({Authorization: `Bearer ${token()}`});
        const suffix = () => siteId() ? `?site_id=${encodeURIComponent(siteId())}` : "";
        function setStatus(text) { status.textContent = text; }
        function requireToken() {
            if (!token()) window.location.href = loginUrl;
        }
        function renderPrincipal(user) {
            document.getElementById("user-email").textContent = user.email || "-";
            document.getElementById("user-site").textContent = user.site_id || "-";
            document.getElementById("user-role").textContent = user.role || "-";
            const select = document.getElementById("site-id");
            if (select.options.length) return;
            select.append(new Option("Service defaults", ""));
            for (const entry of user.sites || []) {
                if (!entry.site_id) continue;
                const roles = (entry.roles || []).join(", ");
                const siteName = entry.site_name ? ` - ${entry.site_name}` : "";
                const registration = entry.registered ? "registered" : "not registered";
                const option = new Option(`${entry.site_id}${siteName} (${roles || "no roles"}; ${registration})`, entry.site_id);
                if (entry.site_id !== user.site_id) {
                    option.disabled = true;
                    option.textContent += " - login required";
                }
                select.append(option);
            }
            select.value = user.site_id || "";
        }
        function fill(config) {
            const effective = config.effective || {};
            for (const field of fields) {
                const el = document.getElementById(field);
                if (effective[field] !== undefined) el.value = String(effective[field]);
            }
            document.getElementById("effective").textContent = JSON.stringify(config, null, 2);
        }
        function payload() {
            const out = {};
            for (const field of fields) {
                const el = document.getElementById(field);
                if (field === "automatic_training") out[field] = el.value === "true";
                else if (field === "model_name") out[field] = el.value;
                else if (field === "minimum_coverage_ratio") out[field] = Number(el.value);
                else out[field] = Number.parseInt(el.value, 10);
            }
            return out;
        }
        async function jsonFetch(url, options = {}) {
            requireToken();
            const response = await fetch(url, {...options, headers: {...auth(), "Content-Type": "application/json", ...(options.headers || {})}});
            const body = await response.json();
            if (!response.ok) throw new Error(JSON.stringify(body.detail || body));
            return body;
        }
        async function ensurePrincipal() {
            if (!principal) {
                principal = await jsonFetch("/auth/me");
                renderPrincipal(principal);
            }
            return principal;
        }
        async function loadAll() {
            await ensurePrincipal();
            const config = await jsonFetch(`/ops/config${suffix()}`);
            fill(config);
            const readiness = await jsonFetch(`/ops/training/readiness${suffix()}`);
            document.getElementById("readiness").textContent = JSON.stringify(readiness, null, 2);
            setStatus("Loaded.");
        }
        document.getElementById("load").addEventListener("click", () => loadAll().catch(error => setStatus(error.message)));
        document.getElementById("config-form").addEventListener("submit", async event => {
            event.preventDefault();
            try {
                fill(await jsonFetch(`/ops/config${suffix()}`, {method: "PATCH", body: JSON.stringify(payload())}));
                await loadAll();
                setStatus("Saved. Scheduler changes are immediate; training-quality changes apply on the next run and usually require retraining.");
            } catch (error) { setStatus(error.message); }
        });
        document.getElementById("train").addEventListener("click", async () => {
            try {
                const result = await jsonFetch(`/ops/train${suffix()}`, {method: "POST", body: "{}"});
                document.getElementById("readiness").textContent = JSON.stringify(result, null, 2);
                setStatus("Training request completed.");
            } catch (error) { setStatus(error.message); }
        });
        document.getElementById("site-id").addEventListener("change", () => loadAll().catch(error => setStatus(error.message)));
        document.getElementById("logout").addEventListener("click", () => {
            localStorage.removeItem("m3l2_token");
            localStorage.removeItem("m3l2_principal");
            window.location.href = loginUrl;
        });
        loadAll().catch(error => {
            localStorage.removeItem("m3l2_token");
            localStorage.removeItem("m3l2_principal");
            window.location.href = loginUrl;
        });
    </script>
</body>
</html>"""
    )


@app.get("/ops/config", tags=["m3l2-ops"])
def get_operator_config(
    site_id: str | None = None,
    principal: SitePrincipal = Depends(require_roles("site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    if site_id:
        require_same_site_for_ops(principal, site_id)
    return config_payload(session, site_id=site_id)


@app.patch("/ops/config", tags=["m3l2-ops"])
def patch_operator_config(
    patch: dict[str, Any] = Body(...),
    site_id: str | None = None,
    principal: SitePrincipal = Depends(require_roles("site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    if site_id:
        require_same_site_for_ops(principal, site_id)
    try:
        update_config(session, patch, site_id=site_id, updated_by_email=principal.email)
    except ValueError as exc:
        detail = exc.args[0] if exc.args and isinstance(exc.args[0], list) else str(exc)
        raise HTTPException(status_code=422, detail=detail) from exc
    scheduler_state = configure_scheduler_jobs() if site_id is None else None
    payload = config_payload(session, site_id=site_id)
    payload["scheduler"] = scheduler_state
    return payload


@app.get("/ops/training/readiness", tags=["m3l2-ops"])
def get_training_readiness(
    site_id: str | None = None,
    principal: SitePrincipal = Depends(require_roles("site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    if site_id:
        require_same_site_for_ops(principal, site_id)
    return training_readiness(session, site_id=site_id)


@app.post("/ops/train", tags=["m3l2-ops"])
def train_now(
    site_id: str | None = None,
    principal: SitePrincipal = Depends(require_roles("site_admin")),
) -> dict[str, Any]:
    if site_id:
        require_same_site_for_ops(principal, site_id)
    return train_model(force=True, triggered_by_email=principal.email, site_id=site_id)


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
