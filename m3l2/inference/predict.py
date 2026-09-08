from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from m3l2.app.db import ExecutionRecord, SessionLocal, SiteProfile, SiteSnapshot, SiteStatusSnapshot, create_tables, utc_now
from m3l2.app.schemas import PredictRequest
from m3l2.inference.cache import cache_state, get_valid_cache, store_cache
from m3l2.training.registry import get_active_model
from m3l2.training.train import FEATURE_COLUMNS, TARGET

logger = logging.getLogger(__name__)

_LOADED_MODEL: dict[str, Any] = {"version": None, "path": None, "pipeline": None}


def _schema_dump(value: Any, *, exclude_none: bool = False) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=exclude_none)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=exclude_none)
    return value


def _coerce_request(request: PredictRequest | dict[str, Any]) -> PredictRequest:
    if isinstance(request, PredictRequest):
        return request
    return PredictRequest(**request)


def _parse_duration(value: str) -> timedelta:
    match = re.fullmatch(r"(\d+)([hm])", value.strip().lower())
    if not match:
        raise ValueError(f"Unsupported duration: {value}")
    amount = int(match.group(1))
    return timedelta(hours=amount) if match.group(2) == "h" else timedelta(minutes=amount)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _floor_to_step(value: datetime, step: timedelta) -> datetime:
    value = _ensure_utc(value)
    step_seconds = max(int(step.total_seconds()), 1)
    epoch_seconds = int(value.timestamp())
    return datetime.fromtimestamp(epoch_seconds - (epoch_seconds % step_seconds), tz=timezone.utc)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump") or hasattr(value, "dict"):
        return _jsonable(_schema_dump(value, exclude_none=True))
    if isinstance(value, datetime):
        return _ensure_utc(value).isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _quality(freshness: str, coverage: float, metrics: dict[str, Any] | None) -> dict[str, Any]:
    n_train = (metrics or {}).get("n_train") or 0
    if n_train >= 100 and coverage >= 0.9:
        confidence = "high"
    elif n_train >= 20 and coverage >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "forecast_quality": "baseline",
        "freshness": freshness,
        "coverage": coverage,
        "confidence": confidence,
    }


def _latest_context(session: Session, site_id: str) -> ExecutionRecord | None:
    return session.execute(
        select(ExecutionRecord)
        .where(ExecutionRecord.site_id == site_id)
        .order_by(desc(ExecutionRecord.start_ts), desc(ExecutionRecord.id))
    ).scalars().first()


def _latest_site_status(session: Session, site_id: str) -> SiteStatusSnapshot | None:
    return session.execute(
        select(SiteStatusSnapshot)
        .where(SiteStatusSnapshot.site_id == site_id)
        .order_by(desc(SiteStatusSnapshot.timestamp), desc(SiteStatusSnapshot.id))
    ).scalars().first()


def _latest_l2_site_snapshot(session: Session, site_id: str) -> SiteSnapshot | None:
    return session.execute(
        select(SiteSnapshot)
        .where(SiteSnapshot.site_id == site_id)
        .order_by(desc(SiteSnapshot.ts), desc(SiteSnapshot.id))
    ).scalars().first()


def _site_profile(session: Session, site_id: str) -> SiteProfile | None:
    return session.execute(select(SiteProfile).where(SiteProfile.site_id == site_id)).scalar_one_or_none()


def _site_ids(session: Session, requested: list[str] | None) -> list[str]:
    if requested:
        return list(dict.fromkeys(requested))
    sites: set[str] = set()
    for model, column in (
        (ExecutionRecord, ExecutionRecord.site_id),
        (SiteProfile, SiteProfile.site_id),
        (SiteStatusSnapshot, SiteStatusSnapshot.site_id),
        (SiteSnapshot, SiteSnapshot.site_id),
    ):
        sites.update(site for site in session.execute(select(column).distinct()).scalars().all() if site)
    return sorted(sites)


def _duration_s(row: ExecutionRecord | None) -> float:
    if row and row.stop_ts and row.start_ts:
        return max(float((row.stop_ts - row.start_ts).total_seconds()), 0.0)
    return 0.0


def _workload_dict(request: PredictRequest) -> dict[str, Any]:
    return _schema_dump(request.workload, exclude_none=True) if request.workload else {}


def _workload_duration_s(workload: dict[str, Any], context: ExecutionRecord | None) -> float:
    if workload.get("duration_s") is not None:
        return float(workload["duration_s"])
    duration = (workload.get("time_requirements") or {}).get("duration")
    if duration:
        return _parse_duration(duration).total_seconds()
    return _duration_s(context)


def _workload_cpu_hours(workload: dict[str, Any], context: ExecutionRecord | None) -> float | None:
    if workload.get("cpu_hours") is not None:
        return float(workload["cpu_hours"])
    resources = workload.get("resource_requirements") or {}
    duration_s = _workload_duration_s(workload, context)
    if resources.get("cpu") is not None and duration_s > 0:
        return float(resources["cpu"]) * float(resources.get("instances") or 1) * duration_s / 3600.0
    if context and context.work:
        return float(context.work) / 3600.0
    return None


def _workload_gpu_hours(workload: dict[str, Any], context: ExecutionRecord | None) -> float | None:
    if workload.get("gpu_hours") is not None:
        return float(workload["gpu_hours"])
    resources = workload.get("resource_requirements") or {}
    duration_s = _workload_duration_s(workload, context)
    if resources.get("gpu") is not None and duration_s > 0:
        return float(resources["gpu"]) * float(resources.get("instances") or 1) * duration_s / 3600.0
    return None


def _base_feature_row(site_id: str, context: ExecutionRecord | None, ts: datetime, workload: dict[str, Any]) -> dict[str, Any]:
    resources = workload.get("resource_requirements") or {}
    duration_s = _workload_duration_s(workload, context)
    inferred_work = None
    if resources.get("cpu") is not None and duration_s > 0:
        inferred_work = float(resources["cpu"]) * float(resources.get("instances") or 1) * duration_s
    work = workload.get("work", inferred_work if inferred_work is not None else (context.work if context else 0.0))
    return {
        "site_id": site_id,
        "ri_type": workload.get("ri_type", context.ri_type if context else "unknown"),
        "duration_s": float(duration_s),
        "work": float(work or 0.0),
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "records_count_site_24h": 1.0 if context else 0.0,
        "rolling_energy_mean_site_24h": float(context.energy_wh if context and context.energy_wh else 0.0),
        "rolling_work_mean_site_24h": float(context.work if context and context.work else 0.0),
    }


def _capacity(status: SiteStatusSnapshot | None, profile: SiteProfile | None) -> dict[str, Any]:
    return {
        "compute_capacity": profile.compute_capacity if profile else None,
        "gpu_capacity": profile.gpu_capacity if profile else None,
        "storage_capacity": profile.storage_capacity if profile else None,
        "free_cpu_capacity": status.free_cpu_capacity if status else None,
        "free_gpu_capacity": status.free_gpu_capacity if status else None,
        "queue_length": status.queue_length if status else None,
        "provisioning_delay_s": status.provisioning_delay_s if status else None,
    }


def _efficiency(
    context: ExecutionRecord | None,
    status: SiteStatusSnapshot | None,
    workload: dict[str, Any],
    predicted_energy: float,
) -> dict[str, Any]:
    cpu_hours = _workload_cpu_hours(workload, context)
    gpu_hours = _workload_gpu_hours(workload, context)
    energy_per_cpu_hour = None if not cpu_hours else predicted_energy / cpu_hours
    energy_per_gpu_hour = None if not gpu_hours else predicted_energy / gpu_hours
    carbon_g = None
    if status and status.carbon_intensity is not None:
        carbon_g = (predicted_energy / 1000.0) * status.carbon_intensity
    return {
        "energy_per_cpu_hour_wh": energy_per_cpu_hour,
        "energy_per_gpu_hour_wh": energy_per_gpu_hour,
        "workload_class": workload.get("workload_class"),
        "expected_workload_energy_wh": predicted_energy if workload else None,
        "expected_workload_carbon_g": carbon_g if workload else None,
    }


def _status_forecast(status: SiteStatusSnapshot | None, profile: SiteProfile | None, timestamps: list[datetime]) -> list[dict[str, Any]]:
    if status is None:
        return [
            {
                "ts": ts.isoformat(),
                "operational_status": "unknown",
                "availability": None,
                "free_cpu_capacity": profile.compute_capacity if profile else None,
                "free_gpu_capacity": profile.gpu_capacity if profile else None,
                "queue_length": None,
                "provisioning_delay_s": None,
                "maintenance_flag": False,
            }
            for ts in timestamps
        ]
    return [
        {
            "ts": ts.isoformat(),
            "operational_status": status.operational_status,
            "availability": status.node_availability,
            "free_cpu_capacity": status.free_cpu_capacity,
            "free_gpu_capacity": status.free_gpu_capacity,
            "queue_length": status.queue_length,
            "provisioning_delay_s": status.provisioning_delay_s,
            "maintenance_flag": status.maintenance_flag,
            "scheduled_maintenance": status.scheduled_maintenance,
            "load_index": status.load_index,
        }
        for ts in timestamps
    ]


def _latest_l2_status_metadata(session: Session, site_id: str) -> dict[str, Any]:
    snapshot = _latest_l2_site_snapshot(session, site_id)
    if snapshot is None:
        return {"availability": None, "usage": None, "efficiency": None, "snapshot_ts": None}
    return {
        "availability": snapshot.availability or {},
        "usage": snapshot.usage or {},
        "efficiency": snapshot.efficiency or {},
        "snapshot_ts": snapshot.ts.isoformat() if snapshot.ts else None,
    }


def _status_freshness(status: SiteStatusSnapshot | None, generated_at: datetime, step_delta: timedelta) -> dict[str, Any]:
    if status is None or status.timestamp is None:
        return {"site_status": "absent", "site_status_age_s": None, "site_status_stale": True}
    ts = _ensure_utc(status.timestamp)
    age_s = max((generated_at - ts).total_seconds(), 0.0)
    stale = bool(status.stale_flag or age_s > max(step_delta.total_seconds() * 2, 3600.0))
    return {"site_status": "stale" if stale else "fresh", "site_status_age_s": age_s, "site_status_stale": stale}


def _feasibility(status: SiteStatusSnapshot | None, profile: SiteProfile | None, workload: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if status is None:
        reasons.append("missing_site_status")
    else:
        if status.maintenance_flag:
            reasons.append("maintenance_flag")
        if str(status.operational_status or "").upper() not in {"UP", "AVAILABLE", "OK"}:
            reasons.append(f"operational_status_{status.operational_status}")

    resources = workload.get("resource_requirements") or {}
    instances = float(resources.get("instances") or 1)
    requested_cpu = resources.get("cpu")
    requested_gpu = resources.get("gpu")
    requested_storage = resources.get("storage_gb")
    if requested_cpu is not None and status and status.free_cpu_capacity is not None:
        if float(requested_cpu) * instances > float(status.free_cpu_capacity):
            reasons.append("insufficient_free_cpu_capacity")
    if requested_gpu is not None and status and status.free_gpu_capacity is not None:
        if float(requested_gpu) * instances > float(status.free_gpu_capacity):
            reasons.append("insufficient_free_gpu_capacity")
    if requested_storage is not None and profile and profile.storage_capacity is not None:
        if float(requested_storage) * instances > float(profile.storage_capacity):
            reasons.append("insufficient_storage_capacity")

    if not workload and not reasons:
        return {"status": "unknown", "reasons": ["no_workload_requirements"]}
    if status is None:
        return {"status": "unknown", "reasons": reasons}
    return {"status": "infeasible" if reasons else "feasible", "reasons": reasons}


def _site_context_signature(session: Session, site_ids: list[str]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for site_id in sorted(site_ids):
        latest_exec = _latest_context(session, site_id)
        status = _latest_site_status(session, site_id)
        profile = _site_profile(session, site_id)
        snapshot = _latest_l2_site_snapshot(session, site_id)
        context[site_id] = {
            "latest_exec_unit_id": latest_exec.exec_unit_id if latest_exec else None,
            "latest_exec_start_ts": latest_exec.start_ts.isoformat() if latest_exec and latest_exec.start_ts else None,
            "latest_exec_ingested_at": latest_exec.ingested_at.isoformat() if latest_exec and latest_exec.ingested_at else None,
            "profile_updated_at": profile.updated_at.isoformat() if profile and profile.updated_at else None,
            "site_status_ts": status.timestamp.isoformat() if status and status.timestamp else None,
            "site_status_stale_flag": status.stale_flag if status else None,
            "l2_snapshot_ts": snapshot.ts.isoformat() if snapshot and snapshot.ts else None,
        }
    return context


def request_signature(
    request: PredictRequest,
    site_ids: list[str],
    forecast_start: datetime,
    target: str,
    session: Session,
) -> str:
    payload = {
        "candidate_site_ids": sorted(site_ids),
        "forecast_start_time": forecast_start.isoformat(),
        "horizon": request.horizon,
        "step": request.step,
        "target": target,
        "workload": _workload_dict(request),
        "site_context": _site_context_signature(session, site_ids),
    }
    return hashlib.sha256(json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _load_pipeline(model_row: Any) -> Any:
    if (
        _LOADED_MODEL["version"] == model_row.version
        and _LOADED_MODEL["path"] == model_row.path
        and _LOADED_MODEL["pipeline"] is not None
    ):
        return _LOADED_MODEL["pipeline"]
    model_bundle = joblib.load(model_row.path)
    pipeline = model_bundle["pipeline"] if isinstance(model_bundle, dict) else model_bundle
    _LOADED_MODEL.update({"version": model_row.version, "path": model_row.path, "pipeline": pipeline})
    return pipeline


def _result_from_forecast(
    session: Session,
    site_id: str,
    forecast_rows: list[dict[str, Any]],
    request: PredictRequest,
    generated_at: datetime,
    valid_until: datetime,
    timestamps: list[datetime],
    model_metrics: dict[str, Any] | None,
    cache_status: str,
    signature: str,
) -> dict[str, Any]:
    context = _latest_context(session, site_id)
    status = _latest_site_status(session, site_id)
    profile = _site_profile(session, site_id)
    workload = _workload_dict(request)
    warnings: list[str] = []
    if context is None:
        warnings.append("missing_execution_history")
    freshness = _status_freshness(status, generated_at, _parse_duration(request.step))
    if freshness["site_status_stale"]:
        warnings.append("stale_or_missing_site_status")

    forecast = [{**point, "unit": point.get("unit", "Wh")} for point in forecast_rows]
    predicted_total = float(sum(point["value"] for point in forecast))
    quality = _quality(cache_status, 1.0 if context else 0.0, model_metrics)
    estimates = _efficiency(context, status, workload, predicted_total)
    cache_info = {
        "status": cache_status,
        "request_signature": signature,
        "valid_until": valid_until.isoformat(),
    }
    return {
        "site_id": site_id,
        "target": TARGET,
        "forecast": forecast,
        "energy_forecast": forecast,
        "site_status_forecast": _status_forecast(status, profile, timestamps),
        "latest_site_status": _latest_l2_status_metadata(session, site_id),
        "capacity": _capacity(status, profile),
        "feasibility": _feasibility(status, profile, workload),
        "workload_estimates": estimates,
        "efficiency": estimates,
        "quality": quality,
        "freshness": freshness,
        "warnings": warnings,
        "cache": cache_info,
    }


def _response(
    request: PredictRequest,
    model_row: Any,
    generated_at: datetime,
    valid_until: datetime,
    forecast_start: datetime,
    horizon: str,
    step: str,
    results: list[dict[str, Any]],
    signature: str,
    warnings: list[str],
    cache_status: str,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "request_id": request.request_id,
        "prediction_id": str(uuid4()),
        "generated_at": generated_at.isoformat(),
        "created_at": generated_at.isoformat(),
        "valid_until": valid_until.isoformat(),
        "model_name": model_row.model_name,
        "model_version": model_row.version,
        "target": TARGET,
        "forecast_start_time": forecast_start.isoformat(),
        "horizon": horizon,
        "step": step,
        "results": results,
        "predictions": results,
        "warnings": warnings,
        "cache": {"status": cache_status, "request_signature": signature},
    }


def _predict_with_session(request: PredictRequest | dict[str, Any], session: Session) -> dict[str, Any]:
    request = _coerce_request(request)
    horizon_delta = _parse_duration(request.horizon)
    step_delta = _parse_duration(request.step)
    if step_delta.total_seconds() <= 0 or horizon_delta.total_seconds() <= 0:
        raise ValueError("horizon and step must be positive")
    if step_delta > horizon_delta:
        raise ValueError("step must not be longer than horizon")

    model_row = get_active_model(session, TARGET)
    if model_row is None:
        return {"status": "no_active_model", "detail": "No active energy_wh model is registered."}

    generated_at = utc_now()
    forecast_start = _ensure_utc(request.forecast_start_time) if request.forecast_start_time else _floor_to_step(generated_at, step_delta)
    valid_until = forecast_start + step_delta
    sites = _site_ids(session, request.candidate_site_ids)
    warnings: list[str] = []
    if not sites:
        warnings.append("no_candidate_sites")

    signature = request_signature(request, sites, forecast_start, TARGET, session)
    n_steps = int(horizon_delta.total_seconds() // step_delta.total_seconds())
    timestamps = [forecast_start + (step_delta * idx) for idx in range(1, n_steps + 1)]
    use_cache = bool(request.cache.use_cache)

    if use_cache and sites:
        cached_results = []
        cached_valid_until: list[datetime] = []
        cache_states = []
        for site_id in sites:
            state = cache_state(session, site_id, TARGET, request.horizon, request.step, model_row.version, signature)
            cache_states.append(state)
            cached = get_valid_cache(session, site_id, TARGET, request.horizon, request.step, model_row.version, signature)
            if cached is None:
                cached_results = []
                break
            cached_results.append(
                _result_from_forecast(
                    session,
                    site_id,
                    cached.predictions,
                    request,
                    generated_at,
                    cached.valid_until,
                    timestamps,
                    model_row.metrics,
                    "cached",
                    signature,
                )
            )
            cached_valid_until.append(cached.valid_until)
        if cached_results and len(cached_results) == len(sites):
            return _response(
                request,
                model_row,
                generated_at,
                min(cached_valid_until),
                forecast_start,
                request.horizon,
                request.step,
                cached_results,
                signature,
                warnings,
                "cached",
            )
        if "stale" in cache_states:
            warnings.append("cached_forecast_stale_refreshed")
        elif "absent" in cache_states:
            warnings.append("cached_forecast_absent_refreshed")

    pipeline = _load_pipeline(model_row)
    contexts = {site_id: _latest_context(session, site_id) for site_id in sites}
    feature_rows: list[dict[str, Any]] = []
    row_keys: list[tuple[str, datetime]] = []
    workload = _workload_dict(request)
    for site_id in sites:
        for ts in timestamps:
            row_keys.append((site_id, ts))
            feature_rows.append(_base_feature_row(site_id, contexts[site_id], ts, workload))

    predicted_by_site: dict[str, list[dict[str, Any]]] = {site_id: [] for site_id in sites}
    if feature_rows:
        frame = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        values = pipeline.predict(frame)
        for (site_id, ts), value in zip(row_keys, values):
            predicted_by_site[site_id].append({"ts": ts.isoformat(), "value": max(float(value), 0.0), "unit": "Wh"})

    results = []
    for site_id in sites:
        forecast_rows = predicted_by_site[site_id]
        quality = _quality("fresh", 1.0 if contexts[site_id] else 0.0, model_row.metrics)
        store_cache(
            session,
            site_id,
            TARGET,
            request.horizon,
            request.step,
            model_row.version,
            signature,
            forecast_rows,
            quality,
            generated_at,
            valid_until,
            forecast_start,
        )
        results.append(
            _result_from_forecast(
                session,
                site_id,
                forecast_rows,
                request,
                generated_at,
                valid_until,
                timestamps,
                model_row.metrics,
                "fresh",
                signature,
            )
        )
    session.commit()

    return _response(
        request,
        model_row,
        generated_at,
        valid_until,
        forecast_start,
        request.horizon,
        request.step,
        results,
        signature,
        warnings,
        "fresh",
    )


def predict(request: PredictRequest | dict[str, Any]) -> dict[str, Any]:
    create_tables()
    with SessionLocal() as session:
        return _predict_with_session(request, session)


def predict_many(requests: list[PredictRequest | dict[str, Any]]) -> list[dict[str, Any]]:
    create_tables()
    with SessionLocal() as session:
        return [_predict_with_session(request, session) for request in requests]
