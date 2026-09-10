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

from m3l2.app.db import (
    ExecutionRecord,
    RegisteredSite,
    SessionLocal,
    SiteProfile,
    SiteSnapshot,
    SiteStatusSnapshot,
    create_tables,
    utc_now,
)
from m3l2.app.schemas import PredictRequest
from m3l2.inference.cache import cache_state, get_valid_cache, store_cache
from m3l2.training.registry import get_active_model
from m3l2.training.train import FEATURE_COLUMNS, TARGET, TARGET_COLUMNS

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


def _registered_site_maps(session: Session) -> tuple[dict[str, str], dict[str, str]]:
    public_to_training: dict[str, str] = {}
    training_to_public: dict[str, str] = {}
    for row in session.execute(select(RegisteredSite)).scalars().all():
        metadata = row.site_metadata or {}
        training_site_id = metadata.get("execution_records_site_id") or row.site_id
        public_to_training[row.site_id] = training_site_id
        training_to_public.setdefault(training_site_id, row.site_id)
    return public_to_training, training_to_public


def _known_site_ids(session: Session) -> set[str]:
    sites: set[str] = set()
    for model, column in (
        (ExecutionRecord, ExecutionRecord.site_id),
        (SiteProfile, SiteProfile.site_id),
        (SiteStatusSnapshot, SiteStatusSnapshot.site_id),
        (SiteSnapshot, SiteSnapshot.site_id),
    ):
        sites.update(site for site in session.execute(select(column).distinct()).scalars().all() if site)
    return sites


def _resolve_candidate_sites(session: Session, requested: list[str] | None) -> tuple[list[dict[str, str | None]], list[str]]:
    public_to_training, training_to_public = _registered_site_maps(session)
    known_sites = _known_site_ids(session)
    candidates = list(dict.fromkeys(requested or sorted(set(public_to_training) | known_sites)))
    resolved: list[dict[str, str | None]] = []
    missing: list[str] = []
    seen_training: set[str] = set()

    for candidate in candidates:
        training_site_id = public_to_training.get(candidate, candidate)
        registered_site_id = candidate if candidate in public_to_training else training_to_public.get(candidate)
        is_known = candidate in public_to_training or training_site_id in known_sites or candidate in known_sites
        if not is_known:
            missing.append(candidate)
            continue
        if training_site_id in seen_training:
            continue
        seen_training.add(training_site_id)
        if candidate in public_to_training:
            response_site_id = candidate
            resolution = "registered_site_mapping" if training_site_id != candidate else "registered_site_direct"
        else:
            response_site_id = candidate
            resolution = "execution_records_site_id" if registered_site_id else "direct_site_id"
        resolved.append(
            {
                "site_id": response_site_id,
                "training_site_id": training_site_id,
                "registered_site_id": registered_site_id,
                "requested_site_id": candidate,
                "site_id_resolution": resolution,
            }
        )
    return resolved, missing


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


def _base_l2_feature_row(site_id: str, status: SiteStatusSnapshot | None, profile: SiteProfile | None, ts: datetime) -> dict[str, Any]:
    availability = status.node_availability if status and status.node_availability is not None else 1.0
    free_cpu = status.free_cpu_capacity if status and status.free_cpu_capacity is not None else (profile.compute_capacity if profile else 0.0)
    queue_length = status.queue_length if status and status.queue_length is not None else 0
    load_index = status.load_index if status and status.load_index is not None else 0.0
    return {
        "site_id": site_id,
        "ri_type": (status.ri_type if status else None) or (profile.ri_type if profile else "unknown"),
        "timestamp": ts,
        "hour": ts.hour,
        "day_of_week": ts.weekday(),
        "records_count_site_24h": 1.0 if status else 0.0,
        "rolling_availability_mean_site_24h": float(availability),
        "rolling_free_cpu_mean_site_24h": float(free_cpu or 0.0),
        "rolling_queue_mean_site_24h": float(queue_length or 0),
        "rolling_load_mean_site_24h": float(load_index or 0.0),
    }


def _clip(value: Any, lower: float | None = None, upper: float | None = None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if lower is not None:
        parsed = max(parsed, lower)
    if upper is not None:
        parsed = min(parsed, upper)
    return parsed


def _payload_number(payload: dict[str, Any] | None, names: tuple[str, ...]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _submitted_capacity(snapshot: SiteSnapshot | None) -> dict[str, float | None]:
    if snapshot is None:
        return {}
    payload = snapshot.raw_json or {}
    capabilities = snapshot.capabilities or payload.get("capabilities")
    return {
        "compute_capacity": _first_known(
            _payload_number(capabilities, ("compute_capacity", "total_vcpus", "vcpus_total", "cpu_capacity")),
            _payload_number(payload, ("compute_capacity", "total_vcpus", "vcpus_total", "cpu_capacity")),
        ),
        "gpu_capacity": _first_known(
            _payload_number(capabilities, ("gpu_capacity", "total_gpus", "gpus_total")),
            _payload_number(payload, ("gpu_capacity", "total_gpus", "gpus_total")),
        ),
        "storage_capacity": _first_known(
            _payload_number(capabilities, ("storage_capacity", "storage_gb", "total_disk_gb", "disk_gb_total")),
            _payload_number(payload, ("storage_capacity", "storage_gb", "total_disk_gb", "disk_gb_total")),
        ),
    }


def _first_known(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _predicted_operational_status(score: float | None, maintenance_flag: bool, availability: float | None) -> str:
    if maintenance_flag:
        return "MAINTENANCE"
    score = 1.0 if score is None else score
    if score < 0.25 or (availability is not None and availability < 0.25):
        return "DOWN"
    if score < 0.75 or (availability is not None and availability < 0.95):
        return "DEGRADED"
    return "UP"


def _status_prediction_row(
    ts: datetime,
    values: Any,
    status: SiteStatusSnapshot | None,
    profile: SiteProfile | None,
    submitted_capacity: dict[str, float | None] | None = None,
    target_columns: list[str] | None = None,
) -> dict[str, Any]:
    targets = target_columns or TARGET_COLUMNS
    raw = dict(zip(targets, values))
    submitted_capacity = submitted_capacity or {}
    maintenance_flag = bool((_clip(raw.get("maintenance_flag"), 0.0, 1.0) or 0.0) >= 0.5)
    availability = _clip(raw.get("node_availability"), 0.0, 1.0)
    compute_capacity = _clip(raw.get("compute_capacity"), 0.0, None)
    gpu_capacity = _clip(raw.get("gpu_capacity"), 0.0, None)
    storage_capacity = _clip(raw.get("storage_capacity"), 0.0, None)
    free_cpu = _clip(raw.get("free_cpu_capacity"), 0.0, None)
    free_gpu = _clip(raw.get("free_gpu_capacity"), 0.0, None)
    queue_length = int(round(_clip(raw.get("queue_length"), 0.0, None) or 0.0))
    provisioning_delay = _clip(raw.get("provisioning_delay_s"), 0.0, None)
    load_index = _clip(raw.get("load_index"), 0.0, 1.0)
    return {
        "ts": ts.isoformat(),
        "operational_status": _predicted_operational_status(
            _clip(raw.get("operational_status_score"), 0.0, 1.0),
            maintenance_flag,
            availability,
        ),
        "availability": availability,
        "node_availability": availability,
        "link_availability": _clip(raw.get("link_availability"), 0.0, 1.0),
        "compute_capacity": _first_known(
            compute_capacity,
            profile.compute_capacity if profile else None,
            submitted_capacity.get("compute_capacity"),
        ),
        "gpu_capacity": _first_known(
            gpu_capacity,
            profile.gpu_capacity if profile else None,
            submitted_capacity.get("gpu_capacity"),
        ),
        "storage_capacity": _first_known(
            storage_capacity,
            profile.storage_capacity if profile else None,
            submitted_capacity.get("storage_capacity"),
        ),
        "free_cpu_capacity": free_cpu if free_cpu is not None else (profile.compute_capacity if profile else None),
        "free_gpu_capacity": free_gpu if free_gpu is not None else (profile.gpu_capacity if profile else None),
        "queue_length": queue_length,
        "provisioning_delay_s": provisioning_delay,
        "maintenance_flag": maintenance_flag,
        "scheduled_maintenance": status.scheduled_maintenance if status else None,
        "load_index": load_index,
        "inference_source": "model",
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


def _capacity_from_status_forecast(
    status_forecast: list[dict[str, Any]],
    status: SiteStatusSnapshot | None,
    profile: SiteProfile | None,
    submitted_capacity: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    first = status_forecast[0] if status_forecast else {}
    submitted_capacity = submitted_capacity or {}
    return {
        "compute_capacity": _first_known(
            first.get("compute_capacity"),
            profile.compute_capacity if profile else None,
            submitted_capacity.get("compute_capacity"),
        ),
        "gpu_capacity": _first_known(
            first.get("gpu_capacity"),
            profile.gpu_capacity if profile else None,
            submitted_capacity.get("gpu_capacity"),
        ),
        "storage_capacity": _first_known(
            first.get("storage_capacity"),
            profile.storage_capacity if profile else None,
            submitted_capacity.get("storage_capacity"),
        ),
        "free_cpu_capacity": _first_known(first.get("free_cpu_capacity"), status.free_cpu_capacity if status else None),
        "free_gpu_capacity": _first_known(first.get("free_gpu_capacity"), status.free_gpu_capacity if status else None),
        "queue_length": _first_known(first.get("queue_length"), status.queue_length if status else None),
        "provisioning_delay_s": _first_known(
            first.get("provisioning_delay_s"),
            status.provisioning_delay_s if status else None,
        ),
    }


def _fill_forecast_capacity(
    status_forecast: list[dict[str, Any]],
    profile: SiteProfile | None,
    submitted_capacity: dict[str, float | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in status_forecast:
        enriched = dict(row)
        enriched["compute_capacity"] = _first_known(
            enriched.get("compute_capacity"),
            profile.compute_capacity if profile else None,
            submitted_capacity.get("compute_capacity"),
        )
        enriched["gpu_capacity"] = _first_known(
            enriched.get("gpu_capacity"),
            profile.gpu_capacity if profile else None,
            submitted_capacity.get("gpu_capacity"),
        )
        enriched["storage_capacity"] = _first_known(
            enriched.get("storage_capacity"),
            profile.storage_capacity if profile else None,
            submitted_capacity.get("storage_capacity"),
        )
        rows.append(enriched)
    return rows


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
                "compute_capacity": profile.compute_capacity if profile else None,
                "gpu_capacity": profile.gpu_capacity if profile else None,
                "storage_capacity": profile.storage_capacity if profile else None,
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
            "compute_capacity": profile.compute_capacity if profile else None,
            "gpu_capacity": profile.gpu_capacity if profile else None,
            "storage_capacity": profile.storage_capacity if profile else None,
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


def _feasibility_from_status_forecast(
    status_forecast: list[dict[str, Any]],
    profile: SiteProfile | None,
    workload: dict[str, Any],
) -> dict[str, Any]:
    if not status_forecast:
        return {"status": "unknown", "reasons": ["missing_site_status_forecast"]}
    first = status_forecast[0]
    reasons: list[str] = []
    if first.get("maintenance_flag"):
        reasons.append("predicted_maintenance_flag")
    operational_status = str(first.get("operational_status") or "").upper()
    if operational_status not in {"UP", "AVAILABLE", "OK"}:
        reasons.append(f"predicted_operational_status_{operational_status or 'UNKNOWN'}")

    resources = workload.get("resource_requirements") or {}
    instances = float(resources.get("instances") or 1)
    requested_cpu = resources.get("cpu")
    requested_gpu = resources.get("gpu")
    requested_storage = resources.get("storage_gb")
    free_cpu = first.get("free_cpu_capacity")
    free_gpu = first.get("free_gpu_capacity")
    storage_capacity = first.get("storage_capacity", profile.storage_capacity if profile else None)
    if requested_cpu is not None and free_cpu is not None:
        if float(requested_cpu) * instances > float(free_cpu):
            reasons.append("predicted_insufficient_free_cpu_capacity")
    if requested_gpu is not None and free_gpu is not None:
        if float(requested_gpu) * instances > float(free_gpu):
            reasons.append("predicted_insufficient_free_gpu_capacity")
    if requested_storage is not None and storage_capacity is not None:
        if float(requested_storage) * instances > float(storage_capacity):
            reasons.append("predicted_insufficient_storage_capacity")

    if not workload and not reasons:
        return {"status": "unknown", "reasons": ["no_workload_requirements"]}
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
    site: dict[str, str | None],
    forecast_rows: list[dict[str, Any]],
    request: PredictRequest,
    generated_at: datetime,
    valid_until: datetime,
    timestamps: list[datetime],
    model_metrics: dict[str, Any] | None,
    cache_status: str,
    signature: str,
) -> dict[str, Any]:
    site_id = site["site_id"] or site["training_site_id"]
    training_site_id = site["training_site_id"] or site_id
    context = _latest_context(session, training_site_id)
    status = _latest_site_status(session, training_site_id)
    profile = _site_profile(session, training_site_id)
    submitted_capacity = _submitted_capacity(_latest_l2_site_snapshot(session, training_site_id))
    workload = _workload_dict(request)
    warnings: list[str] = []
    freshness = _status_freshness(status, generated_at, _parse_duration(request.step))
    if freshness["site_status_stale"]:
        warnings.append("stale_or_missing_site_status")

    status_forecast = _fill_forecast_capacity(forecast_rows, profile, submitted_capacity)
    forecast = [
        {
            "ts": point["ts"],
            "value": float(point["availability"]) if point.get("availability") is not None else 0.0,
            "unit": "ratio",
        }
        for point in status_forecast
    ]
    quality = _quality(cache_status, 1.0 if status else 0.0, model_metrics)
    estimates: dict[str, Any] = {}
    cache_info = {
        "status": cache_status,
        "request_signature": signature,
        "valid_until": valid_until.isoformat(),
    }
    return {
        "site_id": site_id,
        "training_site_id": training_site_id,
        "registered_site_id": site.get("registered_site_id"),
        "requested_site_id": site.get("requested_site_id"),
        "site_id_resolution": site.get("site_id_resolution"),
        "target": TARGET,
        "forecast": forecast,
        "energy_forecast": [],
        "site_status_forecast": status_forecast,
        "latest_site_status": _latest_l2_status_metadata(session, site_id) if site_id else None,
        "capacity": _capacity_from_status_forecast(status_forecast, status, profile, submitted_capacity),
        "feasibility": _feasibility_from_status_forecast(status_forecast, profile, workload),
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
        return {"status": "no_active_model", "detail": "No active l2_site_status model is registered."}

    generated_at = utc_now()
    forecast_start = _ensure_utc(request.forecast_start_time) if request.forecast_start_time else _floor_to_step(generated_at, step_delta)
    valid_until = forecast_start + step_delta
    sites, missing_sites = _resolve_candidate_sites(session, request.candidate_site_ids)
    training_site_ids = [site["training_site_id"] for site in sites if site["training_site_id"]]
    warnings: list[str] = []
    if missing_sites:
        warnings.append(f"candidate_sites_not_found:{','.join(missing_sites)}")
    if request.candidate_site_ids and not sites:
        return {
            "status": "candidate_sites_not_found",
            "detail": "None of the requested candidate sites could be resolved to registered or training-compatible L2DB data.",
            "missing_site_ids": missing_sites or request.candidate_site_ids,
        }
    if not sites:
        warnings.append("no_candidate_sites")

    signature = request_signature(request, training_site_ids, forecast_start, TARGET, session)
    n_steps = int(horizon_delta.total_seconds() // step_delta.total_seconds())
    timestamps = [forecast_start + (step_delta * idx) for idx in range(1, n_steps + 1)]
    use_cache = bool(request.cache.use_cache)

    if use_cache and sites:
        cached_results = []
        cached_valid_until: list[datetime] = []
        cache_states = []
        for site in sites:
            site_id = site["training_site_id"]
            state = cache_state(session, site_id, TARGET, request.horizon, request.step, model_row.version, signature)
            cache_states.append(state)
            cached = get_valid_cache(session, site_id, TARGET, request.horizon, request.step, model_row.version, signature)
            if cached is None:
                cached_results = []
                break
            cached_results.append(
                _result_from_forecast(
                    session,
                    site,
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
    target_columns = (model_row.feature_schema or {}).get("targets") or TARGET_COLUMNS
    statuses = {site["training_site_id"]: _latest_site_status(session, site["training_site_id"]) for site in sites}
    profiles = {site["training_site_id"]: _site_profile(session, site["training_site_id"]) for site in sites}
    submitted_capacities = {
        site["training_site_id"]: _submitted_capacity(_latest_l2_site_snapshot(session, site["training_site_id"]))
        for site in sites
    }
    feature_rows: list[dict[str, Any]] = []
    row_keys: list[tuple[str, datetime]] = []
    for site in sites:
        site_id = site["training_site_id"]
        for ts in timestamps:
            row_keys.append((site_id, ts))
            feature_rows.append(_base_l2_feature_row(site_id, statuses[site_id], profiles[site_id], ts))

    predicted_by_site: dict[str, list[dict[str, Any]]] = {site["training_site_id"]: [] for site in sites}
    if feature_rows:
        frame = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
        values = pipeline.predict(frame)
        for (site_id, ts), value in zip(row_keys, values):
            predicted_by_site[site_id].append(
                _status_prediction_row(
                    ts,
                    value,
                    statuses[site_id],
                    profiles[site_id],
                    submitted_capacities[site_id],
                    target_columns,
                )
            )

    results = []
    for site in sites:
        site_id = site["training_site_id"]
        forecast_rows = predicted_by_site[site_id]
        quality = _quality("fresh", 1.0 if statuses[site_id] else 0.0, model_row.metrics)
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
                site,
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
