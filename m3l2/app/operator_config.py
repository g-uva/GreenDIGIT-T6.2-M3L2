from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from m3l2.app.config import get_settings
from m3l2.app.db import ModelRegistry, OperatorConfig, SiteStatusSnapshot, TrainingRun, utc_now
from m3l2.training.features import STATUS_REQUIRED_INPUTS

SERVICE_KEY = "service"
SITE_KEY_PREFIX = "site:"
IMPLEMENTED_MODEL = "hist_gradient_boosting_mvp"
PLANNED_MODELS = ("xgb", "lstm")
PREDICTION_TARGET = "l2_site_status"
AGGREGATION_DESCRIPTION = "Hourly site-status snapshots aligned to aggregation_interval_minutes buckets."
SUPPORTED_OVERRIDE_FIELDS = {
    "automatic_training",
    "training_frequency_hours",
    "training_window_hours",
    "min_usable_records",
    "forecast_horizon_hours",
    "forecast_step_minutes",
    "forecast_refresh_minutes",
    "model_name",
    "aggregation_interval_minutes",
    "submission_cadence_minutes",
    "staleness_limit_minutes",
    "minimum_coverage_ratio",
}


def service_defaults() -> dict[str, Any]:
    settings = get_settings()
    return {
        "automatic_training": bool(settings.enable_scheduler),
        "training_frequency_hours": int(settings.train_interval_hours),
        "training_window_hours": int(settings.batch_lookback_hours),
        "min_usable_records": int(settings.min_training_records),
        "forecast_horizon_hours": int(settings.forecast_horizon_hours),
        "forecast_step_minutes": int(settings.forecast_step_minutes),
        "forecast_refresh_minutes": int(settings.forecast_refresh_minutes),
        "model_name": IMPLEMENTED_MODEL,
        "aggregation_interval_minutes": 60,
        "submission_cadence_minutes": 60,
        "staleness_limit_minutes": 120,
        "minimum_coverage_ratio": 0.8,
        "eimps_connection_enabled": False,
    }


def model_options() -> list[dict[str, Any]]:
    return [
        {"name": IMPLEMENTED_MODEL, "enabled": True, "label": IMPLEMENTED_MODEL},
        *[
            {"name": name, "enabled": False, "label": f"{name} - Not yet available"}
            for name in PLANNED_MODELS
        ],
    ]


def _site_key(site_id: str) -> str:
    return f"{SITE_KEY_PREFIX}{site_id}"


def _config_key(site_id: str | None = None) -> str:
    return _site_key(site_id) if site_id else SERVICE_KEY


def _row_settings(session: Session, site_id: str | None = None) -> dict[str, Any]:
    row = session.get(OperatorConfig, _config_key(site_id))
    return dict(row.settings or {}) if row else {}


def validate_config_patch(patch: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    unknown = sorted(set(patch) - SUPPORTED_OVERRIDE_FIELDS)
    for field in unknown:
        errors.append({"loc": [field], "msg": "setting is not supported", "type": "value_error.extra"})

    def integer(field: str, minimum: int, maximum: int | None = None) -> None:
        if field not in patch:
            return
        value = patch[field]
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append({"loc": [field], "msg": f"{field} must be an integer", "type": "type_error.integer"})
            return
        if value < minimum or (maximum is not None and value > maximum):
            limit = f" between {minimum} and {maximum}" if maximum is not None else f" greater than or equal to {minimum}"
            errors.append({"loc": [field], "msg": f"{field} must be{limit}", "type": "value_error.range"})
            return
        values[field] = value

    def ratio(field: str) -> None:
        if field not in patch:
            return
        value = patch[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append({"loc": [field], "msg": f"{field} must be a number", "type": "type_error.float"})
            return
        parsed = float(value)
        if parsed < 0 or parsed > 1:
            errors.append({"loc": [field], "msg": f"{field} must be between 0 and 1", "type": "value_error.range"})
            return
        values[field] = parsed

    if "automatic_training" in patch:
        if not isinstance(patch["automatic_training"], bool):
            errors.append({"loc": ["automatic_training"], "msg": "automatic_training must be a boolean", "type": "type_error.bool"})
        else:
            values["automatic_training"] = patch["automatic_training"]
    if "model_name" in patch:
        if patch["model_name"] != IMPLEMENTED_MODEL:
            errors.append({"loc": ["model_name"], "msg": "selected model is not available", "type": "value_error.disabled"})
        else:
            values["model_name"] = patch["model_name"]

    integer("training_frequency_hours", 1, 168)
    integer("training_window_hours", 1, 24 * 365)
    integer("min_usable_records", 1)
    integer("forecast_horizon_hours", 1, 24 * 30)
    integer("forecast_step_minutes", 1, 24 * 60)
    integer("forecast_refresh_minutes", 1, 24 * 60)
    integer("aggregation_interval_minutes", 1, 24 * 60)
    integer("submission_cadence_minutes", 1, 24 * 60)
    integer("staleness_limit_minutes", 1, 24 * 60 * 30)
    ratio("minimum_coverage_ratio")

    if errors:
        raise ValueError(errors)
    return values


def update_config(
    session: Session,
    patch: dict[str, Any],
    *,
    site_id: str | None = None,
    updated_by_email: str | None = None,
) -> OperatorConfig:
    values = validate_config_patch(patch)
    key = _config_key(site_id)
    row = session.get(OperatorConfig, key)
    if row is None:
        row = OperatorConfig(config_key=key, site_id=site_id, settings={}, updated_at=utc_now(), updated_by_email=updated_by_email)
        session.add(row)
    row.settings = {**(row.settings or {}), **values}
    row.updated_at = utc_now()
    row.updated_by_email = updated_by_email
    session.commit()
    session.refresh(row)
    return row


def effective_config(session: Session, site_id: str | None = None) -> dict[str, Any]:
    defaults = service_defaults()
    service = _row_settings(session)
    site = _row_settings(session, site_id) if site_id else {}
    effective = {**defaults, **service, **site}
    effective["eimps_connection_enabled"] = False
    effective["prediction_target"] = PREDICTION_TARGET
    effective["required_inputs"] = STATUS_REQUIRED_INPUTS
    effective["aggregation_interval"] = {
        "minutes": int(effective["aggregation_interval_minutes"]),
        "description": AGGREGATION_DESCRIPTION,
    }
    effective["extensions_training_policy"] = "Stored extensions are excluded from training until explicitly mapped."
    effective["models"] = model_options()
    return effective


def config_payload(session: Session, site_id: str | None = None) -> dict[str, Any]:
    service_row = session.get(OperatorConfig, SERVICE_KEY)
    site_row = session.get(OperatorConfig, _site_key(site_id)) if site_id else None
    return {
        "scope": "site" if site_id else "service",
        "site_id": site_id,
        "defaults": service_defaults(),
        "service_overrides": (service_row.settings or {}) if service_row else {},
        "site_overrides": (site_row.settings or {}) if site_row else {},
        "effective": effective_config(session, site_id),
        "change_effects": {
            "scheduler": "Automatic training frequency changes are applied to the in-process scheduler immediately.",
            "training": "Training window, minimum usable records, model, aggregation and data-quality changes apply to the next Train now or scheduled training run.",
            "forecast": "Forecast horizon and step apply to new forecast refreshes and predictions that use configured defaults.",
            "retraining_required": True,
        },
    }


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def usable_status_rows(session: Session, site_id: str | None = None) -> list[SiteStatusSnapshot]:
    cfg = effective_config(session, site_id)
    query = select(SiteStatusSnapshot)
    if site_id:
        query = query.where(SiteStatusSnapshot.site_id == site_id)
    rows = session.execute(query.order_by(SiteStatusSnapshot.site_id, SiteStatusSnapshot.timestamp)).scalars().all()
    if not rows:
        return []
    latest_ts = max(_to_utc(row.timestamp) for row in rows if row.timestamp)
    start = latest_ts - timedelta(hours=int(cfg["training_window_hours"]))
    window_rows = [row for row in rows if row.timestamp and _to_utc(row.timestamp) >= start]
    stale_ids = _stale_gap_ids(window_rows, int(cfg["staleness_limit_minutes"]))
    return [row for row in window_rows if row.id not in stale_ids and not row.stale_flag and _row_has_required_inputs(row)]


def _row_has_required_inputs(row: SiteStatusSnapshot) -> bool:
    return all(getattr(row, field) is not None for field in STATUS_REQUIRED_INPUTS)


def _missing_required_counts(rows: list[SiteStatusSnapshot]) -> dict[str, int]:
    return {
        field: sum(1 for row in rows if getattr(row, field) is None)
        for field in STATUS_REQUIRED_INPUTS
        if sum(1 for row in rows if getattr(row, field) is None)
    }


def _stale_gap_ids(rows: list[SiteStatusSnapshot], staleness_limit_minutes: int) -> set[int]:
    stale_ids: set[int] = set()
    for _, group in _rows_by_site(rows).items():
        previous: datetime | None = None
        for row in sorted(group, key=lambda item: _to_utc(item.timestamp)):
            current = _to_utc(row.timestamp)
            if previous is not None:
                gap_minutes = (current - previous).total_seconds() / 60.0
                if gap_minutes > staleness_limit_minutes:
                    stale_ids.add(row.id)
            previous = current
    return stale_ids


def _rows_by_site(rows: list[SiteStatusSnapshot]) -> dict[str, list[SiteStatusSnapshot]]:
    grouped: dict[str, list[SiteStatusSnapshot]] = {}
    for row in rows:
        grouped.setdefault(row.site_id or "unknown-site", []).append(row)
    return grouped


def training_readiness(session: Session, site_id: str | None = None) -> dict[str, Any]:
    cfg = effective_config(session, site_id)
    query = select(SiteStatusSnapshot)
    if site_id:
        query = query.where(SiteStatusSnapshot.site_id == site_id)
    all_rows = session.execute(query.order_by(SiteStatusSnapshot.timestamp)).scalars().all()
    if all_rows:
        latest_ts = max(_to_utc(row.timestamp) for row in all_rows if row.timestamp)
        start = latest_ts - timedelta(hours=int(cfg["training_window_hours"]))
        window_rows = [row for row in all_rows if row.timestamp and _to_utc(row.timestamp) >= start]
    else:
        latest_ts = None
        start = None
        window_rows = []
    stale_gap_ids = _stale_gap_ids(window_rows, int(cfg["staleness_limit_minutes"]))
    usable_rows = [
        row
        for row in window_rows
        if row.id not in stale_gap_ids and not row.stale_flag and _row_has_required_inputs(row)
    ]
    usable_count = len(usable_rows)
    expected_records = 0
    first_window_ts = min((_to_utc(row.timestamp) for row in window_rows), default=None)
    for group in _rows_by_site(window_rows).values():
        first_site_ts = min(_to_utc(row.timestamp) for row in group)
        latest_site_ts = max(_to_utc(row.timestamp) for row in group)
        span_minutes = max((latest_site_ts - first_site_ts).total_seconds() / 60.0, 0)
        expected_records += max(1, int(span_minutes // int(cfg["submission_cadence_minutes"])) + 1)
    coverage = min(1.0, usable_count / expected_records) if expected_records else 0.0
    missing_counts = _missing_required_counts(window_rows)
    reasons = []
    if usable_count < int(cfg["min_usable_records"]):
        reasons.append("not_enough_usable_records")
    if coverage < float(cfg["minimum_coverage_ratio"]):
        reasons.append("coverage_below_minimum")
    if missing_counts:
        reasons.append("missing_required_fields")
    if not window_rows:
        reasons.append("no_site_telemetry_in_training_window")
    last_run_query = select(TrainingRun)
    if site_id:
        last_run_query = last_run_query.where(TrainingRun.site_id == site_id)
    last_run = session.execute(last_run_query.order_by(desc(TrainingRun.started_at), desc(TrainingRun.id))).scalars().first()
    active = session.execute(
        select(ModelRegistry).where(ModelRegistry.target == PREDICTION_TARGET, ModelRegistry.active.is_(True))
    ).scalars().first()
    return {
        "available": not reasons,
        "reason": "ready" if not reasons else ", ".join(dict.fromkeys(reasons)),
        "site_id": site_id,
        "target": PREDICTION_TARGET,
        "required_inputs": STATUS_REQUIRED_INPUTS,
        "aggregation_interval_minutes": int(cfg["aggregation_interval_minutes"]),
        "submission_cadence_minutes": int(cfg["submission_cadence_minutes"]),
        "staleness_limit_minutes": int(cfg["staleness_limit_minutes"]),
        "total_records": len(all_rows),
        "window_records": len(window_rows),
        "usable_records": usable_count,
        "expected_records": expected_records,
        "coverage": coverage,
        "minimum_coverage_ratio": float(cfg["minimum_coverage_ratio"]),
        "missing_required_fields": missing_counts,
        "stale_records": sum(1 for row in window_rows if row.stale_flag) + len(stale_gap_ids),
        "training_window_start": start.isoformat() if start else None,
        "observed_window_start": first_window_ts.isoformat() if first_window_ts else None,
        "training_window_end": latest_ts.isoformat() if latest_ts else None,
        "last_run": training_run_payload(last_run) if last_run else None,
        "active_model_version": active.version if active else None,
    }


def training_run_payload(row: TrainingRun) -> dict[str, Any]:
    return {
        "id": row.id,
        "site_id": row.site_id,
        "status": row.status,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "triggered_by_email": row.triggered_by_email,
        "model_version": row.model_version,
        "detail": row.detail or {},
        "error": row.error,
    }
