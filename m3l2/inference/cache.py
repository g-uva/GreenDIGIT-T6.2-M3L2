from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from m3l2.app.db import ForecastCache, utc_now


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def get_valid_cache(
    session: Session,
    site_id: str,
    target: str,
    horizon: str,
    step: str,
    model_version: str,
    request_signature: str,
) -> ForecastCache | None:
    return session.execute(
        select(ForecastCache)
        .where(
            ForecastCache.site_id == site_id,
            ForecastCache.target == target,
            ForecastCache.horizon == horizon,
            ForecastCache.step == step,
            ForecastCache.model_version == model_version,
            ForecastCache.request_signature == request_signature,
            ForecastCache.valid_until > utc_now(),
        )
        .order_by(ForecastCache.created_at.desc())
    ).scalars().first()


def get_latest_cache(
    session: Session,
    site_id: str,
    target: str,
    horizon: str,
    step: str,
    model_version: str,
    request_signature: str,
) -> ForecastCache | None:
    return session.execute(
        select(ForecastCache)
        .where(
            ForecastCache.site_id == site_id,
            ForecastCache.target == target,
            ForecastCache.horizon == horizon,
            ForecastCache.step == step,
            ForecastCache.model_version == model_version,
            ForecastCache.request_signature == request_signature,
        )
        .order_by(ForecastCache.created_at.desc())
    ).scalars().first()


def store_cache(
    session: Session,
    site_id: str,
    target: str,
    horizon: str,
    step: str,
    model_version: str,
    request_signature: str,
    predictions: list[dict[str, Any]],
    quality: dict[str, Any],
    created_at: datetime,
    valid_until: datetime,
    forecast_start_ts: datetime,
) -> ForecastCache:
    row = session.execute(
        select(ForecastCache).where(
            ForecastCache.site_id == site_id,
            ForecastCache.target == target,
            ForecastCache.horizon == horizon,
            ForecastCache.step == step,
            ForecastCache.model_version == model_version,
            ForecastCache.request_signature == request_signature,
        )
    ).scalars().first()
    if row is None:
        row = ForecastCache(
            site_id=site_id,
            target=target,
            horizon=horizon,
            step=step,
            model_version=model_version,
            request_signature=request_signature,
            created_at=created_at,
            valid_until=valid_until,
            forecast_start_ts=forecast_start_ts,
            predictions=predictions,
            quality=quality,
        )
        session.add(row)
        return row

    row.created_at = created_at
    row.valid_until = valid_until
    row.forecast_start_ts = forecast_start_ts
    row.predictions = predictions
    row.quality = quality
    return row


def cache_state(
    session: Session,
    site_id: str,
    target: str,
    horizon: str,
    step: str,
    model_version: str,
    request_signature: str,
) -> str:
    latest = get_latest_cache(
        session,
        site_id,
        target,
        horizon,
        step,
        model_version,
        request_signature,
    )
    if latest is None:
        return "absent"
    return "fresh" if _ensure_utc(latest.valid_until) > utc_now() else "stale"
