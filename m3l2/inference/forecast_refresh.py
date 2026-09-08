from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from sqlalchemy import select

from m3l2.app.config import get_settings
from m3l2.app.db import ExecutionRecord, SessionLocal, SiteProfile, SiteSnapshot, SiteStatusSnapshot, create_tables, utc_now
from m3l2.app.schemas import PredictRequest
from m3l2.inference.predict import predict

logger = logging.getLogger(__name__)


def _known_site_ids() -> list[str]:
    sites: set[str] = set()
    with SessionLocal() as session:
        for column in (
            ExecutionRecord.site_id,
            SiteProfile.site_id,
            SiteStatusSnapshot.site_id,
            SiteSnapshot.site_id,
        ):
            sites.update(site for site in session.execute(select(column).distinct()).scalars().all() if site)
    return sorted(sites)


def refresh_forecasts(site_ids: list[str] | None = None, force: bool = True) -> dict[str, Any]:
    create_tables()
    settings = get_settings()
    sites = list(dict.fromkeys(site_ids or _known_site_ids()))
    if not sites:
        return {"status": "no_sites", "refreshed": 0}

    step_minutes = max(int(settings.forecast_step_minutes), 1)
    forecast_start = utc_now().replace(second=0, microsecond=0)
    forecast_start = forecast_start - timedelta(minutes=forecast_start.minute % step_minutes)
    request = PredictRequest(
        request_id=f"forecast-refresh-{forecast_start.isoformat()}",
        candidate_site_ids=sites,
        forecast_start_time=forecast_start,
        horizon=f"{settings.forecast_horizon_hours}h",
        step=f"{settings.forecast_step_minutes}m",
        cache={"use_cache": not force},
        include_site_status=True,
    )
    result = predict(request)
    if result.get("status") == "no_active_model":
        return {"status": "no_active_model", "refreshed": 0, "detail": result.get("detail")}
    refreshed = len(result.get("results") or result.get("predictions") or [])
    logger.info("Forecast refresh completed for %s sites", refreshed)
    return {
        "status": "refreshed",
        "refreshed": refreshed,
        "model_version": result.get("model_version"),
        "forecast_start_time": result.get("forecast_start_time"),
        "valid_until": result.get("valid_until"),
    }
