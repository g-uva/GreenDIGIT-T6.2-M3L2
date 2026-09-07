from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from m3l2.app.config import get_settings
from m3l2.app.db import ExecutionRecord, RegisteredSite, SessionLocal, SiteSnapshot, utc_now
from m3l2.site_adapter.auth import SitePrincipal, require_roles, require_same_site
from m3l2.site_adapter.client import SiteAdapterClient
from m3l2.site_adapter.schemas import SiteRegistrationRequest, SiteSnapshotIn, WorkloadSubmissionRequest

router = APIRouter(prefix="/l2/sites", tags=["l2-site-adapter"])


def get_db() -> Session:
    with SessionLocal() as session:
        yield session


def model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        data = model.model_dump()
    else:
        data = model.dict()
    return jsonable_encoder(data)


def _to_utc(value: datetime | None = None) -> datetime:
    value = value or utc_now()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _dt(value: datetime | None) -> str | None:
    return _to_utc(value).isoformat() if value else None


def _site_to_dict(site: RegisteredSite) -> dict[str, Any]:
    return {
        "id": site.id,
        "site_id": site.site_id,
        "site_name": site.site_name,
        "ri_type": site.ri_type,
        "adapter_base_url": site.adapter_base_url,
        "contact_email": site.contact_email,
        "auth_type": site.auth_type,
        "auth_config": site.auth_config or {},
        "enabled": site.enabled,
        "registered_at": _dt(site.registered_at),
        "last_seen_at": _dt(site.last_seen_at),
        "metadata": site.site_metadata or {},
    }


def snapshot_to_dict(snapshot: SiteSnapshot) -> dict[str, Any]:
    return {
        "id": snapshot.id,
        "site_id": snapshot.site_id,
        "ts": _dt(snapshot.ts),
        "capabilities": snapshot.capabilities or {},
        "availability": snapshot.availability or {},
        "usage": snapshot.usage or {},
        "efficiency": snapshot.efficiency or {},
        "status": snapshot.status or {},
        "source": snapshot.source,
        "submitted_by_email": snapshot.submitted_by_email,
        "quality": snapshot.quality or {},
        "raw_json": snapshot.raw_json,
    }


def latest_snapshot(session: Session, site_id: str) -> SiteSnapshot | None:
    return session.execute(
        select(SiteSnapshot).where(SiteSnapshot.site_id == site_id).order_by(desc(SiteSnapshot.ts), desc(SiteSnapshot.id))
    ).scalars().first()


def _load_site(session: Session, site_id: str) -> RegisteredSite:
    site = session.execute(select(RegisteredSite).where(RegisteredSite.site_id == site_id)).scalar_one_or_none()
    if site is None:
        raise HTTPException(status_code=404, detail=f"Registered site not found: {site_id}")
    if not site.enabled:
        raise HTTPException(status_code=409, detail=f"Registered site is disabled: {site_id}")
    return site


def _execution_site_exists(session: Session, site_id: str | None) -> bool:
    if not site_id:
        return False
    count = session.scalar(select(func.count()).select_from(ExecutionRecord).where(ExecutionRecord.site_id == site_id))
    return bool(count)


def _registration_matches_known_site(session: Session, payload: SiteRegistrationRequest) -> bool:
    metadata = payload.metadata or {}
    candidates = {
        payload.site_id,
        payload.site_name,
        metadata.get("eimps_site_name"),
        metadata.get("metricsdb_site_id"),
        metadata.get("execution_records_site_id"),
    }
    return any(_execution_site_exists(session, str(candidate)) for candidate in candidates if candidate)


def _auth_config(payload: SiteRegistrationRequest) -> dict[str, Any]:
    if payload.auth_type == "egi_checkin":
        settings = get_settings()
        return {
            "issuer": settings.egi_checkin_issuer,
            "audience": settings.egi_checkin_audience,
            "implemented": False,
            **(payload.auth_config or {}),
        }
    return payload.auth_config or {}


def store_snapshot(
    session: Session,
    site_id: str,
    payload: dict[str, Any],
    source: str,
    ts: datetime | None = None,
    raw_json: dict[str, Any] | None = None,
    submitted_by_email: str | None = None,
) -> SiteSnapshot:
    snapshot = SiteSnapshot(
        site_id=site_id,
        ts=_to_utc(ts),
        capabilities=jsonable_encoder(payload.get("capabilities") or {}),
        availability=jsonable_encoder(payload.get("availability") or {}),
        usage=jsonable_encoder(payload.get("usage") or {}),
        efficiency=jsonable_encoder(payload.get("efficiency") or {}),
        status=jsonable_encoder(payload.get("status") or {}),
        source=source,
        submitted_by_email=submitted_by_email,
        quality=jsonable_encoder(payload.get("quality") or {}),
        raw_json=jsonable_encoder(raw_json if raw_json is not None else payload),
    )
    session.add(snapshot)
    site = session.execute(select(RegisteredSite).where(RegisteredSite.site_id == site_id)).scalar_one_or_none()
    if site:
        site.last_seen_at = utc_now()
    session.commit()
    session.refresh(snapshot)
    return snapshot


async def pull_and_store_snapshot(
    session: Session,
    site: RegisteredSite,
    start: datetime | None = None,
    end: datetime | None = None,
    step: str = "1h",
    submitted_by_email: str | None = None,
) -> SiteSnapshot:
    try:
        pulled = await SiteAdapterClient(site.adapter_base_url).pull_snapshot(
            start=_to_utc(start) if start else None,
            end=_to_utc(end) if end else None,
            step=step,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"L3 Site Adapter pull failed: {exc}") from exc
    payload = {
        **pulled,
        "status": pulled.get("availability", {}),
        "quality": {"freshness": "fresh"},
    }
    return store_snapshot(
        session,
        site.site_id,
        payload,
        source="pull",
        ts=utc_now(),
        raw_json=pulled,
        submitted_by_email=submitted_by_email,
    )


@router.post("/register")
def register_site(
    payload: SiteRegistrationRequest,
    principal: SitePrincipal = Depends(require_roles("site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    require_same_site(principal, payload.site_id)
    if not _registration_matches_known_site(session, payload):
        raise HTTPException(
            status_code=400,
            detail="site_id must match an existing MetricsDB/EIMPS site or provide approved mapping",
        )

    existing = session.execute(select(RegisteredSite).where(RegisteredSite.site_id == payload.site_id)).scalar_one_or_none()
    values = {
        "site_id": payload.site_id,
        "site_name": payload.site_name,
        "ri_type": payload.ri_type,
        "adapter_base_url": payload.adapter_base_url.rstrip("/"),
        "contact_email": payload.contact_email,
        "auth_type": payload.auth_type,
        "auth_config": _auth_config(payload),
        "enabled": payload.enabled,
        "site_metadata": payload.metadata or {},
    }
    if existing is None:
        existing = RegisteredSite(**values, registered_at=utc_now())
        session.add(existing)
    else:
        for key, value in values.items():
            setattr(existing, key, value)
    session.commit()
    session.refresh(existing)
    return _site_to_dict(existing)


@router.get("")
def list_sites(
    principal: SitePrincipal = Depends(require_roles("reader", "publisher", "site_admin")),
    session: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    rows = (
        session.execute(select(RegisteredSite).where(RegisteredSite.site_id == principal.site_id).order_by(RegisteredSite.site_id))
        .scalars()
        .all()
    )
    return [_site_to_dict(row) for row in rows]


@router.get("/{site_id}")
def get_site(
    site_id: str,
    principal: SitePrincipal = Depends(require_roles("reader", "publisher", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    require_same_site(principal, site_id)
    return _site_to_dict(_load_site(session, site_id))


@router.post("/{site_id}/snapshots")
def push_snapshot(
    site_id: str,
    payload: SiteSnapshotIn,
    principal: SitePrincipal = Depends(require_roles("publisher", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    require_same_site(principal, site_id)
    _load_site(session, site_id)
    body = model_dump(payload)
    snapshot = store_snapshot(
        session,
        site_id,
        body,
        source="push",
        ts=payload.ts,
        raw_json=body,
        submitted_by_email=principal.email,
    )
    return snapshot_to_dict(snapshot)


@router.post("/{site_id}/pull")
async def pull_snapshot_endpoint(
    site_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    step: str = "1h",
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    require_same_site(principal, site_id)
    site = _load_site(session, site_id)
    snapshot = await pull_and_store_snapshot(session, site, start=start, end=end, step=step, submitted_by_email=principal.email)
    return snapshot_to_dict(snapshot)


@router.get("/{site_id}/latest")
async def latest(
    site_id: str,
    refresh: bool = False,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    require_same_site(principal, site_id)
    site = _load_site(session, site_id)
    snapshot = (
        await pull_and_store_snapshot(session, site, submitted_by_email=principal.email)
        if refresh
        else latest_snapshot(session, site_id)
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No snapshots stored for site: {site_id}")
    return snapshot_to_dict(snapshot)


async def _latest_section(
    site_id: str,
    section: str,
    refresh: bool,
    principal: SitePrincipal,
    session: Session,
) -> dict[str, Any]:
    require_same_site(principal, site_id)
    site = _load_site(session, site_id)
    snapshot = (
        await pull_and_store_snapshot(session, site, submitted_by_email=principal.email)
        if refresh
        else latest_snapshot(session, site_id)
    )
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No snapshots stored for site: {site_id}")
    return getattr(snapshot, section) or {}


@router.get("/{site_id}/capabilities")
async def get_capabilities(
    site_id: str,
    refresh: bool = False,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return await _latest_section(site_id, "capabilities", refresh, principal, session)


@router.get("/{site_id}/availability")
async def get_availability(
    site_id: str,
    refresh: bool = False,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return await _latest_section(site_id, "availability", refresh, principal, session)


@router.get("/{site_id}/usage")
async def get_usage(
    site_id: str,
    refresh: bool = False,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return await _latest_section(site_id, "usage", refresh, principal, session)


@router.get("/{site_id}/efficiency")
async def get_efficiency(
    site_id: str,
    refresh: bool = False,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return await _latest_section(site_id, "efficiency", refresh, principal, session)


async def forward_workload_to_site(session: Session, site_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    site = _load_site(session, site_id)
    try:
        site_response = await SiteAdapterClient(site.adapter_base_url).submit_workload(payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"L3 Site Adapter workload submission failed: {exc}") from exc
    return {"site_id": site_id, "forwarded": True, "site_response": site_response}


@router.post("/{site_id}/submit-workload", include_in_schema=False)
async def submit_workload(
    site_id: str,
    payload: WorkloadSubmissionRequest,
    principal: SitePrincipal = Depends(require_roles("publisher", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    require_same_site(principal, site_id)
    return await forward_workload_to_site(session, site_id, model_dump(payload))
