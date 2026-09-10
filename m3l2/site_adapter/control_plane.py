from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from m3l2.app.db import RegisteredSite, SessionLocal, SiteSnapshot, SiteStatusSnapshot, utc_now
from m3l2.ingestion.site_adapter import normalise_site_status
from m3l2.site_adapter.auth import SitePrincipal, current_principal, require_roles, require_same_site
from m3l2.site_adapter.schemas import SiteSnapshotIn

router = APIRouter(prefix="/l2/sites", tags=["l2-site-adapter"], dependencies=[Depends(current_principal)])


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
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
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


def _snapshot_status_payload(site_id: str, payload: dict[str, Any], ts: datetime | None) -> dict[str, Any]:
    status_ts = ts or payload.get("ts") or payload.get("timestamp")
    return {
        **{
            key: value
            for key, value in payload.items()
            if key not in {"ts", "timestamp", "capabilities", "availability", "usage", "efficiency", "status", "quality"}
        },
        "site_id": site_id,
        "timestamp": status_ts,
        **(payload.get("availability") or {}),
        **(payload.get("usage") or {}),
        **(payload.get("efficiency") or {}),
        **(payload.get("status") or {}),
        **(payload.get("quality") or {}),
    }


def _load_site(session: Session, site_id: str) -> RegisteredSite:
    site = session.execute(select(RegisteredSite).where(RegisteredSite.site_id == site_id)).scalar_one_or_none()
    if site is None:
        raise HTTPException(status_code=404, detail=f"Registered site not found: {site_id}")
    if not site.enabled:
        raise HTTPException(status_code=409, detail=f"Registered site is disabled: {site_id}")
    return site


def _ensure_site_registered(session: Session, site_id: str, principal: SitePrincipal, payload: dict[str, Any]) -> RegisteredSite:
    site = session.execute(select(RegisteredSite).where(RegisteredSite.site_id == site_id)).scalar_one_or_none()
    if site is not None:
        if not site.enabled:
            raise HTTPException(status_code=409, detail=f"Registered site is disabled: {site_id}")
        return site

    ri_type = str(payload.get("ri_type") or (payload.get("capabilities") or {}).get("ri_type") or "unknown").lower()
    site = RegisteredSite(
        site_id=site_id,
        site_name=site_id,
        ri_type=ri_type,
        adapter_base_url=f"auto://{site_id}",
        contact_email=principal.email,
        auth_type="jwt",
        auth_config={},
        enabled=True,
        registered_at=utc_now(),
        site_metadata={"registration_source": "first_snapshot"},
    )
    session.add(site)
    session.flush()
    return site


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
    try:
        status = normalise_site_status(_snapshot_status_payload(site_id, payload, ts))
    except ValueError:
        status = None
    if status is not None:
        status.pop("_warnings", None)
        status["raw_json"] = jsonable_encoder({"source_schema": "l2_site_snapshot", **(raw_json if raw_json is not None else payload)})
        session.add(SiteStatusSnapshot(**status, ingested_at=utc_now()))
    site = session.execute(select(RegisteredSite).where(RegisteredSite.site_id == site_id)).scalar_one_or_none()
    if site:
        site.last_seen_at = utc_now()
    session.commit()
    session.refresh(snapshot)
    return snapshot


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
    body = model_dump(payload)
    _ensure_site_registered(session, site_id, principal, body)
    snapshot_ts = payload.ts or body.get("timestamp")
    snapshot = store_snapshot(
        session,
        site_id,
        body,
        source="push",
        ts=snapshot_ts,
        raw_json=body,
        submitted_by_email=principal.email,
    )
    return snapshot_to_dict(snapshot)


@router.get("/{site_id}/latest")
def latest(
    site_id: str,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    require_same_site(principal, site_id)
    _load_site(session, site_id)
    snapshot = latest_snapshot(session, site_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No snapshots stored for site: {site_id}")
    return snapshot_to_dict(snapshot)


def _latest_section(
    site_id: str,
    section: str,
    principal: SitePrincipal,
    session: Session,
) -> dict[str, Any]:
    require_same_site(principal, site_id)
    _load_site(session, site_id)
    snapshot = latest_snapshot(session, site_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No snapshots stored for site: {site_id}")
    return getattr(snapshot, section) or {}


@router.get("/{site_id}/capabilities")
def get_capabilities(
    site_id: str,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return _latest_section(site_id, "capabilities", principal, session)


@router.get("/{site_id}/availability")
def get_availability(
    site_id: str,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return _latest_section(site_id, "availability", principal, session)


@router.get("/{site_id}/usage")
def get_usage(
    site_id: str,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return _latest_section(site_id, "usage", principal, session)


@router.get("/{site_id}/efficiency")
def get_efficiency(
    site_id: str,
    principal: SitePrincipal = Depends(require_roles("reader", "site_admin")),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return _latest_section(site_id, "efficiency", principal, session)
