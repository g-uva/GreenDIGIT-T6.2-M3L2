from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from m3l2.app.db import AuthUser, RegisteredSite, SessionLocal, SiteSnapshot
from m3l2.app.main import app
from m3l2.site_adapter.auth import create_site_jwt


def test_health_works(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_no_active_model(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    with TestClient(app) as client:
        response = client.post("/predict", json={"site_ids": None, "horizon": "24h", "step": "1h", "use_cache": True})
    assert response.status_code == 503
    assert response.json()["status"] == "no_active_model"


def _seed_site() -> None:
    with SessionLocal() as session:
        session.add(AuthUser(email="reader@uth.gr", password_hash="unused", enabled=True))
        session.add(AuthUser(email="publisher@uth.gr", password_hash="unused", enabled=True))
        session.add(AuthUser(email="other@uth.gr", password_hash="unused", enabled=True))
        session.add(
            RegisteredSite(
                site_id="SLICES-GR-UTH",
                site_name="SLICES-GR-UTH",
                ri_type="grid",
                adapter_base_url="http://127.0.0.1:8000/mock-l3/sites/SLICES-GR-UTH",
                contact_email="reader@uth.gr",
            )
        )
        session.add(
            SiteSnapshot(
                site_id="SLICES-GR-UTH",
                ts=datetime(2026, 6, 30, tzinfo=timezone.utc),
                availability={"status": "up"},
                usage={"cpu_utilization": 0.42},
                efficiency={"energy_per_cpu_hour_wh": 120.0},
                source="push",
                submitted_by_email="legacy@uth.gr",
                raw_json={"availability": {"status": "up"}},
            )
        )
        session.commit()


def _auth_header(email: str = "reader@uth.gr", site_id: str = "SLICES-GR-UTH", role: str = "reader") -> dict[str, str]:
    token = create_site_jwt(email, site_id, role, "test-secret")
    return {"Authorization": f"Bearer {token}"}


def test_l2_site_reads_require_bearer_token(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    _seed_site()

    with TestClient(app) as client:
        assert client.get("/l2/sites").status_code == 401
        assert client.get("/l2/sites/SLICES-GR-UTH").status_code == 401
        assert client.get("/l2/sites/SLICES-GR-UTH/latest").status_code == 401
        assert client.get("/l2/sites/SLICES-GR-UTH/availability").status_code == 401


def test_l2_site_reads_return_authenticated_site_data(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    _seed_site()

    with TestClient(app) as client:
        sites = client.get("/l2/sites", headers=_auth_header())
        latest = client.get("/l2/sites/SLICES-GR-UTH/latest", headers=_auth_header())
        availability = client.get("/l2/sites/SLICES-GR-UTH/availability", headers=_auth_header())

    assert sites.status_code == 200
    assert sites.json()[0]["site_id"] == "SLICES-GR-UTH"
    assert latest.status_code == 200
    assert latest.json()["availability"] == {"status": "up"}
    assert latest.json()["submitted_by_email"] == "legacy@uth.gr"
    assert availability.status_code == 200
    assert availability.json() == {"status": "up"}


def test_l2_site_reads_reject_other_site_token(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    _seed_site()

    with TestClient(app) as client:
        response = client.get("/l2/sites/SLICES-GR-UTH/latest", headers=_auth_header("other@uth.gr", "OTHER-SITE"))

    assert response.status_code == 403


def test_snapshot_submission_records_submitter_email(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    _seed_site()

    payload = {
        "ts": "2026-09-07T00:00:00Z",
        "availability": {"status": "up"},
        "usage": {"load_index": 0.2},
        "efficiency": {"pue_estimate": 1.2},
        "quality": {"mock": True},
    }
    with TestClient(app) as client:
        response = client.post(
            "/l2/sites/SLICES-GR-UTH/snapshots",
            headers=_auth_header("publisher@uth.gr", role="publisher"),
            json=payload,
        )

    assert response.status_code == 200
    assert response.json()["submitted_by_email"] == "publisher@uth.gr"
