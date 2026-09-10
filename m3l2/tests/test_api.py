from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy import select

from m3l2.app.db import AuthUser, OperatorConfig, RegisteredSite, SessionLocal, SiteProfile, SiteSnapshot, SiteStatusSnapshot
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


def test_legacy_predict_routes_are_hidden_from_openapi(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    with TestClient(app) as client:
        schema = client.get("/openapi.json").json()

    assert "/l2/predict" in schema["paths"]
    assert "/predict" not in schema["paths"]
    assert "/predict/batch" not in schema["paths"]
    operation_tags = [
        tag
        for path in schema["paths"].values()
        for operation in path.values()
        for tag in operation.get("tags", ["default"])
    ]
    assert "default" not in operation_tags


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
                capabilities={"max_cpu": 32},
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


def test_l2_site_adapter_endpoints_require_bearer_token(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    _seed_site()

    snapshot_payload = {
        "ts": "2026-09-07T00:00:00Z",
        "availability": {"status": "up"},
    }
    register_payload = {
        "site_id": "SLICES-GR-UTH",
        "site_name": "SLICES-GR-UTH",
        "ri_type": "grid",
        "adapter_base_url": "http://127.0.0.1:8000/mock-l3/sites/SLICES-GR-UTH",
        "contact_email": "reader@uth.gr",
    }
    workload_payload = {
        "workload_id": "workload-1",
        "workload_type": "batch",
        "requirements": {},
        "metadata": {},
    }

    with TestClient(app) as client:
        checks = [
            client.post("/l2/predict", json={"site_ids": ["SLICES-GR-UTH"], "horizon": "1h", "step": "1h"}),
            client.get("/l2/sites"),
            client.get("/l2/sites/SLICES-GR-UTH"),
            client.get("/l2/sites/SLICES-GR-UTH/latest"),
            client.get("/l2/sites/SLICES-GR-UTH/capabilities"),
            client.get("/l2/sites/SLICES-GR-UTH/availability"),
            client.get("/l2/sites/SLICES-GR-UTH/usage"),
            client.get("/l2/sites/SLICES-GR-UTH/efficiency"),
            client.post("/l2/sites/SLICES-GR-UTH/pull"),
            client.post("/l2/sites/SLICES-GR-UTH/snapshots", json=snapshot_payload),
            client.post("/l2/sites/SLICES-GR-UTH/submit-workload", json=workload_payload),
            client.post("/l2/sites/register", json=register_payload),
        ]

    assert {response.status_code for response in checks} == {401}


def test_l2_predict_requires_model_after_bearer_token(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    _seed_site()

    with TestClient(app) as client:
        response = client.post(
            "/l2/predict",
            headers=_auth_header(),
            json={"site_ids": ["SLICES-GR-UTH"], "horizon": "1h", "step": "1h"},
        )

    assert response.status_code == 503
    assert response.json()["status"] == "no_active_model"


def test_site_registration_and_telemetry_submission_do_not_require_eimps_records(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    with SessionLocal() as session:
        session.add(AuthUser(email="admin@uth.gr", password_hash="unused", enabled=True))
        session.commit()

    register_payload = {
        "site_id": "NEW-SITE",
        "site_name": "New Site",
        "ri_type": "network",
        "adapter_base_url": "http://127.0.0.1:8000/mock-l3/sites/NEW-SITE",
        "contact_email": "admin@uth.gr",
    }
    telemetry_payload = {
        "timestamp": "2026-09-07T01:00:00Z",
        "ri_type": "network",
        "operational_status": "UP",
        "maintenance_flag": False,
        "node_availability": 1.0,
        "link_availability": 1.0,
        "free_cpu_capacity": 8,
        "queue_length": 0,
        "load_index": 0.1,
    }

    with TestClient(app) as client:
        registered = client.post(
            "/l2/sites/register",
            headers=_auth_header("admin@uth.gr", "NEW-SITE", "site_admin"),
            json=register_payload,
        )
        submitted = client.post(
            "/l2/sites/NEW-SITE/snapshots",
            headers=_auth_header("admin@uth.gr", "NEW-SITE", "site_admin"),
            json=telemetry_payload,
        )

    assert registered.status_code == 200
    assert submitted.status_code == 200
    with SessionLocal() as session:
        status = session.execute(select(SiteStatusSnapshot).where(SiteStatusSnapshot.site_id == "NEW-SITE")).scalars().first()
    assert status is not None
    assert status.node_availability == 1.0


def test_operator_config_requires_site_admin(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    _seed_site()

    with TestClient(app) as client:
        no_token = client.patch("/ops/config", json={"training_frequency_hours": 2})
        publisher = client.patch(
            "/ops/config",
            headers=_auth_header("publisher@uth.gr", role="publisher"),
            json={"training_frequency_hours": 2},
        )

    assert no_token.status_code == 401
    assert publisher.status_code == 403


def test_train_now_requires_site_admin_and_records_operator(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    _seed_site()
    captured = {}

    def fake_train_model(force: bool, triggered_by_email: str | None = None, site_id: str | None = None):
        captured["force"] = force
        captured["triggered_by_email"] = triggered_by_email
        captured["site_id"] = site_id
        return {"status": "trained", "model_version": "test-model"}

    monkeypatch.setattr("m3l2.app.main.train_model", fake_train_model)

    with TestClient(app) as client:
        publisher = client.post("/ops/train", headers=_auth_header("publisher@uth.gr", role="publisher"))
        admin = client.post("/ops/train?site_id=SLICES-GR-UTH", headers=_auth_header(role="site_admin"))

    assert publisher.status_code == 403
    assert admin.status_code == 200
    assert captured == {"force": True, "triggered_by_email": "reader@uth.gr", "site_id": "SLICES-GR-UTH"}


def test_operator_config_persists_service_and_site_overrides(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    _seed_site()

    with TestClient(app) as client:
        service = client.patch(
            "/ops/config",
            headers=_auth_header(role="site_admin"),
            json={"training_frequency_hours": 4, "min_usable_records": 3, "minimum_coverage_ratio": 0.5},
        )
        site = client.patch(
            "/ops/config?site_id=SLICES-GR-UTH",
            headers=_auth_header(role="site_admin"),
            json={"min_usable_records": 2},
        )
        effective = client.get("/ops/config?site_id=SLICES-GR-UTH", headers=_auth_header(role="site_admin"))
        disabled_model = client.patch(
            "/ops/config",
            headers=_auth_header(role="site_admin"),
            json={"model_name": "xgb"},
        )

    assert service.status_code == 200
    assert site.status_code == 200
    assert effective.status_code == 200
    body = effective.json()
    assert body["service_overrides"]["training_frequency_hours"] == 4
    assert body["site_overrides"]["min_usable_records"] == 2
    assert body["effective"]["min_usable_records"] == 2
    assert body["effective"]["model_name"] == "hist_gradient_boosting_mvp"
    assert disabled_model.status_code == 422
    with SessionLocal() as session:
        assert session.get(OperatorConfig, "service").settings["min_usable_records"] == 3


def test_operator_config_scheduler_changes_are_reported(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    _seed_site()

    with TestClient(app) as client:
        disabled = client.patch(
            "/ops/config",
            headers=_auth_header(role="site_admin"),
            json={"automatic_training": False},
        )
        enabled = client.patch(
            "/ops/config",
            headers=_auth_header(role="site_admin"),
            json={"automatic_training": True, "training_frequency_hours": 3, "forecast_refresh_minutes": 7},
        )

    assert disabled.status_code == 200
    assert disabled.json()["scheduler"]["status"] == "disabled"
    assert enabled.status_code == 200
    assert enabled.json()["scheduler"]["status"] == "scheduled"
    assert enabled.json()["scheduler"]["jobs"] == ["m3l2_forecast_refresh", "m3l2_train"]


def test_operator_config_ui_marks_unavailable_features_disabled(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")

    with TestClient(app) as client:
        landing = client.get("/")
        response = client.get("/ops/config/ui")

    assert landing.status_code == 200
    assert "Login to config" in landing.text
    assert "/auth/login?next=/ops/config/ui&role=site_admin" in landing.text
    assert response.status_code == 200
    text = response.text
    assert 'id="token"' not in text
    assert "localStorage.getItem(\"m3l2_token\")" in text
    assert "/auth/me" in text
    assert "Configuration scope" in text
    assert "Connect to EIMPS" in text
    assert "Not yet available" in text
    assert '<option value="xgb" disabled>' in text
    assert '<option value="lstm" disabled>' in text


def test_browser_login_issues_token_and_me_lists_user_sites(temp_database, tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    allowed = tmp_path / "allowed_emails.txt"
    allowed.write_text(
        "operator@uth.gr,SLICES-GR-UTH,site_admin|publisher\n"
        "operator@uth.gr,OTHER-SITE,reader\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALLOWED_EMAILS_PATH", str(allowed))

    with TestClient(app) as client:
        login_page = client.get("/auth/login?next=/ops/config/ui&role=site_admin")
        token_response = client.post(
            "/auth/token",
            json={
                "email": "operator@uth.gr",
                "password": "correct horse battery staple",
                "site_id": "SLICES-GR-UTH",
                "role": "site_admin",
            },
        )
        token = token_response.json()["access_token"]
        me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

    assert login_page.status_code == 200
    assert 'const nextUrl = "/ops/config/ui";' in login_page.text
    assert 'localStorage.setItem("m3l2_token", body.access_token)' in login_page.text
    assert token_response.status_code == 200
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == "operator@uth.gr"
    assert body["site_id"] == "SLICES-GR-UTH"
    assert body["role"] == "site_admin"
    assert body["roles_for_current_site"] == ["publisher", "site_admin"]
    assert {"site_id": "OTHER-SITE", "roles": ["reader"], "registered": False} in body["sites"]
    assert {"site_id": "SLICES-GR-UTH", "roles": ["publisher", "site_admin"], "registered": False} in body["sites"]


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
    with SessionLocal() as session:
        status = session.execute(
            select(SiteStatusSnapshot)
            .where(SiteStatusSnapshot.site_id == "SLICES-GR-UTH")
            .order_by(SiteStatusSnapshot.timestamp.desc())
        ).scalars().first()

    assert status is not None
    assert status.operational_status == "UP"
    assert status.load_index == 0.2


def test_flat_uth_snapshot_submission_is_training_compatible(temp_database, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    _seed_site()

    payload = {
        "timestamp": "2026-09-07T01:00:00Z",
        "ri_type": "Grid",
        "operational_status": "DEGRADED",
        "maintenance_flag": False,
        "node_availability": 0.95,
        "link_availability": 0.98,
        "cpu_util_avg": 72.0,
        "queue_length": 3,
        "remaining_jobs": 8,
        "load_index": 0.7,
        "energy_consumed": 100.0,
        "pue_estimate": 1.2,
        "carbon_intensity": 250.0,
        "update_frequency": 3600,
        "data_confidence": 0.9,
        "coverage_ratio": 0.95,
        "stale_flag": False,
    }
    with TestClient(app) as client:
        response = client.post(
            "/l2/sites/SLICES-GR-UTH/snapshots",
            headers=_auth_header("publisher@uth.gr", role="publisher"),
            json=payload,
        )

    assert response.status_code == 200
    assert response.json()["ts"] == "2026-09-07T01:00:00+00:00"
    with SessionLocal() as session:
        status = session.execute(
            select(SiteStatusSnapshot)
            .where(SiteStatusSnapshot.site_id == "SLICES-GR-UTH")
            .order_by(SiteStatusSnapshot.timestamp.desc())
        ).scalars().first()

    assert status is not None
    assert status.ri_type == "grid"
    assert status.operational_status == "DEGRADED"
    assert status.node_availability == 0.95
    assert status.queue_length == 3


def test_site_status_batch_validation_prevents_partial_persistence(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    payload = [
        {
            "site_id": "SITE-OK",
            "ri_type": "grid",
            "timestamp": "2026-09-08T07:00:00Z",
            "node_availability": 0.9,
        },
        {
            "site_id": "SITE-BAD",
            "ri_type": "grid",
            "node_availability": 1.2,
        },
    ]

    with TestClient(app) as client:
        response = client.post("/site-status", json=payload)

    assert response.status_code == 422
    fields = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert ("body", 1, "timestamp") in fields
    assert ("body", 1, "node_availability") in fields
    with SessionLocal() as session:
        count = session.query(SiteStatusSnapshot).count()
    assert count == 0


def test_site_profile_alias_conflict_returns_field_error(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    payload = {
        "site_id": "SLICES-GR-UTH",
        "site": "OTHER-SITE",
        "ri_type": "grid",
    }

    with TestClient(app) as client:
        response = client.post("/site-profiles?adapter_type=iot", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "site_id"]
    with SessionLocal() as session:
        count = session.query(SiteProfile).count()
    assert count == 0


def test_profile_extensions_are_warned_and_returned(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    payload = {
        "site_id": "SLICES-GR-UTH",
        "ri_type": "grid",
        "location": "UTH",
        "local_owner": "uth",
        "extensions": {"sensor_generation": "v2"},
    }

    with TestClient(app) as client:
        submit = client.post("/site-profiles", json=payload)
        listed = client.get("/site-profiles")

    assert submit.status_code == 200
    assert submit.json()["warnings"] == [{"index": 0, "fields": ["local_owner", "sensor_generation"]}]
    profile = listed.json()[0]
    assert profile["extensions"] == {"local_owner": "uth", "sensor_generation": "v2"}


def test_iot_status_round_trip_preserves_uth_fields_and_zero_false_values(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    payload = {
        "site": "SLICES-GR-UTH",
        "ri_type": "network",
        "ts": "2026-09-08T07:00:00Z",
        "alive_nodes": 0,
        "total_nodes": 10,
        "active_links": 0,
        "total_links": 5,
        "cpu_utilization": 0,
        "stability": 1.0,
        "stale": False,
        "lab_phase": "pilot",
    }

    with TestClient(app) as client:
        submit = client.post("/site-status?adapter_type=iot", json=payload)
        latest = client.get("/site-status/latest?site_id=SLICES-GR-UTH")

    assert submit.status_code == 200
    assert submit.json()["warnings"] == [{"index": 0, "fields": ["lab_phase"]}]
    status = latest.json()[0]
    assert status["ri_type"] == "network"
    assert status["node_availability"] == 0.0
    assert status["link_availability"] == 0.0
    assert status["cpu_util_avg"] == 0.0
    assert status["stability_score"] == 1.0
    assert status["stale_flag"] is False
    assert status["extensions"] == {"lab_phase": "pilot"}


def test_iot_status_rejects_conflicting_explicit_and_derived_availability(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    payload = {
        "site_id": "SLICES-GR-UTH",
        "ri_type": "network",
        "timestamp": "2026-09-08T07:00:00Z",
        "node_availability": 0.5,
        "alive_nodes": 9,
        "total_nodes": 10,
    }

    with TestClient(app) as client:
        response = client.post("/site-status?adapter_type=iot", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "node_availability"]


def test_iot_adapter_does_not_accept_iot_as_ri_type(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    payload = {
        "site_id": "SLICES-GR-UTH",
        "ri_type": "iot",
        "timestamp": "2026-09-08T07:00:00Z",
    }

    with TestClient(app) as client:
        response = client.post("/site-status?adapter_type=iot", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "ri_type"]
