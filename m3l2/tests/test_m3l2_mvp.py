from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from m3l2.app.db import (
    AuthUser,
    ExecutionRecord,
    ForecastCache,
    ModelRegistry,
    OperatorConfig,
    RegisteredSite,
    SessionLocal,
    SiteProfile,
    SiteStatusSnapshot,
    utc_now,
)
from m3l2.app.main import app
from m3l2.inference.predict import predict
from m3l2.site_adapter.auth import create_site_jwt
from m3l2.training.train import train_model


def _seed_records(count: int = 8) -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with SessionLocal() as session:
        session.add(AuthUser(email="reader@uth.gr", password_hash="unused", enabled=True))
        for idx in range(count):
            start = base + timedelta(hours=idx)
            site_id = "site-a" if idx % 2 == 0 else "site-b"
            session.add(
                ExecutionRecord(
                    exec_unit_id=f"exec-{idx}",
                    site_id=site_id,
                    ri_type="cloud",
                    start_ts=start,
                    stop_ts=start + timedelta(minutes=30 + idx),
                    status="finished",
                    energy_wh=20.0 + idx,
                    work=100.0 + (idx * 10),
                    work_type="cpu_time",
                )
            )
            session.add(
                SiteStatusSnapshot(
                    site_id=site_id,
                    ri_type="cloud",
                    timestamp=start,
                    operational_status="UP",
                    node_availability=1.0,
                    link_availability=1.0,
                    free_cpu_capacity=16 + (idx % 3),
                    free_gpu_capacity=1,
                    queue_length=idx % 2,
                    provisioning_delay_s=20 + idx,
                    load_index=0.2 + (idx * 0.01),
                    carbon_intensity=250,
                )
            )
        for site_id in ("site-a", "site-b"):
            session.add(SiteProfile(site_id=site_id, ri_type="cloud", compute_capacity=32, gpu_capacity=2, storage_capacity=500))
            session.add(
                SiteStatusSnapshot(
                    site_id=site_id,
                    ri_type="cloud",
                    timestamp=base + timedelta(hours=count),
                    operational_status="UP",
                    node_availability=1.0,
                    link_availability=1.0,
                    free_cpu_capacity=16,
                    free_gpu_capacity=1,
                    queue_length=1,
                    provisioning_delay_s=30,
                    load_index=0.2,
                    carbon_intensity=250,
                )
            )
        session.add(
            RegisteredSite(
                site_id="PUBLIC-SITE-A",
                site_name="Public Site A",
                ri_type="cloud",
                adapter_base_url="http://example.test/sites/PUBLIC-SITE-A",
                contact_email="operator@example.test",
                site_metadata={"execution_records_site_id": "site-a"},
            )
        )
        session.commit()


def _prepare_training(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("M3L2_MIN_TRAINING_RECORDS", "4")
    monkeypatch.setenv("M3L2_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("M3L2_FORECAST_STEP_MINUTES", "60")
    monkeypatch.setenv("M3L2_FORECAST_HORIZON_HOURS", "2")
    _seed_records()
    with SessionLocal() as session:
        session.add(
            OperatorConfig(
                config_key="service",
                settings={"submission_cadence_minutes": 120, "minimum_coverage_ratio": 0.5},
            )
        )
        session.commit()


def _predict_payload(workload_id: str = "wl-1") -> dict[str, Any]:
    return {
        "request_id": f"req-{workload_id}",
        "candidate_site_ids": ["site-a"],
        "forecast_start_time": "2026-12-01T00:00:00Z",
        "horizon": "2h",
        "step": "1h",
        "workload": {
            "workload_id": workload_id,
            "workload_type": "batch",
            "time_requirements": {"duration": "1h"},
            "resource_requirements": {"cpu": 2, "memory_gb": 4, "storage_gb": 20, "instances": 1},
            "metadata": {"queue": "short"},
            "extensions": {"application": "fixture"},
        },
        "cache": {"use_cache": True},
        "include_site_status": True,
    }


def _key_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _key_shape(child) for key, child in value.items()}
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def test_unchanged_data_skips_training(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    first = train_model(force=True)
    second = train_model(force=False)

    assert first["status"] == "trained"
    assert second["status"] == "skipped_unchanged_data"
    assert second["model_version"] == first["model_version"]


def test_forced_training_still_runs(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    first = train_model(force=True)
    second = train_model(force=True)

    assert first["status"] == "trained"
    assert second["status"] == "trained"
    assert second["model_version"] != first["model_version"]


def test_changed_training_data_triggers_retraining(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    first = train_model(force=True)
    with SessionLocal() as session:
        row = session.execute(select(SiteStatusSnapshot).where(SiteStatusSnapshot.site_id == "site-a")).scalars().first()
        row.free_cpu_capacity = 999.0
        session.commit()

    second = train_model(force=False)

    assert second["status"] == "trained"
    assert second["model_version"] != first["model_version"]
    assert second["training_data_fingerprint"] != first["training_data_fingerprint"]


def test_hgbr_model_trains_and_predicts(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    result = train_model(force=True)

    bundle = joblib.load(result["path"])
    assert bundle["feature_schema"]["target"] == "l2_site_status"
    assert bundle["pipeline"].named_steps["model"].__class__.__name__ == "MultiOutputRegressor"

    prediction = predict(_predict_payload())
    assert prediction["status"] == "ok"
    assert prediction["model_name"] == "hist_gradient_boosting_mvp"
    assert prediction["target"] == "l2_site_status"
    assert prediction["results"][0]["forecast"][0]["unit"] == "ratio"
    assert prediction["results"][0]["site_status_forecast"][0]["inference_source"] == "model"


def test_workload_specific_cache_isolation(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    train_model(force=True)

    first = predict(_predict_payload("wl-a"))
    second = predict(_predict_payload("wl-b"))

    assert first["cache"]["request_signature"] != second["cache"]["request_signature"]
    with SessionLocal() as session:
        count = session.scalar(
            select(func.count())
            .select_from(ForecastCache)
            .where(ForecastCache.request_signature.in_([first["cache"]["request_signature"], second["cache"]["request_signature"]]))
        )
        assert count == 2


def test_model_version_cache_invalidation(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    trained = train_model(force=True)
    first = predict(_predict_payload("wl-a"))

    with SessionLocal() as session:
        active = session.execute(select(ModelRegistry).where(ModelRegistry.active.is_(True))).scalar_one()
        active.active = False
        session.add(
            ModelRegistry(
                model_name=active.model_name,
                target=active.target,
                version=f"{active.version}-manual",
                path=active.path,
                trained_at=utc_now(),
                training_window_start=active.training_window_start,
                training_window_end=active.training_window_end,
                metrics=active.metrics,
                feature_schema=active.feature_schema,
                training_data_fingerprint=active.training_data_fingerprint,
                active=True,
            )
        )
        session.commit()

    second = predict(_predict_payload("wl-a"))

    assert trained["model_version"] == first["model_version"]
    assert second["model_version"].endswith("-manual")
    assert second["cache"]["status"] == "fresh"
    with SessionLocal() as session:
        count = session.scalar(
            select(func.count())
            .select_from(ForecastCache)
            .where(ForecastCache.request_signature == first["cache"]["request_signature"])
        )
        assert count == 2


def test_fresh_and_cached_responses_have_same_structure(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    train_model(force=True)

    fresh = predict(_predict_payload("wl-a"))
    cached = predict(_predict_payload("wl-a"))

    assert fresh["cache"]["status"] == "fresh"
    assert cached["cache"]["status"] == "cached"
    assert _key_shape(fresh) == _key_shape(cached)


def test_successful_typed_predict_request(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    train_model(force=True)
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")

    with TestClient(app) as client:
        response = client.post("/predict", json=_predict_payload("typed"))

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-typed"
    assert body["results"][0]["site_id"] == "site-a"
    assert body["results"][0]["energy_forecast"] == []
    assert body["results"][0]["site_status_forecast"][0]["inference_source"] == "model"


def test_predict_resolves_registered_site_to_training_site(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    train_model(force=True)
    payload = _predict_payload("mapped")
    payload["candidate_site_ids"] = ["PUBLIC-SITE-A"]

    prediction = predict(payload)

    result = prediction["results"][0]
    assert result["site_id"] == "PUBLIC-SITE-A"
    assert result["training_site_id"] == "site-a"
    assert result["registered_site_id"] == "PUBLIC-SITE-A"
    assert result["site_id_resolution"] == "registered_site_mapping"
    assert result["target"] == "l2_site_status"


def test_public_and_training_site_ids_reuse_cache_entry(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    train_model(force=True)
    public_payload = _predict_payload("alias")
    public_payload["candidate_site_ids"] = ["PUBLIC-SITE-A"]
    raw_payload = _predict_payload("alias")
    raw_payload["candidate_site_ids"] = ["site-a"]

    public_response = predict(public_payload)
    raw_response = predict(raw_payload)

    assert public_response["cache"]["request_signature"] == raw_response["cache"]["request_signature"]
    assert raw_response["cache"]["status"] == "cached"
    with SessionLocal() as session:
        count = session.scalar(
            select(func.count())
            .select_from(ForecastCache)
            .where(ForecastCache.request_signature == public_response["cache"]["request_signature"])
        )
    assert count == 1


def test_predict_returns_clear_error_for_unknown_site(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    train_model(force=True)
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    payload = _predict_payload("missing")
    payload["candidate_site_ids"] = ["UNKNOWN-SITE"]

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 404
    body = response.json()
    assert body["status"] == "candidate_sites_not_found"
    assert body["missing_site_ids"] == ["UNKNOWN-SITE"]


def test_successful_authenticated_l2_predict_request(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    train_model(force=True)
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    token = create_site_jwt("reader@uth.gr", "PUBLIC-SITE-A", "reader", "test-secret")
    payload = _predict_payload("l2-mapped")
    payload["candidate_site_ids"] = ["PUBLIC-SITE-A"]

    with TestClient(app) as client:
        response = client.post("/l2/predict", headers={"Authorization": f"Bearer {token}"}, json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["results"][0]["site_id"] == "PUBLIC-SITE-A"
    assert body["results"][0]["training_site_id"] == "site-a"


def test_invalid_workload_time_and_resource_inputs(temp_database, monkeypatch):
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")
    payload = _predict_payload("bad")
    payload["workload"]["time_requirements"]["duration"] = "0h"
    payload["workload"]["resource_requirements"]["cpu"] = -1

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_successful_batch_prediction(temp_database, monkeypatch, tmp_path):
    _prepare_training(monkeypatch, tmp_path)
    train_model(force=True)
    monkeypatch.setenv("M3L2_ENABLE_SCHEDULER", "false")

    with TestClient(app) as client:
        response = client.post("/predict/batch", json=[_predict_payload("batch-a"), _predict_payload("batch-b")])

    assert response.status_code == 200
    assert [item["request_id"] for item in response.json()] == ["req-batch-a", "req-batch-b"]
