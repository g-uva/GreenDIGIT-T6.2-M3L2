from __future__ import annotations

from datetime import datetime, timedelta, timezone

from m3l2.app.db import SessionLocal, SiteStatusSnapshot
from m3l2.training.train import train_model


def test_train_model_not_enough_data(temp_database):
    result = train_model(force=True)
    assert result["status"] == "not_enough_data"
    assert result["n_records"] == 0


def test_train_model_uses_submitted_site_telemetry_without_eimps(temp_database, monkeypatch, tmp_path):
    monkeypatch.setenv("M3L2_MIN_TRAINING_RECORDS", "4")
    monkeypatch.setenv("M3L2_MODEL_DIR", str(tmp_path / "models"))
    base = datetime(2026, 9, 7, tzinfo=timezone.utc)
    with SessionLocal() as session:
        for idx in range(6):
            session.add(
                SiteStatusSnapshot(
                    site_id="SLICES-GR-UTH",
                    ri_type="network",
                    timestamp=base + timedelta(hours=idx),
                    operational_status="UP",
                    maintenance_flag=False,
                    node_availability=1.0,
                    link_availability=1.0,
                    free_cpu_capacity=16 - idx,
                    queue_length=idx % 2,
                    provisioning_delay_s=20 + idx,
                    load_index=0.2 + idx * 0.01,
                )
            )
        session.commit()

    result = train_model(force=True, triggered_by_email="operator@uth.gr", site_id="SLICES-GR-UTH")

    assert result["status"] == "trained"
    assert result["target"] == "l2_site_status"
    assert result["n_records"] == 6
    assert result["readiness"]["available"] is True
