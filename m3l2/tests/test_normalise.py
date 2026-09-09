from __future__ import annotations

from m3l2.ingestion.normalise import normalise_execution_record
from m3l2.ingestion.site_adapter import normalise_site_profile, normalise_site_status


def test_normalise_cloud_payload():
    result = normalise_execution_record(
        {
            "ExecUnitID": "job-1",
            "Site": "site-a",
            "StartExecTime": "2026-01-01T00:00:00Z",
            "StopExecTime": "2026-01-01T01:00:00Z",
            "Energy_wh": "12.5",
            "cloud_type": "vm",
            "CPUTime": 99,
        }
    )
    assert result["exec_unit_id"] == "job-1"
    assert result["site_id"] == "site-a"
    assert result["ri_type"] == "cloud"
    assert result["work_type"] == "cpu_time"
    assert result["energy_wh"] == 12.5


def test_normalise_network_payload():
    result = normalise_execution_record(
        {
            "exec_unit_id": "net-1",
            "site": "site-b",
            "start_ts": "2026-01-01T00:00:00+00:00",
            "network_type": "wan",
            "AmountOfDataTransferred": 123,
        }
    )
    assert result["ri_type"] == "network"
    assert result["work_type"] == "data_transfer"


def test_normalise_unknown_payload_derives_id():
    result = normalise_execution_record({"site_name": "x", "start_ts": "2026-01-01T00:00:00Z"})
    assert result["exec_unit_id"].startswith("derived-")
    assert result["ri_type"] == "unknown"


def test_normalise_uth_site_adapter_schema_fields():
    profile = normalise_site_profile(
        {
            "site_id": "SLICES-GR-UTH",
            "ri_type": "Grid",
            "location": "UTH",
            "compute_capacity": 128,
            "storage_capacity": 2048,
            "network_topology": "Mesh",
            "link_capacities": {"core": 10000},
            "supported_workload_types": ["Batch", "Stream", "ML"],
            "energy_capabilities": {"metering": True},
            "static_pue_baseline": 1.2,
        }
    )
    status = normalise_site_status(
        {
            "site_id": "SLICES-GR-UTH",
            "ri_type": "Grid",
            "timestamp": "2026-09-08T07:00:00Z",
            "operational_status": "DEGRADED",
            "maintenance_flag": False,
            "scheduled_maintenance": {"start": None, "end": None},
            "node_availability": 0.95,
            "link_availability": 0.98,
            "stability_score": 0.99,
            "packet_loss": 0.1,
            "network_jitter": 2.0,
            "network_utilization": 42.0,
            "available_bandwidth": 1000.0,
            "cpu_util_avg": 55.0,
            "queue_length": 3,
            "remaining_jobs": 7,
            "load_index": 0.61,
            "energy_consumed": 123.4,
            "pue_estimate": 1.2,
            "carbon_intensity": 250.0,
            "energy_per_task_proxy": 12.3,
            "update_frequency": 3600,
            "data_confidence": 0.9,
            "coverage_ratio": 0.95,
            "stale_flag": False,
        }
    )

    assert profile["ri_type"] == "grid"
    assert profile["compute_capacity"] == 128.0
    assert status["ri_type"] == "grid"
    assert status["operational_status"] == "DEGRADED"
    assert status["node_availability"] == 0.95
    assert status["queue_length"] == 3
