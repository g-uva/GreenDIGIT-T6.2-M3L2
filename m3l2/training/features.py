from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from m3l2.app.db import ExecutionRecord, SiteSnapshot, SiteStatusSnapshot
from m3l2.ingestion.site_adapter import normalise_site_status

ENERGY_FEATURE_COLUMNS = [
    "site_id",
    "ri_type",
    "start_ts",
    "duration_s",
    "energy_wh",
    "work",
    "hour",
    "day_of_week",
    "records_count_site_24h",
    "rolling_energy_mean_site_24h",
    "rolling_work_mean_site_24h",
]

STATUS_TARGET_COLUMNS = [
    "operational_status_score",
    "maintenance_flag",
    "node_availability",
    "link_availability",
    "compute_capacity",
    "gpu_capacity",
    "storage_capacity",
    "free_cpu_capacity",
    "free_gpu_capacity",
    "queue_length",
    "provisioning_delay_s",
    "load_index",
]

STATUS_REQUIRED_INPUTS = [
    "site_id",
    "timestamp",
    "operational_status",
    "node_availability",
    "link_availability",
    "free_cpu_capacity",
    "queue_length",
    "load_index",
]

STATUS_FEATURE_COLUMNS = [
    "site_id",
    "ri_type",
    "timestamp",
    "hour",
    "day_of_week",
    "records_count_site_24h",
    "rolling_availability_mean_site_24h",
    "rolling_free_cpu_mean_site_24h",
    "rolling_queue_mean_site_24h",
    "rolling_load_mean_site_24h",
]

FEATURE_COLUMNS = ENERGY_FEATURE_COLUMNS


def build_training_frame(session: Session) -> pd.DataFrame:
    rows = session.execute(select(ExecutionRecord)).scalars().all()
    if not rows:
        return pd.DataFrame(columns=ENERGY_FEATURE_COLUMNS)

    data = [
        {
            "site_id": row.site_id or "unknown-site",
            "ri_type": row.ri_type or "unknown",
            "start_ts": row.start_ts,
            "stop_ts": row.stop_ts,
            "energy_wh": row.energy_wh,
            "work": row.work,
        }
        for row in rows
    ]
    df = pd.DataFrame(data)
    df["start_ts"] = pd.to_datetime(df["start_ts"], utc=True)
    df["stop_ts"] = pd.to_datetime(df["stop_ts"], utc=True)
    df = df[df["energy_wh"].notna() & (df["energy_wh"] > 0)].copy()
    if df.empty:
        return pd.DataFrame(columns=ENERGY_FEATURE_COLUMNS)

    durations = (df["stop_ts"] - df["start_ts"]).dt.total_seconds()
    valid_durations = durations[durations.notna() & (durations >= 0)]
    median_duration = float(valid_durations.median()) if not valid_durations.empty else 0.0
    df["duration_s"] = durations.fillna(median_duration).clip(lower=0)
    df["work"] = pd.to_numeric(df["work"], errors="coerce").fillna(0.0)
    df["hour"] = df["start_ts"].dt.hour
    df["day_of_week"] = df["start_ts"].dt.dayofweek
    df.sort_values(["site_id", "start_ts"], inplace=True)

    rolling_frames = []
    for _, group in df.groupby("site_id", sort=False):
        group = group.sort_values("start_ts").set_index("start_ts")
        group["records_count_site_24h"] = group["energy_wh"].rolling("24h", min_periods=1).count().to_numpy()
        group["rolling_energy_mean_site_24h"] = group["energy_wh"].rolling("24h", min_periods=1).mean().to_numpy()
        group["rolling_work_mean_site_24h"] = group["work"].rolling("24h", min_periods=1).mean().to_numpy()
        rolling_frames.append(group.reset_index())

    df = pd.concat(rolling_frames, ignore_index=True).sort_values("start_ts")
    return df[ENERGY_FEATURE_COLUMNS].reset_index(drop=True)


def _status_score(value: str | None) -> float:
    return {
        "DOWN": 0.0,
        "MAINTENANCE": 0.0,
        "DEGRADED": 0.5,
        "UP": 1.0,
        "AVAILABLE": 1.0,
        "OK": 1.0,
    }.get(str(value or "UP").upper(), 0.5)


def _payload_number(payload: dict | None, names: tuple[str, ...]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for name in names:
        value = payload.get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _capacity_targets(payload: dict | None) -> dict[str, float | None]:
    capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
    return {
        "compute_capacity": _first_not_none(
            _payload_number(payload, ("compute_capacity", "total_vcpus", "vcpus_total", "cpu_capacity")),
            _payload_number(capabilities, ("compute_capacity", "total_vcpus", "vcpus_total", "cpu_capacity")),
        ),
        "gpu_capacity": _first_not_none(
            _payload_number(payload, ("gpu_capacity", "total_gpus", "gpus_total")),
            _payload_number(capabilities, ("gpu_capacity", "total_gpus", "gpus_total")),
        ),
        "storage_capacity": _first_not_none(
            _payload_number(payload, ("storage_capacity", "storage_gb", "total_disk_gb", "disk_gb_total")),
            _payload_number(capabilities, ("storage_capacity", "storage_gb", "total_disk_gb", "disk_gb_total")),
        ),
    }


def _status_training_row(row: SiteStatusSnapshot) -> dict:
    capacity = _capacity_targets(row.raw_json)
    return {
        "site_id": row.site_id or "unknown-site",
        "ri_type": row.ri_type or "unknown",
        "timestamp": row.timestamp,
        "operational_status_score": _status_score(row.operational_status),
        "maintenance_flag": 1.0 if row.maintenance_flag else 0.0,
        "node_availability": row.node_availability,
        "link_availability": row.link_availability,
        **capacity,
        "free_cpu_capacity": row.free_cpu_capacity,
        "free_gpu_capacity": row.free_gpu_capacity,
        "queue_length": row.queue_length,
        "provisioning_delay_s": row.provisioning_delay_s,
        "load_index": row.load_index,
        "cpu_util_avg": row.cpu_util_avg,
    }


def _snapshot_status_payload(snapshot: SiteSnapshot) -> dict:
    return {
        "site_id": snapshot.site_id,
        "timestamp": snapshot.ts,
        **(snapshot.availability or {}),
        **(snapshot.usage or {}),
        **(snapshot.efficiency or {}),
        **(snapshot.status or {}),
        **(snapshot.quality or {}),
    }


def _snapshot_training_row(snapshot: SiteSnapshot) -> dict | None:
    try:
        status = normalise_site_status(_snapshot_status_payload(snapshot))
    except ValueError:
        return None
    capacity = _capacity_targets(snapshot.raw_json)
    return {
        "site_id": status["site_id"] or "unknown-site",
        "ri_type": status.get("ri_type") or "unknown",
        "timestamp": status["timestamp"],
        "operational_status_score": _status_score(status.get("operational_status")),
        "maintenance_flag": 1.0 if status.get("maintenance_flag") else 0.0,
        "node_availability": status.get("node_availability"),
        "link_availability": status.get("link_availability"),
        **capacity,
        "free_cpu_capacity": status.get("free_cpu_capacity"),
        "free_gpu_capacity": status.get("free_gpu_capacity"),
        "queue_length": status.get("queue_length"),
        "provisioning_delay_s": status.get("provisioning_delay_s"),
        "load_index": status.get("load_index"),
        "cpu_util_avg": status.get("cpu_util_avg"),
    }


def build_site_status_training_frame(session: Session, site_id: str | None = None) -> pd.DataFrame:
    from m3l2.app.operator_config import usable_status_rows

    rows = usable_status_rows(session, site_id)
    snapshots = session.execute(select(SiteSnapshot)).scalars().all()
    if site_id:
        snapshots = [snapshot for snapshot in snapshots if snapshot.site_id == site_id]
    if not rows and not snapshots:
        return pd.DataFrame(columns=STATUS_FEATURE_COLUMNS + STATUS_TARGET_COLUMNS)

    data = [_status_training_row(row) for row in rows]
    existing_keys = {(item["site_id"], item["timestamp"]) for item in data}
    for snapshot in snapshots:
        if (snapshot.site_id, snapshot.ts) in existing_keys:
            continue
        item = _snapshot_training_row(snapshot)
        if item is not None:
            data.append(item)
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["timestamp"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=STATUS_FEATURE_COLUMNS + STATUS_TARGET_COLUMNS)

    df["node_availability"] = pd.to_numeric(df["node_availability"], errors="coerce")
    df["link_availability"] = pd.to_numeric(df["link_availability"], errors="coerce")
    df["compute_capacity"] = pd.to_numeric(df["compute_capacity"], errors="coerce")
    df["gpu_capacity"] = pd.to_numeric(df["gpu_capacity"], errors="coerce")
    df["storage_capacity"] = pd.to_numeric(df["storage_capacity"], errors="coerce")
    df["free_cpu_capacity"] = pd.to_numeric(df["free_cpu_capacity"], errors="coerce")
    df["free_gpu_capacity"] = pd.to_numeric(df["free_gpu_capacity"], errors="coerce")
    df["queue_length"] = pd.to_numeric(df["queue_length"], errors="coerce")
    df["provisioning_delay_s"] = pd.to_numeric(df["provisioning_delay_s"], errors="coerce")
    df["load_index"] = pd.to_numeric(df["load_index"], errors="coerce")
    df["cpu_util_avg"] = pd.to_numeric(df["cpu_util_avg"], errors="coerce")
    df = df.dropna(subset=["node_availability", "link_availability", "free_cpu_capacity", "queue_length", "load_index"])
    if df.empty:
        return pd.DataFrame(columns=STATUS_FEATURE_COLUMNS + STATUS_TARGET_COLUMNS)

    df["node_availability"] = df["node_availability"].clip(lower=0.0, upper=1.0)
    df["link_availability"] = df["link_availability"].clip(lower=0.0, upper=1.0)
    df["free_cpu_capacity"] = df["free_cpu_capacity"].clip(lower=0.0)
    df["free_gpu_capacity"] = df["free_gpu_capacity"].fillna(0.0).clip(lower=0.0)
    df["compute_capacity"] = df["compute_capacity"].fillna(df["free_cpu_capacity"]).clip(lower=0.0)
    df["gpu_capacity"] = df["gpu_capacity"].fillna(df["free_gpu_capacity"]).clip(lower=0.0)
    df["storage_capacity"] = df["storage_capacity"].fillna(0.0).clip(lower=0.0)
    df["queue_length"] = df["queue_length"].fillna(0.0).clip(lower=0.0)
    df["provisioning_delay_s"] = df["provisioning_delay_s"].fillna(0.0).clip(lower=0.0)
    inferred_load = (df["cpu_util_avg"].fillna(0.0) / 100.0) + (df["queue_length"] / 50.0)
    df["load_index"] = df["load_index"].fillna(inferred_load).fillna(0.0).clip(lower=0.0, upper=1.0)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df.sort_values(["site_id", "timestamp"], inplace=True)

    rolling_frames = []
    for _, group in df.groupby("site_id", sort=False):
        group = group.sort_values("timestamp").set_index("timestamp")
        group["records_count_site_24h"] = group["node_availability"].rolling("24h", min_periods=1).count().to_numpy()
        group["rolling_availability_mean_site_24h"] = group["node_availability"].rolling("24h", min_periods=1).mean().to_numpy()
        group["rolling_free_cpu_mean_site_24h"] = group["free_cpu_capacity"].rolling("24h", min_periods=1).mean().to_numpy()
        group["rolling_queue_mean_site_24h"] = group["queue_length"].rolling("24h", min_periods=1).mean().to_numpy()
        group["rolling_load_mean_site_24h"] = group["load_index"].rolling("24h", min_periods=1).mean().to_numpy()
        rolling_frames.append(group.reset_index())

    df = pd.concat(rolling_frames, ignore_index=True).sort_values("timestamp")
    return df[STATUS_FEATURE_COLUMNS + STATUS_TARGET_COLUMNS].reset_index(drop=True)
