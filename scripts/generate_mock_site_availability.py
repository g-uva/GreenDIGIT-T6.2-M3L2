#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_site_inputs(path: Path) -> tuple[dict[str, dict], datetime]:
    sites: dict[str, dict] = defaultdict(lambda: {"max_ncores": 0.0, "rows": 0})
    first_ts: datetime | None = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            site_id = row["site_id"]
            sites[site_id]["max_ncores"] = max(sites[site_id]["max_ncores"], float(row.get("ncores") or 0))
            sites[site_id]["rows"] += 1
            ts = parse_ts(row["bucket_15m"])
            first_ts = ts if first_ts is None else min(first_ts, ts)
    if first_ts is None:
        raise ValueError(f"No rows in {path}")
    return sites, first_ts.replace(minute=0, second=0, microsecond=0)


def profile_for(site_id: str, index: int, max_ncores: float) -> dict:
    ri_types = ["network", "cloud", "grid"]
    ri_type = ri_types[index % len(ri_types)]
    compute_capacity = max(32.0, math.ceil(max_ncores * 1.25 / 16.0) * 16.0)
    gpu_capacity = 4.0 if ri_type == "cloud" else 1.0 if ri_type == "grid" else 0.0
    return {
        "site_id": site_id,
        "ri_type": ri_type,
        "location": f"MOCK-{index + 1}",
        "compute_capacity": compute_capacity,
        "gpu_capacity": gpu_capacity,
        "storage_capacity": 2048.0 + index * 1024.0,
        "network_topology": "Mesh" if ri_type == "network" else "Hybrid",
        "link_capacities": {"uplink_mbps": 1000 + index * 500},
        "supported_workload_types": ["batch", "stream", "ml"] if ri_type != "grid" else ["batch", "cpu"],
        "energy_capabilities": {"metering": "mock", "cpu": True, "gpu": gpu_capacity > 0},
        "static_pue_baseline": round(1.15 + index * 0.05, 3),
    }


def status_rows(profile: dict, start: datetime, hours: int, index: int) -> list[dict]:
    rows = []
    compute = float(profile["compute_capacity"])
    gpu = float(profile["gpu_capacity"] or 0)
    total_nodes = 12 + index * 6
    total_links = 8 + index * 3
    maintenance_start = start + timedelta(hours=24 + index * 12)
    maintenance_end = maintenance_start + timedelta(hours=2)

    for hour in range(hours):
        ts = start + timedelta(hours=hour)
        daily = (math.sin((hour % 24) / 24.0 * 2.0 * math.pi - math.pi / 2.0) + 1.0) / 2.0
        weekly = (math.sin(hour / 168.0 * 2.0 * math.pi + index) + 1.0) / 2.0
        in_maintenance = maintenance_start <= ts < maintenance_end
        degraded = not in_maintenance and hour % (37 + index * 5) == 0
        operational_status = "DOWN" if in_maintenance else "DEGRADED" if degraded else "UP"

        alive_nodes = total_nodes - (2 if degraded else 0) - (total_nodes if in_maintenance else 0)
        active_links = total_links - (1 if degraded else 0) - (total_links if in_maintenance else 0)
        cpu_util = 100.0 if in_maintenance else min(95.0, 20.0 + 55.0 * daily + 15.0 * weekly)
        gpu_util = None if gpu == 0 else min(95.0, 10.0 + 60.0 * weekly)
        queue_length = 0 if in_maintenance else max(0, int((cpu_util - 55.0) / 8.0) + (2 if degraded else 0))
        provisioning_delay = 0.0 if in_maintenance else 30.0 + queue_length * 45.0 + (120.0 if degraded else 0.0)
        free_cpu = 0.0 if in_maintenance else max(0.0, compute * (1.0 - cpu_util / 100.0))
        free_gpu = None if gpu == 0 else (0.0 if in_maintenance else max(0.0, gpu * (1.0 - (gpu_util or 0) / 100.0)))
        node_availability = max(0.0, alive_nodes / total_nodes)
        link_availability = max(0.0, active_links / total_links)
        carbon_intensity = 120.0 + 80.0 * daily + index * 20.0
        pue = profile["static_pue_baseline"] + 0.03 * weekly

        rows.append(
            {
                "site_id": profile["site_id"],
                "ri_type": profile["ri_type"],
                "timestamp": ts.isoformat(),
                "operational_status": operational_status,
                "maintenance_flag": str(in_maintenance).lower(),
                "scheduled_maintenance": json.dumps({"start": maintenance_start.isoformat(), "end": maintenance_end.isoformat()}),
                "node_availability": round(node_availability, 4),
                "link_availability": round(link_availability, 4),
                "stability_score": round(0.98 - (0.2 if degraded else 0.0) - (0.98 if in_maintenance else 0.0), 4),
                "packet_loss": round(0.2 + (3.0 if degraded else 0.0) + 0.5 * weekly, 4),
                "network_jitter": round(2.0 + (20.0 if degraded else 0.0) + 4.0 * daily, 4),
                "network_utilization": round(min(100.0, 25.0 + 50.0 * daily), 4),
                "available_bandwidth": round(max(0.0, profile["link_capacities"]["uplink_mbps"] * (1.0 - daily * 0.75)), 4),
                "cpu_util_avg": round(cpu_util, 4),
                "gpu_util_avg": "" if gpu_util is None else round(gpu_util, 4),
                "free_cpu_capacity": round(free_cpu, 4),
                "free_gpu_capacity": "" if free_gpu is None else round(free_gpu, 4),
                "queue_length": queue_length,
                "remaining_jobs": 0 if in_maintenance else max(1, int(cpu_util / 12.0)),
                "provisioning_delay_s": round(provisioning_delay, 4),
                "load_index": round(min(1.0, cpu_util / 100.0 + queue_length / 50.0), 4),
                "energy_consumed": round(compute * cpu_util / 100.0 * pue * 12.5, 4),
                "pue_estimate": round(pue, 4),
                "carbon_intensity": round(carbon_intensity, 4),
                "energy_per_task_proxy": round((compute * max(cpu_util, 1.0) / 100.0) * 8.0, 4),
                "update_frequency": 3600,
                "data_confidence": 0.5 if in_maintenance else 0.75 if degraded else 0.92,
                "coverage_ratio": round(node_availability, 4),
                "stale_flag": "false",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate mock Site Adapter profile/status data.")
    parser.add_argument("--source", default="raw_data/summary_sites_15m.csv")
    parser.add_argument("--profiles-out", default="raw_data/mock_site_profiles.json")
    parser.add_argument("--status-out", default="raw_data/mock_site_status_snapshots.csv")
    parser.add_argument("--hours", type=int, default=168)
    args = parser.parse_args()

    sites, start = read_site_inputs(Path(args.source))
    profiles = [profile_for(site_id, idx, meta["max_ncores"]) for idx, (site_id, meta) in enumerate(sorted(sites.items()))]
    status = []
    for idx, profile in enumerate(profiles):
        status.extend(status_rows(profile, start, args.hours, idx))

    profiles_path = Path(args.profiles_out)
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    profiles_path.write_text(json.dumps(profiles, indent=2) + "\n")

    status_path = Path(args.status_out)
    with status_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(status[0].keys()))
        writer.writeheader()
        writer.writerows(status)

    print(json.dumps({"profiles": len(profiles), "status_rows": len(status), "hours": args.hours}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
