#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def default_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def snapshot(site_id: str, ts: datetime, index: int, step_minutes: int) -> dict:
    day_fraction = ((ts.hour * 60 + ts.minute) / 1440.0) % 1.0
    yearly_fraction = ((ts.timetuple().tm_yday - 1) / 365.0) % 1.0
    daily = (math.sin(day_fraction * 2.0 * math.pi - math.pi / 2.0) + 1.0) / 2.0
    seasonal = (math.sin(yearly_fraction * 2.0 * math.pi) + 1.0) / 2.0
    degraded = index % max(1, int((24 * 60) / step_minutes * 14)) == 0

    cpu_util = min(95.0, 8.0 + 62.0 * daily + 12.0 * seasonal)
    node_availability = 0.9 if degraded else 1.0
    link_availability = 0.8 if degraded else 1.0
    queue_length = 2 if degraded else max(0, int((cpu_util - 70.0) / 10.0))
    load_index = min(1.0, cpu_util / 100.0 + queue_length / 50.0)
    available_bandwidth = max(0.0, 1000.0 * (1.0 - daily * 0.75))
    energy_consumed = 64.0 * cpu_util / 100.0 * 1.2 * 12.5
    timestamp = ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    return {
        "ts": ts.isoformat(),
        "capabilities": {
            "mock": True,
            "generated_by": "scripts/submit_mock_site_snapshots.py",
            "site_id": site_id,
            "ri_type": "grid",
        },
        "availability": {
            "mock": True,
            "node_availability": round(node_availability, 4),
            "link_availability": round(link_availability, 4),
            "stability_score": round(0.9 if degraded else 1.0, 4),
            "packet_loss": None if degraded else 0.0,
            "network_jitter": 2.5 if degraded else 1.0,
            "network_utilization": round(cpu_util / 10.0, 4),
            "available_bandwidth": round(available_bandwidth, 4),
        },
        "usage": {
            "mock": True,
            "cpu_util_avg": round(cpu_util, 4),
            "queue_length": queue_length,
            "remaining_jobs": queue_length,
            "load_index": round(load_index, 4),
        },
        "efficiency": {
            "mock": True,
            "energy_consumed": round(energy_consumed, 4),
            "pue_estimate": 1.2,
            "carbon_intensity": 250.0,
            "energy_per_task_proxy": 0.0 if queue_length == 0 else round(energy_consumed / queue_length, 4),
        },
        "status": {
            "mock": True,
            "operational_status": "DEGRADED" if degraded else "UP",
            "maintenance_flag": False,
            "scheduled_maintenance": {"start": None, "end": None},
        },
        "quality": {
            "mock": True,
            "timestamp": timestamp,
            "update_frequency": step_minutes * 60,
            "data_confidence": 1.0,
            "coverage_ratio": node_availability,
            "stale_flag": False,
        },
    }


def post_json(url: str, token: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {response.read().decode('utf-8', errors='replace')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit generated mock L2 site snapshots.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--site-id", default="SARA-MATRIX")
    parser.add_argument("--start", default=None, help="UTC start timestamp, default: today at 00:00 UTC")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--step-minutes", type=int, default=60)
    parser.add_argument("--token", default=None)
    parser.add_argument("--token-env", default="TOKEN")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between POSTs.")
    args = parser.parse_args()

    token = args.token or os.getenv(args.token_env)
    if not token and not args.dry_run:
        print(f"Missing bearer token. Set {args.token_env} or pass --token.", file=sys.stderr)
        return 2

    start = parse_ts(args.start) if args.start else default_start()
    total = int(args.days * 24 * 60 / args.step_minutes)
    endpoint = f"{args.base_url.rstrip('/')}/l2/sites/{args.site_id}/snapshots"

    first_payload = snapshot(args.site_id, start, 0, args.step_minutes)
    if args.dry_run:
        print(json.dumps({"endpoint": endpoint, "records": total, "first_payload": first_payload}, indent=2))
        return 0

    submitted = 0
    for index in range(total):
        payload = snapshot(args.site_id, start + timedelta(minutes=args.step_minutes * index), index, args.step_minutes)
        try:
            post_json(endpoint, token or "", payload)
        except (HTTPError, URLError, RuntimeError) as exc:
            print(json.dumps({"submitted": submitted, "failed_at": payload["ts"], "error": str(exc)}), file=sys.stderr)
            return 1
        submitted += 1
        if submitted % 100 == 0:
            print(json.dumps({"submitted": submitted, "total": total}))
        if args.sleep:
            time.sleep(args.sleep)

    print(json.dumps({"submitted": submitted, "site_id": args.site_id, "start": start.isoformat(), "days": args.days}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
