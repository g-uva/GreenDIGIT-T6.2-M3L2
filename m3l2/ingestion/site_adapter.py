from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


ADAPTER_TYPES = {"generic", "iot", "openstack"}
RI_TYPES = {"network", "cloud", "grid"}
OPERATIONAL_STATUSES = {"UP", "DOWN", "DEGRADED", "MAINTENANCE"}

PROFILE_FIELDS = {
    "site_id",
    "ri_type",
    "location",
    "compute_capacity",
    "gpu_capacity",
    "storage_capacity",
    "network_topology",
    "link_capacities",
    "supported_workload_types",
    "energy_capabilities",
    "static_pue_baseline",
}

STATUS_FIELDS = {
    "site_id",
    "ri_type",
    "timestamp",
    "operational_status",
    "maintenance_flag",
    "scheduled_maintenance",
    "node_availability",
    "link_availability",
    "stability_score",
    "packet_loss",
    "network_jitter",
    "network_utilization",
    "available_bandwidth",
    "cpu_util_avg",
    "gpu_util_avg",
    "free_cpu_capacity",
    "free_gpu_capacity",
    "queue_length",
    "remaining_jobs",
    "provisioning_delay_s",
    "load_index",
    "energy_consumed",
    "pue_estimate",
    "carbon_intensity",
    "energy_per_task_proxy",
    "update_frequency",
    "data_confidence",
    "coverage_ratio",
    "stale_flag",
}

PROFILE_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "generic": {
        "site_id": ("site", "site_name"),
        "static_pue_baseline": ("pue",),
    },
    "iot": {
        "site_id": ("site", "site_name"),
        "location": ("facility",),
        "compute_capacity": ("total_nodes", "node_count"),
        "network_topology": ("topology",),
        "static_pue_baseline": ("pue",),
    },
    "openstack": {
        "site_id": ("cloud_name", "name"),
        "location": ("region_name", "availability_zone"),
        "compute_capacity": ("total_vcpus", "vcpus_total", "cpu_capacity"),
        "gpu_capacity": ("total_gpus", "gpus_total"),
        "storage_capacity": ("total_disk_gb", "disk_gb_total"),
        "static_pue_baseline": ("pue",),
    },
}

STATUS_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "generic": {
        "site_id": ("site", "site_name"),
        "timestamp": ("ts",),
        "operational_status": ("state", "status"),
        "cpu_util_avg": ("cpu_utilization", "cpu_utilisation", "cpu_util", "cpu_util_percent"),
        "gpu_util_avg": ("gpu_utilization", "gpu_utilisation", "gpu_util", "gpu_util_percent"),
        "stability_score": ("stability", "stability_index"),
        "stale_flag": ("stale", "is_stale"),
        "network_jitter": ("jitter_ms",),
        "packet_loss": ("packet_loss_percent",),
        "available_bandwidth": ("available_bandwidth_mbps",),
        "energy_consumed": ("energy_wh",),
        "carbon_intensity": ("ci_gco2_kwh",),
        "pue_estimate": ("pue",),
    },
    "iot": {
        "site_id": ("site", "site_name"),
        "timestamp": ("ts", "bucket_15m", "updated_at"),
        "operational_status": ("state", "status"),
        "maintenance_flag": ("maintenance",),
        "cpu_util_avg": ("cpu_utilization", "cpu_utilisation", "cpu_util", "cpu_util_percent"),
        "gpu_util_avg": ("gpu_utilization", "gpu_utilisation", "gpu_util", "gpu_util_percent"),
        "stability_score": ("stability", "stability_index"),
        "stale_flag": ("stale", "is_stale"),
        "network_jitter": ("jitter_ms",),
        "packet_loss": ("packet_loss_percent",),
        "available_bandwidth": ("available_bandwidth_mbps",),
        "energy_consumed": ("energy_wh",),
        "carbon_intensity": ("ci_gco2_kwh",),
        "pue_estimate": ("pue",),
    },
    "openstack": {
        "site_id": ("cloud_name", "name"),
        "timestamp": ("ts", "updated_at"),
        "operational_status": ("state", "status"),
        "maintenance_flag": ("maintenance", "planned_maintenance"),
        "cpu_util_avg": ("cpu_utilization", "cpu_utilisation", "cpu_util", "cpu_util_percent"),
        "gpu_util_avg": ("gpu_utilization", "gpu_utilisation", "gpu_util", "gpu_util_percent"),
        "free_cpu_capacity": ("free_vcpus", "vcpus_free"),
        "free_gpu_capacity": ("free_gpus", "gpus_free"),
        "queue_length": ("pending_vms", "pending_jobs"),
        "remaining_jobs": ("active_vms", "running_jobs"),
        "provisioning_delay_s": ("vm_provisioning_delay_s",),
        "energy_consumed": ("energy_wh",),
        "carbon_intensity": ("ci_gco2_kwh",),
        "pue_estimate": ("pue",),
    },
}

DERIVED_STATUS_FIELDS: dict[str, tuple[str, ...]] = {
    "iot": ("alive_nodes", "active_nodes", "total_nodes", "node_count", "active_links", "total_links"),
    "openstack": ("total_vcpus", "vcpus_total", "compute_capacity", "total_gpus", "gpus_total", "gpu_capacity"),
}


class SiteAdapterValidationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        super().__init__("site adapter submission validation failed")


def _err(field: str, message: str, code: str = "value_error") -> dict[str, Any]:
    return {"loc": [field], "msg": message, "type": code}


def _adapter_type(value: str) -> str:
    normalized = str(value or "generic").strip().lower()
    if normalized not in ADAPTER_TYPES:
        raise SiteAdapterValidationError([_err("adapter_type", "adapter_type must be one of generic, iot or openstack")])
    return normalized


def _equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def _aliases_for(alias_config: dict[str, tuple[str, ...]], field: str) -> tuple[str, ...]:
    return (field, *alias_config.get(field, ()))


def _canonicalise(
    payload: dict[str, Any],
    fields: set[str],
    aliases: dict[str, tuple[str, ...]],
    extra_recognised: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[dict[str, Any]]]:
    mapped: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    recognised_names = set(fields) | set(extra_recognised)
    for field in fields:
        names = _aliases_for(aliases, field)
        recognised_names.update(names)
        present = [(name, payload[name]) for name in names if name in payload]
        if not present:
            continue
        concrete = [(name, value) for name, value in present if value is not None]
        if not concrete:
            mapped[field] = None
            continue
        first_name, first_value = concrete[0]
        for name, value in concrete[1:]:
            if not _equivalent(first_value, value):
                errors.append(
                    _err(
                        field,
                        f"conflicting values for {field}: {first_name} and {name}",
                        "value_error.alias_conflict",
                    )
                )
                break
        mapped[field] = first_value

    explicit_extensions = payload.get("extensions")
    extension_names = [name for name in payload if name not in recognised_names and name != "extensions"]
    extensions = {name: payload[name] for name in extension_names}
    if explicit_extensions is not None:
        if not isinstance(explicit_extensions, dict):
            errors.append(_err("extensions", "extensions must be an object", "type_error.dict"))
        else:
            overlap = sorted(set(extensions).intersection(explicit_extensions))
            if overlap:
                errors.append(_err("extensions", f"extensions duplicates top-level fields: {', '.join(overlap)}"))
            extensions.update(explicit_extensions)
            extension_names.extend(str(name) for name in explicit_extensions)
    return mapped, extensions, sorted(set(extension_names)), errors


def _optional_string(value: Any, field: str, errors: list[dict[str, Any]]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append(_err(field, f"{field} must be a string", "type_error.str"))
        return None
    stripped = value.strip()
    if not stripped:
        errors.append(_err(field, f"{field} must not be empty"))
        return None
    return stripped


def _required_string(value: Any, field: str, errors: list[dict[str, Any]]) -> str | None:
    parsed = _optional_string(value, field, errors)
    if parsed is None and not any(error["loc"] == [field] for error in errors):
        errors.append(_err(field, f"{field} is required", "value_error.missing"))
    return parsed


def _ri_type(value: Any, errors: list[dict[str, Any]]) -> str:
    if value is None:
        return "unknown"
    if not isinstance(value, str):
        errors.append(_err("ri_type", "ri_type must be one of network, cloud or grid", "type_error.enum"))
        return "unknown"
    normalized = value.strip().lower()
    if normalized not in RI_TYPES:
        errors.append(_err("ri_type", "ri_type must be one of network, cloud or grid", "value_error.enum"))
        return "unknown"
    return normalized


def _datetime(value: Any, field: str, errors: list[dict[str, Any]], required: bool = False) -> datetime | None:
    if value is None:
        if required:
            errors.append(_err(field, f"{field} is required", "value_error.missing"))
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        if not value.strip():
            errors.append(_err(field, f"{field} must not be empty"))
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            errors.append(_err(field, f"{field} must be an ISO-8601 timestamp", "value_error.datetime"))
            return None
    else:
        errors.append(_err(field, f"{field} must be an ISO-8601 timestamp", "type_error.datetime"))
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _number(
    value: Any,
    field: str,
    errors: list[dict[str, Any]],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        errors.append(_err(field, f"{field} must be a finite number", "type_error.float"))
        return None
    parsed = float(value)
    if minimum is not None and parsed < minimum:
        errors.append(_err(field, f"{field} must be greater than or equal to {minimum}", "value_error.range"))
    if exclusive_minimum is not None and parsed <= exclusive_minimum:
        errors.append(_err(field, f"{field} must be greater than {exclusive_minimum}", "value_error.range"))
    if maximum is not None and parsed > maximum:
        errors.append(_err(field, f"{field} must be less than or equal to {maximum}", "value_error.range"))
    return parsed


def _integer(value: Any, field: str, errors: list[dict[str, Any]], *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(_err(field, f"{field} must be an integer", "type_error.integer"))
        return None
    if minimum is not None and value < minimum:
        errors.append(_err(field, f"{field} must be greater than or equal to {minimum}", "value_error.range"))
    return value


def _boolean(value: Any, field: str, errors: list[dict[str, Any]]) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        errors.append(_err(field, f"{field} must be a boolean", "type_error.bool"))
        return None
    return value


def _dict(value: Any, field: str, errors: list[dict[str, Any]]) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        errors.append(_err(field, f"{field} must be an object", "type_error.dict"))
        return None
    return value


def _string_list(value: Any, field: str, errors: list[dict[str, Any]]) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        errors.append(_err(field, f"{field} must be a list of non-empty strings", "type_error.list"))
        return None
    return [item.strip() for item in value]


def _operational_status(value: Any, errors: list[dict[str, Any]]) -> str:
    if value is None:
        return "UP"
    if not isinstance(value, str):
        errors.append(_err("operational_status", "operational_status must be one of UP, DOWN, DEGRADED or MAINTENANCE"))
        return "UP"
    normalized = value.strip().upper()
    if normalized not in OPERATIONAL_STATUSES:
        errors.append(_err("operational_status", "operational_status must be one of UP, DOWN, DEGRADED or MAINTENANCE"))
        return "UP"
    return normalized


def _first_present_number(payload: dict[str, Any], names: tuple[str, ...], field: str, errors: list[dict[str, Any]]) -> float | None:
    present = [(name, payload[name]) for name in names if name in payload and payload[name] is not None]
    if not present:
        return None
    first_name, first_value = present[0]
    parsed = _number(first_value, first_name, errors, minimum=0)
    for name, value in present[1:]:
        candidate = _number(value, name, errors, minimum=0)
        if parsed is not None and candidate is not None and not _equivalent(parsed, candidate):
            errors.append(_err(field, f"conflicting values for {field}: {first_name} and {name}", "value_error.alias_conflict"))
    return parsed


def _derive_ratio(
    payload: dict[str, Any],
    *,
    field: str,
    active_names: tuple[str, ...],
    total_names: tuple[str, ...],
    explicit: float | None,
    errors: list[dict[str, Any]],
) -> float | None:
    active = _first_present_number(payload, active_names, field, errors)
    total = _first_present_number(payload, total_names, field, errors)
    if active is None and total is None:
        return explicit
    if active is None or total is None:
        errors.append(_err(field, f"{field} derivation requires both active and total counts"))
        return explicit
    if total <= 0:
        errors.append(_err(field, f"{field} total count must be greater than 0", "value_error.range"))
        return explicit
    if active > total:
        errors.append(_err(field, f"{field} active count must be less than or equal to total count"))
        return explicit
    derived = active / total
    if explicit is not None and not math.isclose(explicit, derived, rel_tol=1e-6, abs_tol=1e-6):
        errors.append(_err(field, f"{field} conflicts with derived value from counts", "value_error.alias_conflict"))
        return explicit
    return derived


def _derive_utilisation(
    payload: dict[str, Any],
    *,
    field: str,
    total_names: tuple[str, ...],
    free_field: str,
    explicit: float | None,
    free_value: float | None,
    errors: list[dict[str, Any]],
) -> float | None:
    total = _first_present_number(payload, total_names, field, errors)
    if total is None:
        return explicit
    if total <= 0:
        errors.append(_err(field, f"{field} total capacity must be greater than 0", "value_error.range"))
        return explicit
    if free_value is None:
        return explicit
    if free_value > total:
        errors.append(_err(free_field, f"{free_field} must be less than or equal to total capacity"))
        return explicit
    derived = 100.0 * (1.0 - free_value / total)
    if explicit is not None and not math.isclose(explicit, derived, rel_tol=1e-6, abs_tol=1e-6):
        errors.append(_err(field, f"{field} conflicts with derived value from capacity counts", "value_error.alias_conflict"))
        return explicit
    return derived


def normalise_site_profile(payload: dict[str, Any], adapter_type: str = "generic") -> dict[str, Any]:
    adapter = _adapter_type(adapter_type)
    mapped, extensions, warning_fields, errors = _canonicalise(payload, PROFILE_FIELDS, PROFILE_ALIASES[adapter])

    profile = {
        "site_id": _required_string(mapped.get("site_id"), "site_id", errors),
        "ri_type": _ri_type(mapped.get("ri_type"), errors),
        "location": _optional_string(mapped.get("location"), "location", errors),
        "compute_capacity": _number(mapped.get("compute_capacity"), "compute_capacity", errors, minimum=0),
        "gpu_capacity": _number(mapped.get("gpu_capacity"), "gpu_capacity", errors, minimum=0),
        "storage_capacity": _number(mapped.get("storage_capacity"), "storage_capacity", errors, minimum=0),
        "network_topology": _optional_string(mapped.get("network_topology"), "network_topology", errors),
        "link_capacities": _dict(mapped.get("link_capacities"), "link_capacities", errors),
        "supported_workload_types": _string_list(mapped.get("supported_workload_types"), "supported_workload_types", errors),
        "energy_capabilities": _dict(mapped.get("energy_capabilities"), "energy_capabilities", errors),
        "static_pue_baseline": _number(mapped.get("static_pue_baseline"), "static_pue_baseline", errors, minimum=1),
        "extensions": extensions,
        "raw_json": payload,
    }
    if errors:
        raise SiteAdapterValidationError(errors)
    profile["_warnings"] = warning_fields
    return profile


def normalise_site_status(payload: dict[str, Any], adapter_type: str = "generic") -> dict[str, Any]:
    adapter = _adapter_type(adapter_type)
    mapped, extensions, warning_fields, errors = _canonicalise(
        payload,
        STATUS_FIELDS,
        STATUS_ALIASES[adapter],
        DERIVED_STATUS_FIELDS.get(adapter, ()),
    )

    node_availability = _number(mapped.get("node_availability"), "node_availability", errors, minimum=0, maximum=1)
    link_availability = _number(mapped.get("link_availability"), "link_availability", errors, minimum=0, maximum=1)
    free_cpu_capacity = _number(mapped.get("free_cpu_capacity"), "free_cpu_capacity", errors, minimum=0)
    free_gpu_capacity = _number(mapped.get("free_gpu_capacity"), "free_gpu_capacity", errors, minimum=0)
    cpu_util_avg = _number(mapped.get("cpu_util_avg"), "cpu_util_avg", errors, minimum=0, maximum=100)
    gpu_util_avg = _number(mapped.get("gpu_util_avg"), "gpu_util_avg", errors, minimum=0, maximum=100)

    if adapter == "iot":
        node_availability = _derive_ratio(
            payload,
            field="node_availability",
            active_names=("alive_nodes", "active_nodes"),
            total_names=("total_nodes", "node_count"),
            explicit=node_availability,
            errors=errors,
        )
        link_availability = _derive_ratio(
            payload,
            field="link_availability",
            active_names=("active_links",),
            total_names=("total_links",),
            explicit=link_availability,
            errors=errors,
        )
    elif adapter == "openstack":
        cpu_util_avg = _derive_utilisation(
            payload,
            field="cpu_util_avg",
            total_names=("total_vcpus", "vcpus_total", "compute_capacity"),
            free_field="free_cpu_capacity",
            explicit=cpu_util_avg,
            free_value=free_cpu_capacity,
            errors=errors,
        )
        gpu_util_avg = _derive_utilisation(
            payload,
            field="gpu_util_avg",
            total_names=("total_gpus", "gpus_total", "gpu_capacity"),
            free_field="free_gpu_capacity",
            explicit=gpu_util_avg,
            free_value=free_gpu_capacity,
            errors=errors,
        )

    maintenance_flag = _boolean(mapped.get("maintenance_flag"), "maintenance_flag", errors)
    stale_flag = _boolean(mapped.get("stale_flag"), "stale_flag", errors)
    operational_status = _operational_status(mapped.get("operational_status"), errors)
    if operational_status == "MAINTENANCE" and maintenance_flag is False:
        errors.append(_err("maintenance_flag", "maintenance_flag must be true when operational_status is MAINTENANCE"))

    status = {
        "site_id": _required_string(mapped.get("site_id"), "site_id", errors),
        "ri_type": _ri_type(mapped.get("ri_type"), errors),
        "timestamp": _datetime(mapped.get("timestamp"), "timestamp", errors, required=True),
        "operational_status": operational_status,
        "maintenance_flag": maintenance_flag if maintenance_flag is not None else False,
        "scheduled_maintenance": _dict(mapped.get("scheduled_maintenance"), "scheduled_maintenance", errors),
        "node_availability": node_availability,
        "link_availability": link_availability,
        "stability_score": _number(mapped.get("stability_score"), "stability_score", errors, minimum=0, maximum=1),
        "packet_loss": _number(mapped.get("packet_loss"), "packet_loss", errors, minimum=0, maximum=100),
        "network_jitter": _number(mapped.get("network_jitter"), "network_jitter", errors, minimum=0),
        "network_utilization": _number(mapped.get("network_utilization"), "network_utilization", errors, minimum=0, maximum=100),
        "available_bandwidth": _number(mapped.get("available_bandwidth"), "available_bandwidth", errors, minimum=0),
        "cpu_util_avg": cpu_util_avg,
        "gpu_util_avg": gpu_util_avg,
        "free_cpu_capacity": free_cpu_capacity,
        "free_gpu_capacity": free_gpu_capacity,
        "queue_length": _integer(mapped.get("queue_length"), "queue_length", errors, minimum=0),
        "remaining_jobs": _integer(mapped.get("remaining_jobs"), "remaining_jobs", errors, minimum=0),
        "provisioning_delay_s": _number(mapped.get("provisioning_delay_s"), "provisioning_delay_s", errors, minimum=0),
        "load_index": _number(mapped.get("load_index"), "load_index", errors, minimum=0, maximum=1),
        "energy_consumed": _number(mapped.get("energy_consumed"), "energy_consumed", errors, minimum=0),
        "pue_estimate": _number(mapped.get("pue_estimate"), "pue_estimate", errors, minimum=1),
        "carbon_intensity": _number(mapped.get("carbon_intensity"), "carbon_intensity", errors, minimum=0),
        "energy_per_task_proxy": _number(mapped.get("energy_per_task_proxy"), "energy_per_task_proxy", errors, minimum=0),
        "update_frequency": _number(mapped.get("update_frequency"), "update_frequency", errors, exclusive_minimum=0),
        "data_confidence": _number(mapped.get("data_confidence"), "data_confidence", errors, minimum=0, maximum=1),
        "coverage_ratio": _number(mapped.get("coverage_ratio"), "coverage_ratio", errors, minimum=0, maximum=1),
        "stale_flag": stale_flag if stale_flag is not None else False,
        "extensions": extensions,
        "raw_json": payload,
    }
    if errors:
        raise SiteAdapterValidationError(errors)
    status["_warnings"] = warning_fields
    return status
