from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import uuid4

import pydantic
from pydantic import BaseModel, Field, root_validator, validator

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic v1 compatibility
    ConfigDict = None


DURATION_RE = re.compile(r"^[1-9]\d*[hm]$")


if int(pydantic.VERSION.split(".", maxsplit=1)[0]) >= 2 and ConfigDict is not None:

    class APIModel(BaseModel):
        model_config = ConfigDict(populate_by_name=True, validate_by_name=True, extra="forbid")

else:

    class APIModel(BaseModel):
        class Config:
            allow_population_by_field_name = True
            extra = "forbid"


class IngestRunRequest(APIModel):
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    sites: list[str] | None = None


class TimeRequirements(APIModel):
    start_time: datetime | None = None
    duration: str | None = None
    deadline: datetime | None = None

    @validator("duration")
    def duration_must_be_positive(cls, value: str | None) -> str | None:
        if value is not None and not DURATION_RE.fullmatch(value):
            raise ValueError("duration must use a positive '<n>h' or '<n>m' value")
        return value

    @root_validator(skip_on_failure=True)
    def deadline_must_follow_start(cls, values: dict[str, Any]) -> dict[str, Any]:
        start = values.get("start_time")
        deadline = values.get("deadline")
        if start is not None and deadline is not None and deadline <= start:
            raise ValueError("deadline must be later than start_time")
        return values


class ResourceRequirements(APIModel):
    cpu: float | None = Field(default=None, ge=0)
    memory_gb: float | None = Field(default=None, ge=0)
    storage_gb: float | None = Field(default=None, ge=0)
    gpu: float | None = Field(default=None, ge=0)
    instances: int = Field(default=1, ge=1)
    flavour: str | None = None


class WorkloadDescriptor(APIModel):
    workload_id: str | None = None
    workload_type: str = "unknown"
    time_requirements: TimeRequirements = Field(default_factory=TimeRequirements)
    resource_requirements: ResourceRequirements = Field(default_factory=ResourceRequirements)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)
    ri_type: str | None = None
    work: float | None = Field(default=None, ge=0)
    duration_s: float | None = Field(default=None, ge=0)
    cpu_hours: float | None = Field(default=None, ge=0)
    gpu_hours: float | None = Field(default=None, ge=0)
    workload_class: str | None = Field(default=None, alias="class")


class CachePreference(APIModel):
    use_cache: bool = True
    allow_stale: bool = False
    max_age_minutes: int | None = Field(default=None, ge=1)


class PredictRequest(APIModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    candidate_site_ids: list[str] | None = Field(default=None, alias="site_ids")
    forecast_start_time: datetime | None = None
    horizon: str = "24h"
    step: str = "1h"
    workload: WorkloadDescriptor | None = None
    cache: CachePreference = Field(default_factory=CachePreference)
    use_cache: bool | None = None
    include_site_status: bool = False

    @validator("horizon", "step")
    def forecast_duration_must_be_positive(cls, value: str) -> str:
        if not DURATION_RE.fullmatch(value):
            raise ValueError("forecast horizon and step must use a positive '<n>h' or '<n>m' value")
        return value

    @root_validator(pre=True)
    def support_legacy_workload_dict(cls, values: dict[str, Any]) -> dict[str, Any]:
        workload = values.get("workload")
        if isinstance(workload, dict):
            legacy = dict(workload)
            if "requirements" in legacy and "resource_requirements" not in legacy:
                requirements = legacy.pop("requirements") or {}
                resource_requirements = {}
                aliases = {
                    "cpu": "cpu",
                    "cpu_cores": "cpu",
                    "memory_gb": "memory_gb",
                    "memory": "memory_gb",
                    "storage_gb": "storage_gb",
                    "storage": "storage_gb",
                    "gpu": "gpu",
                    "gpu_count": "gpu",
                    "instances": "instances",
                    "flavour": "flavour",
                    "flavor": "flavour",
                }
                for key, mapped in aliases.items():
                    if key in requirements:
                        resource_requirements[mapped] = requirements[key]
                legacy["resource_requirements"] = resource_requirements
                legacy.setdefault("extensions", {})["legacy_requirements"] = requirements
            values["workload"] = legacy
        return values

    @root_validator(skip_on_failure=True)
    def sync_legacy_cache_flag(cls, values: dict[str, Any]) -> dict[str, Any]:
        use_cache = values.get("use_cache")
        if use_cache is not None:
            cache = values.get("cache") or CachePreference()
            cache.use_cache = bool(use_cache)
            values["cache"] = cache
        return values

    @property
    def site_ids(self) -> list[str] | None:
        return self.candidate_site_ids


class ForecastPoint(APIModel):
    ts: datetime | str
    value: float
    unit: str = "Wh"


class SiteCapacity(APIModel):
    compute_capacity: float | None = None
    gpu_capacity: float | None = None
    storage_capacity: float | None = None
    free_cpu_capacity: float | None = None
    free_gpu_capacity: float | None = None
    queue_length: int | None = None
    provisioning_delay_s: float | None = None


class Feasibility(APIModel):
    status: str
    reasons: list[str] = Field(default_factory=list)


class SitePrediction(APIModel):
    site_id: str
    training_site_id: str | None = None
    registered_site_id: str | None = None
    requested_site_id: str | None = None
    site_id_resolution: str | None = None
    target: str = "l2_site_status"
    forecast: list[ForecastPoint]
    energy_forecast: list[ForecastPoint] = Field(default_factory=list)
    site_status_forecast: list[dict[str, Any]] = Field(default_factory=list)
    latest_site_status: dict[str, Any] | None = None
    capacity: SiteCapacity | None = None
    feasibility: Feasibility | None = None
    workload_estimates: dict[str, Any] = Field(default_factory=dict)
    efficiency: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)
    freshness: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    cache: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(APIModel):
    status: str = "ok"
    request_id: str
    prediction_id: str
    generated_at: datetime | str
    created_at: datetime | str
    valid_until: datetime | str
    model_name: str
    model_version: str
    target: str = "l2_site_status"
    forecast_start_time: datetime | str
    horizon: str
    step: str
    results: list[SitePrediction]
    predictions: list[SitePrediction]
    warnings: list[str] = Field(default_factory=list)
    cache: dict[str, Any] = Field(default_factory=dict)
