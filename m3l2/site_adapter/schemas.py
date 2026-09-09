from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, root_validator, validator

RIType = Literal["cloud", "network", "grid", "Cloud", "Network", "Grid", "CLOUD", "NETWORK", "GRID"]
AuthType = Literal["jwt", "egi_checkin", "none"]
SnapshotSource = Literal["push", "pull"]


class SiteRegistrationRequest(BaseModel):
    site_id: str
    site_name: str
    ri_type: RIType = "grid"
    adapter_base_url: str
    contact_email: str
    auth_type: AuthType = "jwt"
    auth_config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @validator("ri_type", pre=True)
    def normalise_ri_type(cls, value: Any) -> str:
        return str(value or "grid").lower()


class SiteSnapshotIn(BaseModel):
    ts: datetime | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    availability: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    efficiency: dict[str, Any] = Field(default_factory=dict)
    status: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, Any] = Field(default_factory=dict)

    @root_validator(pre=True)
    def accept_site_adapter_timestamp(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "ts" not in values and "timestamp" in values:
            values["ts"] = values["timestamp"]
        return values

    class Config:
        extra = "allow"


class SiteSnapshotOut(SiteSnapshotIn):
    id: int
    site_id: str
    source: SnapshotSource
    raw_json: dict[str, Any] | None = None


class WorkloadSubmissionRequest(BaseModel):
    workload_id: str
    workload_type: str
    requirements: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
