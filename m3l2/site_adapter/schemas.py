from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, root_validator

SnapshotSource = Literal["push", "pull"]


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
