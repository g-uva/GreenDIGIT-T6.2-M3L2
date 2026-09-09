from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from m3l2.app.schemas import PredictRequest, ResourceRequirements, TimeRequirements, WorkloadDescriptor
from m3l2.inference.predict import predict as run_predict
from m3l2.site_adapter.control_plane import forward_workload_to_site, get_db

router = APIRouter(prefix="/mock-broker", tags=["mock-broker"])


class MockBrokerSubmitRequest(BaseModel):
    workload_id: str
    workload_type: str = "unknown"
    candidate_sites: list[str]
    horizon: str = "24h"
    step: str = "1h"
    start_time: str | None = None
    duration: str | None = None
    deadline: str | None = None
    requirements: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extensions: dict[str, Any] = Field(default_factory=dict)


def _resource_requirements(requirements: dict[str, Any]) -> ResourceRequirements:
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
    mapped = {target: requirements[source] for source, target in aliases.items() if source in requirements}
    return ResourceRequirements(**mapped)


def _first_forecast_value(prediction: dict[str, Any]) -> float | None:
    forecast = prediction.get("forecast") or []
    if not forecast:
        return None
    first = forecast[0]
    if not isinstance(first, dict) or first.get("value") is None:
        return None
    return float(first["value"])


def _latest_availability_status(prediction: dict[str, Any]) -> str:
    status_forecast = prediction.get("site_status_forecast") or []
    if status_forecast and isinstance(status_forecast[0], dict):
        status = status_forecast[0].get("operational_status") or ""
        return str(status).lower()
    latest = prediction.get("latest_site_status") or {}
    availability = latest.get("availability") or {}
    status = availability.get("status") or availability.get("operational_status") or ""
    return str(status).lower()


def _free_cpu_capacity(prediction: dict[str, Any]) -> float:
    capacity = prediction.get("capacity") or {}
    value = capacity.get("free_cpu_capacity")
    return float(value) if value is not None else 0.0


def _select_best_site(predictions: list[dict[str, Any]]) -> tuple[str, dict[str, Any], float]:
    candidates: list[tuple[str, dict[str, Any], float, float]] = []
    for prediction in predictions:
        feasibility = prediction.get("feasibility") or {}
        if feasibility.get("status") == "infeasible":
            continue
        if _latest_availability_status(prediction) in {"down", "maintenance"}:
            continue
        value = _first_forecast_value(prediction)
        if value is None:
            continue
        candidates.append((prediction["site_id"], prediction, value, _free_cpu_capacity(prediction)))
    if not candidates:
        raise HTTPException(status_code=409, detail="No candidate site is available for submission")
    site_id, prediction, availability, _ = max(candidates, key=lambda item: (item[2], item[3]))
    return site_id, prediction, availability


@router.post("/submit")
async def submit_to_best_site(payload: MockBrokerSubmitRequest, session: Session = Depends(get_db)) -> dict[str, Any]:
    prediction_request = PredictRequest(
        candidate_site_ids=payload.candidate_sites,
        horizon=payload.horizon,
        step=payload.step,
        workload=WorkloadDescriptor(
            workload_id=payload.workload_id,
            workload_type=payload.workload_type or payload.metadata.get("workload_type", "unknown"),
            time_requirements=TimeRequirements(
                start_time=payload.start_time,
                duration=payload.duration,
                deadline=payload.deadline,
            ),
            resource_requirements=_resource_requirements(payload.requirements),
            metadata=payload.metadata,
            extensions={**payload.extensions, "legacy_requirements": payload.requirements},
        ),
        include_site_status=True,
    )
    prediction_result = run_predict(prediction_request)
    if prediction_result.get("status") == "no_active_model":
        raise HTTPException(status_code=503, detail=prediction_result["detail"])

    site_id, prediction, first_value = _select_best_site(prediction_result.get("predictions") or [])
    workload_payload = {
        "workload_id": payload.workload_id,
        "workload_type": payload.workload_type or payload.metadata.get("workload_type", "unknown"),
        "requirements": payload.requirements,
        "metadata": payload.metadata,
        "extensions": payload.extensions,
    }
    submission_response = await forward_workload_to_site(session, site_id, workload_payload)
    return {
        "selected_site": site_id,
        "reason": "highest_predicted_l2_availability",
        "prediction_summary": {
            "first_forecast_availability": first_value,
            "selected_prediction": prediction,
        },
        "submission_response": submission_response,
    }
