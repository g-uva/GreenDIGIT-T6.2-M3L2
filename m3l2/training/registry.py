from __future__ import annotations

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from m3l2.app.db import ModelRegistry


def get_active_model(session: Session, target: str = "energy_wh") -> ModelRegistry | None:
    return session.execute(
        select(ModelRegistry)
        .where(ModelRegistry.target == target, ModelRegistry.active.is_(True))
        .order_by(desc(ModelRegistry.trained_at))
    ).scalars().first()


def list_models(session: Session) -> list[ModelRegistry]:
    return session.execute(select(ModelRegistry).order_by(desc(ModelRegistry.trained_at))).scalars().all()


def get_model_by_version(session: Session, version: str) -> ModelRegistry | None:
    return session.execute(select(ModelRegistry).where(ModelRegistry.version == version)).scalar_one_or_none()


def serialise_model(row: ModelRegistry) -> dict:
    return {
        "id": row.id,
        "model_name": row.model_name,
        "target": row.target,
        "version": row.version,
        "path": row.path,
        "trained_at": row.trained_at.isoformat(),
        "training_window_start": row.training_window_start.isoformat() if row.training_window_start else None,
        "training_window_end": row.training_window_end.isoformat() if row.training_window_end else None,
        "metrics": row.metrics,
        "feature_schema": row.feature_schema,
        "training_data_fingerprint": row.training_data_fingerprint,
        "active": row.active,
    }
