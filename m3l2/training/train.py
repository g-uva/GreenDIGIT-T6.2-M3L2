from __future__ import annotations

import logging
import math
import hashlib
import json
from datetime import timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import select

from m3l2.app.config import get_settings
from m3l2.app.db import ModelRegistry, SessionLocal, SiteStatusSnapshot, create_tables, utc_now
from m3l2.training.features import STATUS_FEATURE_COLUMNS, STATUS_TARGET_COLUMNS, build_site_status_training_frame

logger = logging.getLogger(__name__)

TARGET = "l2_site_status"
CATEGORICAL_FEATURES = ["site_id", "ri_type"]
NUMERIC_FEATURES = [
    "hour",
    "day_of_week",
    "records_count_site_24h",
    "rolling_availability_mean_site_24h",
    "rolling_free_cpu_mean_site_24h",
    "rolling_queue_mean_site_24h",
    "rolling_load_mean_site_24h",
]
FEATURE_COLUMNS = STATUS_FEATURE_COLUMNS
TARGET_COLUMNS = STATUS_TARGET_COLUMNS


def _make_pipeline() -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical, CATEGORICAL_FEATURES),
            ("numeric", numeric, NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=80, min_samples_leaf=1, random_state=42)),
            ),
        ]
    )


def _metrics(y_true: pd.DataFrame, y_pred: Any, n_train: int, n_val: int) -> dict[str, Any]:
    if n_val == 0:
        return {"mae": None, "rmse": None, "n_train": n_train, "n_val": n_val}
    errors: list[float] = []
    squared: list[float] = []
    per_target: dict[str, dict[str, float]] = {}
    for index, target in enumerate(TARGET_COLUMNS):
        truth = y_true[target].to_list()
        predicted = [row[index] for row in y_pred]
        target_errors = [float(abs(a - b)) for a, b in zip(truth, predicted)]
        target_squared = [float((a - b) ** 2) for a, b in zip(truth, predicted)]
        errors.extend(target_errors)
        squared.extend(target_squared)
        per_target[target] = {
            "mae": float(sum(target_errors) / len(target_errors)),
            "rmse": float(math.sqrt(sum(target_squared) / len(target_squared))),
        }
    return {
        "mae": float(sum(errors) / len(errors)),
        "rmse": float(math.sqrt(sum(squared) / len(squared))),
        "n_train": n_train,
        "n_val": n_val,
        "targets": per_target,
    }


def _version(now) -> str:
    return f"l2-site-status-{now.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"


def training_data_fingerprint(df: pd.DataFrame) -> str:
    fingerprint_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    if df.empty:
        payload = {"columns": fingerprint_columns, "records": []}
    else:
        canonical = df[fingerprint_columns].copy()
        canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        canonical = canonical.sort_values(fingerprint_columns).reset_index(drop=True)
        payload = {
            "columns": fingerprint_columns,
            "records": canonical.to_dict(orient="records"),
        }
    data = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def train_model(force: bool = False) -> dict[str, Any]:
    create_tables()
    settings = get_settings()
    with SessionLocal() as session:
        df = build_site_status_training_frame(session)
        if len(df) < settings.min_training_records:
            return {
                "status": "not_enough_data",
                "n_records": len(df),
                "min_training_records": settings.min_training_records,
                "target": TARGET,
                "detail": "Not enough training-compatible L2 site status snapshots are available.",
            }

        fingerprint = training_data_fingerprint(df)
        active_model = session.execute(
            select(ModelRegistry).where(ModelRegistry.target == TARGET, ModelRegistry.active.is_(True))
        ).scalars().first()
        if not force and active_model and active_model.training_data_fingerprint == fingerprint:
            return {
                "status": "skipped_unchanged_data",
                "model_version": active_model.version,
                "training_data_fingerprint": fingerprint,
                "n_records": len(df),
            }

        df = df.sort_values("timestamp").reset_index(drop=True)
        val_size = max(1, int(len(df) * 0.2)) if len(df) >= 5 else 0
        train_df = df.iloc[:-val_size] if val_size else df
        val_df = df.iloc[-val_size:] if val_size else df.iloc[0:0]

        pipeline = _make_pipeline()
        pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMNS])
        val_pred = pipeline.predict(val_df[FEATURE_COLUMNS]) if val_size else []
        metrics = _metrics(val_df[TARGET_COLUMNS], val_pred, len(train_df), len(val_df))

        now = utc_now()
        version = _version(now)
        model_dir = Path(settings.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{version}.joblib"
        feature_schema = {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "target": TARGET,
            "targets": TARGET_COLUMNS,
        }
        joblib.dump({"pipeline": pipeline, "feature_schema": feature_schema, "version": version}, model_path)

        for row in session.execute(
            select(ModelRegistry).where(ModelRegistry.target == TARGET, ModelRegistry.active.is_(True))
        ).scalars():
            row.active = False
        registry_row = ModelRegistry(
            model_name="hist_gradient_boosting_mvp",
            target=TARGET,
            version=version,
            path=str(model_path),
            trained_at=now,
            training_window_start=df["timestamp"].min().to_pydatetime(),
            training_window_end=df["timestamp"].max().to_pydatetime(),
            metrics=metrics,
            feature_schema=feature_schema,
            training_data_fingerprint=fingerprint,
            active=True,
        )
        session.add(registry_row)
        session.commit()

        sites = session.execute(select(SiteStatusSnapshot.site_id).distinct()).scalars().all()

    forecast_status = "skipped"
    if sites:
        from m3l2.inference.forecast_refresh import refresh_forecasts

        result = refresh_forecasts(site_ids=[site for site in sites if site], force=True)
        forecast_status = result.get("status", "stored")

    logger.info("Training completed for %s with metrics %s", version, metrics)
    return {
        "status": "trained",
        "target": TARGET,
        "model_version": version,
        "path": str(model_path),
        "metrics": metrics,
        "training_data_fingerprint": fingerprint,
        "forecast_status": forecast_status,
    }
