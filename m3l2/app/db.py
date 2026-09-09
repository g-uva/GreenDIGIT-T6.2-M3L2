from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import JSON

from m3l2.app.config import get_settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ExecutionRecord(Base):
    __tablename__ = "execution_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exec_unit_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    site_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    ri_type: Mapped[str] = mapped_column(String, default="unknown")
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    stop_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str | None] = mapped_column(String, nullable=True)
    energy_wh: Mapped[float | None] = mapped_column(Float, nullable=True)
    work: Mapped[float | None] = mapped_column(Float, nullable=True)
    work_type: Mapped[str] = mapped_column(String, default="unknown")
    owner: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SiteProfile(Base):
    __tablename__ = "site_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    ri_type: Mapped[str] = mapped_column(String, default="unknown")
    location: Mapped[str | None] = mapped_column(String, nullable=True)
    compute_capacity: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_capacity: Mapped[float | None] = mapped_column(Float, nullable=True)
    storage_capacity: Mapped[float | None] = mapped_column(Float, nullable=True)
    network_topology: Mapped[str | None] = mapped_column(String, nullable=True)
    link_capacities: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    supported_workload_types: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    energy_capabilities: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    static_pue_baseline: Mapped[float | None] = mapped_column(Float, nullable=True)
    extensions: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SiteStatusSnapshot(Base):
    __tablename__ = "site_status_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    ri_type: Mapped[str] = mapped_column(String, default="unknown")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    operational_status: Mapped[str] = mapped_column(String, default="UP")
    maintenance_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    scheduled_maintenance: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    node_availability: Mapped[float | None] = mapped_column(Float, nullable=True)
    link_availability: Mapped[float | None] = mapped_column(Float, nullable=True)
    stability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    packet_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    network_jitter: Mapped[float | None] = mapped_column(Float, nullable=True)
    network_utilization: Mapped[float | None] = mapped_column(Float, nullable=True)
    available_bandwidth: Mapped[float | None] = mapped_column(Float, nullable=True)
    cpu_util_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    gpu_util_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    free_cpu_capacity: Mapped[float | None] = mapped_column(Float, nullable=True)
    free_gpu_capacity: Mapped[float | None] = mapped_column(Float, nullable=True)
    queue_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    remaining_jobs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provisioning_delay_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    load_index: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_consumed: Mapped[float | None] = mapped_column(Float, nullable=True)
    pue_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    carbon_intensity: Mapped[float | None] = mapped_column(Float, nullable=True)
    energy_per_task_proxy: Mapped[float | None] = mapped_column(Float, nullable=True)
    update_frequency: Mapped[float | None] = mapped_column(Float, nullable=True)
    data_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    stale_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    extensions: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class OperatorConfig(Base):
    __tablename__ = "operator_configs"

    config_key: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_by_email: Mapped[str | None] = mapped_column(String, nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    triggered_by_email: Mapped[str | None] = mapped_column(String, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String, nullable=True)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)


class RegisteredSite(Base):
    __tablename__ = "registered_sites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    site_name: Mapped[str] = mapped_column(String, nullable=False)
    ri_type: Mapped[str] = mapped_column(String, default="unknown")
    adapter_base_url: Mapped[str] = mapped_column(String, nullable=False)
    contact_email: Mapped[str] = mapped_column(String, nullable=False)
    auth_type: Mapped[str] = mapped_column(String, default="jwt")
    auth_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    site_metadata: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)


class SiteSnapshot(Base):
    __tablename__ = "site_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    capabilities: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    availability: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    efficiency: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    submitted_by_email: Mapped[str | None] = mapped_column(String, nullable=True)
    quality: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    raw_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class AuthUser(Base):
    __tablename__ = "auth_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String)
    target: Mapped[str] = mapped_column(String)
    version: Mapped[str] = mapped_column(String, unique=True)
    path: Mapped[str] = mapped_column(String)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    training_window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    training_window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    feature_schema: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    training_data_fingerprint: Mapped[str | None] = mapped_column(String, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=False)


class ForecastCache(Base):
    __tablename__ = "forecast_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[str | None] = mapped_column(String, index=True)
    target: Mapped[str] = mapped_column(String)
    horizon: Mapped[str] = mapped_column(String)
    step: Mapped[str] = mapped_column(String)
    request_signature: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    forecast_start_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model_version: Mapped[str] = mapped_column(String)
    predictions: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    quality: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


def _normalise_database_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _engine_kwargs(url: str) -> dict[str, Any]:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {}


_engine: Engine | None = None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        url = _normalise_database_url(get_settings().database_url)
        _engine = create_engine(url, future=True, **_engine_kwargs(url))
        SessionLocal.configure(bind=_engine)
    return _engine


def configure_database(database_url: str) -> Engine:
    global _engine
    url = _normalise_database_url(database_url)
    if _engine is not None:
        _engine.dispose()
    _engine = create_engine(url, future=True, **_engine_kwargs(url))
    SessionLocal.configure(bind=_engine)
    return _engine


def create_tables() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_site_snapshot_columns(engine)
    _ensure_site_profile_columns(engine)
    _ensure_site_status_columns(engine)
    _ensure_model_registry_columns(engine)
    _ensure_forecast_cache_columns(engine)


def _ensure_site_snapshot_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("site_snapshots")}
    if "submitted_by_email" in columns:
        return
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE site_snapshots ADD COLUMN submitted_by_email VARCHAR"))


def _json_column_type(engine: Engine) -> str:
    return "JSONB" if engine.dialect.name == "postgresql" else "JSON"


def _ensure_site_profile_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("site_profiles")}
    if "extensions" in columns:
        return
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE site_profiles ADD COLUMN extensions {_json_column_type(engine)}"))


def _ensure_site_status_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("site_status_snapshots")}
    if "extensions" in columns:
        return
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE site_status_snapshots ADD COLUMN extensions {_json_column_type(engine)}"))


def _ensure_model_registry_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("model_registry")}
    if "training_data_fingerprint" in columns:
        return
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE model_registry ADD COLUMN training_data_fingerprint VARCHAR"))


def _ensure_forecast_cache_columns(engine: Engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns("forecast_cache")}
    statements = []
    if "request_signature" not in columns:
        statements.append("ALTER TABLE forecast_cache ADD COLUMN request_signature VARCHAR")
    if "forecast_start_ts" not in columns:
        timestamp_type = "DATETIME" if engine.dialect.name == "sqlite" else "TIMESTAMP WITH TIME ZONE"
        statements.append(f"ALTER TABLE forecast_cache ADD COLUMN forecast_start_ts {timestamp_type}")
    if not statements:
        return
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def db_status() -> str:
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return "ok"
    except Exception:
        return "error"
