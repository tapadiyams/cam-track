# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Environment-driven settings, shared by every service in the pipeline.

All deployment-specific values (hosts, ports, credentials, feature flags)
live here and are read from environment variables / a `.env` file. Fixed,
non-deployment-specific values belong in `constants.py` instead.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration, validated once at startup.

    Every field has a safe local-dev default so `docker-compose up` works
    out of the box; production deployments override via real env vars or
    a mounted `.env` file (see `.env.example`).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # --- Message broker (Redis Streams is the default backend; see
    # docs/decisions/0003-message-queue-choice.md for why, and why Kafka
    # remains a supported alternate backend rather than the default) ---
    stream_backend: str = "redis"  # "redis" or "kafka"
    redis_url: str = "redis://localhost:6379/0"
    kafka_bootstrap_servers: str = "localhost:9092"

    # --- Storage (TimescaleDB) ---
    timescale_dsn: str = "postgresql://camtrack:camtrack@localhost:5432/camtrack"

    # --- Inference ---
    detector_weights_path: str = "models/yolov8n.onnx"
    detector_device: str = "cpu"  # "cpu", "cuda", or "cuda:N"
    detector_use_onnx_runtime: bool = True
    max_batch_size: int = 8
    max_batch_wait_ms: int = 40

    # --- Re-identification ---
    reid_enabled: bool = True
    reid_weights_path: str = "models/reid_resnet18.onnx"

    # --- Ingestion ---
    camera_config_path: str = "configs/cameras.yaml"

    # --- Dashboard ---
    dashboard_host: str = "0.0.0.0"  # noqa: S104 -- intentional for containerized deploy
    dashboard_port: int = 8080

    # --- Observability ---
    log_level: str = "INFO"
    service_name: str = "camtrack"


def get_settings() -> Settings:
    """Return a fresh `Settings` instance.

    Time: O(1) plus the cost of reading env vars / the .env file (bounded by
    the fixed number of declared fields, not by any runtime input).
    Space: O(1) -- a handful of scalar fields.

    Not memoized on purpose: tests frequently monkeypatch environment
    variables between calls and expect a fresh read, not a cached singleton.
    """
    return Settings()
