# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Dashboard service entrypoint: FastAPI app + static analytics UI.

Run with: uvicorn src.dashboard.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.common.logging_utils import configure_logging
from src.config.settings import get_settings
from src.dashboard.api.routes import router
from src.dashboard.db import init_pool

settings = get_settings()
logger = configure_logging("dashboard", settings.log_level)

app = FastAPI(title="cam-track dashboard", version="1.0.0")
app.include_router(router)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


@app.on_event("startup")
def on_startup() -> None:
    init_pool(settings.timescale_dsn)
    logger.info("dashboard started", extra={"port": settings.dashboard_port})
