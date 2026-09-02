# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Loads and validates `configs/cameras.yaml`."""

from __future__ import annotations

import yaml
from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    """One camera source and the zone it belongs to for re-ID scoping."""

    camera_id: str
    source: str  # RTSP URL, local video file, or webcam index as a string
    zone: str
    fps_cap: int = Field(default=15, gt=0)


def load_camera_configs(path: str) -> list[CameraConfig]:
    """Parse `path` (see configs/cameras.yaml) into `CameraConfig` objects.

    Time: O(c) where c is the number of camera entries in the file.
    Space: O(c).

    Raises `FileNotFoundError` if `path` does not exist and `ValueError` if
    the file exists but has no `cameras` key or fails schema validation --
    both are fail-fast on startup rather than silently running zero
    cameras.
    """
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    entries = raw.get("cameras")
    if not entries:
        raise ValueError(f"{path} has no 'cameras' entries.")

    return [CameraConfig.model_validate(entry) for entry in entries]
