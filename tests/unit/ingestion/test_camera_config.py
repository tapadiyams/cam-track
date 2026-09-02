# Authored by: Shubham Tapadiya
# Created: 2026-09-02
# Updated: 2026-09-02
"""Tests for src/ingestion/camera_config.py."""

import textwrap

import pytest

from src.ingestion.camera_config import load_camera_configs


def _write_yaml(tmp_path, content: str):
    path = tmp_path / "cameras.yaml"
    path.write_text(textwrap.dedent(content))
    return str(path)


def test_load_camera_configs_parses_valid_file(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        cameras:
          - camera_id: cam-01
            source: rtsp://example/stream
            zone: store-front
            fps_cap: 20
        """,
    )
    configs = load_camera_configs(path)
    assert len(configs) == 1
    assert configs[0].camera_id == "cam-01"
    assert configs[0].fps_cap == 20


def test_load_camera_configs_applies_default_fps_cap(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        cameras:
          - camera_id: cam-02
            source: sample.mp4
            zone: warehouse
        """,
    )
    configs = load_camera_configs(path)
    assert configs[0].fps_cap == 15


def test_load_camera_configs_rejects_empty_cameras_list(tmp_path):
    path = _write_yaml(tmp_path, "cameras: []\n")
    with pytest.raises(ValueError, match="no 'cameras' entries"):
        load_camera_configs(path)


def test_load_camera_configs_rejects_missing_required_field(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        cameras:
          - camera_id: cam-03
            zone: warehouse
        """,
    )
    with pytest.raises(Exception):
        load_camera_configs(path)


def test_load_camera_configs_raises_file_not_found_for_missing_path():
    with pytest.raises(FileNotFoundError):
        load_camera_configs("/nonexistent/path/cameras.yaml")
