from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_interval_is_one_second():
    toml = (ROOT / "settings.toml").read_text()
    settings = (ROOT / "src/pmai_core/settings.py").read_text()
    assert "result_interval_seconds = 1.0" in toml
    assert "result_interval_seconds: float = 1.0" in settings
    assert "result_interval_seconds = 5.0" not in toml
    assert "result_interval_seconds: float = 5.0" not in settings


def test_snapshot_encodes_one_jpeg_and_null_preview():
    src = (ROOT / "src/pmai_core/api/vision/vision_controller.py").read_text()
    assert src.count("frame_to_jpeg_base64(") == 1
    assert "preview_jpeg_base64=None" in src or "preview_jpeg_base64=None," in src
    schema = (ROOT / "src/pmai_core/api/vision/schemas.py").read_text()
    assert "default=480" in schema
    assert "le=960" in schema
