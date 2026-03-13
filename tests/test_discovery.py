"""Tests for USB camera discovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pmai_core.camera.discovery import _parse_v4l2_devices, discover_usb_cameras
from pmai_core.domain.camera import CameraStatus


class TestParseV4L2Devices:
    def test_returns_empty_when_command_not_found(self) -> None:
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _parse_v4l2_devices() == {}

    def test_parses_typical_output(self) -> None:
        mock_output = (
            "USB Camera (usb-0000:00:14.0-1):\n"
            "\t/dev/video0\n"
            "\t/dev/video1\n"
            "\n"
            "Integrated Camera (usb-0000:00:14.0-5):\n"
            "\t/dev/video2\n"
        )
        mock_result = MagicMock(returncode=0, stdout=mock_output)
        with patch("subprocess.run", return_value=mock_result):
            devices = _parse_v4l2_devices()
            assert devices["/dev/video0"] == "USB Camera (usb-0000:00:14.0-1)"
            assert devices["/dev/video1"] == "USB Camera (usb-0000:00:14.0-1)"
            assert devices["/dev/video2"] == "Integrated Camera (usb-0000:00:14.0-5)"


class TestDiscoverUSBCameras:
    @patch("pmai_core.camera.discovery.Path.glob", return_value=[])
    def test_no_devices_returns_empty(self, _mock_glob: MagicMock) -> None:
        cameras = discover_usb_cameras()
        assert cameras == []

    @patch("pmai_core.camera.discovery._can_open_with_opencv", return_value=True)
    @patch("pmai_core.camera.discovery._is_capture_device", return_value=True)
    @patch("pmai_core.camera.discovery._parse_v4l2_devices", return_value={})
    def test_discovers_valid_camera(
        self,
        _v4l2: MagicMock,
        _capture: MagicMock,
        _opencv: MagicMock,
    ) -> None:
        from pathlib import Path

        fake_dev = Path("/dev/video0")
        with patch("pmai_core.camera.discovery.Path.glob", return_value=[fake_dev]):
            cameras = discover_usb_cameras()

        assert len(cameras) == 1
        assert cameras[0].camera_id == "usb_0"
        assert cameras[0].status == CameraStatus.DISCOVERED
        assert cameras[0].device_path == "/dev/video0"
