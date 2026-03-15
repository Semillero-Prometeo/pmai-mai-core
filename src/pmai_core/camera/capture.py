"""Threaded per-camera frame capture with a bounded queue.

Supports three source types:
- **V4L2**: local USB camera via ``/dev/video*`` (production on real hardware)
- **stream**: network URL — RTSP, HTTP MJPEG, etc. (development / IP cameras)
- **file**: local video file, optionally looped (testing / offline development)
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from queue import Full, Queue
from typing import TYPE_CHECKING

import cv2
import numpy as np
import structlog
from numpy.typing import NDArray

from pmai_core.domain.camera import CameraSourceType

if TYPE_CHECKING:
    from pmai_core.domain.camera import CameraInfo

logger = structlog.get_logger(__name__)

FrameType = NDArray[np.uint8]

QUEUE_MAX_SIZE = 4

_MAX_CONSECUTIVE_FAILURES = 50


def _open_v4l2(info: CameraInfo) -> cv2.VideoCapture:

    cap = cv2.VideoCapture(info.device_path, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open V4L2 camera {info.device_path}")

    w, h = info.resolution

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, info.fps)

    # prueba de frame
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but no frame received")

    return cap

def _open_stream(info: CameraInfo) -> cv2.VideoCapture:
    """Open a network stream (RTSP, HTTP MJPEG, etc.)."""
    url = info.device_path
    if not url:
        raise ValueError(f"Camera {info.camera_id}: stream URL is empty")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream {url}")
    return cap


def _open_file(info: CameraInfo) -> cv2.VideoCapture:
    """Open a local video file."""
    path = info.device_path
    if not path:
        raise ValueError(f"Camera {info.camera_id}: file path is empty")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {path}")
    return cap


_OPENERS = {
    CameraSourceType.V4L2: _open_v4l2,
    CameraSourceType.STREAM: _open_stream,
    CameraSourceType.FILE: _open_file,
}


class CameraCapture:
    """Reads frames from a camera source in a background thread.

    Each captured frame is pushed to an internal ``Queue`` that the pipeline
    engine consumes.  If the queue is full the oldest frame is silently
    dropped so the capture thread never blocks on a slow consumer.
    """

    def __init__(self, camera_info: CameraInfo) -> None:
        self._info = camera_info
        self._cap: cv2.VideoCapture | None = None
        self._queue: Queue[tuple[FrameType, float]] = Queue(maxsize=QUEUE_MAX_SIZE)
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._logged_first_frame = False
        self._consecutive_failures = 0

    @property
    def camera_id(self) -> str:
        return self._info.camera_id

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        opener = _OPENERS.get(self._info.source_type)
        if opener is None:
            raise ValueError(f"Unsupported source type: {self._info.source_type}")

        self._cap = opener(self._info)

        self._running.set()
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"cam-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "camera_capture_started",
            camera_id=self.camera_id,
            source_type=self._info.source_type,
            source=self._info.device_path,
        )

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("camera_capture_stopped", camera_id=self.camera_id)

    def get_frame(self, timeout: float = 1.0) -> tuple[FrameType, float] | None:
        """Return ``(frame, timestamp)`` or ``None`` if nothing is available."""
        try:
            return self._queue.get(timeout=timeout)
        except Exception:
            return None

    def _read_loop(self) -> None:
        assert self._cap is not None
        read_count = 0
        while self._running.is_set():
            if self._info.source_type == CameraSourceType.V4L2:
                read_count += 1
                if read_count == 1 or (read_count % 100 == 0):
                    logger.debug(
                        "v4l2_read_attempt",
                        camera_id=self.camera_id,
                        device=self._info.device_path,
                        location="CameraCapture._read_loop",
                        read_count=read_count,
                    )
            ret, frame = self._cap.read()

            if not ret or frame is None:
                # For file sources, loop back to the beginning.
                if self._info.source_type == CameraSourceType.FILE and self._info.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                if self._info.source_type == CameraSourceType.V4L2:
                    logger.warning(
                        "v4l2_read_failed",
                        camera_id=self.camera_id,
                        device=self._info.device_path,
                        location="CameraCapture._read_loop",
                        consecutive_failures=self._consecutive_failures + 1,
                        hint="OpenCV stderr 'select() timeout' is from this read()",
                    )

                self._consecutive_failures += 1
                if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "camera_giving_up",
                        camera_id=self.camera_id,
                        device=self._info.device_path,
                        location="CameraCapture._read_loop",
                        failures=self._consecutive_failures,
                    )
                    self._running.clear()
                    return
                if self._consecutive_failures % 50 == 0:
                    logger.error(
                        "frame_read_failed_many_times",
                        camera_id=self.camera_id,
                        device=self._info.device_path,
                        location="CameraCapture._read_loop",
                        failures=self._consecutive_failures,
                    )
                time.sleep(0.05)
                continue

            if not self._logged_first_frame:
                h, w = frame.shape[:2]
                logger.info(
                    "camera_first_frame",
                    camera_id=self.camera_id,
                    resolution=f"{w}x{h}",
                )
                self._logged_first_frame = True

            self._consecutive_failures = 0

            typed_frame: FrameType = np.asarray(frame, dtype=np.uint8)
            ts = time.time()
            try:
                self._queue.put_nowait((typed_frame, ts))
            except Full:
                with contextlib.suppress(Exception):
                    self._queue.get_nowait()
                with contextlib.suppress(Full):
                    self._queue.put_nowait((typed_frame, ts))

            # Throttle file-based sources to approximate real-time playback.
            if self._info.source_type == CameraSourceType.FILE and self._info.fps > 0:
                time.sleep(1.0 / self._info.fps)
