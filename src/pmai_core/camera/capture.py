"""Threaded per-camera frame capture with a bounded queue."""

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

if TYPE_CHECKING:
    from pmai_core.domain.camera import CameraInfo

logger = structlog.get_logger(__name__)

FrameType = NDArray[np.uint8]

QUEUE_MAX_SIZE = 4


class CameraCapture:
    """Reads frames from a single USB camera in a background thread.

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

    @property
    def camera_id(self) -> str:
        return self._info.camera_id

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        idx_match = re.search(r"\d+$", self._info.device_path)
        if idx_match is None:
            raise ValueError(f"Cannot parse device index from {self._info.device_path}")
        idx = int(idx_match.group())

        # Prefer V4L2 explicitly inside Linux containers.
        self._cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._info.device_path}")

        w, h = self._info.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_FPS, self._info.fps)

        self._running.set()
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"cam-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info("camera_capture_started", camera_id=self.camera_id)

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
        while self._running.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("frame_read_failed", camera_id=self.camera_id)
                time.sleep(0.05)
                continue

            ts = time.time()
            try:
                self._queue.put_nowait((frame, ts))
            except Full:
                with contextlib.suppress(Exception):
                    self._queue.get_nowait()
                with contextlib.suppress(Full):
                    self._queue.put_nowait((frame, ts))
