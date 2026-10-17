"""
CameraService — webcam capture management.

Responsibilities:
  - Discover and open the configured camera index
  - Provide frames on demand (thread-safe)
  - Measure actual capture FPS
  - Handle failure gracefully (open fails, frame read fails)
  - Release resources cleanly on stop/shutdown

This service does NOT run inference — it only captures frames.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraService:
    """
    Background webcam capture service.

    Captures frames continuously in a background thread. The latest
    frame is always available via `get_latest_frame()` without blocking.

    Args:
        camera_index: OpenCV device index (0 = default webcam).
        width:        Requested frame width (camera may override).
        height:       Requested frame height (camera may override).
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # FPS measurement
        self._capture_fps: float = 0.0
        self._frame_count: int = 0
        self._fps_start: float = time.monotonic()

        # Error state
        self._error: Optional[str] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """
        Open camera and start background capture thread.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        if self._running:
            logger.warning("CameraService.start() called but already running.")
            return True

        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self._error = f"Failed to open camera at index {self._camera_index}"
            logger.error(self._error)
            return False

        # Request resolution (camera may not honour it exactly)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            f"Camera {self._camera_index} opened: {actual_w}x{actual_h}"
        )

        self._running = True
        self._error = None
        self._frame_count = 0
        self._fps_start = time.monotonic()

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._frame_lock:
            self._latest_frame = None
        logger.info("CameraService stopped and resources released.")

    # ── Capture loop (background thread) ─────────────────────────────────────

    def _capture_loop(self) -> None:
        """Continuously read frames and store the latest one."""
        consecutive_failures = 0
        MAX_FAILURES = 10

        while self._running:
            if self._cap is None:
                break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning(
                    f"Frame read failed ({consecutive_failures}/{MAX_FAILURES})"
                )
                if consecutive_failures >= MAX_FAILURES:
                    self._error = "Camera stopped producing frames — device disconnected?"
                    logger.error(self._error)
                    break
                time.sleep(0.1)
                continue

            consecutive_failures = 0

            with self._frame_lock:
                self._latest_frame = frame

            # FPS accounting
            self._frame_count += 1
            elapsed = time.monotonic() - self._fps_start
            if elapsed >= 2.0:
                self._capture_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start = time.monotonic()

        self._running = False
        logger.info("Capture loop exited.")

    # ── Public accessors ──────────────────────────────────────────────────────

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent captured frame (thread-safe)."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def capture_fps(self) -> float:
        return round(self._capture_fps, 1)

    @property
    def error(self) -> Optional[str]:
        return self._error

    @staticmethod
    def enumerate_cameras(max_index: int = 5) -> list[int]:
        """Probe camera indices and return those that open successfully."""
        available = []
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available.append(idx)
            cap.release()
        return available
