"""
CameraChannel — self-contained per-camera detection pipeline (Sprint 4).

Each configured camera gets one CameraChannel instance which bundles:
  - CameraService        (capture thread)
  - YOLOWorldDetectionService (shared, serialized via inference_lock)
  - ClassificationPolicy
  - TemporalConfirmation
  - AlertManager
  - Detection loop background thread
  - WebSocket client set for real-time push

The shared detector is passed in from AppState so only one YOLO model
is ever loaded (keeps VRAM constant regardless of camera count).

Thread safety:
  - _frame_lock   protects latest raw + annotated frames
  - _event_lock   protects latest DetectionEvent
  - _ws_lock      protects the WebSocket client set
  - inference_lock (external, shared) serializes GPU predict() calls
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Set

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── DetectionEvent ─────────────────────────────────────────────────────────────

@dataclass
class DetectionEvent:
    """Snapshot of the pipeline state at a single inference tick."""
    timestamp: float
    status: str              # "clear" | "evaluating" | "alert" | "cooldown" | "error"
    confirmation_ratio: str  # e.g. "3/5"
    hit_count: int
    cooldown_remaining: float
    inference_ms: float
    capture_fps: float
    detections: list[dict]   # [{class_name, confidence, label, bbox}]
    alert_fired: bool = False
    alert_class: Optional[str] = None
    alert_confidence: Optional[float] = None

    def to_json(self, cam_id: str = "") -> str:
        return json.dumps({
            "type": "detection",
            "cam_id": cam_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "confirmation_ratio": self.confirmation_ratio,
            "hit_count": self.hit_count,
            "cooldown_remaining": round(self.cooldown_remaining, 1),
            "inference_ms": round(self.inference_ms, 1),
            "capture_fps": self.capture_fps,
            "detections": self.detections,
            "alert_fired": self.alert_fired,
            "alert_class": self.alert_class,
            "alert_confidence": self.alert_confidence,
        })


# ── CameraChannel ──────────────────────────────────────────────────────────────

class CameraChannel:
    """
    Self-contained per-camera detection pipeline.

    Args:
        cam_id:         Unique string identifier (e.g. "cam0").
        label:          Human-readable camera name (e.g. "Front Door").
        camera_index:   OpenCV device index.
        width, height:  Requested capture resolution.
        detector:       Shared YOLOWorldDetectionService instance.
        inference_lock: threading.Lock shared across all channels to
                        serialize GPU predict() calls.
        policy:         ClassificationPolicy instance.
        confirmation:   TemporalConfirmation instance (one per channel).
        alert_manager:  AlertManager instance (one per channel).
        inference_fps:  Target inference rate (frames per second).
        jpeg_quality:   JPEG quality for MJPEG stream (1-100).
    """

    def __init__(
        self,
        cam_id: str,
        label: str,
        camera_index: int,
        width: int,
        height: int,
        detector,
        inference_lock: threading.Lock,
        policy,
        confirmation,
        alert_manager,
        inference_fps: int = 10,
        jpeg_quality: int = 70,
        classifier_verifier=None,
    ) -> None:
        self.cam_id = cam_id
        self.label = label
        self.camera_index = camera_index
        self._cam_width = width
        self._cam_height = height

        # Services
        self._detector = detector
        self._inference_lock = inference_lock
        self._policy = policy
        self._confirmation = confirmation
        self._alert_manager = alert_manager
        self._inference_fps = inference_fps
        self._jpeg_quality = jpeg_quality
        self._classifier = classifier_verifier

        # Camera service (created on start)
        self._camera = None

        # Frame storage
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_annotated: Optional[np.ndarray] = None
        self._frame_counter: int = 0
        self._frame_lock = threading.Lock()

        # Latest event
        self._latest_event: Optional[DetectionEvent] = None
        self._event_lock = threading.Lock()

        # WebSocket clients subscribed to this channel
        self._ws_clients: Set[Any] = set()
        self._ws_lock: Optional[asyncio.Lock] = None  # created after loop starts

        # Loop control
        self._loop_running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop) -> bool:
        """
        Open camera and start the detection loop.

        Args:
            loop: The running asyncio event loop (for WS broadcast).

        Returns:
            True if camera opened successfully, False otherwise.
        """
        from backend.services.camera_service import CameraService

        self._asyncio_loop = loop
        self._ws_lock = asyncio.Lock()

        self._camera = CameraService(
            camera_index=self.camera_index,
            width=self._cam_width,
            height=self._cam_height,
        )
        if not self._camera.start():
            logger.error(
                f"[{self.cam_id}] Cannot open camera {self.camera_index}: {self._camera.error}"
            )
            return False

        logger.info(f"[{self.cam_id}] Camera {self.camera_index} opened.")
        self._loop_running = True
        self._loop_thread = threading.Thread(
            target=self._detection_loop,
            name=f"detection-{self.cam_id}",
            daemon=True,
        )
        self._loop_thread.start()
        logger.info(f"[{self.cam_id}] Detection loop started.")
        return True

    def stop(self) -> None:
        """Stop detection loop and release the camera."""
        self._loop_running = False
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
            self._loop_thread = None
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        logger.info(f"[{self.cam_id}] Stopped.")

    # ── Frame access ───────────────────────────────────────────────────────────

    def set_frames(self, raw: np.ndarray, annotated: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = raw
            self._latest_annotated = annotated
            self._frame_counter += 1

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_annotated.copy() if self._latest_annotated is not None else None

    def get_frame_counter(self) -> int:
        with self._frame_lock:
            return self._frame_counter

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ── Event access ───────────────────────────────────────────────────────────

    def set_event(self, event: DetectionEvent) -> None:
        with self._event_lock:
            self._latest_event = event

    def get_event(self) -> Optional[DetectionEvent]:
        with self._event_lock:
            return self._latest_event

    # ── WebSocket helpers ──────────────────────────────────────────────────────

    async def register_ws(self, ws: Any) -> None:
        async with self._ws_lock:
            self._ws_clients.add(ws)
        logger.info(f"[{self.cam_id}] WS client connected. Total: {len(self._ws_clients)}")

    async def unregister_ws(self, ws: Any) -> None:
        async with self._ws_lock:
            self._ws_clients.discard(ws)
        logger.info(f"[{self.cam_id}] WS client disconnected. Total: {len(self._ws_clients)}")

    async def broadcast(self, message: str) -> None:
        """Send a JSON string to all WebSocket clients subscribed to this channel."""
        if not self._ws_clients:
            return
        async with self._ws_lock:
            clients = set(self._ws_clients)
        dead: Set[Any] = set()
        for ws in clients:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._ws_lock:
                self._ws_clients -= dead

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._loop_running

    @property
    def camera_ok(self) -> bool:
        return self._camera is not None and self._camera.is_running

    @property
    def capture_fps(self) -> float:
        return self._camera.capture_fps if self._camera else 0.0

    # ── Detection loop ─────────────────────────────────────────────────────────

    def _detection_loop(self) -> None:
        """
        Background thread: captures frame, runs inference (serialized via lock),
        builds DetectionEvent, annotates frame, and schedules WS broadcast.
        """
        from backend.services.classification_policy import ClassLabel

        inference_interval = 1.0 / max(1, self._inference_fps)
        last_inference_time = 0.0
        last_inference_ms = 1.0
        last_detections = []
        last_adjusted_confidences = {}

        # Colour map
        COLOR_TARGET = (0, 0, 255)
        COLOR_PET    = (0, 200, 0)
        COLOR_OTHER  = (180, 180, 180)
        COLOR_ALERT  = (0, 0, 255)
        COLOR_CLEAR  = (0, 200, 0)
        COLOR_EVAL   = (0, 165, 255)

        while self._loop_running:
            frame = self._camera.get_latest_frame() if self._camera else None
            if frame is None:
                if self._camera and not getattr(self._camera, "_running", True):
                    # Camera died — emit error event
                    err_event = DetectionEvent(
                        timestamp=time.time(),
                        status="error",
                        detections=[],
                        hit_count=0,
                        confirmation_ratio="0/0",
                        cooldown_remaining=0.0,
                        alert_fired=False,
                        alert_class=None,
                        alert_confidence=None,
                        inference_ms=last_inference_ms,
                        capture_fps=0.0,
                    )
                    self.set_event(err_event)
                    if self._asyncio_loop and self._asyncio_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self.broadcast(err_event.to_json(self.cam_id)),
                            self._asyncio_loop,
                        )
                    time.sleep(1.0)
                    continue
                time.sleep(0.01)
                continue

            now = time.monotonic()
            run_inference = (now - last_inference_time) >= inference_interval

            if run_inference:
                last_inference_time = now

                # Serialize GPU access across all channels
                with self._inference_lock:
                    detections = self._detector.predict(frame)
                    last_inference_ms = self._detector.last_inference_ms
                last_detections = detections

                # Classification & confirmation (probability-based)
                # Run second-stage classifier verification on target detections
                adjusted_confidences = {}  # index -> adjusted conf
                for i, d in enumerate(detections):
                    if self._policy.is_target(d.class_name) and self._classifier is not None:
                        adj_conf, cls_pred = self._classifier.verify(
                            frame=frame,
                            bbox_x1=d.bbox.x1, bbox_y1=d.bbox.y1,
                            bbox_x2=d.bbox.x2, bbox_y2=d.bbox.y2,
                            detected_class=d.class_name,
                            detected_confidence=d.confidence,
                        )
                        adjusted_confidences[i] = adj_conf
                    elif self._policy.is_target(d.class_name):
                        adjusted_confidences[i] = d.confidence
                    else:
                        adjusted_confidences[i] = d.confidence

                last_adjusted_confidences = adjusted_confidences

                # Identify all target detections and select the highest percentage
                target_entries = [
                    (i, d, adjusted_confidences.get(i, d.confidence))
                    for i, d in enumerate(detections)
                    if self._policy.is_target(d.class_name)
                ]

                if target_entries:
                    # Sort/pick the detection with highest confidence percentage
                    best_target_idx, best_target_det, highest_target_conf = max(
                        target_entries, key=lambda item: item[2]
                    )
                else:
                    best_target_idx, best_target_det, highest_target_conf = None, None, 0.0

                self._confirmation.record(highest_target_conf)

                # Status
                confirmed   = self._confirmation.is_confirmed()
                in_cooldown = self._alert_manager.in_cooldown
                hits        = self._confirmation.hit_count

                if confirmed and not in_cooldown:
                    status = "alert"
                elif confirmed and in_cooldown:
                    status = "cooldown"
                elif hits > 0:
                    status = "evaluating"
                else:
                    status = "clear"

                # Build detection payload
                det_list = []
                for i, d in enumerate(detections):
                    label = self._policy.classify(d.class_name)
                    conf = adjusted_confidences.get(i, d.confidence)
                    det_list.append({
                        "class_name": d.class_name,
                        "confidence": round(conf, 3),
                        "raw_confidence": round(d.confidence, 3),
                        "label": label.name.lower(),
                        "bbox": {
                            "x1": d.bbox.x1, "y1": d.bbox.y1,
                            "x2": d.bbox.x2, "y2": d.bbox.y2,
                        },
                    })

                # Alert trigger: use the target animal with the highest percentage
                alert_fired = False
                alert_class = None
                alert_conf  = None
                if confirmed and not in_cooldown and best_target_det is not None:
                    fired = self._alert_manager.trigger(
                        class_name=best_target_det.class_name,
                        confidence=highest_target_conf,
                        frame=frame,
                    )
                    if fired:
                        alert_fired = True
                        alert_class = best_target_det.class_name
                        alert_conf  = round(highest_target_conf, 3)
                        logger.warning(f"[{self.cam_id}] ALERT FIRED: {alert_class} ({alert_conf:.1%})")

                event = DetectionEvent(
                    timestamp=time.time(),
                    status=status,
                    confirmation_ratio=self._confirmation.summary(),
                    hit_count=hits,
                    cooldown_remaining=self._alert_manager.cooldown_remaining,
                    inference_ms=last_inference_ms,
                    capture_fps=self._camera.capture_fps if self._camera else 0.0,
                    detections=det_list,
                    alert_fired=alert_fired,
                    alert_class=alert_class,
                    alert_confidence=alert_conf,
                )
                self.set_event(event)

                if self._asyncio_loop and self._asyncio_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.broadcast(event.to_json(self.cam_id)),
                        self._asyncio_loop,
                    )

            # ── Annotate frame ─────────────────────────────────────────────────
            annotated = frame.copy()
            h, w = annotated.shape[:2]

            for i, det in enumerate(last_detections):
                label = self._policy.classify(det.class_name)
                if label == ClassLabel.TARGET:
                    color = COLOR_TARGET
                elif label == ClassLabel.PET:
                    color = COLOR_PET
                else:
                    color = COLOR_OTHER
                x1, y1, x2, y2 = (
                    int(det.bbox.x1), int(det.bbox.y1),
                    int(det.bbox.x2), int(det.bbox.y2),
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                conf = last_adjusted_confidences.get(i, det.confidence)
                text = f"{det.class_name} {conf:.0%}"
                cv2.putText(
                    annotated, text, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
                )

            evt = self.get_event()
            if evt:
                bar_h = 48
                cv2.rectangle(annotated, (0, 0), (w, bar_h), (15, 15, 20), -1)
                if evt.status == "alert":
                    s_text  = f"[{self.label}] ALERT — {evt.confirmation_ratio}"
                    s_color = COLOR_ALERT
                elif evt.status == "cooldown":
                    s_text  = f"[{self.label}] COOLDOWN {evt.cooldown_remaining:.0f}s"
                    s_color = COLOR_EVAL
                elif evt.status == "evaluating":
                    s_text  = f"[{self.label}] EVALUATING — {evt.confirmation_ratio}"
                    s_color = COLOR_EVAL
                elif evt.status == "error":
                    s_text  = f"[{self.label}] CAMERA ERROR"
                    s_color = (0, 80, 200)
                else:
                    s_text  = f"[{self.label}] CLEAR — {evt.confirmation_ratio}"
                    s_color = COLOR_CLEAR
                cv2.putText(
                    annotated, s_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, s_color, 2, cv2.LINE_AA,
                )
                fps_text = f"AI:{1000/last_inference_ms:.0f}fps  CAM:{self._camera.capture_fps:.0f}fps"
                cv2.putText(
                    annotated, fps_text, (w - 210, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
                )

            self.set_frames(raw=frame, annotated=annotated)
