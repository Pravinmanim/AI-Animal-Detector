"""
AppState — shared application state for the FastAPI layer (Sprint 4).

Multi-camera edition: AppState now holds a dict of CameraChannel objects
(one per enabled camera). A single shared YOLOWorldDetectionService and
inference_lock are passed to every channel so GPU access is serialized.

Backward compatibility:
  - get_event(), get_annotated_frame(), etc. delegate to the first channel
    (cam0) so existing single-camera callers continue to work.

Thread safety:
  - Each CameraChannel manages its own locks internally.
  - inference_lock (threading.Lock) serializes GPU predict() across channels.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AppState:
    """
    Singleton-style holder for all live services and shared state.

    Created once in main.py lifespan and stored on app.state.
    """

    def __init__(self) -> None:
        # Shared detector (one model, GPU serialized via inference_lock)
        self.detector = None
        self.inference_lock = threading.Lock()

        # Per-camera channels: {"cam0": CameraChannel, "cam1": CameraChannel, ...}
        self.channels: Dict[str, Any] = {}

        # Config
        self.config: dict = {}
        self.config_path: str = ""

    # ── Channel helpers ───────────────────────────────────────────────────────

    def get_channel(self, cam_id: str):
        """Return a CameraChannel by id, or None if not found."""
        return self.channels.get(cam_id)

    def first_channel(self):
        """Return the first channel (cam0), or None. Used for backward compat."""
        if not self.channels:
            return None
        return next(iter(self.channels.values()))

    def list_cameras(self) -> list[dict]:
        """Return a summary list of all channels for /cameras endpoint."""
        result = []
        for cam_id, ch in self.channels.items():
            evt = ch.get_event()
            result.append({
                "id":          cam_id,
                "label":       ch.label,
                "index":       ch.camera_index,
                "camera_ok":   ch.camera_ok,
                "is_running":  ch.is_running,
                "status":      evt.status if evt else "starting",
                "capture_fps": ch.capture_fps,
            })
        return result

    # ── Backward-compat proxies (delegate to first channel) ───────────────────

    def get_event(self):
        ch = self.first_channel()
        return ch.get_event() if ch else None

    def get_annotated_frame(self):
        ch = self.first_channel()
        return ch.get_annotated_frame() if ch else None

    def get_frame_counter(self) -> int:
        ch = self.first_channel()
        return ch.get_frame_counter() if ch else 0

    def get_raw_frame(self):
        ch = self.first_channel()
        return ch.get_raw_frame() if ch else None

    # ── Config update ─────────────────────────────────────────────────────────

    def update_config(self, new_settings: dict) -> None:
        """Update live configuration and persist to disk."""
        if "target_classes" in new_settings or "pet_classes" in new_settings:
            classes_cfg = self.config.get("classes", {})
            if "target_classes" in new_settings:
                classes_cfg["target_classes"] = new_settings["target_classes"]
            if "pet_classes" in new_settings:
                classes_cfg["pet_classes"] = new_settings["pet_classes"]

            # Rebuild policy for all channels
            from backend.services.classification_policy import ClassificationPolicy
            new_policy = ClassificationPolicy.from_config(
                target_classes=classes_cfg.get("target_classes", []),
                pet_classes=classes_cfg.get("pet_classes", []),
            )
            for ch in self.channels.values():
                ch._policy = new_policy

        if "cooldown_seconds" in new_settings or "sound_enabled" in new_settings:
            alert_cfg = self.config.get("alert", {})
            for ch in self.channels.values():
                if "cooldown_seconds" in new_settings:
                    val = new_settings["cooldown_seconds"]
                    alert_cfg["cooldown_seconds"] = val
                    if ch._alert_manager:
                        ch._alert_manager.cooldown_seconds = val
                if "sound_enabled" in new_settings:
                    val = new_settings["sound_enabled"]
                    alert_cfg["sound_enabled"] = val
                    if ch._alert_manager:
                        ch._alert_manager.sound_enabled = val

        # Persist to disk
        if self.config_path:
            import json
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
                f.write("\n")

    # ── Lifecycle (delegated to main.py) ──────────────────────────────────────

    def start_all_channels(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start detection loop for all channels."""
        for cam_id, ch in self.channels.items():
            ok = ch.start(loop)
            if not ok:
                logger.warning(f"Channel {cam_id} failed to start (camera unavailable).")

    def stop_all_channels(self) -> None:
        """Stop all channels cleanly."""
        for ch in self.channels.values():
            ch.stop()

