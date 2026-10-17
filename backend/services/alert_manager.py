"""
AlertManager — handles confirmed-alert actions.

On a confirmed detection:
  1. Play local audio alert (pygame WAV, or pyttsx3 TTS fallback)
  2. Save a screenshot with timestamp
  3. Record a history entry (in-memory list + JSON file)
  4. Start cooldown timer — no repeat sound during cooldown
  5. After cooldown expires: new alert allowed if still confirmed

Fully offline. No cloud calls. No continuous recording.
Screenshots only on confirmed alerts.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlertRecord:
    timestamp: str
    class_name: str
    confidence: float
    confirmed: bool
    screenshot_path: Optional[str] = None


class AlertManager:
    """
    Manages sound alerts, screenshots, history, and cooldown.

    Args:
        sound_file:       Path to WAV file for audio alert.
        screenshots_dir:  Directory for saving alert screenshots.
        history_file:     JSON file for persisting alert history.
        cooldown_seconds: Seconds to wait before re-alerting.
        sound_enabled:    If False, skip audio (visual + screenshot still occur).
    """

    def __init__(
        self,
        sound_file: str,
        screenshots_dir: str,
        history_file: str,
        cooldown_seconds: float = 30.0,
        sound_enabled: bool = True,
    ) -> None:
        self._sound_file = sound_file
        self._screenshots_dir = Path(screenshots_dir)
        self._history_file = Path(history_file)
        self._cooldown_seconds = cooldown_seconds
        self._sound_enabled = sound_enabled

        self._screenshots_dir.mkdir(parents=True, exist_ok=True)

        self._history: list[AlertRecord] = []
        self._last_alert_time: float = 0.0
        self._lock = threading.Lock()

        # Load history from disk if it exists
        self._load_history()

        # Initialize pygame mixer (audio subsystem)
        self._pygame_ok = False
        self._init_audio()

    # ── Audio init ────────────────────────────────────────────────────────────

    def _init_audio(self) -> None:
        try:
            import pygame
            pygame.mixer.init()
            self._pygame_ok = True
            logger.info("pygame mixer initialized for audio alerts.")
        except Exception as exc:
            logger.warning(f"pygame mixer init failed: {exc} — will use pyttsx3 fallback.")

    # ── Cooldown ─────────────────────────────────────────────────────────────

    @property
    def in_cooldown(self) -> bool:
        """True if we are still within the cooldown window."""
        return (time.monotonic() - self._last_alert_time) < self._cooldown_seconds

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining in cooldown, 0.0 if not in cooldown."""
        elapsed = time.monotonic() - self._last_alert_time
        remaining = self._cooldown_seconds - elapsed
        return max(0.0, remaining)

    # ── Main trigger ─────────────────────────────────────────────────────────

    def trigger(
        self,
        class_name: str,
        confidence: float,
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Attempt to fire an alert.

        Returns True if an alert was fired, False if suppressed by cooldown.

        Args:
            class_name:  Detected class (e.g. 'snake').
            confidence:  Detection confidence score.
            frame:       Current BGR frame for screenshot (optional).
        """
        with self._lock:
            if self.in_cooldown:
                logger.debug(
                    f"Alert suppressed (cooldown): {cooldown_remaining:.1f}s remaining"
                    if (cooldown_remaining := self.cooldown_remaining) else "cooldown"
                )
                return False

            self._last_alert_time = time.monotonic()

        # Everything below is outside the lock (I/O can be slow)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path: Optional[str] = None

        # 1. Save screenshot
        if frame is not None:
            screenshot_path = self._save_screenshot(frame, class_name, timestamp)

        # 2. Play audio alert
        if self._sound_enabled:
            self._play_alert()

        # 3. Record history
        record = AlertRecord(
            timestamp=timestamp,
            class_name=class_name,
            confidence=round(confidence, 4),
            confirmed=True,
            screenshot_path=screenshot_path,
        )
        with self._lock:
            self._history.append(record)
        self._save_history()

        logger.info(
            f"🔊 ALERT FIRED: {class_name} ({confidence:.1%}) | "
            f"screenshot={screenshot_path} | cooldown={self._cooldown_seconds}s"
        )
        return True

    # ── Screenshot ────────────────────────────────────────────────────────────

    def _save_screenshot(
        self,
        frame: np.ndarray,
        class_name: str,
        timestamp: str,
    ) -> Optional[str]:
        """Save alert screenshot. Returns file path or None on failure."""
        try:
            # Sanitize class name for filename (no path traversal)
            safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in class_name)
            filename = f"alert_{timestamp}_{safe_name}.jpg"
            path = self._screenshots_dir / filename
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            logger.info(f"Screenshot saved: {path}")
            return str(path)
        except Exception as exc:
            logger.error(f"Screenshot save failed: {exc}")
            return None

    # ── Audio ─────────────────────────────────────────────────────────────────

    def _play_alert(self) -> None:
        """Play WAV alert sound. Falls back to pyttsx3 TTS on failure."""
        def _play() -> None:
            # Try pygame WAV playback first
            if self._pygame_ok and os.path.isfile(self._sound_file):
                try:
                    import pygame
                    pygame.mixer.music.load(self._sound_file)
                    pygame.mixer.music.play()
                    # Wait for playback to finish (non-blocking for the main thread)
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                    return
                except Exception as exc:
                    logger.warning(f"pygame playback failed: {exc} — trying pyttsx3.")

            # Fallback: pyttsx3 TTS
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say("Animal alert! Animal detected!")
                engine.runAndWait()
                del engine
            except Exception as exc:
                logger.error(f"All audio playback failed: {exc}")

        # Play in background thread so main detection loop isn't blocked
        threading.Thread(target=_play, daemon=True).start()

    # ── History persistence ───────────────────────────────────────────────────

    def _save_history(self) -> None:
        """Persist history to JSON. Fails silently (non-critical)."""
        try:
            with self._lock:
                data = [asdict(r) for r in self._history]
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"History save failed: {exc}")

    def _load_history(self) -> None:
        """Load history from JSON on startup."""
        if not self._history_file.exists():
            return
        try:
            with open(self._history_file, encoding="utf-8") as f:
                data = json.load(f)
            self._history = [AlertRecord(**r) for r in data]
            logger.info(f"Loaded {len(self._history)} history records from {self._history_file}")
        except Exception as exc:
            logger.warning(f"History load failed (starting fresh): {exc}")
            self._history = []

    def get_history(self) -> list[AlertRecord]:
        with self._lock:
            return list(self._history)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
        self._save_history()
        logger.info("Alert history cleared.")
