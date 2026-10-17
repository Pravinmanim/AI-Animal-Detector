"""
AI Animal Guard — Sprint 1 Standalone Detection Script
=======================================================
Pipeline:
  Webcam → YOLO-World (GPU/CPU) → ClassificationPolicy → TemporalConfirmation
  → AlertManager → console output + local sound + screenshot

No web layer. Proves the core detection logic works reliably in isolation
before wrapping in FastAPI (Sprint 2).

Usage:
    python standalone/detect.py
    python standalone/detect.py --config backend/config.json
    python standalone/detect.py --cpu       # Force CPU mode
    python standalone/detect.py --camera 1  # Use camera index 1

Press 'q' in the OpenCV window to quit, or Ctrl+C in the terminal.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Ensure backend package is importable ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services.camera_service import CameraService
from backend.services.classification_policy import ClassificationPolicy, ClassLabel
from backend.services.detection_service import YOLOWorldDetectionService
from backend.services.temporal_confirmation import TemporalConfirmation
from backend.services.alert_manager import AlertManager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("detect")

# ── Colours for on-screen overlay ─────────────────────────────────────────────
COLOR_TARGET = (0, 0, 255)    # Red — dangerous animal
COLOR_PET    = (0, 200, 0)    # Green — pet (never alert)
COLOR_OTHER  = (180, 180, 180) # Grey — ignored
COLOR_ALERT  = (0, 0, 255)
COLOR_CLEAR  = (0, 200, 0)
COLOR_EVAL   = (0, 165, 255)   # Orange — evaluating


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def draw_overlay(
    frame: np.ndarray,
    detections,
    policy: ClassificationPolicy,
    confirmation: TemporalConfirmation,
    alert_manager: AlertManager,
    inference_ms: float,
    capture_fps: float,
) -> np.ndarray:
    """Draw bounding boxes, labels, status bar onto frame."""
    overlay = frame.copy()

    # ── Bounding boxes ────────────────────────────────────────────────────────
    for det in detections:
        label = policy.classify(det.class_name)
        if label == ClassLabel.TARGET:
            color = COLOR_TARGET
        elif label == ClassLabel.PET:
            color = COLOR_PET
        else:
            color = COLOR_OTHER

        x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        text = f"{det.class_name} {det.confidence:.0%}"
        cv2.putText(
            overlay, text,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )

    # ── Status bar (top of frame) ─────────────────────────────────────────────
    h, w = overlay.shape[:2]
    bar_h = 52
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)

    # Confirmation status
    confirmed = confirmation.is_confirmed()
    in_cooldown = alert_manager.in_cooldown
    hits = confirmation.hit_count
    window = confirmation.window_size

    if confirmed and not in_cooldown:
        status_text = f"🔴 ALERT — {confirmation.summary()}"
        status_color = COLOR_ALERT
    elif confirmed and in_cooldown:
        status_text = f"⏳ COOLDOWN {alert_manager.cooldown_remaining:.0f}s — {confirmation.summary()}"
        status_color = COLOR_EVAL
    elif hits > 0:
        status_text = f"🟡 EVALUATING — {confirmation.summary()}"
        status_color = COLOR_EVAL
    else:
        status_text = f"🟢 CLEAR — {confirmation.summary()}"
        status_color = COLOR_CLEAR

    cv2.putText(
        overlay, status_text,
        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA,
    )

    # FPS info (right side)
    fps_text = f"AI: {1000/inference_ms:.1f} FPS ({inference_ms:.0f}ms) | CAM: {capture_fps:.0f} FPS"
    cv2.putText(
        overlay, fps_text,
        (w - 400, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )

    return overlay


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    # ── Extract config values ─────────────────────────────────────────────────
    model_cfg = config["model"]
    classes_cfg = config["classes"]
    temporal_cfg = config["temporal_confirmation"]
    alert_cfg = config["alert"]
    camera_cfg = config["camera"]

    camera_index = args.camera if args.camera is not None else camera_cfg["index"]
    force_cpu = args.cpu or (model_cfg.get("device") == "cpu")
    inference_fps_target = model_cfg["inference_fps"]
    inference_interval = 1.0 / inference_fps_target  # seconds between inferences

    # ── Build services ────────────────────────────────────────────────────────
    logger.info("=== AI Animal Guard — Sprint 1 Standalone ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Target animals: {classes_cfg['target_classes']}")
    logger.info(f"Pet classes (never alert): {classes_cfg['pet_classes']}")
    logger.info(f"Inference rate target: {inference_fps_target} FPS")

    # Detection service
    logger.info("Loading YOLO-World model…")
    try:
        detector = YOLOWorldDetectionService(
            model_name=model_cfg["name"],
            all_classes=classes_cfg["all_classes"],
            conf_thresholds=classes_cfg["confidence_thresholds"],
            class_prompts=classes_cfg.get("class_prompts"),
            force_cpu=force_cpu,
        )
    except RuntimeError as exc:
        logger.critical(str(exc))
        sys.exit(1)

    logger.info(f"Model loaded on: {detector.device.upper()}")

    # Classification policy
    policy = ClassificationPolicy.from_config(
        target_classes=classes_cfg["target_classes"],
        pet_classes=classes_cfg["pet_classes"],
    )

    # Temporal confirmation (one per target class — shared for simplicity in Sprint 1)
    confirmation = TemporalConfirmation(
        window_size=temporal_cfg["window_size"],
        required_hits=temporal_cfg["required_hits"],
    )

    # Alert manager
    base_dir = Path(args.config).parent
    alert_manager = AlertManager(
        sound_file=str(base_dir / alert_cfg["sound_file"]),
        screenshots_dir=str(base_dir / alert_cfg["screenshots_dir"]),
        history_file=str(base_dir / "alert_history.json"),
        cooldown_seconds=alert_cfg["cooldown_seconds"],
        sound_enabled=alert_cfg["sound_enabled"],
    )

    # Camera service
    camera = CameraService(
        camera_index=camera_index,
        width=camera_cfg["width"],
        height=camera_cfg["height"],
    )

    logger.info(f"Opening camera {camera_index}…")
    if not camera.start():
        logger.critical(f"Cannot open camera {camera_index}: {camera.error}")
        logger.info(f"Available cameras: {CameraService.enumerate_cameras()}")
        sys.exit(1)

    logger.info("Camera open. Starting detection loop. Press 'q' to quit.")

    # ── Detection loop ────────────────────────────────────────────────────────
    last_inference_time = 0.0
    last_inference_ms = 0.0
    last_detections = []

    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            run_inference = (now - last_inference_time) >= inference_interval

            if run_inference:
                last_inference_time = now
                detections = detector.predict(frame)
                last_inference_ms = detector.last_inference_ms
                last_detections = detections

                # ── Classification & confirmation ─────────────────────────────
                target_found = any(
                    policy.is_target(d.class_name) for d in detections
                )
                confirmation.record(target_found)

                # ── Console output ────────────────────────────────────────────
                ai_fps = 1000 / last_inference_ms if last_inference_ms > 0 else 0
                status = "🔴 ALERT" if confirmation.is_confirmed() else (
                    "🟡 EVAL" if confirmation.hit_count > 0 else "🟢 CLEAR"
                )
                detected_names = [
                    f"{d.class_name}({d.confidence:.0%})" for d in detections
                ]
                logger.info(
                    f"{status} | {confirmation.summary()} | "
                    f"{ai_fps:.1f} FPS ({last_inference_ms:.0f}ms) | "
                    f"CAM {camera.capture_fps} FPS | "
                    f"{'NONE' if not detected_names else ', '.join(detected_names)}"
                )

                # ── Pet safety check ──────────────────────────────────────────
                for det in detections:
                    if policy.is_pet(det.class_name):
                        logger.info(
                            f"  → PET IGNORED: {det.class_name} ({det.confidence:.0%}) — no alert."
                        )

                # ── Alert trigger ─────────────────────────────────────────────
                if confirmation.is_confirmed() and not alert_manager.in_cooldown:
                    # Find the highest-confidence target detection
                    targets = [d for d in detections if policy.is_target(d.class_name)]
                    best = max(targets, key=lambda d: d.confidence) if targets else None
                    fired = alert_manager.trigger(
                        class_name=best.class_name if best else "unknown",
                        confidence=best.confidence if best else 0.0,
                        frame=frame,
                    )
                    if fired:
                        logger.warning(
                            f"🔊 ALERT FIRED: {best.class_name if best else 'unknown'} "
                            f"— cooldown {alert_cfg['cooldown_seconds']}s started"
                        )

                # ── Camera error check ────────────────────────────────────────
                if camera.error:
                    logger.error(f"Camera error: {camera.error}")
                    break

            # ── OpenCV display window ─────────────────────────────────────────
            display = draw_overlay(
                frame=frame,
                detections=last_detections,
                policy=policy,
                confirmation=confirmation,
                alert_manager=alert_manager,
                inference_ms=last_inference_ms if last_inference_ms > 0 else 1.0,
                capture_fps=camera.capture_fps,
            )

            cv2.imshow("AI Animal Guard — Sprint 1", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Quit requested via keyboard.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        history = alert_manager.get_history()
        logger.info(f"Session complete. Total alerts fired: {len(history)}")
        for rec in history[-5:]:
            logger.info(f"  [{rec.timestamp}] {rec.class_name} ({rec.confidence:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Animal Guard — Standalone Detector")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "backend" / "config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU inference (skip GPU)",
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera index (overrides config)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
