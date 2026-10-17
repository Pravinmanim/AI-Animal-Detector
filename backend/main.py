"""
AI Animal Guard — FastAPI application entry point (Sprint 4).

Start the server with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

Or use the convenience launcher:
    python run.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.api.routes import router
from backend.api.state import AppState

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Config path ───────────────────────────────────────────────────────────────
CONFIG_PATH = ROOT / "backend" / "config.json"


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Lifespan: startup & shutdown ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load config, initialise shared detector, create one CameraChannel
             per enabled camera, start all detection loops.
    Shutdown: stop all channels cleanly.
    """
    state: AppState = app.state.app_state
    config = _load_config(CONFIG_PATH)
    state.config = config
    state.config_path = str(CONFIG_PATH)

    logger.info("=== AI Animal Guard — Sprint 4 FastAPI Server ===")
    logger.info(f"Config: {CONFIG_PATH}")

    # ── Imports (heavy, only at startup) ─────────────────────────────────────
    from backend.services.alert_manager import AlertManager
    from backend.services.camera_channel import CameraChannel
    from backend.services.classification_policy import ClassificationPolicy
    from backend.services.classifier_verifier import ClassifierVerifier
    from backend.services.detection_service import YOLOWorldDetectionService
    from backend.services.temporal_confirmation import TemporalConfirmation

    model_cfg    = config["model"]
    classes_cfg  = config["classes"]
    temporal_cfg = config["temporal_confirmation"]
    alert_cfg    = config["alert"]

    # ── Shared detector (one model for all cameras) ───────────────────────────
    logger.info("Loading YOLO-World model…")
    force_cpu = model_cfg.get("device") == "cpu"
    try:
        state.detector = YOLOWorldDetectionService(
            model_name=model_cfg["name"],
            all_classes=classes_cfg["all_classes"],
            conf_thresholds=classes_cfg["confidence_thresholds"],
            class_prompts=classes_cfg.get("class_prompts"),
            force_cpu=force_cpu,
        )
    except RuntimeError as exc:
        logger.critical(str(exc))
        sys.exit(1)
    logger.info(f"Model ready on: {state.detector.device.upper()}")

    # ── Classifier verifier (second-stage, optional) ──────────────────────────
    classifier_cfg = config.get("classifier", {})
    classifier_verifier = None
    if classifier_cfg.get("enabled", False):
        weights_path = str(ROOT / classifier_cfg["weights"])
        classifier_verifier = ClassifierVerifier(
            weights_path=weights_path,
            imgsz=classifier_cfg.get("imgsz", 224),
            min_confidence=classifier_cfg.get("min_confidence", 0.6),
            device="auto",
        )
        if classifier_verifier.enabled:
            logger.info(
                f"Classifier verifier active: {classifier_cfg['weights']} "
                f"({len(classifier_verifier.known_classes)} classes)"
            )
        else:
            logger.warning("Classifier verifier failed to load — running without it.")
            classifier_verifier = None
    else:
        logger.info("Classifier verifier disabled in config.")

    # ── Build camera list from config ─────────────────────────────────────────
    # Support new `cameras[]` array or fall back to legacy `camera` block
    cameras_cfg = config.get("cameras")
    if not cameras_cfg:
        legacy = config.get("camera", {"index": 0, "width": 640, "height": 480})
        cameras_cfg = [{
            "id": "cam0", "label": "Camera 1",
            "index": legacy["index"],
            "width": legacy.get("width", 640),
            "height": legacy.get("height", 480),
            "enabled": True,
        }]

    backend_dir = CONFIG_PATH.parent

    for cam_cfg in cameras_cfg:
        if not cam_cfg.get("enabled", True):
            logger.info(f"Skipping disabled camera: {cam_cfg.get('id', '?')}")
            continue

        cam_id = cam_cfg["id"]
        label  = cam_cfg.get("label", cam_id)

        # Each channel gets its own policy + confirmation + alert manager
        policy = ClassificationPolicy.from_config(
            target_classes=classes_cfg["target_classes"],
            pet_classes=classes_cfg["pet_classes"],
        )
        confirmation = TemporalConfirmation(
            window_size=temporal_cfg["window_size"],
            required_hits=temporal_cfg["required_hits"],
        )
        alert_mgr = AlertManager(
            sound_file=str(backend_dir / alert_cfg["sound_file"]),
            screenshots_dir=str(backend_dir / alert_cfg["screenshots_dir"]),
            history_file=str(backend_dir / f"alert_history_{cam_id}.json"),
            cooldown_seconds=alert_cfg["cooldown_seconds"],
            sound_enabled=alert_cfg["sound_enabled"],
        )

        channel = CameraChannel(
            cam_id=cam_id,
            label=label,
            camera_index=cam_cfg["index"],
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            detector=state.detector,
            inference_lock=state.inference_lock,
            policy=policy,
            confirmation=confirmation,
            alert_manager=alert_mgr,
            inference_fps=model_cfg.get("inference_fps", 10),
            jpeg_quality=model_cfg.get("jpeg_quality", 70),
            classifier_verifier=classifier_verifier,
        )
        state.channels[cam_id] = channel
        logger.info(f"Channel created: {cam_id} ({label}) → device index {cam_cfg['index']}")

    if not state.channels:
        logger.critical("No cameras configured or all disabled. Exiting.")
        sys.exit(1)

    # ── Start all channels ────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    state.start_all_channels(loop)

    active = sum(1 for ch in state.channels.values() if ch.is_running)
    logger.info(f"Server ready. {active}/{len(state.channels)} cameras active.")
    logger.info("Dashboard: http://localhost:8000  |  Press Ctrl+C to stop.")

    yield  # ← server is running

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down…")
    state.stop_all_channels()
    logger.info("Shutdown complete.")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Animal Guard",
        description="Real-time wildlife intrusion detection API — multi-camera edition",
        version="4.0.0",
        lifespan=lifespan,
    )

    # Attach shared state
    app.state.app_state = AppState()

    # API routes
    app.include_router(router)

    # Serve frontend static files (index.html, style.css, app.js)
    frontend_dir = ROOT / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    else:
        logger.warning(f"Frontend directory not found: {frontend_dir}")

    return app


app = create_app()

