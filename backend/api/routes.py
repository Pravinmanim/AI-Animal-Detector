"""
FastAPI routes — all HTTP endpoints and the WebSocket handler (Sprint 4).

Routes:
  GET  /              → redirect to /index.html (served by StaticFiles)
  GET  /cameras       → list all configured cameras + status
  GET  /stream/{cam_id}  → MJPEG annotated video stream for one camera
  GET  /stream           → redirect to /stream/cam0 (backward compat)
  WS   /ws/{cam_id}      → real-time JSON event push for one camera
  WS   /ws               → redirect to /ws/cam0 (backward compat)
  GET  /status           → aggregate pipeline state (all cameras)
  GET  /alerts           → full alert history (cam0 by default)
  GET  /alerts/{cam_id}  → alert history for a specific camera
  DELETE /alerts         → clear alert history (all cameras)
  GET  /config           → current config
  PUT  /config           → update config live
  GET  /screenshots/{filename} → serve saved screenshots
  PUT  /cameras/{cam_id}/enable → enable or disable a camera channel
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_state(request: Request):
    return request.app.state.app_state


def _get_channel(state, cam_id: str):
    """Return channel or raise 404."""
    ch = state.get_channel(cam_id)
    if ch is None:
        raise KeyError(cam_id)
    return ch


# ── Root redirect ─────────────────────────────────────────────────────────────

@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/index.html")


# ── Camera list ───────────────────────────────────────────────────────────────

@router.get("/cameras", summary="List all configured cameras")
async def list_cameras(request: Request):
    state = _get_state(request)
    return JSONResponse(state.list_cameras())


# ── MJPEG video stream ────────────────────────────────────────────────────────

async def _mjpeg_stream(state, cam_id: str, request: Request):
    """Shared MJPEG generator for a given camera channel."""
    try:
        channel = _get_channel(state, cam_id)
    except KeyError:
        return JSONResponse({"error": f"Camera '{cam_id}' not found."}, status_code=404)

    jpeg_quality = state.config.get("model", {}).get("jpeg_quality", 70)

    async def frame_generator():
        last_frame_counter = -1
        while True:
            if await request.is_disconnected():
                break
            current_counter = channel.get_frame_counter()
            if current_counter == last_frame_counter:
                await asyncio.sleep(0.03)
                continue
            frame = channel.get_annotated_frame()
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            last_frame_counter = current_counter
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            ret, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ret:
                await asyncio.sleep(0.03)
                continue
            jpg_bytes = buf.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpg_bytes
                + b"\r\n"
            )
            await asyncio.sleep(0.01)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/stream/{cam_id}", summary="Annotated MJPEG video stream for a camera")
async def stream_cam(cam_id: str, request: Request):
    state = _get_state(request)
    return await _mjpeg_stream(state, cam_id, request)


@router.get("/stream", summary="Annotated MJPEG stream (cam0 — backward compat)")
async def stream_default(request: Request):
    """Backward-compatible: redirects to /stream/cam0."""
    state = _get_state(request)
    first = state.first_channel()
    cam_id = first.cam_id if first else "cam0"
    return await _mjpeg_stream(state, cam_id, request)


# ── WebSocket ─────────────────────────────────────────────────────────────────

async def _ws_handler(state, cam_id: str, websocket: WebSocket):
    """Shared WS handler for a given camera channel."""
    ch = state.get_channel(cam_id)
    if ch is None:
        await websocket.close(code=1008, reason=f"Camera '{cam_id}' not found")
        return

    await websocket.accept()
    await ch.register_ws(websocket)

    # Send current state immediately on connect
    event = ch.get_event()
    if event:
        try:
            await websocket.send_text(event.to_json(cam_id))
        except Exception:
            pass

    try:
        while True:
            # Keep connection alive; background loop handles all sends
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ch.unregister_ws(websocket)


@router.websocket("/ws/{cam_id}")
async def websocket_cam(cam_id: str, websocket: WebSocket):
    """Real-time event push for a specific camera."""
    state = websocket.app.state.app_state
    await _ws_handler(state, cam_id, websocket)


@router.websocket("/ws")
async def websocket_default(websocket: WebSocket):
    """Backward-compat WebSocket — connects to the first camera (cam0)."""
    state = websocket.app.state.app_state
    first = state.first_channel()
    cam_id = first.cam_id if first else "cam0"
    await _ws_handler(state, cam_id, websocket)


# ── Status snapshot ───────────────────────────────────────────────────────────

@router.get("/status", summary="Aggregate pipeline status (all cameras)")
async def get_status(request: Request):
    state = _get_state(request)
    if not state.channels:
        return JSONResponse({"status": "starting", "message": "No cameras initialized."})

    cameras_status = []
    for cam_id, ch in state.channels.items():
        event = ch.get_event()
        cameras_status.append({
            "cam_id":           cam_id,
            "label":            ch.label,
            "status":           event.status if event else "starting",
            "confirmation_ratio": event.confirmation_ratio if event else "0/0",
            "hit_count":        event.hit_count if event else 0,
            "cooldown_remaining": event.cooldown_remaining if event else 0.0,
            "inference_ms":     event.inference_ms if event else 0.0,
            "capture_fps":      ch.capture_fps,
            "detection_count":  len(event.detections) if event else 0,
            "detections":       event.detections if event else [],
            "camera_ok":        ch.camera_ok,
            "device":           state.detector.device.upper() if state.detector else "N/A",
        })

    # Overall status: "alert" if any cam is alerting, else worst non-clear status
    all_statuses = [c["status"] for c in cameras_status]
    if "alert" in all_statuses:
        overall = "alert"
    elif "evaluating" in all_statuses:
        overall = "evaluating"
    elif "cooldown" in all_statuses:
        overall = "cooldown"
    elif "error" in all_statuses:
        overall = "error"
    else:
        overall = "clear"

    return JSONResponse({
        "overall_status": overall,
        "camera_count":   len(cameras_status),
        "cameras":        cameras_status,
    })


# ── Alert history ─────────────────────────────────────────────────────────────

def _format_alerts(history):
    return [
        {
            "timestamp":       r.timestamp,
            "class_name":      r.class_name,
            "confidence":      r.confidence,
            "confirmed":       r.confirmed,
            "screenshot_path": r.screenshot_path,
            "screenshot_url": (
                f"/screenshots/{Path(r.screenshot_path).name}"
                if r.screenshot_path else None
            ),
        }
        for r in reversed(history)
    ]


@router.get("/alerts", summary="Alert history (first camera)")
async def get_alerts(request: Request):
    state = _get_state(request)
    first = state.first_channel()
    if not first or not first._alert_manager:
        return JSONResponse([])
    return JSONResponse(_format_alerts(first._alert_manager.get_history()))


@router.get("/alerts/{cam_id}", summary="Alert history for a specific camera")
async def get_alerts_cam(cam_id: str, request: Request):
    state = _get_state(request)
    ch = state.get_channel(cam_id)
    if ch is None:
        return JSONResponse({"error": f"Camera '{cam_id}' not found"}, status_code=404)
    if not ch._alert_manager:
        return JSONResponse([])
    return JSONResponse(_format_alerts(ch._alert_manager.get_history()))


@router.delete("/alerts", summary="Clear alert history (all cameras)")
async def clear_alerts(request: Request):
    state = _get_state(request)
    for ch in state.channels.values():
        if ch._alert_manager:
            ch._alert_manager.clear_history()
    return JSONResponse({"message": "Alert history cleared for all cameras."})


# ── Config ────────────────────────────────────────────────────────────────────

@router.get("/config", summary="Current configuration")
async def get_config(request: Request):
    state = _get_state(request)
    return JSONResponse(state.config)


@router.put("/config", summary="Update configuration")
async def update_config(request: Request):
    state = _get_state(request)
    try:
        new_settings = await request.json()
        state.update_config(new_settings)
        return JSONResponse({"message": "Configuration updated successfully", "config": state.config})
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)


# ── Camera enable/disable ─────────────────────────────────────────────────────

@router.put("/cameras/{cam_id}/enable", summary="Enable or disable a camera channel")
async def set_camera_enabled(cam_id: str, request: Request):
    """
    Body: {"enabled": true|false}
    Enabling starts the channel's detection loop if it was stopped.
    Disabling stops it.
    """
    state = _get_state(request)
    ch = state.get_channel(cam_id)
    if ch is None:
        return JSONResponse({"error": f"Camera '{cam_id}' not found"}, status_code=404)
    try:
        body = await request.json()
        enabled = bool(body.get("enabled", True))
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    if enabled and not ch.is_running:
        import asyncio
        loop = asyncio.get_event_loop()
        ok = ch.start(loop)
        return JSONResponse({"cam_id": cam_id, "enabled": ok, "message": "Started" if ok else "Failed to open camera"})
    elif not enabled and ch.is_running:
        ch.stop()
        return JSONResponse({"cam_id": cam_id, "enabled": False, "message": "Stopped"})
    else:
        return JSONResponse({"cam_id": cam_id, "enabled": ch.is_running, "message": "No change"})


# ── Screenshot serving ────────────────────────────────────────────────────────

@router.get("/screenshots/{filename}", summary="Serve a saved screenshot")
async def get_screenshot(filename: str, request: Request):
    """Serve alert screenshot images from the backend/screenshots directory."""
    state = _get_state(request)
    cfg = state.config.get("alert", {})
    screenshots_dir = Path(state.config_path).parent / cfg.get("screenshots_dir", "screenshots")
    safe_name = Path(filename).name
    img_path = screenshots_dir / safe_name

    if not img_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)

    from fastapi.responses import FileResponse
    return FileResponse(str(img_path), media_type="image/jpeg")

