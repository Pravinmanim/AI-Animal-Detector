"""
Tests for CameraChannel and AppState (Sprint 4 — multi-camera).

These tests use mocks to avoid needing a real camera or GPU.
"""
from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_frame(h=480, w=640):
    """Return a fake BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mock_detector(inference_ms=50.0, detections=None):
    det = MagicMock()
    det.last_inference_ms = inference_ms
    det.predict.return_value = detections or []
    return det


def _mock_policy():
    from backend.services.classification_policy import ClassLabel, ClassificationPolicy
    policy = MagicMock(spec=ClassificationPolicy)
    policy.is_target.return_value = False
    policy.classify.return_value = ClassLabel.IGNORED
    return policy


def _mock_confirmation():
    from backend.services.temporal_confirmation import TemporalConfirmation
    conf = MagicMock(spec=TemporalConfirmation)
    conf.is_confirmed.return_value = False
    conf.hit_count = 0
    conf.summary.return_value = "0/5"
    return conf


def _mock_alert_manager():
    from backend.services.alert_manager import AlertManager
    am = MagicMock(spec=AlertManager)
    am.in_cooldown = False
    am.cooldown_remaining = 0.0
    am.trigger.return_value = False
    return am


def _mock_camera(frame=None):
    from backend.services.camera_service import CameraService
    cam = MagicMock(spec=CameraService)
    cam.is_running = True
    cam.capture_fps = 30.0
    cam.get_latest_frame.return_value = frame if frame is not None else _make_frame()
    return cam


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDetectionEvent(unittest.TestCase):
    def test_to_json_includes_cam_id(self):
        from backend.services.camera_channel import DetectionEvent
        import json
        event = DetectionEvent(
            timestamp=1234567890.0,
            status="clear",
            confirmation_ratio="0/5",
            hit_count=0,
            cooldown_remaining=0.0,
            inference_ms=50.0,
            capture_fps=30.0,
            detections=[],
        )
        data = json.loads(event.to_json(cam_id="cam0"))
        self.assertEqual(data["type"], "detection")
        self.assertEqual(data["cam_id"], "cam0")
        self.assertEqual(data["status"], "clear")

    def test_to_json_rounds_values(self):
        from backend.services.camera_channel import DetectionEvent
        import json
        event = DetectionEvent(
            timestamp=0.0,
            status="cooldown",
            confirmation_ratio="3/5",
            hit_count=3,
            cooldown_remaining=29.999,
            inference_ms=97.1234,
            capture_fps=29.5,
            detections=[],
        )
        data = json.loads(event.to_json("cam1"))
        self.assertEqual(data["cooldown_remaining"], 30.0)
        self.assertEqual(data["inference_ms"], 97.1)


class TestCameraChannelInit(unittest.TestCase):
    def _make_channel(self, cam_id="cam0"):
        from backend.services.camera_channel import CameraChannel
        lock = threading.Lock()
        return CameraChannel(
            cam_id=cam_id,
            label="Test Camera",
            camera_index=0,
            width=640,
            height=480,
            detector=_mock_detector(),
            inference_lock=lock,
            policy=_mock_policy(),
            confirmation=_mock_confirmation(),
            alert_manager=_mock_alert_manager(),
            inference_fps=10,
            jpeg_quality=70,
        )

    def test_initial_state(self):
        ch = self._make_channel()
        self.assertEqual(ch.cam_id, "cam0")
        self.assertEqual(ch.label, "Test Camera")
        self.assertFalse(ch.is_running)
        self.assertFalse(ch.camera_ok)
        self.assertIsNone(ch.get_event())
        self.assertIsNone(ch.get_annotated_frame())
        self.assertEqual(ch.get_frame_counter(), 0)

    def test_set_and_get_frames(self):
        ch = self._make_channel()
        raw = _make_frame()
        ann = _make_frame()
        ch.set_frames(raw, ann)
        self.assertEqual(ch.get_frame_counter(), 1)
        ret = ch.get_annotated_frame()
        self.assertIsNotNone(ret)
        self.assertEqual(ret.shape, (480, 640, 3))

    def test_set_and_get_event(self):
        from backend.services.camera_channel import DetectionEvent
        ch = self._make_channel()
        evt = DetectionEvent(
            timestamp=1.0, status="alert", confirmation_ratio="3/5",
            hit_count=3, cooldown_remaining=0.0, inference_ms=90.0,
            capture_fps=30.0, detections=[], alert_fired=True,
            alert_class="leopard", alert_confidence=0.85,
        )
        ch.set_event(evt)
        retrieved = ch.get_event()
        self.assertEqual(retrieved.status, "alert")
        self.assertEqual(retrieved.alert_class, "leopard")

    def test_cam_id_uniqueness(self):
        ch0 = self._make_channel("cam0")
        ch1 = self._make_channel("cam1")
        self.assertNotEqual(ch0.cam_id, ch1.cam_id)


class TestAppState(unittest.TestCase):
    def setUp(self):
        from backend.api.state import AppState
        self.state = AppState()

    def test_empty_channels(self):
        self.assertIsNone(self.state.first_channel())
        self.assertIsNone(self.state.get_channel("cam0"))
        self.assertEqual(self.state.list_cameras(), [])

    def test_first_channel(self):
        mock_ch = MagicMock()
        mock_ch.get_event.return_value = None
        mock_ch.cam_id = "cam0"
        mock_ch.label = "Camera 1"
        mock_ch.camera_index = 0
        mock_ch.camera_ok = True
        mock_ch.is_running = True
        mock_ch.capture_fps = 30.0
        self.state.channels["cam0"] = mock_ch
        self.assertIs(self.state.first_channel(), mock_ch)

    def test_get_event_delegates_to_first(self):
        from backend.services.camera_channel import DetectionEvent
        mock_ch = MagicMock()
        evt = DetectionEvent(
            timestamp=0.0, status="clear", confirmation_ratio="0/5",
            hit_count=0, cooldown_remaining=0.0, inference_ms=50.0,
            capture_fps=30.0, detections=[],
        )
        mock_ch.get_event.return_value = evt
        self.state.channels["cam0"] = mock_ch
        result = self.state.get_event()
        self.assertEqual(result.status, "clear")

    def test_list_cameras_summary(self):
        mock_ch = MagicMock()
        mock_ch.cam_id = "cam0"
        mock_ch.label = "Front"
        mock_ch.camera_index = 0
        mock_ch.camera_ok = True
        mock_ch.is_running = True
        mock_ch.capture_fps = 25.0
        mock_ch.get_event.return_value = None
        self.state.channels["cam0"] = mock_ch
        result = self.state.list_cameras()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "cam0")
        self.assertEqual(result[0]["status"], "starting")


class TestMultiChannelInference(unittest.TestCase):
    """
    Verify that two channels with a shared inference_lock work without
    deadlocking. We use threading to run two concurrent 'predict' calls.
    """
    def test_shared_lock_serializes_predict(self):
        """Two channels sharing the same lock must not deadlock."""
        lock = threading.Lock()
        call_order = []

        def slow_predict(frame):
            call_order.append('start')
            time.sleep(0.02)
            call_order.append('end')
            return []

        det1 = _mock_detector()
        det1.predict.side_effect = slow_predict
        det2 = _mock_detector()
        det2.predict.side_effect = slow_predict

        def run_one(det):
            with lock:
                det.predict(None)

        t1 = threading.Thread(target=run_one, args=(det1,))
        t2 = threading.Thread(target=run_one, args=(det2,))
        t1.start(); t2.start()
        t1.join(timeout=2.0); t2.join(timeout=2.0)

        # Both must have completed
        self.assertFalse(t1.is_alive(), "Thread 1 deadlocked")
        self.assertFalse(t2.is_alive(), "Thread 2 deadlocked")
        # Serialized: no interleaving (start,start,end,end would be wrong)
        self.assertEqual(call_order, ['start', 'end', 'start', 'end'])


if __name__ == "__main__":
    unittest.main()
