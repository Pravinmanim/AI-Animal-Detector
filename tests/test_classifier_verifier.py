"""
Unit tests for ClassifierVerifier — second-stage crop-and-verify.

Tests cover:
  - Crop padding logic
  - Match/mismatch confidence adjustment
  - Bypass for unknown classes
  - Disabled verifier behavior

Run with:
    python -m pytest tests/test_classifier_verifier.py -v
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.services.classifier_verifier import ClassifierVerifier


# ─── Crop padding tests ──────────────────────────────────────────────────────

class TestCropWithPadding:

    def test_normal_crop(self):
        """Basic crop within frame bounds."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = ClassifierVerifier._crop_with_padding(
            frame, 100, 100, 200, 200, padding_ratio=0.15
        )
        assert crop is not None
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_crop_with_padding_extends_region(self):
        """Crop with padding should be larger than raw bbox."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Raw bbox: 100x100
        crop_padded = ClassifierVerifier._crop_with_padding(
            frame, 100, 100, 200, 200, padding_ratio=0.15
        )
        crop_raw = ClassifierVerifier._crop_with_padding(
            frame, 100, 100, 200, 200, padding_ratio=0.0
        )
        assert crop_padded.shape[0] >= crop_raw.shape[0]
        assert crop_padded.shape[1] >= crop_raw.shape[1]

    def test_crop_clamps_to_frame_bounds(self):
        """Crop near edges should not exceed frame dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = ClassifierVerifier._crop_with_padding(
            frame, 0, 0, 50, 50, padding_ratio=0.5
        )
        assert crop is not None
        assert crop.shape[0] <= 480
        assert crop.shape[1] <= 640

    def test_crop_at_frame_edge(self):
        """Crop at the far edge should clamp correctly."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = ClassifierVerifier._crop_with_padding(
            frame, 600, 450, 640, 480, padding_ratio=0.3
        )
        assert crop is not None

    def test_invalid_bbox_returns_none(self):
        """Zero-area bbox should return None."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        crop = ClassifierVerifier._crop_with_padding(
            frame, 100, 100, 100, 100, padding_ratio=0.0
        )
        assert crop is None


# ─── Verifier behavior when disabled ─────────────────────────────────────────

class TestDisabledVerifier:

    def test_non_existent_weights_disables(self):
        """Missing weights file should gracefully disable the verifier."""
        verifier = ClassifierVerifier(
            weights_path="/nonexistent/path/model.pt",
            imgsz=224,
        )
        assert verifier.enabled is False
        assert verifier.known_classes == []

    def test_disabled_verify_returns_original_conf(self):
        """Disabled verifier should return the original confidence unchanged."""
        verifier = ClassifierVerifier(weights_path="/nonexistent/model.pt")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        adj_conf, cls_pred = verifier.verify(
            frame, 100, 100, 200, 200,
            detected_class="elephant",
            detected_confidence=0.85,
        )
        assert adj_conf == 0.85
        assert cls_pred is None


# ─── Can-verify logic ────────────────────────────────────────────────────────

class TestCanVerify:

    def test_can_verify_known_class(self):
        """Should return True for a class in the classifier vocabulary."""
        verifier = ClassifierVerifier(weights_path="/nonexistent/model.pt")
        # Manually set class names for testing
        verifier._class_names = ["cat", "dog", "elephant", "cow", "horse", "lion"]
        verifier._enabled = True
        assert verifier.can_verify("elephant") is True
        assert verifier.can_verify("ELEPHANT") is True  # case insensitive

    def test_cannot_verify_unknown_class(self):
        """Should return False for a class NOT in the classifier vocabulary."""
        verifier = ClassifierVerifier(weights_path="/nonexistent/model.pt")
        verifier._class_names = ["cat", "dog", "elephant"]
        verifier._enabled = True
        assert verifier.can_verify("snake") is False
        assert verifier.can_verify("tiger") is False

    def test_unknown_class_bypassed_in_verify(self):
        """verify() should return original confidence for unknown classes."""
        verifier = ClassifierVerifier(weights_path="/nonexistent/model.pt")
        verifier._class_names = ["cat", "dog", "elephant"]
        verifier._enabled = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        adj_conf, cls_pred = verifier.verify(
            frame, 100, 100, 200, 200,
            detected_class="snake",
            detected_confidence=0.75,
        )
        assert adj_conf == 0.75  # Unchanged
        assert cls_pred is None


# ─── Confidence adjustment logic (mocked model) ──────────────────────────────

class TestConfidenceAdjustment:
    """Tests with mocked YOLO model to verify match/mismatch logic."""

    def _make_verifier_with_mock(self, top1_name: str, top1_conf: float):
        """Create a verifier with a mocked classifier model."""
        verifier = ClassifierVerifier(weights_path="/nonexistent/model.pt")
        verifier._class_names = ["cat", "cow", "dog", "elephant", "horse", "lion"]
        verifier._enabled = True
        verifier._device = "cpu"

        # Mock model predict
        mock_probs = MagicMock()
        mock_probs.top1 = verifier._class_names.index(top1_name)
        mock_probs.top1conf = top1_conf

        mock_result = MagicMock()
        mock_result.probs = mock_probs

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        verifier._model = mock_model

        return verifier

    def test_match_averages_confidence(self):
        """When classifier agrees, confidence should be averaged."""
        verifier = self._make_verifier_with_mock("elephant", 0.92)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        adj_conf, cls_pred = verifier.verify(
            frame, 100, 100, 200, 200,
            detected_class="elephant",
            detected_confidence=0.80,
        )
        expected = (0.80 + 0.92) / 2.0  # 0.86
        assert abs(adj_conf - expected) < 1e-6
        assert cls_pred == "elephant"

    def test_mismatch_halves_confidence(self):
        """When classifier disagrees, confidence should be halved."""
        verifier = self._make_verifier_with_mock("dog", 0.88)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        adj_conf, cls_pred = verifier.verify(
            frame, 100, 100, 200, 200,
            detected_class="elephant",
            detected_confidence=0.70,
        )
        expected = 0.70 * 0.5  # 0.35
        assert abs(adj_conf - expected) < 1e-6
        assert cls_pred == "dog"

    def test_low_classifier_confidence_treated_as_mismatch(self):
        """If classifier agrees but below min_confidence, treat as mismatch."""
        verifier = self._make_verifier_with_mock("elephant", 0.4)  # Below 0.6 threshold
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        adj_conf, cls_pred = verifier.verify(
            frame, 100, 100, 200, 200,
            detected_class="elephant",
            detected_confidence=0.80,
        )
        # Classifier says elephant but with only 0.4 confidence (< 0.6 threshold)
        # → treated as mismatch → halved
        expected = 0.80 * 0.5
        assert abs(adj_conf - expected) < 1e-6
