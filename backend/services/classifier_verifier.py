"""
ClassifierVerifier — second-stage crop-and-classify verification.

Takes a YOLO detection (bbox + class_name) and verifies it against
a trained image classifier. Adjusts confidence based on agreement:
  - Match:    averaged confidence (detector + classifier)
  - Mismatch: halved confidence (likely false positive)
  - Unknown:  bypassed (class not in classifier vocabulary)

The classifier model is loaded once and shared across all camera
channels via the same pattern as the main detector.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ClassifierVerifier:
    """
    Second-stage classifier for verifying YOLO detections.

    Args:
        weights_path:   Path to the trained classifier weights (.pt).
        imgsz:          Input image size for the classifier (default 224).
        min_confidence: Minimum classifier confidence to count as a match.
        device:         Device string ('cuda', 'cpu', or 'auto').
    """

    def __init__(
        self,
        weights_path: str,
        imgsz: int = 224,
        min_confidence: float = 0.6,
        device: str = "auto",
    ) -> None:
        self._imgsz = imgsz
        self._min_confidence = min_confidence
        self._model = None
        self._class_names: list[str] = []
        self._enabled = False

        self._load_model(weights_path, device)

    def _load_model(self, weights_path: str, device: str) -> None:
        """Load the YOLOv8 classifier model."""
        path = Path(weights_path)
        if not path.exists():
            logger.warning(
                f"Classifier weights not found: {path} — "
                "second-stage verification disabled."
            )
            return

        try:
            from ultralytics import YOLO
            import torch

            self._model = YOLO(str(path))

            # Resolve device
            if device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                actual_device = device
            self._device = actual_device

            # Extract class names from the model
            if hasattr(self._model, "names") and self._model.names:
                self._class_names = [
                    name.lower().strip()
                    for name in self._model.names.values()
                ]
            else:
                self._class_names = []

            self._enabled = True
            logger.info(
                f"ClassifierVerifier loaded: {path.name} on {actual_device}, "
                f"{len(self._class_names)} classes: {self._class_names}"
            )
        except Exception as exc:
            logger.error(f"Failed to load classifier: {exc}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Whether the classifier is loaded and ready."""
        return self._enabled

    @property
    def known_classes(self) -> list[str]:
        """List of class names the classifier can recognize."""
        return list(self._class_names)

    def can_verify(self, class_name: str) -> bool:
        """Check if the classifier knows this class."""
        return self._enabled and class_name.lower().strip() in self._class_names

    def verify(
        self,
        frame: np.ndarray,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        detected_class: str,
        detected_confidence: float,
        padding_ratio: float = 0.15,
    ) -> tuple[float, Optional[str]]:
        """
        Verify a detection by cropping and classifying.

        Args:
            frame:                Full BGR frame from the camera.
            bbox_x1/y1/x2/y2:    Bounding box coordinates.
            detected_class:       Class name from the YOLO detector.
            detected_confidence:  Confidence from the YOLO detector.
            padding_ratio:        Extra padding around the crop (0.15 = 15%).

        Returns:
            (adjusted_confidence, classifier_prediction)
            - If the classifier matches: confidence is averaged.
            - If the classifier disagrees: confidence is halved.
            - If the class is unknown to the classifier: original confidence returned.
        """
        class_lower = detected_class.lower().strip()

        # Bypass if classifier can't verify this class
        if not self.can_verify(class_lower):
            return detected_confidence, None

        # Crop the bounding box region with padding
        crop = self._crop_with_padding(
            frame, bbox_x1, bbox_y1, bbox_x2, bbox_y2, padding_ratio
        )
        if crop is None or crop.size == 0:
            return detected_confidence, None

        # Run classifier
        try:
            results = self._model.predict(
                source=crop,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )

            if not results or not hasattr(results[0], "probs") or results[0].probs is None:
                return detected_confidence, None

            probs = results[0].probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            top1_name = self._class_names[top1_idx] if top1_idx < len(self._class_names) else "unknown"

            # Compare classifier prediction vs detector prediction
            if top1_name == class_lower and top1_conf >= self._min_confidence:
                # Both models agree — average the confidence (boost)
                adjusted = (detected_confidence + top1_conf) / 2.0
                logger.debug(
                    f"Classifier MATCH: {class_lower} "
                    f"(det={detected_confidence:.2f}, cls={top1_conf:.2f}) → {adjusted:.2f}"
                )
                return adjusted, top1_name
            else:
                # Models disagree — reduce confidence (likely false positive)
                adjusted = detected_confidence * 0.5
                logger.debug(
                    f"Classifier MISMATCH: det={class_lower} vs cls={top1_name}({top1_conf:.2f}) "
                    f"→ halved to {adjusted:.2f}"
                )
                return adjusted, top1_name

        except Exception as exc:
            logger.error(f"Classifier inference error: {exc}")
            return detected_confidence, None

    @staticmethod
    def _crop_with_padding(
        frame: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        padding_ratio: float = 0.15,
    ) -> Optional[np.ndarray]:
        """Crop a region from the frame with extra padding for context."""
        h, w = frame.shape[:2]

        # Calculate padding
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = box_w * padding_ratio
        pad_y = box_h * padding_ratio

        # Apply padding, clamped to frame bounds
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        return frame[cy1:cy2, cx1:cx2]
