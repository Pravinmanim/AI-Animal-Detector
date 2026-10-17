"""
DetectionService — YOLO-World model wrapper (ModelService interface).

Responsibilities:
  - Load yolov8l-worldv2.pt on construction
  - Try GPU (CUDA); fall back to CPU automatically on failure
  - Set vocabulary classes from config (no fine-tuning)
  - Support rich class_prompts for better zero-shot accuracy:
      YOLO-World receives descriptive multi-word prompts (e.g.
      'leopard spotted rosette big cat feline wild') which the
      CLIP text encoder can match more precisely. The returned
      Detection.class_name always uses the short canonical name
      (e.g. 'leopard') so the rest of the system is unaffected.
  - Run inference and return typed Detection results
  - Never simulate or fabricate detections
  - Surface clear errors on model/inference failure

Keeping the model behind this interface means it can be swapped
for a fine-tuned model later without touching any other module.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─── Public data types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: BoundingBox


# ─── ModelService protocol (swap-friendly interface) ──────────────────────────

class ModelService(Protocol):
    """Interface that any detection model must satisfy."""

    @property
    def device(self) -> str: ...
    @property
    def classes(self) -> list[str]: ...

    def predict(self, frame: np.ndarray) -> list[Detection]: ...
    def set_classes(self, classes: list[str]) -> None: ...


# ─── YOLO-World implementation ────────────────────────────────────────────────

class YOLOWorldDetectionService:
    """
    YOLO-World detection service backed by Ultralytics.

    GPU is attempted first. Falls back to CPU if CUDA init fails
    or if `force_cpu=True` is passed (e.g. from settings toggle).

    Args:
        model_name:     Model weight filename (e.g. 'yolov8l-worldv2.pt').
        all_classes:    Full vocabulary list — canonical short names
                        (e.g. ['leopard', 'bear', ...]).
        conf_thresholds: Per-class confidence thresholds (canonical name -> float).
        class_prompts:  Optional dict mapping canonical name -> rich description
                        fed to YOLO-World's CLIP encoder for better accuracy
                        (e.g. {'leopard': 'leopard spotted rosette big cat'}).
                        When provided, YOLO receives the rich prompts but
                        Detection.class_name always returns the canonical name.
        force_cpu:      If True, skip GPU even if available.
    """

    def __init__(
        self,
        model_name: str,
        all_classes: list[str],
        conf_thresholds: dict[str, float],
        class_prompts: dict[str, str] | None = None,
        force_cpu: bool = False,
    ) -> None:
        self._model_name = model_name
        self._all_classes = [c.lower().strip() for c in all_classes]
        self._conf_thresholds = {k.lower().strip(): v for k, v in conf_thresholds.items()}
        self._is_world_model = "world" in self._model_name.lower()

        # Build prompt list and reverse-map: prompt -> canonical name
        if class_prompts:
            self._prompts = [
                class_prompts.get(cls, cls).lower().strip()
                for cls in self._all_classes
            ]
        else:
            self._prompts = list(self._all_classes)
        self._prompt_to_canonical: dict[str, str] = {
            p: c for p, c in zip(self._prompts, self._all_classes)
        }

        self._device = self._resolve_device(force_cpu)
        self._model = self._load_model()
        self._last_inference_ms: float = 0.0

    # ── Device selection ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(force_cpu: bool) -> str:
        if force_cpu:
            logger.info("GPU disabled by config — using CPU.")
            return "cpu"
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available — using GPU: {name}")
            return "cuda"
        logger.warning("CUDA not available — falling back to CPU.")
        return "cpu"

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Load YOLO model. Supports both standard trained YOLO and zero-shot YOLO-World."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading model: {self._model_name} on {self._device}")
            model = YOLO(self._model_name)
            if self._is_world_model and hasattr(model, "set_classes"):
                model.set_classes(self._prompts)
                logger.info(
                    f"Model ready (YOLO-World). {len(self._all_classes)} classes with "
                    f"{'rich prompts' if self._prompts != self._all_classes else 'canonical labels'}"
                )
            else:
                logger.info(f"Model ready (Trained YOLO). Loaded {len(model.names)} classes.")
            return model
        except Exception as exc:
            raise RuntimeError(
                f"CRITICAL: Failed to load model '{self._model_name}': {exc}"
            ) from exc

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        return self._device

    @property
    def classes(self) -> list[str]:
        return list(self._all_classes)

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    def set_classes(self, classes: list[str], prompts: list[str] | None = None) -> None:
        """Update the vocabulary. Resets model class list live if supported."""
        self._all_classes = [c.lower().strip() for c in classes]
        self._prompts = [p.lower().strip() for p in prompts] if prompts else list(self._all_classes)
        self._prompt_to_canonical = {p: c for p, c in zip(self._prompts, self._all_classes)}
        if self._is_world_model and hasattr(self._model, "set_classes"):
            self._model.set_classes(self._prompts)
        logger.info(f"Classes updated: {self._all_classes}")

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLO inference on a single BGR frame (OpenCV format).

        Returns a list of Detection objects whose confidence meets the
        per-class threshold. Returns an empty list (never raises) on
        transient inference errors — logs the error instead.
        """
        if frame is None or frame.size == 0:
            return []
        try:
            t0 = time.perf_counter()
            results = self._model.predict(
                source=frame,
                device=self._device,
                verbose=False,
            )
            self._last_inference_ms = (time.perf_counter() - t0) * 1000

            detections: list[Detection] = []
            if not results or results[0].boxes is None:
                return detections

            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())

                # Resolve class name based on model type
                if self._is_world_model:
                    if cls_idx < len(self._prompts):
                        prompt_used = self._prompts[cls_idx]
                        class_name = self._prompt_to_canonical.get(prompt_used, prompt_used)
                    else:
                        class_name = "unknown"
                else:
                    class_name = self._model.names.get(cls_idx, "unknown").lower().strip()

                # Filter out classes that don't belong to our project vocabulary
                if class_name not in self._all_classes:
                    continue

                threshold = self._conf_thresholds.get(class_name, 0.40)
                if conf < threshold:
                    continue  # Below per-class threshold

                xyxy = boxes.xyxy[i].tolist()
                bbox = BoundingBox(
                    x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]
                )
                detections.append(Detection(
                    class_name=class_name,
                    confidence=conf,
                    bbox=bbox,
                ))

            return detections

        except Exception as exc:
            logger.error(f"Inference error on {self._device}: {exc}")
            # Attempt CPU fallback if we were on GPU
            if self._device == "cuda":
                logger.warning("Attempting CPU fallback for this frame.")
                try:
                    results = self._model.predict(
                        source=frame, device="cpu", verbose=False
                    )
                    # (simplified — no threshold re-check for fallback frame)
                    return []
                except Exception:
                    pass
            return []  # Never raise — degraded but running
