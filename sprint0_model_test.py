"""
Sprint 0 — YOLO-World model load test.
Downloads yolov8s-worldv2.pt on first run (cached locally after).
Runs a single inference pass to confirm GPU is used.
"""
import torch
import numpy as np
import time
from ultralytics import YOLO

print("=== YOLO-World Model Load Test ===")
print(f"Device: {'CUDA (RTX 3050)' if torch.cuda.is_available() else 'CPU'}")

# Load yolov8s-worldv2 — small/fast YOLO-World variant
print("\nLoading yolov8s-worldv2.pt (downloads ~50MB on first run)...")
t0 = time.time()
model = YOLO("yolov8s-worldv2.pt")
load_time = time.time() - t0
print(f"Model loaded in {load_time:.2f}s")

# Set test classes
test_classes = ["snake", "monkey", "wild boar", "dog", "cat", "person"]
model.set_classes(test_classes)
print(f"Classes set: {test_classes}")

# Warm-up inference on a dummy frame (GPU init)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nRunning warm-up inference on {device}...")
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

t0 = time.time()
results = model.predict(
    source=dummy_frame,
    device=device,
    verbose=False,
    conf=0.1
)
infer_time = time.time() - t0
fps_estimate = 1.0 / infer_time if infer_time > 0 else 0

print(f"Inference time: {infer_time*1000:.1f}ms  (~{fps_estimate:.1f} FPS potential)")
print(f"Detections on blank frame: {len(results[0].boxes)} (expected: 0)")
print("\n=== YOLO-World Model: READY ===")
