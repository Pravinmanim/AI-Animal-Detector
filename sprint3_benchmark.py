"""
Sprint 3 — GPU vs CPU Benchmark
================================
Measures REAL inference performance of yolov8l-worldv2.pt on:
  - CUDA (RTX 3050 6GB)
  - CPU  (fallback)

Reports:
  - Cold-start latency (first frame — CUDA JIT compile)
  - Warm median / mean / p95 latency over N frames
  - Effective AI inference FPS
  - GPU VRAM used (if pynvml available)

Then auto-tunes backend/config.json → model.inference_fps
based on real GPU throughput.

Usage:
    python sprint3_benchmark.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = ROOT / "backend" / "config.json"
MODEL_NAME    = "yolov8l-worldv2.pt"   # large model
WARMUP_FRAMES = 5    # frames to discard (CUDA JIT compile happens here)
BENCH_FRAMES  = 30   # frames to measure per device
FRAME_W, FRAME_H = 640, 480

# Classes from config
with open(CONFIG_PATH) as f:
    _cfg = json.load(f)
ALL_CLASSES = _cfg["classes"]["all_classes"]
CONF_THRESHOLDS = _cfg["classes"]["confidence_thresholds"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_test_frame() -> np.ndarray:
    """Synthetic BGR frame (640x480, green gradient) — camera-independent."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :, 1] = np.linspace(30, 120, FRAME_W, dtype=np.uint8)
    return frame


def load_model(device: str):
    from ultralytics import YOLO
    model = YOLO(MODEL_NAME)
    model.set_classes([c.lower() for c in ALL_CLASSES])
    return model


def run_inference(model, frame: np.ndarray, device: str) -> float:
    """Run one inference, return wall-clock ms."""
    t0 = time.perf_counter()
    model.predict(source=frame, device=device, verbose=False)
    return (time.perf_counter() - t0) * 1000.0


def bench_device(device: str) -> dict:
    """Load model, warm-up, benchmark, return stats dict."""
    print(f"\n{'='*55}")
    print(f"  Benchmarking device: {device.upper()}")
    print(f"{'='*55}")

    frame = make_test_frame()

    print(f"  Loading {MODEL_NAME} ...", end=" ", flush=True)
    t_load = time.perf_counter()
    model = load_model(device)
    load_ms = (time.perf_counter() - t_load) * 1000
    print(f"done ({load_ms:.0f} ms)")

    # Cold-start (1st ever inference)
    print(f"  Cold-start inference ...", end=" ", flush=True)
    cold_ms = run_inference(model, frame, device)
    print(f"{cold_ms:.0f} ms")

    # Warm-up: discard these frames
    print(f"  Warm-up ({WARMUP_FRAMES} frames) ...", end=" ", flush=True)
    for _ in range(WARMUP_FRAMES):
        run_inference(model, frame, device)
    print("done")

    # Benchmark
    print(f"  Benchmarking {BENCH_FRAMES} frames ...", end=" ", flush=True)
    latencies: list[float] = []
    for _ in range(BENCH_FRAMES):
        latencies.append(run_inference(model, frame, device))
    print("done")

    med   = statistics.median(latencies)
    mean  = statistics.mean(latencies)
    p95   = sorted(latencies)[int(len(latencies) * 0.95)]
    mn    = min(latencies)
    mx    = max(latencies)
    fps   = 1000.0 / med

    # VRAM (GPU only, optional)
    vram_mb = None
    if device == "cuda":
        try:
            vram_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
        except Exception:
            pass

    result = {
        "device":        device.upper(),
        "model":         MODEL_NAME,
        "cold_start_ms": round(cold_ms, 1),
        "median_ms":     round(med,  1),
        "mean_ms":       round(mean, 1),
        "p95_ms":        round(p95,  1),
        "min_ms":        round(mn,   1),
        "max_ms":        round(mx,   1),
        "fps":           round(fps,  1),
        "vram_used_mb":  vram_mb,
    }

    print(f"\n  Results ({device.upper()}):")
    print(f"    Cold-start : {cold_ms:>8.1f} ms")
    print(f"    Median     : {med:>8.1f} ms  ({fps:.1f} FPS)")
    print(f"    Mean       : {mean:>8.1f} ms")
    print(f"    P95        : {p95:>8.1f} ms")
    print(f"    Min / Max  : {mn:.1f} / {mx:.1f} ms")
    if vram_mb is not None:
        print(f"    VRAM used  : {vram_mb} MB")

    return result


def tune_inference_fps(gpu_fps: float) -> int:
    """Set inference_fps to ~50% of GPU throughput. Clamp 4-20."""
    return max(4, min(20, int(gpu_fps * 0.50)))


def print_comparison(gpu: dict, cpu: dict) -> None:
    speedup = cpu["median_ms"] / gpu["median_ms"] if gpu["median_ms"] > 0 else 0
    print(f"\n{'='*55}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<22} {'GPU':>10}  {'CPU':>10}")
    print(f"  {'-'*44}")
    print(f"  {'Cold-start (ms)':<22} {gpu['cold_start_ms']:>10}  {cpu['cold_start_ms']:>10}")
    print(f"  {'Median latency (ms)':<22} {gpu['median_ms']:>10}  {cpu['median_ms']:>10}")
    print(f"  {'P95 latency (ms)':<22} {gpu['p95_ms']:>10}  {cpu['p95_ms']:>10}")
    print(f"  {'Throughput (FPS)':<22} {gpu['fps']:>10}  {cpu['fps']:>10}")
    if gpu.get("vram_used_mb"):
        print(f"  {'VRAM used (MB)':<22} {gpu['vram_used_mb']:>10}  {'N/A':>10}")
    print(f"\n  GPU is {speedup:.1f}x faster than CPU (median latency)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "="*55)
    print("  AI Animal Guard -- Sprint 3: GPU vs CPU Benchmark")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Frames : {WARMUP_FRAMES} warm-up + {BENCH_FRAMES} measured")
    print("="*55)

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n  GPU detected: {gpu_name}")
    else:
        print("\n  WARNING: CUDA not available -- GPU benchmark skipped.")

    results = {}

    # GPU
    if cuda_ok:
        results["cuda"] = bench_device("cuda")
    else:
        results["cuda"] = None

    # CPU
    results["cpu"] = bench_device("cpu")

    # Comparison
    if results["cuda"] and results["cpu"]:
        print_comparison(results["cuda"], results["cpu"])

    # Auto-tune config.json
    print(f"\n{'='*55}")
    print("  AUTO-TUNING config.json")
    print(f"{'='*55}")

    best = results.get("cuda") or results.get("cpu")
    if best:
        tuned_fps = tune_inference_fps(best["fps"])
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        old_fps = cfg["model"].get("inference_fps", "?")
        cfg["model"]["inference_fps"] = tuned_fps
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  inference_fps: {old_fps} -> {tuned_fps}  "
              f"(50% of {best['device']} throughput {best['fps']:.1f} FPS)")

    # Write results JSON
    out_path = ROOT / "sprint3_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved: {out_path}")
    print("\n  Sprint 3 COMPLETE.\n")


if __name__ == "__main__":
    main()
