"""Sprint 0 — Final verification script."""
import sys

print("=== Sprint 0 Verification ===")
print(f"Python: {sys.version}")

# 1. PyTorch + CUDA
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    print(f"\n[1] PyTorch: {torch.__version__}")
    print(f"    CUDA available: {cuda_ok}")
    print(f"    GPU: {gpu_name}")
    print(f"    CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"\n[1] PyTorch: FAILED — {e}")

# 2. NumPy
try:
    import numpy as np
    print(f"\n[2] NumPy: {np.__version__}")
except Exception as e:
    print(f"\n[2] NumPy: FAILED — {e}")

# 3. OpenCV + webcam
try:
    import cv2
    print(f"\n[3] OpenCV: {cv2.__version__}")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        h, w = (frame.shape[0], frame.shape[1]) if ret else (0, 0)
        print(f"    Camera 0: OPEN — frame {w}x{h}")
        cap.release()
    else:
        print("    Camera 0: FAILED to open (may be in use by another app)")
except Exception as e:
    print(f"\n[3] OpenCV: FAILED — {e}")

# 4. Ultralytics
try:
    import ultralytics
    print(f"\n[4] Ultralytics: {ultralytics.__version__}")
    # Quick YOLO-World sanity check — just check the class exists, don't download model yet
    from ultralytics import YOLO
    print("    YOLO class importable: OK")
except Exception as e:
    print(f"\n[4] Ultralytics: FAILED — {e}")

# 5. FastAPI
try:
    import fastapi
    print(f"\n[5] FastAPI: {fastapi.__version__}")
except Exception as e:
    print(f"\n[5] FastAPI: FAILED — {e}")

# 6. Uvicorn
try:
    import uvicorn
    print(f"\n[6] Uvicorn: {uvicorn.__version__}")
except Exception as e:
    print(f"\n[6] Uvicorn: FAILED — {e}")

# 7. pygame audio
try:
    import pygame
    pygame.mixer.init()
    print(f"\n[7] pygame: {pygame.__version__} — mixer init OK")
    pygame.mixer.quit()
except Exception as e:
    print(f"\n[7] pygame: FAILED — {e}")

# 8. pyttsx3 TTS fallback
try:
    import pyttsx3
    engine = pyttsx3.init()
    print(f"\n[8] pyttsx3: OK (TTS fallback ready)")
    del engine
except Exception as e:
    print(f"\n[8] pyttsx3: FAILED — {e}")

# 9. websockets
try:
    import websockets
    print(f"\n[9] websockets: {websockets.__version__}")
except Exception as e:
    print(f"\n[9] websockets: FAILED — {e}")

# 10. pydantic
try:
    import pydantic
    print(f"\n[10] pydantic: {pydantic.__version__}")
except Exception as e:
    print(f"\n[10] pydantic: FAILED — {e}")

# 11. aiofiles
try:
    import aiofiles
    print(f"\n[11] aiofiles: OK")
except Exception as e:
    print(f"\n[11] aiofiles: FAILED — {e}")

print("\n=== ALL CHECKS COMPLETE ===")
