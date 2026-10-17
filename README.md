# AI Animal Detector

An intelligent real-time animal detection and intrusion alert system designed to detect and classify animals using computer vision and deep learning models.

## Features
- Real-time video stream & image detection using YOLO-World / custom models.
- FastAPI backend serving detections, alerts, and live stream.
- Web dashboard interface for monitoring alerts and camera feeds.
- Configurable alert zones, sensitivity, and threat classification.

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch & CUDA (recommended for GPU acceleration)

### Installation
```bash
pip install -r backend/requirements.txt
```

### Running the Application
```bash
python run.py
```
This will start the local server and open the web dashboard in your browser (`http://127.0.0.1:8000`).

### Development Server
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
