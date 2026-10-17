"""
AI Animal Guard — Convenience launcher (Sprint 2).

Usage:
    python run.py              # Start server + open browser
    python run.py --no-browser # Start server only (headless)
    python run.py --host 0.0.0.0 --port 8000  # Custom binding

The server runs with hot-reload disabled (production-like) for stability.
For development with reload: uvicorn backend.main:app --reload
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Fix Unicode output on Windows consoles (cp1252 can't handle emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Animal Guard — Launch dashboard server"
    )
    parser.add_argument("--host",       default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port",       default=8000, type=int, help="Port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--reload",     action="store_true", help="Enable uvicorn auto-reload (dev mode)")
    return parser.parse_args()


def open_browser_delayed(url: str, delay: float = 2.5) -> None:
    """Open browser after a short delay to let the server start up."""
    def _open():
        time.sleep(delay)
        print(f"\n  🌐 Opening browser: {url}\n")
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main() -> None:
    args = parse_args()
    url = f"http://{args.host}:{args.port}"

    print("=" * 60)
    print("  [AI Animal Guard] -- Sprint 2")
    print(f"  Dashboard: {url}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # Generate alert sound if it doesn't exist
    sound_path = Path("backend/sounds/alert.wav")
    gen_script = Path("generate_alert_sound.py")
    if not sound_path.exists() and gen_script.exists():
        print("\n  [SOUND] Generating alert sound file...")
        sound_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(gen_script)], check=False)

    if not args.no_browser:
        open_browser_delayed(url)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", "info",
    ]
    if args.reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n  Stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\n  Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
