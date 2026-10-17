"""Generate a 2-second alert tone WAV file for AI Animal Guard."""
import wave
import struct
import math
import os

os.makedirs("backend/sounds", exist_ok=True)

sample_rate = 44100
duration = 2.0
frequency = 880.0   # A5 note
amplitude = 16000
num_samples = int(sample_rate * duration)
samples = []

for i in range(num_samples):
    t = i / sample_rate
    # Envelope: fade in 0.05s, sustain, fade out 0.3s
    env = min(1.0, t / 0.05) * min(1.0, (duration - t) / 0.3)
    val = int(amplitude * env * math.sin(2 * math.pi * frequency * t))
    # Second harmonic for a richer alert sound
    val += int((amplitude * 0.3) * env * math.sin(2 * math.pi * frequency * 2 * t))
    samples.append(max(-32767, min(32767, val)))

out_path = "backend/sounds/alert.wav"
with wave.open(out_path, "w") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    f.writeframes(struct.pack("<" + "h" * len(samples), *samples))

size = os.path.getsize(out_path)
print(f"Alert sound generated: {out_path} ({size:,} bytes, {duration}s @ {sample_rate}Hz)")
