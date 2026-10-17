"""
01_download_datasets.py
Downloads both Kaggle animal datasets using kagglehub.
Run: python train/01_download_datasets.py
"""
import os, sys, shutil
os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_0dbc0e24cccb829b405454a8fc225926")

import kagglehub
from pathlib import Path

RAW = Path(__file__).parent / "raw"
RAW.mkdir(exist_ok=True)

DATASETS = [
    ("alessiocorrado99/animals10", "animals10"),
    ("antobenedetti/animals",      "animals5"),
    ("iamsouravbanerjee/animal-image-dataset-90-different-animals", "animals90"),
]

for handle, name in DATASETS:
    print(f"\n{'='*60}\nDataset: {handle}  ->  raw/{name}/\n{'='*60}")
    dest = RAW / name
    if dest.exists() and any(dest.iterdir()):
        print(f"  {dest} already exists and is populated, skipping download.")
        continue

    path = None
    import time
    for attempt in range(1, 6):
        try:
            print(f"  Downloading (attempt {attempt}/5) ...")
            path = kagglehub.dataset_download(handle)
            print(f"  Downloaded/cached at: {path}")
            break
        except Exception as exc:
            print(f"  [Attempt {attempt} failed]: {exc}")
            if attempt < 5:
                time.sleep(5)
            else:
                raise

    if path:
        print(f"  Copying to {dest} ...")
        shutil.copytree(path, dest, dirs_exist_ok=True)
        print(f"  Done.")

print("\n=== Download complete. Folders in train/raw/ ===")
for f in RAW.iterdir():
    print(f"  {f.name}/")

