"""
02_clean_and_merge.py
Cleans both raw datasets and merges into train/data/{train,val}/<class>/
- Maps Italian folder names -> English
- Removes: corrupted, < 50x50 px, non-RGB, exact duplicates (MD5 hash)
- Keeps only: cow, elephant, horse (targets) + dog, cat (pets) + lion->leopard
- Splits 80/20 train/val, stratified
Run: python train/02_clean_and_merge.py
"""
import hashlib, random, shutil, sys
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from collections import defaultdict
from tqdm import tqdm

random.seed(42)

HERE       = Path(__file__).parent
RAW        = HERE / "raw"
OUT        = HERE / "data"
TRAIN_OUT  = OUT / "train"
VAL_OUT    = OUT / "val"
VAL_RATIO  = 0.20
MIN_DIM    = 50   # pixels

# Italian -> English mapping for animals10
ITALIAN_MAP = {
    "cane": "dog", "gatto": "cat", "cavallo": "horse",
    "farfalla": "butterfly", "gallina": "chicken",
    "mucca": "cow", "pecora": "sheep", "ragno": "spider",
    "scoiattolo": "squirrel", "elefante": "elephant",
    "dog": "dog", "cat": "cat", "horse": "horse",
    "butterfly": "butterfly", "chicken": "chicken",
    "cow": "cow", "sheep": "sheep", "spider": "spider",
    "squirrel": "squirrel", "elephant": "elephant",
}

# antobenedetti/animals folder names
ANTOBENEDETTI_MAP = {
    "cats": "cat", "dogs": "dog", "elephants": "elephant",
    "horses": "horse", "lions": "lion",
    "cat": "cat", "dog": "dog", "elephant": "elephant",
    "horse": "horse", "lion": "lion",
}

# 90 animals dataset folder names -> standard class names
ANIMALS90_MAP = {
    "cat": "cat", "cats": "cat",
    "dog": "dog", "dogs": "dog",
    "cow": "cow", "cattle": "cow",
    "horse": "horse", "horses": "horse",
    "elephant": "elephant", "elephants": "elephant",
    "lion": "lion", "lions": "lion",
    "tiger": "tiger", "tigers": "tiger",
    "leopard": "leopard", "leopards": "leopard",
    "cheetah": "leopard", "jaguar": "leopard", "panther": "leopard",
    "bear": "bear", "bears": "bear", "panda": "bear",
    "snake": "snake", "snakes": "snake", "python": "snake", "viper": "snake",
    "monkey": "monkey", "monkeys": "monkey", "chimpanzee": "monkey", "gorilla": "monkey", "baboon": "monkey",
    "boar": "wild boar", "wild boar": "wild boar", "pig": "wild boar",
    "crocodile": "crocodile", "alligator": "crocodile",
    "wolf": "wolf", "wolves": "wolf", "coyote": "wolf",
    "hyena": "hyena", "hyenas": "hyena",
    "deer": "deer", "antelope": "deer", "moose": "deer", "reindeer": "deer",
}

# All animal classes we want in the final classifier dataset
KEEP_CLASSES = {
    "snake", "monkey", "wild boar", "dog", "cat", "elephant",
    "tiger", "leopard", "lion", "bear", "crocodile", "wolf",
    "hyena", "deer", "horse", "cow"
}

# Class -> role
ROLE = {
    "snake": "target", "monkey": "target", "wild boar": "target",
    "elephant": "target", "tiger": "target", "leopard": "target",
    "lion": "target", "bear": "target", "crocodile": "target",
    "wolf": "target", "hyena": "target", "deer": "target",
    "horse": "target", "cow": "target", "dog": "pet", "cat": "pet",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            if w < MIN_DIM or h < MIN_DIM:
                return False, "too_small"
            if img.mode not in ("RGB", "RGBA", "L"):
                return False, "bad_mode"
        return True, None
    except (UnidentifiedImageError, Exception):
        return False, "corrupted"


def collect_images(root: Path, name_map: dict) -> dict:
    """Walk root recursively, map directory names via name_map, return {class: [Path]}."""
    result = defaultdict(list)
    for folder in root.rglob("*"):
        if folder.is_dir():
            cls = name_map.get(folder.name.lower().strip())
            if cls and cls in KEEP_CLASSES:
                for f in folder.iterdir():
                    if f.is_file() and f.suffix.lower() in IMG_EXTS:
                        result[cls].append(f)
    return dict(result)


# ── Main ──────────────────────────────────────────────────────────────────────

# Gather all images from all datasets
all_images = defaultdict(list)  # {class: [Path]}

animals10_root = RAW / "animals10"
if animals10_root.exists():
    collected = collect_images(animals10_root, ITALIAN_MAP)
    for cls, paths in collected.items():
        all_images[cls].extend(paths)
    print(f"animals10: found {sum(len(v) for v in collected.values())} images across {len(collected)} classes")

animals5_root = RAW / "animals5"
if animals5_root.exists():
    collected = collect_images(animals5_root, ANTOBENEDETTI_MAP)
    for cls, paths in collected.items():
        all_images[cls].extend(paths)
    print(f"animals5:  found {sum(len(v) for v in collected.values())} images across {len(collected)} classes")

animals90_root = RAW / "animals90"
if animals90_root.exists():
    collected = collect_images(animals90_root, ANIMALS90_MAP)
    for cls, paths in collected.items():
        all_images[cls].extend(paths)
    print(f"animals90: found {sum(len(v) for v in collected.values())} images across {len(collected)} classes")

# Filter to only keep classes we care about
all_images = {k: v for k, v in all_images.items() if k in KEEP_CLASSES}

print(f"\n--- Raw totals (before cleaning) ---")
for cls in sorted(all_images):
    print(f"  {cls:12s}: {len(all_images[cls]):5d} images  [{ROLE.get(cls,'?')}]")

# Clean: validate + deduplicate
print("\n--- Cleaning images ---")
clean = defaultdict(list)
stats = defaultdict(lambda: defaultdict(int))

global_hashes = set()

for cls in sorted(all_images):
    paths = all_images[cls]
    random.shuffle(paths)
    for p in tqdm(paths, desc=f"  {cls}", unit="img"):
        ok, reason = is_valid(p)
        if not ok:
            stats[cls][reason] += 1
            continue
        h = md5(p)
        if h in global_hashes:
            stats[cls]["duplicate"] += 1
            continue
        global_hashes.add(h)
        clean[cls].append(p)
        stats[cls]["kept"] += 1

print("\n--- Cleaning report ---")
print(f"{'Class':12s} {'Kept':>6} {'Corrupt':>8} {'TooSmall':>9} {'Dup':>5}")
print("-" * 50)
for cls in sorted(clean):
    s = stats[cls]
    print(f"{cls:12s} {s['kept']:6d} {s['corrupted']:8d} {s['too_small']:9d} {s['duplicate']:5d}")
total_kept = sum(len(v) for v in clean.values())
print(f"\nTotal kept: {total_kept} images across {len(clean)} classes")

# Split 80/20 and copy to output
print("\n--- Splitting & copying ---")
shutil.rmtree(OUT, ignore_errors=True)
TRAIN_OUT.mkdir(parents=True)
VAL_OUT.mkdir(parents=True)

split_stats = {}
for cls, paths in sorted(clean.items()):
    random.shuffle(paths)
    n_val = max(1, int(len(paths) * VAL_RATIO))
    val_paths   = paths[:n_val]
    train_paths = paths[n_val:]
    split_stats[cls] = (len(train_paths), len(val_paths))

    (TRAIN_OUT / cls).mkdir(exist_ok=True)
    (VAL_OUT   / cls).mkdir(exist_ok=True)

    for p in tqdm(train_paths, desc=f"  train/{cls}", unit="img", leave=False):
        shutil.copy2(p, TRAIN_OUT / cls / p.name)
    for p in tqdm(val_paths,   desc=f"  val/{cls}",   unit="img", leave=False):
        shutil.copy2(p, VAL_OUT   / cls / p.name)

print("\n--- Final dataset split ---")
print(f"{'Class':12s} {'Train':>6} {'Val':>6}")
print("-" * 28)
for cls in sorted(split_stats):
    t, v = split_stats[cls]
    print(f"{cls:12s} {t:6d} {v:6d}")

print(f"\n=== Done! Dataset ready in train/data/ ===")
print(f"Classes: {sorted(clean.keys())}")
print(f"Training images: {sum(t for t,v in split_stats.values())}")
print(f"Validation images: {sum(v for t,v in split_stats.values())}")
