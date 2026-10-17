"""
03_train_classifier.py
Trains a YOLOv8s image classifier on the cleaned animal dataset.
Saves best weights to weights/animal_classifier.pt
Run: python train/03_train_classifier.py
"""
import sys, shutil
from pathlib import Path

def main():
    HERE     = Path(__file__).parent
    DATA_DIR = HERE / "data"
    WEIGHTS  = HERE.parent / "weights"
    WEIGHTS.mkdir(exist_ok=True)

    if not DATA_DIR.exists():
        print("ERROR: train/data/ not found. Run 02_clean_and_merge.py first.")
        sys.exit(1)

    # Count classes
    classes = sorted([d.name for d in (DATA_DIR / "train").iterdir() if d.is_dir()])
    print(f"Classes ({len(classes)}): {classes}")
    train_count = sum(len(list((DATA_DIR / "train" / c).iterdir())) for c in classes)
    val_count   = sum(len(list((DATA_DIR / "val" / c).iterdir()))   for c in classes)
    print(f"Training images : {train_count}")
    print(f"Validation images: {val_count}")

    from ultralytics import YOLO

    # Use the small classifier base model (pre-trained on ImageNet)
    model = YOLO("yolov8s-cls.pt")

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n=== Starting training on device: {device} ===")
    results = model.train(
        data=str(DATA_DIR),
        epochs=40,
        imgsz=224,
        batch=32,
        device=device,
        workers=2,
        patience=10,
        cos_lr=True,
        degrees=15.0,
        fliplr=0.5,
        hsv_v=0.4,
        hsv_s=0.6,
        name="animal_classifier",
        project=str(WEIGHTS.parent / "train_runs"),
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best weights to weights/ dir
    best = Path(results.save_dir) / "weights" / "best.pt"
    dest = WEIGHTS / "animal_classifier.pt"
    shutil.copy2(best, dest)
    print(f"\n=== Training complete ===")
    print(f"Best weights saved to: {dest}")

    # Validate
    print("\n=== Validation ===")
    val_results = model.val(data=str(DATA_DIR), imgsz=224, device=0)
    top1 = getattr(val_results, "top1", None)
    top5 = getattr(val_results, "top5", None)
    if top1 is not None:
        print(f"Top-1 accuracy: {top1:.1%}")
    if top5 is not None:
        print(f"Top-5 accuracy: {top5:.1%}")
    print(f"\n=== Classes learned: {classes} ===")
    print("Done! The classifier is ready to use as a secondary verifier.")

if __name__ == "__main__":
    main()

