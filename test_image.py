"""
test_image.py — Test AI Animal Guard detection directly on an image file.

Usage:
    python test_image.py                           # Tests sample images from dataset
    python test_image.py path/to/your/image.jpg   # Tests your specific image file
"""
import sys, json
from pathlib import Path
import cv2
from backend.services.detection_service import YOLOWorldDetectionService
from backend.services.classification_policy import ClassificationPolicy, ClassLabel

def test_single_image(img_path: str):
    p = Path(img_path)
    if not p.exists():
        print(f"Error: File not found: {img_path}")
        return

    with open("backend/config.json") as f:
        cfg = json.load(f)

    print("=" * 60)
    print(f"Testing image: {p.name}")
    print(f"Model: {cfg['model']['name']}")
    print("=" * 60)

    det = YOLOWorldDetectionService(
        model_name=cfg["model"]["name"],
        all_classes=cfg["classes"]["all_classes"],
        conf_thresholds=cfg["classes"]["confidence_thresholds"],
        class_prompts=cfg["classes"].get("class_prompts"),
    )
    policy = ClassificationPolicy.from_config(
        target_classes=cfg["classes"]["target_classes"],
        pet_classes=cfg["classes"]["pet_classes"],
    )

    frame = cv2.imread(str(p))
    if frame is None:
        print(f"Error: Unable to decode image {img_path}")
        return

    detections = det.predict(frame)
    print(f"\nDetections found: {len(detections)}")

    for i, d in enumerate(detections, 1):
        label = policy.classify(d.class_name)
        if label == ClassLabel.TARGET:
            status_tag = "🚨 TARGET ANIMAL (ALERT)"
            color = (0, 0, 255)
        elif label == ClassLabel.PET:
            status_tag = "🐾 PET (SAFE / IGNORED)"
            color = (0, 255, 0)
        else:
            status_tag = "👤 NEUTRAL (SAFE / IGNORED)"
            color = (180, 180, 180)

        print(f"  [{i}] {d.class_name.upper()}: {d.confidence*100:.1f}% confidence -> {status_tag}")
        print(f"      BBox: ({d.bbox.x1:.1f}, {d.bbox.y1:.1f}) to ({d.bbox.x2:.1f}, {d.bbox.y2:.1f})")

        # Draw on frame
        x1, y1 = int(d.bbox.x1), int(d.bbox.y1)
        x2, y2 = int(d.bbox.x2), int(d.bbox.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{d.class_name} {d.confidence:.0%}"
        cv2.putText(frame, caption, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = Path("test_output.jpg")
    cv2.imwrite(str(out_path), frame)
    print(f"\nAnnotated preview saved to: {out_path.resolve()}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_single_image(sys.argv[1])
    else:
        # Default: test a sample cow from validation set
        val_cow = Path("train/data/val/cow")
        if val_cow.exists():
            sample = next(val_cow.iterdir())
            test_single_image(str(sample))
        else:
            print("Usage: python test_image.py <image_path>")
