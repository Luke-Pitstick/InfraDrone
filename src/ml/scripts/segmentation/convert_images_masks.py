# Convert Images and Masks to YOLO Segmentation Labels. Credit: ChatGPT

from pathlib import Path
import cv2

ROOT = Path("/Users/lukepitstick/Projects/Data-Science/InfraDrone/datasets/segmentation/crack_segmentation_dataset/")

splits = ["train", "test"]

# Change this depending on your dataset
CLASS_ID = 0


def resolve_mask_path(mask_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def mask_to_yolo_segments(mask_path, image_path, label_path, class_id=0):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(str(image_path))

    if mask is None:
        print(f"Could not read mask: {mask_path}")
        return

    if img is None:
        print(f"Could not read image: {image_path}")
        return

    h, w = mask.shape[:2]

    # Convert mask to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find mask contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []

    for contour in contours:
        if len(contour) < 3:
            continue

        # Simplify polygon slightly
        epsilon = 0.002 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(contour) < 3:
            continue

        segment = []

        for point in contour:
            x, y = point[0]
            x_norm = x / w
            y_norm = y / h
            segment.extend([x_norm, y_norm])

        line = str(class_id) + " " + " ".join(f"{p:.6f}" for p in segment)
        lines.append(line)

    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w") as f:
        f.write("\n".join(lines))


for split in splits:
    image_dir = ROOT / split / "images"
    mask_dir = ROOT / split / "masks"
    label_dir = ROOT / split / "labels"

    for image_path in image_dir.glob("*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        mask_path = resolve_mask_path(mask_dir, image_path.stem)
        if mask_path is None:
            print(f"No mask found for {image_path.name}")
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        mask_to_yolo_segments(
            mask_path=mask_path,
            image_path=image_path,
            label_path=label_path,
            class_id=CLASS_ID
        )

print("Done converting masks to YOLO segmentation labels.")
