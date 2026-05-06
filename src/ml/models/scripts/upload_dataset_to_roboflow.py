import os
import shutil
import tempfile
from pathlib import Path

from roboflow import Roboflow

API_KEY = "wH1j5rP3QMyWpz2fnAdp"

WORKSPACE_ID = "lukepitstick-zv5nr"
PROJECT_ID = "road-damage-kpbof-fsdbt"

DATASET_PATH = Path(
    "/Users/lukepitstick/Projects/Data-Science/InfraDrone/python/datasets/RD2022/"
)

NUM_WORKERS = 15
DATASET_FORMAT = "yolov8"
PROJECT_TYPE = "object-detection"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_yolo_line(line: str):
    """
    YOLO detection format:
    class_id x_center y_center width height

    All coords should be normalized between 0 and 1.
    Width and height must be > 0.
    """
    parts = line.strip().split()

    if len(parts) != 5:
        return None

    try:
        class_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
    except ValueError:
        return None

    if class_id < 0:
        return None

    # Reject truly broken boxes
    if w <= 0 or h <= 0:
        return None

    # Reject boxes that are way outside normalized YOLO range
    if x < -0.01 or x > 1.01:
        return None
    if y < -0.01 or y > 1.01:
        return None
    if w < -0.01 or w > 1.01:
        return None
    if h < -0.01 or h > 1.01:
        return None

    # Clamp tiny floating point errors
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    if w == 0 or h == 0:
        return None

    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def clean_label_file(src_label: Path, dst_label: Path):
    good_lines = []
    bad_lines = []

    with open(src_label, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            cleaned = is_valid_yolo_line(stripped)

            if cleaned is None:
                bad_lines.append((line_num, stripped))
            else:
                good_lines.append(cleaned)

    dst_label.parent.mkdir(parents=True, exist_ok=True)

    with open(dst_label, "w", encoding="utf-8") as f:
        if good_lines:
            f.write("\n".join(good_lines) + "\n")

    return len(good_lines), bad_lines


def copy_clean_dataset(src_root: Path):
    temp_root = Path(tempfile.mkdtemp(prefix="roboflow_clean_"))
    print(f"Creating cleaned dataset at: {temp_root}")

    total_images = 0
    total_labels = 0
    total_bad_lines = 0
    empty_after_cleaning = 0
    missing_labels = 0

    # Copy top-level files like data.yaml
    for item in src_root.iterdir():
        if item.is_file():
            shutil.copy2(item, temp_root / item.name)

    split_map = {
        "train": "train",
        "valid": "valid",
        "val": "valid",
        "test": "test",
    }

    for src_split, dst_split in split_map.items():
        split_dir = src_root / src_split

        if not split_dir.exists():
            continue

        src_images_dir = split_dir / "images"
        src_labels_dir = split_dir / "labels"

        dst_images_dir = temp_root / dst_split / "images"
        dst_labels_dir = temp_root / dst_split / "labels"

        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)

        if not src_images_dir.exists():
            continue

        for img_path in src_images_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            total_images += 1

            dst_img = dst_images_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            src_label = src_labels_dir / f"{img_path.stem}.txt"
            dst_label = dst_labels_dir / f"{img_path.stem}.txt"

            if not src_label.exists():
                missing_labels += 1
                dst_label.touch()
                continue

            good_count, bad_lines = clean_label_file(src_label, dst_label)

            total_labels += good_count
            total_bad_lines += len(bad_lines)

            if good_count == 0:
                empty_after_cleaning += 1

            if bad_lines:
                print(f"\nBad annotations in {src_label}:")
                for line_num, bad_line in bad_lines[:10]:
                    print(f"  line {line_num}: {bad_line}")

                if len(bad_lines) > 10:
                    print(f"  ... and {len(bad_lines) - 10} more")

    print("\nCleaning summary:")
    print(f"  Images copied: {total_images}")
    print(f"  Valid labels kept: {total_labels}")
    print(f"  Bad annotation lines removed: {total_bad_lines}")
    print(f"  Missing label files: {missing_labels}")
    print(f"  Label files empty after cleaning: {empty_after_cleaning}")

    return temp_root


def upload_dataset(dataset_path: Path):
    rf = Roboflow(api_key=API_KEY)
    workspace = rf.workspace(WORKSPACE_ID)

    workspace.upload_dataset(
        project_name=PROJECT_ID,
        dataset_path=str(DATASET_PATH),
        num_workers=NUM_WORKERS,
        dataset_format=DATASET_FORMAT,
        project_type=PROJECT_TYPE,
    )


def main():
    cleaned_path = copy_clean_dataset(DATASET_PATH)

    try:
        upload_dataset(cleaned_path)
        print("\nUpload complete.")
    finally:
        print(f"Removing cleaned temp dataset: {cleaned_path}")
        shutil.rmtree(cleaned_path, ignore_errors=True)


if __name__ == "__main__":
    main()
