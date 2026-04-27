from pathlib import Path
from ultralytics import YOLO
from PIL import Image


DATASET_ROOT = Path("/Users/lukepitstick/Projects/Data-Science/InfraDrone/python/datasets/RDD2022")
YOLO_ROOT = Path("/Users/lukepitstick/Projects/Data-Science/InfraDrone/python/datasets/RDD2022/yolo")
IMG_SIZE = 640

CLASS_MAP = {
    "longitudinal crack": 0,
    "transverse crack": 1,
    "alligator crack": 2,
    "other corruption": 3,
    "pothole": 4,
}

def convert_dataset(dataset_root: Path, yolo_root: Path, img_size: int, class_map: dict):
    for split in ["train", "val", "test"]:
        for class_name, class_id in class_map.items():
            for img_path in (dataset_root / split / "images").glob("*.jpg"):
                img = Image.open(img_path)
                img = img.resize((img_size, img_size))
                img.save(yolo_root / split / "images" / img_path.name)
                with open(yolo_root / split / "labels" / img_path.with_suffix(".txt").name, "w") as f:
                    f.write(f"{class_id} {img_path.with_suffix('.txt').name}")

convert_dataset(DATASET_ROOT, YOLO_ROOT, IMG_SIZE, CLASS_MAP)