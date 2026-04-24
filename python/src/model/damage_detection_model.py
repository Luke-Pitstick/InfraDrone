from pathlib import Path

from ultralytics import YOLO

# Dataset yaml lives in python/datasets/yolo_pics/ (path inside yaml is relative to that directory).
_PYTHON_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_YAML = _PYTHON_ROOT / "datasets" / "yolo_pics" / "converted.yaml"

model = YOLO("yolo26n.pt")
results = model.train(
    data=str(_DATA_YAML),
    epochs=100,
    imgsz=640,
    device="mps",
    project="runs",
    name="pothole_yolo26",
)
