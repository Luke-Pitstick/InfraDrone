from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import cv2 as cv
import wandb

# Initialize Weights and Biases
wandb.init(
    project="road-damage",
    name="yolo26s-rd2022-img736-b16-native-aug",
    job_type="train",
)

ROOT_DIR = Path("/workspace")

DATASET_PATH = ROOT_DIR / "RD2022"
YOLO_PATH = DATASET_PATH / "yolo"
MODELS_DIR = ROOT_DIR / "InfraDrone/src/ml/models/custom_yolo"

# Hyperparameters
IMG_SIZE = 960
MIN_VISIBILITY = .25
EPOCHS = 100
MODEL_NAME = "yolo26s.yaml"
MODEL_PATH = MODELS_DIR / MODEL_NAME


CLASS_MAP = {
    "crack": 0,
    "pothole": 1,
}

model = YOLO(MODEL_NAME)

model.train(
    data=DATASET_PATH / "converted.yaml",
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=16,
    wandb=True,

    # Geometry
    degrees=10.0,       # rotation
    translate=0.05,     # shift image up/down/left/right
    scale=0.25,         # zoom in/out
    shear=2.0,          # mild perspective-ish slant
    perspective=0.0005, # very mild perspective warp
    flipud=0.0,         # road damage usually shouldn't be upside down
    fliplr=0.5,         # horizontal flip is usually fine

    # Color / lighting
    hsv_h=0.01,         # tiny hue shift
    hsv_s=0.35,         # saturation variation
    hsv_v=0.25,         # brightness variation

    # YOLO mixing augmentations
    mosaic=0.6,
    mixup=0.0,
    copy_paste=0.0,

    # Turn mosaic off near the end
    close_mosaic=10,

    # Regular training stuff
    workers=8,
    device=0,
)