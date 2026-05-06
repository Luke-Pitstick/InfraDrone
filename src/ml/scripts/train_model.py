from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import albumentations as A
import cv2 as cv

ROOT_DIR = Path("workspace")

DATASET_PATH = ROOT_DIR / "InfraDrone/python/datasets/RD2022"
YOLO_PATH = DATASET_PATH / "yolo"

# Hyperparameters
IMG_SIZE = 736
MIN_VISIBILITY = .25
EPOCHS = 100
MODEL_NAME = "yolo26s.pt"

CLASS_MAP = {
    "crack": 0,
    "pothole": 1,
}

albumentations_transforms = A.Compose(
    [
        # Crop out sky/horizon/hood if present; focus on road surface
        A.RandomResizedCrop(
            size=(640, 640), scale=(0.35, 0.85), ratio=(0.75, 1.35), p=0.8
        ),
        # Mild perspective warp, not insane
        A.Perspective(scale=(0.03, 0.12), keep_size=True, fit_output=False, p=0.5),
        # Simulate drone/camera rotation
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=(-0.25, 0.15),  # zoom out more often than in
            rotate_limit=25,
            border_mode=cv.BORDER_CONSTANT,
            p=0.7,
        ),
        # Simulate imperfect drone footage
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ],
            p=0.25,
        ),
        # Lighting differences from outdoor drone footage
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.3
        ),
        # Camera noise/compression
        A.OneOf(
            [
                A.GaussNoise(p=1.0),
                A.ImageCompression(p=1.0),
            ],
            p=0.25,
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=MIN_VISIBILITY,
    ),
)

model = YOLO(MODEL_NAME)

model.train(
    data=DATASET_PATH / "converted.yaml",
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=16,

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