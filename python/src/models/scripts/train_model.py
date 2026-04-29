from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import albumentations as A
import cv2 as cv

ROOT_DIR = Path("workspace")

DATASET_PATH = ROOT_DIR / "RDD2022"
YOLO_PATH = DATASET_PATH / "yolo"

# Hyperparameters
IMG_SIZE = 736
MIN_VISIBILITY = .25
EPOCHS = 100


CLASS_MAP = {
    "longitudinal crack": 0,
    "transverse crack": 1,
    "alligator crack": 2,
    "other corruption": 3,
    "pothole": 4,
}

drone_like_transforms = A.Compose(
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

model = YOLO("yolo26s.pt")

model.train(data= DATASET_PATH / "converted.yaml", 
            epochs=EPOCHS, 
            imgsz=IMG_SIZE, 
            batch=16, workers=8, 
            device=0,
            augmentations=drone_like_transforms
            )