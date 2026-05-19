from pathlib import Path

import albumentations as A
import cv2
from albumentations.core.composition import TransformType

DATASET_ROOT = Path(
    "/Users/lukepitstick/Projects/Data-Science/InfraDrone/python/datasets/RDD2022"
)
YOLO_ROOT = Path(
    "/Users/lukepitstick/Projects/Data-Science/InfraDrone/python/datasets/RDD2022/yolo"
)

IMG_SIZE = 720

CLASS_MAP = {
    "longitudinal crack": 0,
    "transverse crack": 1,
    "alligator crack": 2,
    "other corruption": 3,
    "pothole": 4,
}

DRONE_TRANSFORMS: list[TransformType] = [
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
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
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
            A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
            A.ImageCompression(quality_range=(55, 95), p=1.0),
        ],
        p=0.25,
    ),
]

drone_like_transforms = A.Compose(
    DRONE_TRANSFORMS,
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.25,
    ),
)

img_name = "China_MotorBike_000449.jpg"

img_path = DATASET_ROOT / f"train/images/{img_name}"

image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

augmented_img = drone_like_transforms(image=image)["image"]

cv2.imshow("image", augmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
