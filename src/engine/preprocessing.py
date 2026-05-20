import cv2
import numpy as np

# Preprocessing Functions
def resize_image(img_bgr, size=(640, 640)):
    dsize = (size, size) if isinstance(size, int) else size
    return cv2.resize(img_bgr, dsize, interpolation=cv2.INTER_LINEAR)


def apply_clahe_bgr(img_bgr, clip_limit=3.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def denoise_image(img_bgr):
    return cv2.bilateralFilter(img_bgr, d=6, sigmaColor=40, sigmaSpace=40)


def normalize_for_display(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    return np.clip(img, 0.0, 1.0)


def enhance_for_road_damage(img_bgr, size=(640, 640)):
    resized = resize_image(img_bgr, size=size)
    enhanced = apply_clahe_bgr(resized, clip_limit=2.5, tile_grid_size=(8, 8))
    return denoise_image(enhanced)