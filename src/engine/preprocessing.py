import cv2
import numpy as np


def resize_image(img_bgr, size=(640, 640)):
    """Resize a BGR image to the given width and height.

    Args:
        img_bgr: Input image in BGR order.
        size: Target size as an int (square) or ``(width, height)`` tuple.

    Returns:
        Resized BGR image.
    """
    dsize = (size, size) if isinstance(size, int) else size
    return cv2.resize(img_bgr, dsize, interpolation=cv2.INTER_LINEAR)


def apply_clahe_bgr(img_bgr, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Apply CLAHE contrast enhancement on the L channel in LAB space.

    Args:
        img_bgr: Input image in BGR order.
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid size.

    Returns:
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def denoise_image(img_bgr):
    """Apply edge-preserving bilateral filtering.

    Args:
        img_bgr: Input image in BGR order.

    Returns:
        Denoised BGR image.
    """
    return cv2.bilateralFilter(img_bgr, d=6, sigmaColor=40, sigmaSpace=40)


def normalize_for_display(img_bgr):
    """Scale pixel values to ``[0, 1]`` for matplotlib display.

    Args:
        img_bgr: Input image in BGR order.

    Returns:
        Float array clipped to ``[0, 1]``.
    """
    img = img_bgr.astype(np.float32) / 255.0
    return np.clip(img, 0.0, 1.0)


def enhance_for_road_damage(img_bgr, size=(640, 640)):
    """Resize, boost contrast, and denoise an image for model inference.

    Args:
        img_bgr: Input image in BGR order.
        size: Target resize dimensions.

    Returns:
        Preprocessed BGR image ready for YOLO.
    """
    resized = resize_image(img_bgr, size=size)
    enhanced = apply_clahe_bgr(resized, clip_limit=2.5, tile_grid_size=(8, 8))
    return denoise_image(enhanced)
