import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def _grid_shape(count: int, max_cols: int) -> tuple[int, int]:
    cols = min(max_cols, count)
    rows = (count + cols - 1) // cols
    return rows, cols


def display_matrix(
    matrices: np.ndarray | list[np.ndarray],
    *,
    title: str | None = None,
    titles: list[str] | None = None,
    cmap: str = "gray",
    colorbar: bool = False,
    figsize: tuple[float, float] | None = None,
    max_cols: int = 4,
    suptitle: str | None = None,
) -> None:
    """
    Display one or more 2D numpy arrays with matplotlib.

    Pass a single 2D array or a list of them. Use the toolbar zoom/pan
    controls (magnifying glass / hand icons) to inspect each subplot.
    """
    if isinstance(matrices, np.ndarray):
        if matrices.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {matrices.shape}")
        matrix_list = [matrices]
        plot_titles = [title] if title else None
    else:
        matrix_list = list(matrices)
        for index, matrix in enumerate(matrix_list):
            if matrix.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays, but matrix {index} has shape {matrix.shape}"
                )
        plot_titles = titles

    if not matrix_list:
        raise ValueError("No matrices to display")

    if plot_titles is None:
        plot_titles = [f"Matrix {index}" for index in range(1, len(matrix_list) + 1)]
    elif len(plot_titles) != len(matrix_list):
        raise ValueError(
            f"Got {len(plot_titles)} titles for {len(matrix_list)} matrices"
        )

    count = len(matrix_list)
    rows, cols = _grid_shape(count, max_cols)
    if figsize is None:
        figsize = (3.5 * cols, 3.5 * rows) if count > 1 else (6, 6)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    if suptitle:
        fig.suptitle(suptitle, fontsize=12, y=1.02)

    for axis in axes[count:]:
        axis.axis("off")

    for matrix, plot_title, axis in zip(matrix_list, plot_titles, axes):
        image = axis.imshow(matrix, cmap=cmap, interpolation="nearest")
        axis.set_title(plot_title, fontsize=9)
        axis.set_aspect("equal")
        if colorbar:
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    

def mask_image_to_segment_arrays(mask_path: Path, threshold: int = 127) -> list[np.ndarray]:
    """
    Convert a black/white mask image into one binary array per connected white segment.

    Returns:
        A list of 2D uint8 arrays. Each array has the same HxW shape as the original mask.
        Pixels for that segment are 1, all other pixels are 0.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")

    binary_mask = (mask > threshold).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)

    segment_arrays = []

    # label 0 is background, so start at 1
    for label_id in range(1, num_labels):
        segment = (labels == label_id).astype(np.uint8)
        segment_arrays.append(segment)

    return segment_arrays