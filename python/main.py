import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

# Paths
IMAGE_DIR = "/Users/lukepitstick/Projects/Data-Science/InfraDrone/pics/test"
ANNOTATION_FILE = "/Users/lukepitstick/Projects/Data-Science/InfraDrone/pics/test/_annotations.coco.json"

# Load COCO annotations
coco = COCO(ANNOTATION_FILE)

# Get all image ids
image_ids = coco.getImgIds()

# Pick one image at random
image_id = random.choice(image_ids)

# Load image metadata
image_info = coco.loadImgs(image_id)[0]
image_path = os.path.join(IMAGE_DIR, image_info["file_name"])

# Load image
image = Image.open(image_path).convert("RGB")

# Get annotations for this image
ann_ids = coco.getAnnIds(imgIds=image_id)
anns = coco.loadAnns(ann_ids)

# Build category lookup
categories = coco.loadCats(coco.getCatIds())
cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

# Display image
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(image)

# Draw annotations
for ann in anns:
    # COCO bbox format: [x, y, width, height]
    x, y, w, h = ann["bbox"]

    rect = patches.Rectangle(
        (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)

    category_name = cat_id_to_name.get(ann["category_id"], "unknown")
    ax.text(
        x, y - 5, category_name, fontsize=10, bbox=dict(facecolor="yellow", alpha=0.5)
    )

ax.set_title(image_info["file_name"])
ax.axis("off")
plt.show()
