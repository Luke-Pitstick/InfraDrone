# InfraDrone Engine

InfraDrone analyzes road imagery (typically from drones) to find pavement damage, measure it, and estimate how it may progress over time. The system combines two YOLO-based computer vision models with a physical fatigue model (Paris' Law) to turn raw images into structured damage reports.

## How it works

```
Drone / camera image
        │
        ▼
  Preprocessing (resize, CLAHE, denoise)
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Detection (YOLO26s)          Segmentation (YOLO26s-seg)
  "Where are cracks/potholes?"  "What is the exact crack shape?"
        │                              │
        └──────────────┬───────────────┘
                       ▼
              Post-processing & domain mapping
         (class, subtype, dimensions, confidence)
                       │
                       ▼
              Physical simulation (Paris' Law)
         (stress + crack geometry → growth / severity)
                       │
                       ▼
                 Damage report
```

### 1. Preprocessing

Images are enhanced before inference to handle outdoor lighting, noise, and varying resolution:

- **Resize** to the model input size
- **CLAHE** (contrast-limited adaptive histogram equalization) to bring out surface detail
- **Denoising** to reduce compression and sensor noise

This pipeline lives in `src/engine/preprocessing.py` and is shared between training and inference.

### 2. Detection model (YOLO26s)

The detection model finds **bounding boxes** for road defects. It is trained on the [RDD2022](https://github.com/seunghoonpark/RDD2022)-style dataset with two classes:

| Class ID | Label   |
|----------|---------|
| 0        | crack   |
| 1        | pothole |

Training config: `src/ml/configs/train.yaml` → `detection`  
Training script: `src/ml/scripts/detection/train_model.py`  
Dataset: `datasets/detection/RD2022/`

The `DetectionEngine` in `src/engine/detection.py` runs: **preprocess → infer → post-process**.

### 3. Segmentation model (YOLO26s-seg)

The segmentation model produces **pixel-level crack masks**, which are used for finer measurements (length, width, orientation, branching) that bounding boxes alone cannot capture.

Training config: `src/ml/configs/train.yaml` → `segmentation`  
Training script: `src/ml/scripts/segmentation/train_model.py`  
Dataset: `datasets/segmentation/crack_segmentation_dataset/`

Mask annotations are generated from binary masks via `src/ml/scripts/segmentation/convert_images_masks.py`.

The `SegmentationEngine` in `src/engine/detection.py` returns mask arrays for each detected region.

### 4. Domain model

Raw model outputs are mapped into structured `Damage` objects defined in `src/engine/engine.py`:

- **Type** — crack or pothole
- **Subtype** — e.g. longitudinal / transverse / alligator crack, or pothole size
- **Dimensions** — width and length with unit conversion (cm / inch)
- **Confidence** — model score
- **Stress range** — road class (residential → freeway), used by the fatigue model
- **Severity** — composite rating derived from geometry and simulation

Constants and enums live in `src/engine/constants.py`.

### 5. Physical simulation (Paris' Law)

Paris' Law models **fatigue crack growth** under repeated stress:

```
da/dN = C · (ΔK)^m
```

Where crack length grows per load cycle as a function of stress intensity. InfraDrone uses this to estimate how an observed defect may worsen over time given the road's traffic/stress category (`StressRange` in `constants.py`).

> **Note:** The top-level `Engine` class wires detection and segmentation together; Paris' Law integration and full `predict()` output are still being built out.

## Project layout

```
src/
├── engine/          # Inference pipeline, preprocessing, domain types
│   ├── detection.py
│   ├── preprocessing.py
│   ├── models.py
│   └── engine.py
└── ml/
    ├── configs/     # Shared train.yaml for both models
    ├── scripts/
    │   ├── detection/
    │   └── segmentation/
    ├── models/      # YOLO architecture configs and weights
    └── runs/        # Training outputs (Ultralytics + W&B)

datasets/
├── detection/RD2022/
└── segmentation/crack_segmentation_dataset/
```

## Training

Install dependencies and sync the environment:

```bash
uv sync
```

Train detection:

```bash
uv run python src/ml/scripts/detection/train_model.py
```

Train segmentation:

```bash
uv run python src/ml/scripts/segmentation/train_model.py
```

Both scripts read hyperparameters from `src/ml/configs/train.yaml` and log runs to Weights & Biases.

## Models

| Model            | Task          | Base weights     | Classes / output   |
|------------------|---------------|------------------|--------------------|
| YOLO26s          | Detection     | `yolo26s.pt`     | crack, pothole     |
| YOLO26s-seg      | Segmentation  | `yolo26s-seg.pt` | crack pixel masks  |

Physical simulation: **Paris' Law** (fatigue crack propagation).
