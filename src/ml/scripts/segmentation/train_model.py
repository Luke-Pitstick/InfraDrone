from ultralytics.models import YOLO
import wandb

from ml.utils.utils import get_device, get_project_root, load_train_config, require_path_exists


# Project Settings
PROJECT_ROOT = get_project_root()

# Load config
CONFIG = load_train_config()
PROJECT_CONFIG = CONFIG["project"]
TRAIN_CONFIG = CONFIG["segmentation"]

PROJECT_NAME = PROJECT_CONFIG["name"]
WEIGHTS_DIR = PROJECT_ROOT / PROJECT_CONFIG["weights_dir"]

# Train Settings
DATASET_PATH = require_path_exists(PROJECT_ROOT / TRAIN_CONFIG["dataset"])
RUNS_PATH = PROJECT_ROOT / TRAIN_CONFIG["runs_dir"]
EXPERIMENT_NAME = TRAIN_CONFIG["experiment_name"]

DEVICE = get_device()

EPOCHS = TRAIN_CONFIG["epochs"]
BATCH = TRAIN_CONFIG["batch"]
SAVE_PERIOD = TRAIN_CONFIG["save_period"]
IMGSZ = TRAIN_CONFIG["imgsz"]
MODEL_NAME = TRAIN_CONFIG["model"]
MODEL_PATH = WEIGHTS_DIR / MODEL_NAME
WORKERS = TRAIN_CONFIG["workers"]

# Initialize Weights and Biases
wandb.init(
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    job_type="train",
)

# Initialize Model
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(MODEL_PATH)

# Train Model
model.train(
    project=RUNS_PATH,
    name=EXPERIMENT_NAME,
    data=DATASET_PATH,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMGSZ,
    save=True,
    save_period=SAVE_PERIOD,
    workers=WORKERS,
    device=DEVICE,
)
