import torch
from pathlib import Path
import yaml


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        f"Could not find project root (pyproject.toml) from {Path(__file__).resolve()}"
    )


PROJECT_ROOT = _find_project_root()
TRAIN_CONFIG_PATH = PROJECT_ROOT / "src/ml/configs/train.yaml"


def get_project_root() -> Path:
    """
    Get the project root directory.
    Returns:
        Path: The project root directory.
    """
    return require_path_exists(PROJECT_ROOT)


def check_mps_availability() -> bool:
    """
    Check if MPS is available on the system.
    Returns:
        bool: True if MPS is available, False otherwise.
    """
    return torch.backends.mps.is_available()

def check_cuda_availability() -> bool:
    """
    Check if CUDA is available on the system.
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def get_device():
    """
    Get the device to use for training.
    Returns:
        str: The device to use for training.
    """
    if check_cuda_availability():
        return 0
    elif check_mps_availability():
        return 'mps'
    else:
        return 'cpu'
    
    
def require_path_exists(path: Path) -> Path:
    """
    Require a file or directory path to exist.
    Returns:
        Path: The path if it exists.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path {path} does not exist")
    return path


def check_file_exists(file_path: Path) -> Path:
    """
    Backward-compatible alias for require_path_exists.
    """
    return require_path_exists(file_path)


def load_config(config_path: Path) -> dict:
    """
    Load a YAML configuration file.
    Returns:
        dict: The configuration as a dictionary.
    """
    config_path = require_path_exists(config_path)
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} must contain a YAML mapping")

    return config


def load_train_config() -> dict:
    """
    Load the shared training configuration.
    Returns:
        dict: The training configuration as a dictionary.
    """
    return load_config(TRAIN_CONFIG_PATH)


def prepare_yolo_dataset_yaml(dataset_yaml: Path) -> Path:
    """
    Ensure a YOLO dataset YAML resolves paths from its own directory.
    Fixes stale absolute paths copied from another machine.
    """
    dataset_yaml = require_path_exists(dataset_yaml.resolve())
    dataset_root = dataset_yaml.parent.resolve()

    with open(dataset_yaml, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML {dataset_yaml} must contain a mapping")

    path_value = data.get("path")
    if path_value is not None:
        configured = Path(str(path_value)).expanduser()
        if not configured.is_absolute():
            configured = (dataset_yaml.parent / configured).resolve()
        else:
            configured = configured.resolve()

        if configured != dataset_root:
            data.pop("path", None)
            with open(dataset_yaml, "w", encoding="utf-8") as file:
                yaml.dump(
                    data,
                    file,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                )

            for cache_file in dataset_root.rglob("*.cache"):
                cache_file.unlink(missing_ok=True)

    return dataset_yaml
