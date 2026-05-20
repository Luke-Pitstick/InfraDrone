import torch
from pathlib import Path
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
