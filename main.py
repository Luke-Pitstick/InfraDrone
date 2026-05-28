from src.engine.engine import Engine
from pathlib import Path
import logging


def main() -> None:
    CONFIG_PATH = Path("config.yaml")
    logging.basicConfig(level=logging.INFO)
    engine = Engine(config_path=CONFIG_PATH)
    engine.run()

if __name__ == "__main__":
    main()