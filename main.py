"""
Main entry point for the engine.

This script initializes the engine and runs it in a loop, processing frames and notifying a listener.

Usage:
    uv run python main.py
"""

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