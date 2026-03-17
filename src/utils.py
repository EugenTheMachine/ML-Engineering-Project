"""
This module contains utility functions for loading configuration files and other common tasks.
"""
from pathlib import Path
from typing import Union

import yaml

CFG_PATH = Path("src") / "config.yaml"


def get_cfg(config_path: Union[str, Path] = CFG_PATH) -> dict:
    """
    Loads the configuration from a YAML file.

    Args:
        config_path (Union[str, Path]): The path to the YAML configuration file.
    Returns:
        dict: The configuration as a dictionary.
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
