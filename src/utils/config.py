from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
