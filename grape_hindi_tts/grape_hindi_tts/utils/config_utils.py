"""Configuration management utilities."""

import yaml
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path


class DotDict(dict):
    """Dictionary that supports dot-notation access to nested keys."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return DotDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __repr__(self):
        return f"DotDict({dict.__repr__(self)})"


def load_config(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path for YAML file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge override configuration into base configuration.
    Override values take precedence.

    Args:
        base_config: Base configuration
        override_config: Configuration to override

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def validate_config(
    config: Dict[str, Any],
    required_fields: Optional[list] = None,
) -> bool:
    """
    Validate that configuration contains all required fields.

    Args:
        config: Configuration dictionary to validate
        required_fields: List of required field names (can use dot notation for nested)

    Returns:
        True if valid, raises ValueError otherwise
    """
    if required_fields is None:
        required_fields = [
            "model",
            "data",
            "training",
        ]

    for field in required_fields:
        if "." in field:
            # Handle nested field access like "model.encoder.hidden_dim"
            keys = field.split(".")
            current = config
            for key in keys:
                if key not in current:
                    raise ValueError(f"Missing required config field: {field}")
                current = current[key]
        else:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

    return True


def load_config_with_overrides(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load configuration from file and apply command-line overrides.

    Args:
        config_path: Path to base config YAML
        overrides: Dictionary of overrides (typically from CLI args)

    Returns:
        Merged configuration
    """
    config = load_config(config_path)

    if overrides:
        config = merge_configs(config, overrides)

    return config


def get_nested_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None,
) -> Any:
    """
    Get value from nested dictionary using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path like "model.encoder.hidden_dim"
        default: Default value if not found

    Returns:
        Value at the path, or default if not found
    """
    keys = key_path.split(".")
    current = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def set_nested_value(
    config: Dict[str, Any],
    key_path: str,
    value: Any,
) -> None:
    """
    Set value in nested dictionary using dot notation.

    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated path like "model.encoder.hidden_dim"
        value: Value to set
    """
    keys = key_path.split(".")
    current = config

    # Navigate/create nested structure
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set final value
    current[keys[-1]] = value


def config_to_dotdict(config: Dict[str, Any]) -> DotDict:
    """
    Convert regular dict to DotDict for dot-notation access.

    Args:
        config: Configuration dictionary

    Returns:
        DotDict version with nested DotDicts
    """
    result = DotDict()
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = config_to_dotdict(value)
        else:
            result[key] = value
    return result


def print_config(
    config: Dict[str, Any],
    indent: int = 0,
) -> None:
    """
    Pretty print configuration.

    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"{'  ' * indent}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{'  ' * indent}{key}: {value}")
