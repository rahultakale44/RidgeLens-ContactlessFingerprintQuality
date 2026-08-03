"""
Configuration utilities for RidgeLens.

This module loads and validates settings from config.yaml.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigurationError(RuntimeError):
    """Raised when RidgeLens configuration is missing or invalid."""


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load and validate the RidgeLens YAML configuration.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated configuration dictionary.

    Raises:
        ConfigurationError: When the file is missing or invalid.
    """
    path = Path(config_path)

    if not path.exists():
        raise ConfigurationError(
            f"Configuration file was not found: {path.resolve()}"
        )

    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML configuration in {path}: {error}"
        ) from error
    except OSError as error:
        raise ConfigurationError(
            f"Unable to read configuration file {path}: {error}"
        ) from error

    if not isinstance(config, dict):
        raise ConfigurationError(
            "The configuration root must be a YAML mapping."
        )

    _validate_required_sections(config)
    _validate_weights(config["weights"])
    _validate_thresholds(config["thresholds"])

    return config


def _validate_required_sections(config: dict[str, Any]) -> None:
    """Validate required top-level sections."""
    required_sections = {
        "project",
        "image",
        "thresholds",
        "weights",
        "processing",
        "performance_budget_ms",
        "guidance",
    }

    missing_sections = required_sections.difference(config.keys())

    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ConfigurationError(
            f"Missing required configuration sections: {missing}"
        )


def _validate_weights(weights: dict[str, Any]) -> None:
    """Validate metric names, values, and total weight."""
    expected_metrics = {
        "blur",
        "brightness",
        "glare",
        "roi",
        "ridge",
    }

    if not isinstance(weights, dict):
        raise ConfigurationError(
            "The weights section must be a mapping."
        )

    missing_metrics = expected_metrics.difference(weights.keys())

    if missing_metrics:
        missing = ", ".join(sorted(missing_metrics))
        raise ConfigurationError(
            f"Missing metric weights: {missing}"
        )

    numeric_weights: list[float] = []

    for metric in expected_metrics:
        value = weights[metric]

        if not isinstance(value, (int, float)):
            raise ConfigurationError(
                f"Weight for '{metric}' must be numeric."
            )

        if value < 0:
            raise ConfigurationError(
                f"Weight for '{metric}' cannot be negative."
            )

        numeric_weights.append(float(value))

    total_weight = sum(numeric_weights)

    if abs(total_weight - 1.0) > 1e-6:
        raise ConfigurationError(
            f"Metric weights must total 1.0, but total {total_weight:.4f}."
        )


def _validate_thresholds(thresholds: dict[str, Any]) -> None:
    """Validate all required threshold groups."""
    expected_groups = {
        "blur",
        "brightness",
        "glare",
        "roi",
        "ridge",
        "composite",
    }

    if not isinstance(thresholds, dict):
        raise ConfigurationError(
            "The thresholds section must be a mapping."
        )

    missing_groups = expected_groups.difference(thresholds.keys())

    if missing_groups:
        missing = ", ".join(sorted(missing_groups))
        raise ConfigurationError(
            f"Missing threshold groups: {missing}"
        )


def get_default_config() -> dict[str, Any]:
    """
    Return an independent copy of the default configuration.
    """
    return deepcopy(load_config())


if __name__ == "__main__":
    try:
        loaded_config = load_config()

        print(
            f"{loaded_config['project']['name']} "
            "configuration loaded successfully."
        )
        print(
            f"Version: {loaded_config['project']['version']}"
        )
        print(
            "Composite threshold:",
            loaded_config["thresholds"]["composite"]["minimum_score"],
        )

    except ConfigurationError as error:
        print(f"Configuration error: {error}")
        raise SystemExit(1)