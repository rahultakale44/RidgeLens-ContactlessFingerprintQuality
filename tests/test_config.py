"""
Tests for RidgeLens configuration loading and validation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.config import ConfigurationError, load_config


def test_default_configuration_loads_successfully() -> None:
    """Default configuration should load correctly."""
    config = load_config()

    assert config["project"]["name"] == "RidgeLens"
    assert config["thresholds"]["composite"]["minimum_score"] == 60.0
    assert sum(config["weights"].values()) == pytest.approx(1.0)


def test_missing_configuration_file_raises_error(
    tmp_path: Path,
) -> None:
    """Missing configuration must raise a clear error."""
    missing_path = tmp_path / "missing-config.yaml"

    with pytest.raises(ConfigurationError, match="not found"):
        load_config(missing_path)


def test_invalid_weight_total_raises_error(
    tmp_path: Path,
) -> None:
    """Metric weights must total exactly 1.0."""
    config = load_config()
    config["weights"]["blur"] = 0.80

    invalid_path = tmp_path / "invalid-weights.yaml"

    with invalid_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file)

    with pytest.raises(
        ConfigurationError,
        match="must total 1.0",
    ):
        load_config(invalid_path)


def test_missing_threshold_group_raises_error(
    tmp_path: Path,
) -> None:
    """Every quality metric needs a threshold group."""
    config = load_config()
    del config["thresholds"]["ridge"]

    invalid_path = tmp_path / "missing-threshold.yaml"

    with invalid_path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file)

    with pytest.raises(
        ConfigurationError,
        match="Missing threshold groups",
    ):
        load_config(invalid_path)