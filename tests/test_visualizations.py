"""
Automated tests for RidgeLens visualization utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from app.visualizations import (
    VisualizationError,
    build_metric_rows,
    convert_bgr_to_rgb,
    convert_grayscale_to_rgb,
    create_glare_overlay,
    create_mask_preview,
    create_metric_card_html,
    create_score_ring_svg,
)


@dataclass(frozen=True)
class SyntheticMetric:
    """Minimal visualization-compatible metric result."""

    raw_value: float = 10.0
    normalized_score: float = 0.8
    passed: bool = True
    processing_time_ms: float = 1.0
    message: str = "Synthetic metric result."
    roi_fraction: float = 0.20
    ridge_score: float = 25.0


def create_bgr_image() -> np.ndarray:
    """Create a small deterministic BGR test image."""
    image = np.zeros(
        (120, 160, 3),
        dtype=np.uint8,
    )

    image[:, :, 0] = 20
    image[:, :, 1] = 80
    image[:, :, 2] = 180

    return image


def create_binary_mask() -> np.ndarray:
    """Create a binary test mask."""
    mask = np.zeros(
        (120, 160),
        dtype=np.uint8,
    )

    mask[
        20:80,
        40:120,
    ] = 255

    return mask


def test_bgr_to_rgb_swaps_channels() -> None:
    """BGR conversion should reverse channel order."""
    image = create_bgr_image()

    converted = convert_bgr_to_rgb(
        image
    )

    assert converted.shape == image.shape
    assert converted[0, 0].tolist() == [
        180,
        80,
        20,
    ]


def test_grayscale_to_rgb_creates_three_channels() -> None:
    """Grayscale preview should contain three equal channels."""
    grayscale = np.full(
        (100, 100),
        120,
        dtype=np.uint8,
    )

    converted = convert_grayscale_to_rgb(
        grayscale
    )

    assert converted.shape == (
        100,
        100,
        3,
    )

    assert np.all(
        converted[:, :, 0]
        == converted[:, :, 1]
    )


def test_glare_overlay_preserves_dimensions() -> None:
    """Glare overlay should match the input image."""
    image = create_bgr_image()
    mask = create_binary_mask()

    overlay = create_glare_overlay(
        image,
        mask,
    )

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_mask_preview_creates_rgb_image() -> None:
    """Binary mask preview should be display ready."""
    mask = create_binary_mask()

    preview = create_mask_preview(
        mask
    )

    assert preview.shape == (
        120,
        160,
        3,
    )


def test_score_ring_contains_score_and_status() -> None:
    """Score-ring HTML should contain score and decision."""
    html = create_score_ring_svg(
        score=82.0,
        passed=True,
    )

    assert "82" in html
    assert "READY" in html
    assert "score-pass" in html


def test_failed_score_ring_contains_retake() -> None:
    """Failed score HTML should use fail styling."""
    html = create_score_ring_svg(
        score=40.0,
        passed=False,
    )

    assert "RETAKE" in html
    assert "score-fail" in html


def test_metric_card_contains_metric_data() -> None:
    """Metric card HTML should contain label, status, and message."""
    metric = SyntheticMetric()

    html = create_metric_card_html(
        "blur",
        metric,
    )

    assert "Sharpness" in html
    assert "PASS" in html
    assert "Synthetic metric result" in html


def test_build_metric_rows_returns_five_rows() -> None:
    """Detailed table should include all required metrics."""
    metric_results = {
        "blur": SyntheticMetric(),
        "brightness": SyntheticMetric(),
        "glare": SyntheticMetric(),
        "roi": SyntheticMetric(),
        "ridge": SyntheticMetric(),
    }

    weights = {
        "blur": 0.25,
        "brightness": 0.15,
        "glare": 0.15,
        "roi": 0.20,
        "ridge": 0.25,
    }

    contributions = {
        "blur": 20.0,
        "brightness": 12.0,
        "glare": 12.0,
        "roi": 16.0,
        "ridge": 20.0,
    }

    rows = build_metric_rows(
        metric_results=metric_results,
        weights=weights,
        contributions=contributions,
    )

    assert len(rows) == 5

    assert {
        row["Metric"]
        for row in rows
    } == {
        "Sharpness",
        "Brightness",
        "Glare",
        "Finger Coverage",
        "Ridge Clarity",
    }


def test_missing_metric_row_is_rejected() -> None:
    """Table generation requires all configured metrics."""
    with pytest.raises(
        VisualizationError,
        match="Missing metric result",
    ):
        build_metric_rows(
            metric_results={},
            weights={},
            contributions={},
        )


def test_invalid_glare_mask_is_rejected() -> None:
    """Non-binary masks must not be accepted."""
    image = create_bgr_image()
    invalid_mask = np.full(
        (120, 160),
        100,
        dtype=np.uint8,
    )

    with pytest.raises(
        VisualizationError,
        match="binary mask",
    ):
        create_glare_overlay(
            image,
            invalid_mask,
        )


def test_mismatched_glare_dimensions_are_rejected() -> None:
    """Image and mask dimensions must match."""
    image = create_bgr_image()
    wrong_mask = np.zeros(
        (50, 50),
        dtype=np.uint8,
    )

    with pytest.raises(
        VisualizationError,
        match="dimensions must match",
    ):
        create_glare_overlay(
            image,
            wrong_mask,
        )