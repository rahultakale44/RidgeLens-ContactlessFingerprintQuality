"""
Automated tests for RidgeLens quality metrics.

The tests use deterministic synthetic images so that metric behaviour can be
verified without publishing personal fingerprint photographs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.quality_metrics import (
    MetricResult,
    QualityMetricError,
    check_blur,
    check_brightness,
    evaluate_initial_metrics,
    normalize_blur_score,
    normalize_brightness_score,
)


def create_sharp_pattern(
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """
    Create a high-frequency pattern with strong edges.

    Alternating dark and bright lines simulate repeated intensity transitions
    similar to ridge-like image structure.
    """
    image = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for x_position in range(0, width, 8):
        colour = 230 if (x_position // 8) % 2 == 0 else 25

        cv2.line(
            image,
            (x_position, 0),
            (x_position, height - 1),
            colour,
            thickness=4,
        )

    return image


def create_blurred_pattern() -> np.ndarray:
    """Create a strongly blurred version of the synthetic sharp pattern."""
    sharp_image = create_sharp_pattern()

    return cv2.GaussianBlur(
        sharp_image,
        (51, 51),
        sigmaX=15,
        sigmaY=15,
    )


def test_blur_result_uses_standard_structure() -> None:
    """Blur assessment should return a reusable MetricResult object."""
    image = create_sharp_pattern()

    result = check_blur(
        image,
        minimum_score=10.0,
    )

    assert isinstance(result, MetricResult)
    assert result.name == "blur"
    assert result.raw_value >= 0
    assert 0.0 <= result.normalized_score <= 1.0
    assert isinstance(result.passed, bool)
    assert result.processing_time_ms >= 0
    assert "minimum_score" in result.threshold


def test_sharp_pattern_scores_higher_than_blurred_pattern() -> None:
    """Strong edges should produce higher Laplacian variance than blur."""
    sharp_result = check_blur(
        create_sharp_pattern(),
        minimum_score=10.0,
    )
    blurred_result = check_blur(
        create_blurred_pattern(),
        minimum_score=10.0,
    )

    assert sharp_result.raw_value > blurred_result.raw_value
    assert (
        sharp_result.normalized_score
        >= blurred_result.normalized_score
    )


def test_uniform_image_is_classified_as_blurry() -> None:
    """A completely uniform image contains no useful edge information."""
    uniform_image = np.full(
        (256, 256),
        128,
        dtype=np.uint8,
    )

    result = check_blur(
        uniform_image,
        minimum_score=10.0,
    )

    assert result.raw_value == pytest.approx(0.0)
    assert result.normalized_score == pytest.approx(0.0)
    assert result.passed is False


def test_blur_accepts_bgr_input() -> None:
    """Blur detection should support the project's BGR representation."""
    grayscale = create_sharp_pattern()
    bgr_image = cv2.cvtColor(
        grayscale,
        cv2.COLOR_GRAY2BGR,
    )

    result = check_blur(
        bgr_image,
        minimum_score=10.0,
    )

    assert result.raw_value > 0


def test_negative_blur_threshold_is_rejected() -> None:
    """Negative Laplacian thresholds are invalid."""
    with pytest.raises(
        QualityMetricError,
        match="cannot be negative",
    ):
        check_blur(
            create_sharp_pattern(),
            minimum_score=-1.0,
        )


@pytest.mark.parametrize(
    ("pixel_value", "expected_pass"),
    [
        (20, False),
        (49, False),
        (50, True),
        (128, True),
        (210, True),
        (211, False),
        (245, False),
    ],
)
def test_brightness_decisions_follow_thresholds(
    pixel_value: int,
    expected_pass: bool,
) -> None:
    """Brightness boundaries should follow the configured inclusive range."""
    image = np.full(
        (128, 128),
        pixel_value,
        dtype=np.uint8,
    )

    result = check_brightness(
        image,
        minimum_value=50.0,
        maximum_value=210.0,
    )

    assert result.raw_value == pytest.approx(float(pixel_value))
    assert result.passed is expected_pass


def test_balanced_brightness_receives_highest_score() -> None:
    """The centre of the acceptable range should be considered ideal."""
    midpoint_image = np.full(
        (128, 128),
        130,
        dtype=np.uint8,
    )
    dark_boundary_image = np.full(
        (128, 128),
        50,
        dtype=np.uint8,
    )

    midpoint_result = check_brightness(
        midpoint_image,
        minimum_value=50.0,
        maximum_value=210.0,
    )
    boundary_result = check_brightness(
        dark_boundary_image,
        minimum_value=50.0,
        maximum_value=210.0,
    )

    assert midpoint_result.normalized_score == pytest.approx(1.0)
    assert (
        midpoint_result.normalized_score
        > boundary_result.normalized_score
    )


def test_dark_image_returns_dark_guidance() -> None:
    """Dark images should provide a specific corrective message."""
    image = np.full(
        (100, 100),
        20,
        dtype=np.uint8,
    )

    result = check_brightness(
        image,
        minimum_value=50.0,
        maximum_value=210.0,
    )

    assert result.passed is False
    assert "too dark" in result.message.lower()


def test_bright_image_returns_overexposure_guidance() -> None:
    """Overexposed images should provide lighting guidance."""
    image = np.full(
        (100, 100),
        240,
        dtype=np.uint8,
    )

    result = check_brightness(
        image,
        minimum_value=50.0,
        maximum_value=210.0,
    )

    assert result.passed is False
    assert (
        "too bright" in result.message.lower()
        or "overexposed" in result.message.lower()
    )


def test_invalid_brightness_range_is_rejected() -> None:
    """Minimum brightness must remain below maximum brightness."""
    image = np.full(
        (100, 100),
        128,
        dtype=np.uint8,
    )

    with pytest.raises(
        QualityMetricError,
        match="lower than maximum",
    ):
        check_brightness(
            image,
            minimum_value=210.0,
            maximum_value=50.0,
        )


def test_empty_metric_input_is_rejected() -> None:
    """Empty arrays must produce a controlled metric error."""
    empty_image = np.array(
        [],
        dtype=np.uint8,
    )

    with pytest.raises(
        QualityMetricError,
        match="empty",
    ):
        check_blur(empty_image)


def test_normalize_blur_score_is_clamped() -> None:
    """Blur normalization must always remain between zero and one."""
    assert normalize_blur_score(0.0, 10.0) == pytest.approx(0.0)
    assert normalize_blur_score(10.0, 10.0) == pytest.approx(0.5)
    assert normalize_blur_score(20.0, 10.0) == pytest.approx(1.0)
    assert normalize_blur_score(500.0, 10.0) == pytest.approx(1.0)


def test_normalize_brightness_score_is_clamped() -> None:
    """Exposure normalization must remain between zero and one."""
    assert normalize_brightness_score(
        130.0,
        50.0,
        210.0,
    ) == pytest.approx(1.0)

    assert normalize_brightness_score(
        0.0,
        50.0,
        210.0,
    ) == pytest.approx(0.0)

    assert normalize_brightness_score(
        255.0,
        50.0,
        210.0,
    ) == pytest.approx(0.0)


def test_initial_metric_evaluation_returns_both_results() -> None:
    """Phase 3 convenience evaluation should run both metrics."""
    image = create_sharp_pattern()

    results = evaluate_initial_metrics(image)

    assert set(results.keys()) == {
        "blur",
        "brightness",
    }
    assert isinstance(results["blur"], MetricResult)
    assert isinstance(results["brightness"], MetricResult)


def test_metric_result_can_be_serialized_to_dictionary() -> None:
    """Metric results should be usable by JSON, CSV, and Streamlit layers."""
    image = np.full(
        (128, 128),
        128,
        dtype=np.uint8,
    )

    result = check_brightness(image)
    serialized = result.to_dict()

    assert serialized["name"] == "brightness"
    assert serialized["raw_value"] == pytest.approx(128.0)
    assert "processing_time_ms" in serialized
    assert "message" in serialized