"""
Automated tests for RidgeLens quality metrics.

Synthetic images provide deterministic test inputs without exposing private
fingerprint photographs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.quality_metrics import (
    GlareAnalysis,
    MetricResult,
    QualityMetricError,
    check_blur,
    check_brightness,
    check_glare,
    evaluate_initial_metrics,
    normalize_blur_score,
    normalize_brightness_score,
    normalize_glare_score,
)


def create_sharp_pattern(
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Create a high-frequency line pattern with strong edges."""
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
    """Create a strongly blurred synthetic pattern."""
    return cv2.GaussianBlur(
        create_sharp_pattern(),
        (51, 51),
        sigmaX=15,
        sigmaY=15,
    )


def create_glare_image(
    width: int = 100,
    height: int = 100,
    glare_fraction: float = 0.10,
    background_value: int = 120,
) -> np.ndarray:
    """Create an image with a known fraction of overexposed pixels."""
    total_pixels = width * height
    glare_pixels = int(round(total_pixels * glare_fraction))

    flattened = np.full(
        total_pixels,
        background_value,
        dtype=np.uint8,
    )

    flattened[:glare_pixels] = 255

    return flattened.reshape(height, width)


def test_blur_result_uses_standard_structure() -> None:
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
    midpoint_image = np.full(
        (128, 128),
        130,
        dtype=np.uint8,
    )
    boundary_image = np.full(
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
        boundary_image,
        minimum_value=50.0,
        maximum_value=210.0,
    )

    assert midpoint_result.normalized_score == pytest.approx(1.0)
    assert (
        midpoint_result.normalized_score
        > boundary_result.normalized_score
    )


def test_dark_image_returns_dark_guidance() -> None:
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


def test_glare_result_uses_detailed_structure() -> None:
    image = create_glare_image(
        glare_fraction=0.02,
    )

    analysis = check_glare(
        image,
        pixel_threshold=240,
        maximum_fraction=0.05,
    )

    assert isinstance(analysis, GlareAnalysis)
    assert isinstance(analysis.result, MetricResult)
    assert analysis.result.name == "glare"
    assert analysis.glare_mask.shape == image.shape
    assert analysis.glare_mask.dtype == np.uint8


def test_glare_fraction_is_calculated_correctly() -> None:
    image = create_glare_image(
        glare_fraction=0.10,
    )

    analysis = check_glare(
        image,
        pixel_threshold=240,
        maximum_fraction=0.05,
    )

    assert analysis.result.raw_value == pytest.approx(
        0.10,
        abs=0.001,
    )
    assert analysis.result.details is not None
    assert analysis.result.details["glare_percentage"] == pytest.approx(
        10.0,
        abs=0.01,
    )


def test_low_glare_image_passes() -> None:
    image = create_glare_image(
        glare_fraction=0.03,
    )

    result = check_glare(
        image,
        pixel_threshold=240,
        maximum_fraction=0.05,
    ).result

    assert result.passed is True
    assert result.normalized_score > 0.5


def test_excessive_glare_image_fails() -> None:
    image = create_glare_image(
        glare_fraction=0.10,
    )

    result = check_glare(
        image,
        pixel_threshold=240,
        maximum_fraction=0.05,
    ).result

    assert result.passed is False
    assert "glare" in result.message.lower()


def test_glare_boundary_is_inclusive() -> None:
    image = create_glare_image(
        glare_fraction=0.05,
    )

    result = check_glare(
        image,
        pixel_threshold=240,
        maximum_fraction=0.05,
    ).result

    assert result.passed is True


def test_glare_mask_contains_only_binary_values() -> None:
    image = create_glare_image(
        glare_fraction=0.08,
    )

    glare_mask = check_glare(image).glare_mask
    unique_values = set(np.unique(glare_mask).tolist())

    assert unique_values.issubset({0, 255})


def test_pixels_equal_to_threshold_are_not_glare() -> None:
    image = np.full(
        (100, 100),
        240,
        dtype=np.uint8,
    )

    result = check_glare(
        image,
        pixel_threshold=240,
        maximum_fraction=0.05,
    ).result

    assert result.raw_value == pytest.approx(0.0)
    assert result.passed is True


def test_pixels_above_threshold_are_glare() -> None:
    image = np.full(
        (100, 100),
        241,
        dtype=np.uint8,
    )

    result = check_glare(
        image,
        pixel_threshold=240,
        maximum_fraction=0.05,
    ).result

    assert result.raw_value == pytest.approx(1.0)
    assert result.passed is False


@pytest.mark.parametrize(
    "pixel_threshold",
    [-1, 256],
)
def test_invalid_glare_pixel_threshold_is_rejected(
    pixel_threshold: int,
) -> None:
    image = np.full(
        (100, 100),
        128,
        dtype=np.uint8,
    )

    with pytest.raises(
        QualityMetricError,
        match="between 0 and 255",
    ):
        check_glare(
            image,
            pixel_threshold=pixel_threshold,
        )


@pytest.mark.parametrize(
    "maximum_fraction",
    [0.0, -0.1, 1.1],
)
def test_invalid_maximum_glare_fraction_is_rejected(
    maximum_fraction: float,
) -> None:
    image = np.full(
        (100, 100),
        128,
        dtype=np.uint8,
    )

    with pytest.raises(
        QualityMetricError,
        match="greater than 0 and at most 1",
    ):
        check_glare(
            image,
            maximum_fraction=maximum_fraction,
        )


def test_empty_metric_input_is_rejected() -> None:
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
    assert normalize_blur_score(0.0, 10.0) == pytest.approx(0.0)
    assert normalize_blur_score(10.0, 10.0) == pytest.approx(0.5)
    assert normalize_blur_score(20.0, 10.0) == pytest.approx(1.0)
    assert normalize_blur_score(500.0, 10.0) == pytest.approx(1.0)


def test_normalize_brightness_score_is_clamped() -> None:
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


def test_normalize_glare_score_is_clamped() -> None:
    assert normalize_glare_score(
        0.0,
        0.05,
    ) == pytest.approx(1.0)

    assert normalize_glare_score(
        0.05,
        0.05,
    ) == pytest.approx(0.5)

    assert normalize_glare_score(
        0.10,
        0.05,
    ) == pytest.approx(0.0)

    assert normalize_glare_score(
        0.50,
        0.05,
    ) == pytest.approx(0.0)


def test_initial_metric_evaluation_returns_three_results() -> None:
    image = create_sharp_pattern()

    results = evaluate_initial_metrics(image)

    assert set(results.keys()) == {
        "blur",
        "brightness",
        "glare",
    }

    for result in results.values():
        assert isinstance(result, MetricResult)


def test_metric_result_can_be_serialized_to_dictionary() -> None:
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
    assert "details" in serialized


def test_glare_analysis_summary_excludes_raw_mask() -> None:
    image = create_glare_image(
        glare_fraction=0.10,
    )

    analysis = check_glare(image)
    summary = analysis.to_summary()

    assert "result" in summary
    assert "mask_shape" in summary
    assert "glare_pixels" in summary
    assert "glare_mask" not in summary