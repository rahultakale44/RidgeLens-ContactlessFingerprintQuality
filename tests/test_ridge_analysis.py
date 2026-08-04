"""
Automated tests for RidgeLens Gabor-based ridge analysis.

Synthetic periodic patterns simulate repeated ridge transitions without
publishing real biometric data.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.ridge_analysis import (
    RidgeAnalysis,
    RidgeAnalysisError,
    RidgeResult,
    build_gabor_filter_bank,
    calculate_ridge_score,
    check_ridge_clarity,
    create_ridge_overlay,
    normalize_response_for_display,
    normalize_ridge_score,
    prepare_analysis_mask,
)


def create_ridge_pattern(
    width: int = 384,
    height: int = 384,
    period: int = 8,
) -> np.ndarray:
    """Create a synthetic vertical ridge-and-valley pattern."""
    x_coordinates = np.arange(
        width,
        dtype=np.float32,
    )

    sinusoid = (
        127.5
        + 95.0
        * np.sin(
            2.0
            * np.pi
            * x_coordinates
            / period
        )
    )

    image = np.tile(
        sinusoid,
        (height, 1),
    )

    return np.clip(
        image,
        0,
        255,
    ).astype(np.uint8)


def create_blurred_ridge_pattern() -> np.ndarray:
    """Create a strongly blurred version of the ridge pattern."""
    return cv2.GaussianBlur(
        create_ridge_pattern(),
        (31, 31),
        sigmaX=10,
        sigmaY=10,
    )


def create_ellipse_mask(
    width: int = 384,
    height: int = 384,
) -> np.ndarray:
    """Create a central binary finger-like analysis mask."""
    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.ellipse(
        mask,
        center=(width // 2, height // 2),
        axes=(135, 170),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    return mask


def test_ridge_analysis_uses_standard_structure() -> None:
    """Ridge assessment should return structured outputs."""
    analysis = check_ridge_clarity(
        image=create_ridge_pattern(),
        roi_mask=create_ellipse_mask(),
        minimum_score=15.0,
    )

    assert isinstance(
        analysis,
        RidgeAnalysis,
    )
    assert isinstance(
        analysis.result,
        RidgeResult,
    )
    assert analysis.result.name == "ridge"
    assert analysis.combined_response.shape == (
        384,
        384,
    )
    assert analysis.response_visualization.shape == (
        384,
        384,
    )
    assert analysis.analysis_mask.shape == (
        384,
        384,
    )
    assert len(
        analysis.orientation_responses
    ) == 8


def test_clear_ridges_score_higher_than_blurred_ridges() -> None:
    """A clear periodic pattern should outperform its blurred version."""
    mask = create_ellipse_mask()

    clear_result = check_ridge_clarity(
        create_ridge_pattern(),
        roi_mask=mask,
        minimum_score=15.0,
    ).result

    blurred_result = check_ridge_clarity(
        create_blurred_ridge_pattern(),
        roi_mask=mask,
        minimum_score=15.0,
    ).result

    assert (
        clear_result.ridge_score
        > blurred_result.ridge_score
    )


def test_uniform_image_has_low_ridge_score() -> None:
    """A flat image should contain no meaningful ridge structure."""
    uniform_image = np.full(
        (300, 300),
        128,
        dtype=np.uint8,
    )

    result = check_ridge_clarity(
        uniform_image,
        minimum_score=15.0,
    ).result

    assert result.ridge_score < 1.0
    assert result.passed is False
    assert result.normalized_score < 0.1


def test_clear_ridge_pattern_passes_low_threshold() -> None:
    """Clear synthetic ridges should pass a realistic low threshold."""
    result = check_ridge_clarity(
        create_ridge_pattern(),
        roi_mask=create_ellipse_mask(),
        minimum_score=5.0,
    ).result

    assert result.passed is True
    assert result.normalized_score > 0.5


def test_bgr_input_is_supported() -> None:
    """Ridge analysis should accept the project's BGR representation."""
    grayscale = create_ridge_pattern()
    bgr_image = cv2.cvtColor(
        grayscale,
        cv2.COLOR_GRAY2BGR,
    )

    analysis = check_ridge_clarity(
        image=bgr_image,
        minimum_score=5.0,
    )

    assert analysis.result.ridge_score > 0


def test_full_image_mask_is_created_when_roi_is_missing() -> None:
    """The entire image should be analysed when no ROI mask is supplied."""
    image = create_ridge_pattern()

    analysis = check_ridge_clarity(
        image=image,
        roi_mask=None,
        minimum_score=5.0,
    )

    assert np.all(
        analysis.analysis_mask == 255
    )
    assert (
        np.count_nonzero(
            analysis.analysis_mask
        )
        == image.size
    )


def test_roi_mask_restricts_analysis_region() -> None:
    """The returned mask should match the supplied ROI."""
    image = create_ridge_pattern()
    mask = create_ellipse_mask()

    analysis = check_ridge_clarity(
        image=image,
        roi_mask=mask,
        minimum_score=5.0,
    )

    assert np.array_equal(
        analysis.analysis_mask,
        mask,
    )
    assert analysis.result.details[
        "used_roi_mask"
    ] is True


def test_filter_bank_contains_requested_orientations() -> None:
    """Filter bank size should match orientation count."""
    kernels = build_gabor_filter_bank(
        orientations=12,
        kernel_size=21,
        sigma=4.0,
        wavelength=8.0,
        gamma=0.5,
    )

    assert len(kernels) == 12

    for kernel in kernels:
        assert kernel.shape == (21, 21)
        assert kernel.dtype == np.float32


def test_gabor_kernels_are_approximately_zero_mean() -> None:
    """Uniform intensity should not produce a large DC response."""
    kernels = build_gabor_filter_bank()

    for kernel in kernels:
        assert float(
            abs(np.mean(kernel))
        ) < 1e-6


def test_response_visualization_is_uint8() -> None:
    """Diagnostic response must be suitable for display."""
    analysis = check_ridge_clarity(
        create_ridge_pattern(),
        minimum_score=5.0,
    )

    visualization = (
        analysis.response_visualization
    )

    assert visualization.dtype == np.uint8
    assert visualization.ndim == 2
    assert visualization.min() >= 0
    assert visualization.max() <= 255


def test_visualization_is_zero_outside_roi() -> None:
    """Pixels outside the finger mask should not be visualized."""
    image = create_ridge_pattern()
    mask = create_ellipse_mask()

    analysis = check_ridge_clarity(
        image=image,
        roi_mask=mask,
        minimum_score=5.0,
    )

    assert np.all(
        analysis.response_visualization[
            mask == 0
        ]
        == 0
    )


def test_ridge_overlay_preserves_dimensions() -> None:
    """Diagnostic heatmap overlay must match the BGR image."""
    grayscale = create_ridge_pattern()
    image_bgr = cv2.cvtColor(
        grayscale,
        cv2.COLOR_GRAY2BGR,
    )
    mask = create_ellipse_mask()

    analysis = check_ridge_clarity(
        grayscale,
        roi_mask=mask,
        minimum_score=5.0,
    )

    overlay = create_ridge_overlay(
        image_bgr=image_bgr,
        response_visualization=(
            analysis.response_visualization
        ),
        analysis_mask=mask,
    )

    assert overlay.shape == image_bgr.shape
    assert overlay.dtype == np.uint8


def test_ridge_summary_excludes_raw_arrays() -> None:
    """Summary should be serializable without large image arrays."""
    analysis = check_ridge_clarity(
        create_ridge_pattern(),
        roi_mask=create_ellipse_mask(),
        minimum_score=5.0,
    )

    summary = analysis.to_summary()

    assert "result" in summary
    assert "response_shape" in summary
    assert "mask_shape" in summary
    assert "orientation_count" in summary
    assert "analysed_pixels" in summary
    assert "combined_response" not in summary
    assert "orientation_responses" not in summary


@pytest.mark.parametrize(
    ("ridge_score", "expected"),
    [
        (0.0, 0.0),
        (7.5, 0.25),
        (15.0, 0.5),
        (30.0, 1.0),
        (100.0, 1.0),
    ],
)
def test_ridge_normalization(
    ridge_score: float,
    expected: float,
) -> None:
    """Ridge normalization should follow threshold reference points."""
    result = normalize_ridge_score(
        ridge_score=ridge_score,
        minimum_score=15.0,
    )

    assert result == pytest.approx(
        expected,
        abs=0.001,
    )


@pytest.mark.parametrize(
    "minimum_score",
    [0.0, -1.0],
)
def test_invalid_ridge_threshold_is_rejected(
    minimum_score: float,
) -> None:
    """Ridge threshold must be positive."""
    with pytest.raises(
        RidgeAnalysisError,
        match="greater than zero",
    ):
        check_ridge_clarity(
            image=create_ridge_pattern(),
            minimum_score=minimum_score,
        )


@pytest.mark.parametrize(
    "orientations",
    [0, 1],
)
def test_invalid_orientation_count_is_rejected(
    orientations: int,
) -> None:
    """At least two Gabor orientations are required."""
    with pytest.raises(
        RidgeAnalysisError,
        match="At least two",
    ):
        build_gabor_filter_bank(
            orientations=orientations,
        )


@pytest.mark.parametrize(
    "kernel_size",
    [2, 10, 20],
)
def test_invalid_kernel_size_is_rejected(
    kernel_size: int,
) -> None:
    """Gabor kernel size must be odd and at least three."""
    with pytest.raises(
        RidgeAnalysisError,
        match="odd integer",
    ):
        build_gabor_filter_bank(
            kernel_size=kernel_size,
        )


def test_empty_image_is_rejected() -> None:
    """Empty image arrays should raise controlled errors."""
    with pytest.raises(
        RidgeAnalysisError,
        match="empty",
    ):
        check_ridge_clarity(
            np.array([], dtype=np.uint8)
        )


def test_invalid_channel_count_is_rejected() -> None:
    """Four-channel image input is outside the module contract."""
    invalid_image = np.zeros(
        (100, 100, 4),
        dtype=np.uint8,
    )

    with pytest.raises(
        RidgeAnalysisError,
        match="grayscale image or a three-channel BGR image",
    ):
        check_ridge_clarity(
            invalid_image
        )


def test_mismatched_roi_dimensions_are_rejected() -> None:
    """ROI and input image dimensions must match."""
    image = create_ridge_pattern()
    wrong_mask = np.full(
        (100, 100),
        255,
        dtype=np.uint8,
    )

    with pytest.raises(
        RidgeAnalysisError,
        match="dimensions must match",
    ):
        check_ridge_clarity(
            image=image,
            roi_mask=wrong_mask,
        )


def test_empty_roi_mask_is_rejected() -> None:
    """A mask without foreground pixels cannot define analysis area."""
    image = create_ridge_pattern()
    empty_mask = np.zeros_like(
        image
    )

    with pytest.raises(
        RidgeAnalysisError,
        match="no foreground pixels",
    ):
        check_ridge_clarity(
            image=image,
            roi_mask=empty_mask,
        )


def test_calculate_ridge_score_returns_statistics() -> None:
    """The scoring helper should provide explainable statistics."""
    response = np.tile(
        np.linspace(
            0.0,
            1.0,
            200,
            dtype=np.float32,
        ),
        (200, 1),
    )
    mask = np.full(
        (200, 200),
        255,
        dtype=np.uint8,
    )

    score, statistics = calculate_ridge_score(
        combined_response=response,
        analysis_mask=mask,
    )

    assert score > 0
    assert set(
        statistics.keys()
    ) == {
        "mean",
        "std",
        "p10",
        "p90",
        "dynamic_range",
    }


def test_constant_response_visualization_is_safe() -> None:
    """Constant responses should produce a valid black diagnostic image."""
    response = np.full(
        (100, 100),
        0.5,
        dtype=np.float32,
    )
    mask = np.full(
        (100, 100),
        255,
        dtype=np.uint8,
    )

    visualization = normalize_response_for_display(
        response,
        mask,
    )

    assert visualization.dtype == np.uint8
    assert np.count_nonzero(
        visualization
    ) == 0