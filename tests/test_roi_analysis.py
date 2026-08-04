"""
Automated tests for RidgeLens finger ROI analysis.

Synthetic images provide deterministic masks and finger-like regions without
publishing personal biometric photographs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.roi_analysis import (
    ROIAnalysis,
    ROIAnalysisError,
    ROIResult,
    check_roi_completeness,
    clean_candidate_mask,
    create_grayscale_candidate_mask,
    create_roi_overlay,
    create_skin_candidate_mask,
    normalize_roi_score,
    retain_largest_contour,
)


def create_finger_like_bgr_image(
    width: int = 640,
    height: int = 480,
    axes: tuple[int, int] = (90, 170),
    background: tuple[int, int, int] = (30, 30, 30),
    finger_colour: tuple[int, int, int] = (95, 145, 205),
) -> np.ndarray:
    """Create a synthetic finger-shaped ellipse on a dark background."""
    image = np.full(
        (height, width, 3),
        background,
        dtype=np.uint8,
    )

    cv2.ellipse(
        image,
        center=(width // 2, height // 2),
        axes=axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=finger_colour,
        thickness=-1,
    )

    return image


def create_binary_ellipse_mask(
    width: int = 400,
    height: int = 300,
    axes: tuple[int, int] = (70, 120),
) -> np.ndarray:
    """Create a binary mask containing one large ellipse."""
    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.ellipse(
        mask,
        center=(width // 2, height // 2),
        axes=axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    return mask


def test_roi_analysis_uses_standard_structure() -> None:
    """ROI assessment should return structured result and masks."""
    image = create_finger_like_bgr_image()

    analysis = check_roi_completeness(
        image,
        minimum_fraction=0.15,
    )

    assert isinstance(analysis, ROIAnalysis)
    assert isinstance(analysis.result, ROIResult)
    assert analysis.result.name == "roi"
    assert analysis.finger_mask.shape == image.shape[:2]
    assert analysis.candidate_mask.shape == image.shape[:2]
    assert analysis.result.processing_time_ms >= 0


def test_large_finger_region_passes() -> None:
    """A sufficiently large synthetic finger should pass coverage."""
    image = create_finger_like_bgr_image(
        axes=(115, 190),
    )

    result = check_roi_completeness(
        image,
        minimum_fraction=0.15,
    ).result

    assert result.roi_fraction >= 0.15
    assert result.passed is True
    assert result.normalized_score >= 0.5


def test_small_finger_region_fails() -> None:
    """A small finger region should fail minimum coverage."""
    image = create_finger_like_bgr_image(
        axes=(35, 60),
    )

    result = check_roi_completeness(
        image,
        minimum_fraction=0.15,
    ).result

    assert result.roi_fraction < 0.15
    assert result.passed is False
    assert (
        "too small" in result.message.lower()
        or "closer" in result.message.lower()
    )


def test_blank_image_does_not_produce_valid_roi() -> None:
    """A uniform frame should not be accepted as a finger region."""
    image = np.full(
        (400, 400, 3),
        30,
        dtype=np.uint8,
    )

    analysis = check_roi_completeness(
        image,
        minimum_fraction=0.15,
    )

    assert analysis.result.passed is False
    assert analysis.result.roi_fraction < 0.15


def test_skin_candidate_mask_is_binary() -> None:
    """Skin-colour detection must return a binary mask."""
    image = create_finger_like_bgr_image()

    mask = create_skin_candidate_mask(image)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()).issubset(
        {0, 255}
    )


def test_skin_candidate_detects_synthetic_finger() -> None:
    """The synthetic skin-colour ellipse should create foreground pixels."""
    image = create_finger_like_bgr_image()

    mask = create_skin_candidate_mask(image)

    assert np.count_nonzero(mask) > 0


def test_grayscale_candidate_mask_is_binary() -> None:
    """Grayscale fallback must generate a binary foreground mask."""
    image = create_finger_like_bgr_image()
    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    mask = create_grayscale_candidate_mask(
        grayscale
    )

    assert mask.shape == grayscale.shape
    assert set(np.unique(mask).tolist()).issubset(
        {0, 255}
    )


def test_morphological_cleanup_removes_isolated_noise() -> None:
    """Small isolated foreground noise should be removed."""
    mask = create_binary_ellipse_mask()

    mask[5, 5] = 255
    mask[10, 385] = 255
    mask[290, 10] = 255

    cleaned = clean_candidate_mask(mask)

    assert cleaned[5, 5] == 0
    assert cleaned[10, 385] == 0
    assert cleaned[290, 10] == 0
    assert np.count_nonzero(cleaned) > 0


def test_largest_contour_is_retained() -> None:
    """Only the largest plausible connected component should remain."""
    mask = create_binary_ellipse_mask()

    cv2.circle(
        mask,
        (25, 25),
        10,
        255,
        thickness=-1,
    )

    retained, contour, bounding_box = retain_largest_contour(
        mask
    )

    assert contour is not None
    assert bounding_box is not None
    assert retained[25, 25] == 0
    assert np.count_nonzero(retained) > 0


def test_no_contours_returns_empty_result() -> None:
    """An empty candidate mask should return no contour."""
    mask = np.zeros(
        (200, 200),
        dtype=np.uint8,
    )

    retained, contour, bounding_box = retain_largest_contour(
        mask
    )

    assert np.count_nonzero(retained) == 0
    assert contour is None
    assert bounding_box is None


def test_roi_overlay_preserves_image_dimensions() -> None:
    """Diagnostic overlay must match original image dimensions."""
    image = create_finger_like_bgr_image()
    analysis = check_roi_completeness(image)

    overlay = create_roi_overlay(
        image_bgr=image,
        finger_mask=analysis.finger_mask,
        bounding_box=analysis.bounding_box,
    )

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8


def test_roi_summary_excludes_raw_arrays() -> None:
    """Serializable summary must not expose NumPy masks or contours."""
    analysis = check_roi_completeness(
        create_finger_like_bgr_image()
    )

    summary = analysis.to_summary()

    assert "result" in summary
    assert "mask_shape" in summary
    assert "foreground_pixels" in summary
    assert "bounding_box" in summary
    assert "contour_found" in summary
    assert "finger_mask" not in summary
    assert "candidate_mask" not in summary
    assert "contour" not in summary


@pytest.mark.parametrize(
    ("roi_fraction", "expected"),
    [
        (0.0, 0.0),
        (0.075, 0.0),
        (0.15, 0.5),
        (0.30, 1.0),
        (0.80, 1.0),
    ],
)
def test_roi_normalization(
    roi_fraction: float,
    expected: float,
) -> None:
    """ROI normalization should follow configured reference points."""
    score = normalize_roi_score(
        roi_fraction=roi_fraction,
        minimum_fraction=0.15,
    )

    assert score == pytest.approx(
        expected,
        abs=0.001,
    )


@pytest.mark.parametrize(
    "minimum_fraction",
    [0.0, -0.1, 1.1],
)
def test_invalid_minimum_roi_fraction_is_rejected(
    minimum_fraction: float,
) -> None:
    """ROI threshold must remain inside the valid fractional range."""
    image = create_finger_like_bgr_image()

    with pytest.raises(
        ROIAnalysisError,
        match="greater than 0 and at most 1",
    ):
        check_roi_completeness(
            image,
            minimum_fraction=minimum_fraction,
        )


def test_empty_roi_input_is_rejected() -> None:
    """Empty image input should produce a controlled error."""
    with pytest.raises(
        ROIAnalysisError,
        match="empty",
    ):
        check_roi_completeness(
            np.array([], dtype=np.uint8)
        )


def test_invalid_channel_count_is_rejected() -> None:
    """Four-channel input is outside the module contract."""
    image = np.zeros(
        (100, 100, 4),
        dtype=np.uint8,
    )

    with pytest.raises(
        ROIAnalysisError,
        match="grayscale image or a three-channel BGR image",
    ):
        check_roi_completeness(image)


def test_grayscale_input_is_supported() -> None:
    """ROI analysis should also support grayscale captures."""
    bgr_image = create_finger_like_bgr_image()
    grayscale = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2GRAY,
    )

    analysis = check_roi_completeness(
        grayscale,
        minimum_fraction=0.15,
    )

    assert analysis.finger_mask.shape == grayscale.shape
    assert 0.0 <= analysis.result.roi_fraction <= 1.0