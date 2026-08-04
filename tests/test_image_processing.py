"""
Tests for RidgeLens image loading and preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.image_processing import (
    ImageProcessingError,
    apply_clahe_enhancement,
    convert_to_grayscale,
    load_image_from_bytes,
    load_image_from_path,
    resize_preserving_aspect_ratio,
)


def create_encoded_test_image(
    width: int = 640,
    height: int = 480,
    brightness: int = 120,
) -> bytes:
    """Create a valid synthetic JPEG image for isolated tests."""
    image = np.full(
        (height, width, 3),
        brightness,
        dtype=np.uint8,
    )

    cv2.line(
        image,
        (50, 50),
        (width - 50, height - 50),
        (30, 30, 30),
        thickness=8,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
    )

    assert success is True

    return encoded.tobytes()


def test_load_image_from_bytes_returns_preprocessed_outputs() -> None:
    """Valid image bytes should produce all required representations."""
    image_bytes = create_encoded_test_image()

    result = load_image_from_bytes(
        image_bytes,
        source_name="sample.jpg",
    )

    assert result.original_bgr.shape == (
        480,
        640,
        3,
    )

    assert result.resized_bgr.shape == (
        480,
        640,
        3,
    )

    assert result.grayscale.shape == (
        480,
        640,
    )

    assert result.enhanced_grayscale.shape == (
        480,
        640,
    )

    assert result.metadata.source_name == "sample.jpg"
    assert result.metadata.was_resized is False
    assert result.metadata.file_size_bytes == len(image_bytes)


def test_large_image_is_resized_without_distortion() -> None:
    """Large images should be reduced while preserving aspect ratio."""
    image_bytes = create_encoded_test_image(
        width=2400,
        height=1200,
    )

    result = load_image_from_bytes(
        image_bytes,
        source_name="large.jpg",
    )

    processed_height, processed_width = (
        result.resized_bgr.shape[:2]
    )

    assert processed_width == 640
    assert processed_height == 320

    assert (
        processed_width / processed_height
        == pytest.approx(2.0)
    )

    assert result.metadata.was_resized is True

    assert result.metadata.original_width == 2400
    assert result.metadata.original_height == 1200
    assert result.metadata.processed_width == 640
    assert result.metadata.processed_height == 320


def test_resize_does_not_upscale_small_image() -> None:
    """Images below configured limits should not be enlarged."""
    image = np.zeros(
        (200, 300, 3),
        dtype=np.uint8,
    )

    resized, was_resized = resize_preserving_aspect_ratio(
        image,
        maximum_width=640,
        maximum_height=640,
    )

    assert resized.shape == image.shape
    assert was_resized is False
    assert resized is not image


def test_resize_wide_image_preserves_aspect_ratio() -> None:
    """A wide image should fit within limits without distortion."""
    image = np.zeros(
        (600, 1800, 3),
        dtype=np.uint8,
    )

    resized, was_resized = resize_preserving_aspect_ratio(
        image,
        maximum_width=640,
        maximum_height=640,
    )

    resized_height, resized_width = resized.shape[:2]

    assert resized_width == 640
    assert resized_height == 213
    assert was_resized is True

    assert (
        resized_width / resized_height
        == pytest.approx(
            1800 / 600,
            rel=0.01,
        )
    )


def test_resize_tall_image_preserves_aspect_ratio() -> None:
    """A tall image should fit within limits without distortion."""
    image = np.zeros(
        (1800, 600, 3),
        dtype=np.uint8,
    )

    resized, was_resized = resize_preserving_aspect_ratio(
        image,
        maximum_width=640,
        maximum_height=640,
    )

    resized_height, resized_width = resized.shape[:2]

    assert resized_height == 640
    assert resized_width == 213
    assert was_resized is True

    assert (
        resized_height / resized_width
        == pytest.approx(
            1800 / 600,
            rel=0.01,
        )
    )


def test_convert_to_grayscale_returns_single_channel() -> None:
    """BGR conversion should produce an 8-bit grayscale array."""
    image = np.full(
        (100, 100, 3),
        128,
        dtype=np.uint8,
    )

    grayscale = convert_to_grayscale(
        image
    )

    assert grayscale.shape == (
        100,
        100,
    )

    assert grayscale.dtype == np.uint8


def test_clahe_returns_same_dimensions() -> None:
    """CLAHE must preserve grayscale dimensions and data type."""
    grayscale = np.tile(
        np.arange(
            256,
            dtype=np.uint8,
        ),
        (128, 1),
    )

    enhanced = apply_clahe_enhancement(
        grayscale,
        clip_limit=2.0,
        grid_size=8,
    )

    assert enhanced.shape == grayscale.shape
    assert enhanced.dtype == np.uint8


def test_empty_bytes_raise_clear_error() -> None:
    """An empty upload should be rejected."""
    with pytest.raises(
        ImageProcessingError,
        match="empty",
    ):
        load_image_from_bytes(
            b"",
            source_name="empty.jpg",
        )


def test_corrupted_bytes_raise_clear_error() -> None:
    """Random bytes should not be accepted as an image."""
    with pytest.raises(
        ImageProcessingError,
        match="Unable to decode",
    ):
        load_image_from_bytes(
            b"this-is-not-an-image",
            source_name="corrupted.jpg",
        )


def test_unsupported_extension_is_rejected(
    tmp_path: Path,
) -> None:
    """Files outside the configured image formats must be rejected."""
    unsupported_path = (
        tmp_path
        / "fingerprint.gif"
    )

    unsupported_path.write_bytes(
        create_encoded_test_image()
    )

    with pytest.raises(
        ImageProcessingError,
        match="Unsupported image extension",
    ):
        load_image_from_path(
            unsupported_path
        )


def test_missing_image_path_raises_error(
    tmp_path: Path,
) -> None:
    """A missing path should produce a clear message."""
    missing_path = (
        tmp_path
        / "missing.jpg"
    )

    with pytest.raises(
        ImageProcessingError,
        match="not found",
    ):
        load_image_from_path(
            missing_path
        )