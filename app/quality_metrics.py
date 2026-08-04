"""
Fingerprint image-quality metrics for RidgeLens.

This module implements independent and explainable quality checks. Each metric
returns its raw measurement, normalized score, decision, threshold information,
and processing time.

Phase 3 currently includes:

1. Blur and sharpness assessment using Laplacian variance.
2. Brightness assessment using grayscale mean intensity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.config import get_default_config


class QualityMetricError(RuntimeError):
    """Raised when a quality metric receives invalid input or settings."""


@dataclass(frozen=True)
class MetricResult:
    """Standard result returned by every RidgeLens quality metric."""

    name: str
    raw_value: float
    normalized_score: float
    passed: bool
    threshold: dict[str, float]
    processing_time_ms: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the metric result."""
        return asdict(self)


def check_blur(
    image: np.ndarray,
    minimum_score: float | None = None,
    config: dict[str, Any] | None = None,
) -> MetricResult:
    """
    Measure image sharpness using variance of the Laplacian response.

    The Laplacian operator highlights rapid intensity changes such as edges and
    fingerprint ridge boundaries. A sharp image produces stronger variation in
    the Laplacian response, while a blurred image produces weaker variation.

    Args:
        image:
            A grayscale uint8 image or a three-channel BGR image.
        minimum_score:
            Optional blur acceptance threshold. When omitted, the configured
            threshold is used.
        config:
            Optional RidgeLens configuration dictionary.

    Returns:
        MetricResult containing Laplacian variance, normalized score, pass/fail
        decision, processing time, and explanation.

    Raises:
        QualityMetricError:
            If the image or threshold is invalid.
    """
    start_time = perf_counter()
    active_config = config or get_default_config()

    grayscale = _ensure_grayscale(image)

    configured_threshold = float(
        active_config["thresholds"]["blur"]["minimum_score"]
    )
    threshold = (
        configured_threshold
        if minimum_score is None
        else float(minimum_score)
    )

    if threshold < 0:
        raise QualityMetricError(
            "Blur threshold cannot be negative."
        )

    laplacian_response = cv2.Laplacian(
        grayscale,
        cv2.CV_64F,
    )
    blur_score = float(laplacian_response.var())

    normalized_score = normalize_blur_score(
        blur_score=blur_score,
        minimum_score=threshold,
    )

    passed = blur_score >= threshold

    if passed:
        message = (
            "Image sharpness is acceptable for biometric processing."
        )
    else:
        message = (
            "Image appears blurry. Hold the phone and finger steady, "
            "allow the camera to focus, and capture again."
        )

    elapsed_ms = (perf_counter() - start_time) * 1000.0

    return MetricResult(
        name="blur",
        raw_value=round(blur_score, 4),
        normalized_score=round(normalized_score, 4),
        passed=passed,
        threshold={
            "minimum_score": threshold,
        },
        processing_time_ms=round(elapsed_ms, 4),
        message=message,
    )


def check_brightness(
    image: np.ndarray,
    minimum_value: float | None = None,
    maximum_value: float | None = None,
    config: dict[str, Any] | None = None,
) -> MetricResult:
    """
    Evaluate capture exposure using mean grayscale intensity.

    Pixel values range from 0 to 255:

    - Lower values represent darker pixels.
    - Higher values represent brighter pixels.

    Args:
        image:
            A grayscale uint8 image or a three-channel BGR image.
        minimum_value:
            Optional lower acceptable brightness limit.
        maximum_value:
            Optional upper acceptable brightness limit.
        config:
            Optional RidgeLens configuration dictionary.

    Returns:
        MetricResult containing mean intensity, normalized score, decision,
        processing time, and user-facing explanation.

    Raises:
        QualityMetricError:
            If the image or configured brightness range is invalid.
    """
    start_time = perf_counter()
    active_config = config or get_default_config()

    grayscale = _ensure_grayscale(image)

    brightness_config = active_config["thresholds"]["brightness"]

    minimum = (
        float(brightness_config["minimum_value"])
        if minimum_value is None
        else float(minimum_value)
    )
    maximum = (
        float(brightness_config["maximum_value"])
        if maximum_value is None
        else float(maximum_value)
    )

    _validate_brightness_range(
        minimum_value=minimum,
        maximum_value=maximum,
    )

    brightness = float(np.mean(grayscale))

    normalized_score = normalize_brightness_score(
        brightness=brightness,
        minimum_value=minimum,
        maximum_value=maximum,
    )

    too_dark = brightness < minimum
    too_bright = brightness > maximum
    passed = not too_dark and not too_bright

    if too_dark:
        message = (
            "Image is too dark. Move to a brighter location or increase "
            "soft, indirect lighting before capturing again."
        )
    elif too_bright:
        message = (
            "Image is too bright or overexposed. Reduce direct lighting "
            "and avoid pointing a torch directly at the finger."
        )
    else:
        message = (
            "Image brightness is within the acceptable range."
        )

    elapsed_ms = (perf_counter() - start_time) * 1000.0

    return MetricResult(
        name="brightness",
        raw_value=round(brightness, 4),
        normalized_score=round(normalized_score, 4),
        passed=passed,
        threshold={
            "minimum_value": minimum,
            "maximum_value": maximum,
        },
        processing_time_ms=round(elapsed_ms, 4),
        message=message,
    )


def normalize_blur_score(
    blur_score: float,
    minimum_score: float,
) -> float:
    """
    Normalize Laplacian variance into a score between 0.0 and 1.0.

    The configured minimum threshold maps to 0.5. A score twice the minimum
    threshold maps to 1.0. This provides a gradual confidence scale instead of
    returning only a binary decision.
    """
    blur_score = float(blur_score)
    minimum_score = float(minimum_score)

    if blur_score < 0:
        raise QualityMetricError(
            "Blur score cannot be negative."
        )

    if minimum_score < 0:
        raise QualityMetricError(
            "Blur threshold cannot be negative."
        )

    if minimum_score == 0:
        return 1.0 if blur_score > 0 else 0.0

    normalized = blur_score / (minimum_score * 2.0)

    return _clamp(normalized)


def normalize_brightness_score(
    brightness: float,
    minimum_value: float,
    maximum_value: float,
) -> float:
    """
    Convert brightness into a 0.0–1.0 quality score.

    The centre of the acceptable brightness interval receives the maximum
    score. The score gradually decreases as brightness approaches either
    boundary and continues decreasing outside the acceptable range.

    This triangular scoring function rewards balanced exposure rather than
    treating every accepted intensity as equally ideal.
    """
    brightness = float(brightness)
    minimum_value = float(minimum_value)
    maximum_value = float(maximum_value)

    _validate_brightness_range(
        minimum_value=minimum_value,
        maximum_value=maximum_value,
    )

    if not 0 <= brightness <= 255:
        raise QualityMetricError(
            "Brightness must be between 0 and 255."
        )

    midpoint = (minimum_value + maximum_value) / 2.0
    half_range = (maximum_value - minimum_value) / 2.0

    if half_range == 0:
        return 1.0 if brightness == midpoint else 0.0

    distance_from_midpoint = abs(brightness - midpoint)
    normalized = 1.0 - (distance_from_midpoint / half_range)

    return _clamp(normalized)


def evaluate_initial_metrics(
    image: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, MetricResult]:
    """
    Run all metrics currently implemented in Phase 3.

    This convenience function will later be expanded and integrated into the
    complete RidgeLens quality pipeline.
    """
    active_config = config or get_default_config()

    return {
        "blur": check_blur(
            image=image,
            config=active_config,
        ),
        "brightness": check_brightness(
            image=image,
            config=active_config,
        ),
    }


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Validate an image and return an 8-bit grayscale representation.
    """
    if not isinstance(image, np.ndarray):
        raise QualityMetricError(
            "Quality metric input must be a NumPy array."
        )

    if image.size == 0:
        raise QualityMetricError(
            "Quality metric input image is empty."
        )

    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] == 3:
        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        raise QualityMetricError(
            "Expected a grayscale image or a three-channel BGR image."
        )

    if grayscale.shape[0] < 2 or grayscale.shape[1] < 2:
        raise QualityMetricError(
            "Image dimensions are too small for quality assessment."
        )

    if grayscale.dtype != np.uint8:
        grayscale = cv2.convertScaleAbs(grayscale)

    return grayscale


def _validate_brightness_range(
    minimum_value: float,
    maximum_value: float,
) -> None:
    """Validate configured brightness boundaries."""
    if not 0 <= minimum_value <= 255:
        raise QualityMetricError(
            "Minimum brightness must be between 0 and 255."
        )

    if not 0 <= maximum_value <= 255:
        raise QualityMetricError(
            "Maximum brightness must be between 0 and 255."
        )

    if minimum_value >= maximum_value:
        raise QualityMetricError(
            "Minimum brightness must be lower than maximum brightness."
        )


def _clamp(value: float) -> float:
    """Restrict a numeric score to the inclusive range 0.0–1.0."""
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    demonstration_image = np.full(
        (256, 256),
        128,
        dtype=np.uint8,
    )

    cv2.line(
        demonstration_image,
        (20, 20),
        (235, 235),
        20,
        thickness=4,
    )

    results = evaluate_initial_metrics(demonstration_image)

    print("RidgeLens Phase 3 metric demonstration")

    for metric_name, result in results.items():
        print(
            f"{metric_name}: "
            f"value={result.raw_value}, "
            f"score={result.normalized_score}, "
            f"passed={result.passed}, "
            f"time={result.processing_time_ms} ms"
        )