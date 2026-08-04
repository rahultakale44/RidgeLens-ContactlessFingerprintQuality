"""
Fingerprint image-quality metrics for RidgeLens.

Every metric returns a standard MetricResult containing:

- Raw measurement
- Normalized quality score
- PASS or FAIL decision
- Threshold information
- Processing time
- Corrective guidance

Implemented metrics:

1. Blur and sharpness assessment using Laplacian variance.
2. Brightness assessment using grayscale mean intensity.
3. Glare detection using the fraction of overexposed pixels.
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
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable metric representation."""
        return asdict(self)


@dataclass(frozen=True)
class GlareAnalysis:
    """Detailed glare output including the binary overexposure mask."""

    result: MetricResult
    glare_mask: np.ndarray

    def to_summary(self) -> dict[str, Any]:
        """Return glare information without the raw NumPy mask."""
        return {
            "result": self.result.to_dict(),
            "mask_shape": list(self.glare_mask.shape),
            "glare_pixels": int(np.count_nonzero(self.glare_mask)),
        }


def check_blur(
    image: np.ndarray,
    minimum_score: float | None = None,
    config: dict[str, Any] | None = None,
) -> MetricResult:
    """
    Measure image sharpness using variance of the Laplacian response.
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
        details={
            "method": "variance_of_laplacian",
        },
    )


def check_brightness(
    image: np.ndarray,
    minimum_value: float | None = None,
    maximum_value: float | None = None,
    config: dict[str, Any] | None = None,
) -> MetricResult:
    """
    Evaluate capture exposure using mean grayscale intensity.
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
        details={
            "too_dark": too_dark,
            "too_bright": too_bright,
            "method": "mean_grayscale_intensity",
        },
    )


def check_glare(
    image: np.ndarray,
    pixel_threshold: int | None = None,
    maximum_fraction: float | None = None,
    config: dict[str, Any] | None = None,
) -> GlareAnalysis:
    """
    Detect overexposed image regions that may hide fingerprint ridges.

    A pixel is considered glare affected when its grayscale value is greater
    than the configured pixel threshold.

    Glare fraction:

        overexposed pixels / total image pixels

    Args:
        image:
            Grayscale uint8 image or three-channel BGR image.
        pixel_threshold:
            Pixel intensity above which a pixel is marked as glare.
        maximum_fraction:
            Maximum allowed fraction of glare-affected pixels.
        config:
            Optional RidgeLens configuration dictionary.

    Returns:
        GlareAnalysis containing MetricResult and binary glare mask.

    Raises:
        QualityMetricError:
            If the image or threshold values are invalid.
    """
    start_time = perf_counter()
    active_config = config or get_default_config()

    grayscale = _ensure_grayscale(image)
    glare_config = active_config["thresholds"]["glare"]

    threshold = (
        int(glare_config["pixel_threshold"])
        if pixel_threshold is None
        else int(pixel_threshold)
    )
    allowed_fraction = (
        float(glare_config["maximum_fraction"])
        if maximum_fraction is None
        else float(maximum_fraction)
    )

    _validate_glare_settings(
        pixel_threshold=threshold,
        maximum_fraction=allowed_fraction,
    )

    glare_mask = np.where(
        grayscale > threshold,
        255,
        0,
    ).astype(np.uint8)

    glare_pixel_count = int(np.count_nonzero(glare_mask))
    total_pixel_count = int(grayscale.size)

    glare_fraction = (
        glare_pixel_count / total_pixel_count
        if total_pixel_count > 0
        else 0.0
    )

    normalized_score = normalize_glare_score(
        glare_fraction=glare_fraction,
        maximum_fraction=allowed_fraction,
    )

    passed = glare_fraction <= allowed_fraction

    if passed:
        message = (
            "No excessive glare detected. Highlight coverage is within "
            "the acceptable range."
        )
    else:
        message = (
            "Excessive glare detected. Change the finger or camera angle "
            "and avoid direct light reflections."
        )

    elapsed_ms = (perf_counter() - start_time) * 1000.0

    result = MetricResult(
        name="glare",
        raw_value=round(glare_fraction, 6),
        normalized_score=round(normalized_score, 4),
        passed=passed,
        threshold={
            "pixel_threshold": float(threshold),
            "maximum_fraction": allowed_fraction,
        },
        processing_time_ms=round(elapsed_ms, 4),
        message=message,
        details={
            "glare_fraction": round(glare_fraction, 6),
            "glare_percentage": round(glare_fraction * 100.0, 4),
            "glare_pixel_count": glare_pixel_count,
            "total_pixel_count": total_pixel_count,
            "method": "overexposed_pixel_fraction",
        },
    )

    return GlareAnalysis(
        result=result,
        glare_mask=glare_mask,
    )


def normalize_blur_score(
    blur_score: float,
    minimum_score: float,
) -> float:
    """
    Normalize Laplacian variance into a score between 0.0 and 1.0.
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
    Convert brightness into a balanced 0.0–1.0 quality score.
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


def normalize_glare_score(
    glare_fraction: float,
    maximum_fraction: float,
) -> float:
    """
    Convert glare fraction into a 0.0–1.0 quality score.

    No glare receives 1.0.

    The maximum accepted glare fraction maps to 0.5.

    Twice the maximum fraction maps to 0.0.
    """
    glare_fraction = float(glare_fraction)
    maximum_fraction = float(maximum_fraction)

    if not 0 <= glare_fraction <= 1:
        raise QualityMetricError(
            "Glare fraction must be between 0.0 and 1.0."
        )

    if not 0 < maximum_fraction <= 1:
        raise QualityMetricError(
            "Maximum glare fraction must be greater than 0 and at most 1."
        )

    normalized = 1.0 - (
        glare_fraction / (maximum_fraction * 2.0)
    )

    return _clamp(normalized)


def evaluate_initial_metrics(
    image: np.ndarray,
    config: dict[str, Any] | None = None,
) -> dict[str, MetricResult]:
    """
    Run all currently implemented RidgeLens quality metrics.
    """
    active_config = config or get_default_config()

    glare_analysis = check_glare(
        image=image,
        config=active_config,
    )

    return {
        "blur": check_blur(
            image=image,
            config=active_config,
        ),
        "brightness": check_brightness(
            image=image,
            config=active_config,
        ),
        "glare": glare_analysis.result,
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


def _validate_glare_settings(
    pixel_threshold: int,
    maximum_fraction: float,
) -> None:
    """Validate glare-detection parameters."""
    if not 0 <= pixel_threshold <= 255:
        raise QualityMetricError(
            "Glare pixel threshold must be between 0 and 255."
        )

    if not 0 < maximum_fraction <= 1:
        raise QualityMetricError(
            "Maximum glare fraction must be greater than 0 and at most 1."
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

    cv2.circle(
        demonstration_image,
        (200, 60),
        18,
        255,
        thickness=-1,
    )

    results = evaluate_initial_metrics(demonstration_image)
    glare_output = check_glare(demonstration_image)

    print("RidgeLens Phase 4 metric demonstration")

    for metric_name, result in results.items():
        print(
            f"{metric_name}: "
            f"value={result.raw_value}, "
            f"score={result.normalized_score}, "
            f"passed={result.passed}, "
            f"time={result.processing_time_ms} ms"
        )

    print(
        "Glare mask pixels:",
        int(np.count_nonzero(glare_output.glare_mask)),
    )