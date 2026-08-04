"""
Gabor-based fingerprint ridge-clarity assessment for RidgeLens.

Fingerprint ridges contain repeated directional intensity transitions.
Orientation-sensitive Gabor filters respond strongly when their orientation
matches the local ridge direction.

This module:

1. Validates grayscale or BGR input.
2. Builds a zero-mean multi-orientation Gabor filter bank.
3. Applies every filter to the image.
4. Aggregates the strongest response at each pixel.
5. Measures ridge response inside the detected finger ROI.
6. Produces a normalized ridge-quality score.
7. Returns a diagnostic ridge-response image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi
from time import perf_counter
from typing import Any, Sequence

import cv2
import numpy as np

from app.config import get_default_config


class RidgeAnalysisError(RuntimeError):
    """Raised when ridge analysis receives invalid input or parameters."""


@dataclass(frozen=True)
class RidgeResult:
    """Serializable result of fingerprint ridge-clarity assessment."""

    name: str
    ridge_score: float
    normalized_score: float
    passed: bool
    threshold: dict[str, float]
    processing_time_ms: float
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RidgeAnalysis:
    """Complete ridge output including diagnostic response images."""

    result: RidgeResult
    combined_response: np.ndarray
    response_visualization: np.ndarray
    analysis_mask: np.ndarray
    orientation_responses: tuple[np.ndarray, ...]

    def to_summary(self) -> dict[str, Any]:
        """
        Return a serializable summary without raw NumPy arrays.
        """
        return {
            "result": self.result.to_dict(),
            "response_shape": list(self.combined_response.shape),
            "mask_shape": list(self.analysis_mask.shape),
            "orientation_count": len(self.orientation_responses),
            "analysed_pixels": int(
                np.count_nonzero(self.analysis_mask)
            ),
        }


def check_ridge_clarity(
    image: np.ndarray,
    roi_mask: np.ndarray | None = None,
    minimum_score: float | None = None,
    orientations: int = 8,
    kernel_size: int = 21,
    sigma: float = 4.0,
    wavelength: float = 8.0,
    gamma: float = 0.5,
    config: dict[str, Any] | None = None,
) -> RidgeAnalysis:
    """
    Evaluate fingerprint ridge clarity with a Gabor filter bank.

    Args:
        image:
            Grayscale uint8 image or three-channel BGR image. The
            CLAHE-enhanced grayscale representation is recommended.
        roi_mask:
            Optional binary finger mask. When supplied, ridge statistics are
            calculated only inside the detected finger region.
        minimum_score:
            Optional acceptance threshold. The configured ridge threshold is
            used when omitted.
        orientations:
            Number of equally spaced Gabor orientations over 0 to pi.
        kernel_size:
            Odd Gabor-kernel width and height.
        sigma:
            Gaussian envelope standard deviation.
        wavelength:
            Expected ridge wavelength in pixels.
        gamma:
            Spatial aspect ratio of the Gabor kernel.
        config:
            Optional RidgeLens configuration dictionary.

    Returns:
        RidgeAnalysis containing the structured metric result, combined
        response, display visualization, mask, and per-orientation responses.

    Raises:
        RidgeAnalysisError:
            If image, ROI mask, threshold, or filter parameters are invalid.
    """
    start_time = perf_counter()
    active_config = config or get_default_config()

    grayscale = _prepare_grayscale(image)

    configured_threshold = float(
        active_config["thresholds"]["ridge"]["minimum_score"]
    )

    threshold = (
        configured_threshold
        if minimum_score is None
        else float(minimum_score)
    )

    _validate_ridge_threshold(threshold)
    _validate_gabor_parameters(
        orientations=orientations,
        kernel_size=kernel_size,
        sigma=sigma,
        wavelength=wavelength,
        gamma=gamma,
    )

    analysis_mask = prepare_analysis_mask(
        roi_mask=roi_mask,
        image_shape=grayscale.shape,
    )

    kernels = build_gabor_filter_bank(
        orientations=orientations,
        kernel_size=kernel_size,
        sigma=sigma,
        wavelength=wavelength,
        gamma=gamma,
    )

    orientation_responses: list[np.ndarray] = []

    for kernel in kernels:
        response = cv2.filter2D(
            grayscale.astype(np.float32),
            ddepth=cv2.CV_32F,
            kernel=kernel,
            borderType=cv2.BORDER_REFLECT,
        )

        absolute_response = np.abs(response)
        orientation_responses.append(absolute_response)

    stacked_responses = np.stack(
        orientation_responses,
        axis=0,
    )

    combined_response = np.max(
        stacked_responses,
        axis=0,
    )

    ridge_score, statistics = calculate_ridge_score(
        combined_response=combined_response,
        analysis_mask=analysis_mask,
    )

    normalized_score = normalize_ridge_score(
        ridge_score=ridge_score,
        minimum_score=threshold,
    )

    passed = ridge_score >= threshold

    if passed:
        message = (
            "Fingerprint ridge structure is sufficiently clear for "
            "biometric processing."
        )
    else:
        message = (
            "Fingerprint ridges are unclear. Improve focus, move the finger "
            "closer, and use soft side lighting before capturing again."
        )

    response_visualization = normalize_response_for_display(
        combined_response=combined_response,
        analysis_mask=analysis_mask,
    )

    elapsed_ms = (perf_counter() - start_time) * 1000.0

    analysed_pixel_count = int(
        np.count_nonzero(analysis_mask)
    )
    total_pixel_count = int(analysis_mask.size)

    result = RidgeResult(
        name="ridge",
        ridge_score=round(ridge_score, 4),
        normalized_score=round(normalized_score, 4),
        passed=passed,
        threshold={
            "minimum_score": threshold,
        },
        processing_time_ms=round(elapsed_ms, 4),
        message=message,
        details={
            "orientation_count": orientations,
            "kernel_size": kernel_size,
            "sigma": float(sigma),
            "wavelength": float(wavelength),
            "gamma": float(gamma),
            "analysed_pixel_count": analysed_pixel_count,
            "total_pixel_count": total_pixel_count,
            "used_roi_mask": roi_mask is not None,
            "response_mean": round(
                statistics["mean"],
                4,
            ),
            "response_std": round(
                statistics["std"],
                4,
            ),
            "response_p10": round(
                statistics["p10"],
                4,
            ),
            "response_p90": round(
                statistics["p90"],
                4,
            ),
            "response_dynamic_range": round(
                statistics["dynamic_range"],
                4,
            ),
            "method": (
                "multi_orientation_zero_mean_gabor_response"
            ),
        },
    )

    return RidgeAnalysis(
        result=result,
        combined_response=combined_response,
        response_visualization=response_visualization,
        analysis_mask=analysis_mask,
        orientation_responses=tuple(
            orientation_responses
        ),
    )


def build_gabor_filter_bank(
    orientations: int = 8,
    kernel_size: int = 21,
    sigma: float = 4.0,
    wavelength: float = 8.0,
    gamma: float = 0.5,
) -> tuple[np.ndarray, ...]:
    """
    Build equally spaced, zero-mean Gabor kernels.

    A zero-mean kernel reduces response to uniform brightness and makes the
    filter more sensitive to ridge-like intensity variation.
    """
    _validate_gabor_parameters(
        orientations=orientations,
        kernel_size=kernel_size,
        sigma=sigma,
        wavelength=wavelength,
        gamma=gamma,
    )

    kernels: list[np.ndarray] = []

    for orientation_index in range(orientations):
        theta = (
            orientation_index
            * pi
            / orientations
        )

        kernel = cv2.getGaborKernel(
            ksize=(kernel_size, kernel_size),
            sigma=float(sigma),
            theta=float(theta),
            lambd=float(wavelength),
            gamma=float(gamma),
            psi=0.0,
            ktype=cv2.CV_32F,
        )

        kernel = kernel.astype(np.float32)
        kernel -= float(np.mean(kernel))

        absolute_sum = float(
            np.sum(np.abs(kernel))
        )

        if absolute_sum > 0:
            kernel /= absolute_sum

        kernels.append(kernel)

    return tuple(kernels)


def calculate_ridge_score(
    combined_response: np.ndarray,
    analysis_mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """
    Calculate ridge clarity from Gabor response statistics.

    The score uses the response's 10th-to-90th percentile dynamic range inside
    the ROI. Clear ridge patterns create strong and varying directional
    responses, while flat or heavily blurred regions produce weaker ranges.
    """
    _validate_response_image(combined_response)
    _validate_binary_mask(analysis_mask)

    if combined_response.shape != analysis_mask.shape:
        raise RidgeAnalysisError(
            "Combined response and analysis mask must have matching dimensions."
        )

    selected_values = combined_response[
        analysis_mask > 0
    ]

    if selected_values.size == 0:
        raise RidgeAnalysisError(
            "Analysis mask contains no foreground pixels."
        )

    mean_value = float(
        np.mean(selected_values)
    )
    std_value = float(
        np.std(selected_values)
    )
    percentile_10 = float(
        np.percentile(selected_values, 10)
    )
    percentile_90 = float(
        np.percentile(selected_values, 90)
    )

    dynamic_range = max(
        0.0,
        percentile_90 - percentile_10,
    )

    ridge_score = dynamic_range * 255.0

    statistics = {
        "mean": mean_value,
        "std": std_value,
        "p10": percentile_10,
        "p90": percentile_90,
        "dynamic_range": dynamic_range,
    }

    return float(ridge_score), statistics


def normalize_ridge_score(
    ridge_score: float,
    minimum_score: float,
) -> float:
    """
    Normalize ridge clarity into the range 0.0–1.0.

    The minimum accepted threshold maps to 0.5.

    Twice the minimum threshold maps to 1.0.
    """
    ridge_score = float(ridge_score)
    minimum_score = float(minimum_score)

    if ridge_score < 0:
        raise RidgeAnalysisError(
            "Ridge score cannot be negative."
        )

    _validate_ridge_threshold(minimum_score)

    normalized = (
        ridge_score
        / (minimum_score * 2.0)
    )

    return _clamp(normalized)


def prepare_analysis_mask(
    roi_mask: np.ndarray | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """
    Validate an ROI mask or create a full-image analysis mask.
    """
    height, width = image_shape

    if height < 2 or width < 2:
        raise RidgeAnalysisError(
            "Image dimensions are too small for ridge analysis."
        )

    if roi_mask is None:
        return np.full(
            image_shape,
            255,
            dtype=np.uint8,
        )

    if not isinstance(roi_mask, np.ndarray):
        raise RidgeAnalysisError(
            "ROI mask must be a NumPy array."
        )

    if roi_mask.shape != image_shape:
        raise RidgeAnalysisError(
            "ROI mask dimensions must match the input image."
        )

    if roi_mask.ndim != 2:
        raise RidgeAnalysisError(
            "ROI mask must be single-channel."
        )

    if roi_mask.dtype != np.uint8:
        mask = cv2.convertScaleAbs(
            roi_mask
        )
    else:
        mask = roi_mask.copy()

    mask = np.where(
        mask > 0,
        255,
        0,
    ).astype(np.uint8)

    if np.count_nonzero(mask) == 0:
        raise RidgeAnalysisError(
            "ROI mask contains no foreground pixels."
        )

    return mask


def normalize_response_for_display(
    combined_response: np.ndarray,
    analysis_mask: np.ndarray,
) -> np.ndarray:
    """
    Convert floating-point Gabor response into an 8-bit diagnostic image.
    """
    _validate_response_image(combined_response)
    _validate_binary_mask(analysis_mask)

    if combined_response.shape != analysis_mask.shape:
        raise RidgeAnalysisError(
            "Response and mask dimensions must match."
        )

    selected_values = combined_response[
        analysis_mask > 0
    ]

    if selected_values.size == 0:
        raise RidgeAnalysisError(
            "Cannot visualize an empty analysis region."
        )

    lower = float(
        np.percentile(selected_values, 2)
    )
    upper = float(
        np.percentile(selected_values, 98)
    )

    if upper <= lower:
        visualization = np.zeros_like(
            combined_response,
            dtype=np.uint8,
        )
    else:
        scaled = (
            (combined_response - lower)
            / (upper - lower)
        )

        scaled = np.clip(
            scaled,
            0.0,
            1.0,
        )

        visualization = (
            scaled * 255.0
        ).astype(np.uint8)

    visualization[
        analysis_mask == 0
    ] = 0

    return visualization


def create_ridge_overlay(
    image_bgr: np.ndarray,
    response_visualization: np.ndarray,
    analysis_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Create a coloured ridge-response overlay for the Streamlit interface.
    """
    _validate_bgr_image(image_bgr)
    _validate_display_image(
        response_visualization
    )

    if image_bgr.shape[:2] != response_visualization.shape:
        raise RidgeAnalysisError(
            "Image and ridge-response dimensions must match."
        )

    heatmap = cv2.applyColorMap(
        response_visualization,
        cv2.COLORMAP_TURBO,
    )

    if analysis_mask is not None:
        mask = prepare_analysis_mask(
            roi_mask=analysis_mask,
            image_shape=response_visualization.shape,
        )

        heatmap[
            mask == 0
        ] = 0

    overlay = cv2.addWeighted(
        image_bgr,
        0.62,
        heatmap,
        0.38,
        0,
    )

    return overlay


def _prepare_grayscale(
    image: np.ndarray,
) -> np.ndarray:
    """Validate image input and produce uint8 grayscale."""
    if not isinstance(image, np.ndarray):
        raise RidgeAnalysisError(
            "Ridge-analysis input must be a NumPy array."
        )

    if image.size == 0:
        raise RidgeAnalysisError(
            "Ridge-analysis input image is empty."
        )

    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] == 3:
        grayscale = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        raise RidgeAnalysisError(
            "Expected a grayscale image or a three-channel BGR image."
        )

    if (
        grayscale.shape[0] < 2
        or grayscale.shape[1] < 2
    ):
        raise RidgeAnalysisError(
            "Image dimensions are too small for ridge analysis."
        )

    if grayscale.dtype != np.uint8:
        grayscale = cv2.convertScaleAbs(
            grayscale
        )

    return grayscale


def _validate_gabor_parameters(
    orientations: int,
    kernel_size: int,
    sigma: float,
    wavelength: float,
    gamma: float,
) -> None:
    """Validate Gabor-filter-bank parameters."""
    if not isinstance(orientations, int):
        raise RidgeAnalysisError(
            "Orientation count must be an integer."
        )

    if orientations < 2:
        raise RidgeAnalysisError(
            "At least two Gabor orientations are required."
        )

    if not isinstance(kernel_size, int):
        raise RidgeAnalysisError(
            "Gabor kernel size must be an integer."
        )

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise RidgeAnalysisError(
            "Gabor kernel size must be an odd integer of at least 3."
        )

    if sigma <= 0:
        raise RidgeAnalysisError(
            "Gabor sigma must be greater than zero."
        )

    if wavelength <= 0:
        raise RidgeAnalysisError(
            "Gabor wavelength must be greater than zero."
        )

    if gamma <= 0:
        raise RidgeAnalysisError(
            "Gabor gamma must be greater than zero."
        )


def _validate_ridge_threshold(
    minimum_score: float,
) -> None:
    """Validate ridge acceptance threshold."""
    if minimum_score <= 0:
        raise RidgeAnalysisError(
            "Minimum ridge score must be greater than zero."
        )


def _validate_response_image(
    response: np.ndarray,
) -> None:
    """Validate a floating-point single-channel response image."""
    if not isinstance(response, np.ndarray):
        raise RidgeAnalysisError(
            "Gabor response must be a NumPy array."
        )

    if response.size == 0:
        raise RidgeAnalysisError(
            "Gabor response is empty."
        )

    if response.ndim != 2:
        raise RidgeAnalysisError(
            "Gabor response must be single-channel."
        )


def _validate_binary_mask(
    mask: np.ndarray,
) -> None:
    """Validate an 8-bit binary mask."""
    if not isinstance(mask, np.ndarray):
        raise RidgeAnalysisError(
            "Analysis mask must be a NumPy array."
        )

    if mask.size == 0:
        raise RidgeAnalysisError(
            "Analysis mask is empty."
        )

    if mask.ndim != 2:
        raise RidgeAnalysisError(
            "Analysis mask must be single-channel."
        )

    unique_values = set(
        np.unique(mask).tolist()
    )

    if not unique_values.issubset(
        {0, 255}
    ):
        raise RidgeAnalysisError(
            "Analysis mask must contain only 0 and 255."
        )


def _validate_bgr_image(
    image_bgr: np.ndarray,
) -> None:
    """Validate a three-channel BGR image."""
    if not isinstance(image_bgr, np.ndarray):
        raise RidgeAnalysisError(
            "Expected BGR image as a NumPy array."
        )

    if image_bgr.size == 0:
        raise RidgeAnalysisError(
            "BGR image is empty."
        )

    if (
        image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
    ):
        raise RidgeAnalysisError(
            "Expected a three-channel BGR image."
        )


def _validate_display_image(
    image: np.ndarray,
) -> None:
    """Validate an 8-bit single-channel visualization."""
    if not isinstance(image, np.ndarray):
        raise RidgeAnalysisError(
            "Response visualization must be a NumPy array."
        )

    if image.size == 0:
        raise RidgeAnalysisError(
            "Response visualization is empty."
        )

    if image.ndim != 2:
        raise RidgeAnalysisError(
            "Response visualization must be single-channel."
        )

    if image.dtype != np.uint8:
        raise RidgeAnalysisError(
            "Response visualization must use uint8 values."
        )


def _clamp(value: float) -> float:
    """Restrict a score to the inclusive range 0.0–1.0."""
    return max(
        0.0,
        min(1.0, float(value)),
    )


if __name__ == "__main__":
    demo_image = np.full(
        (400, 400),
        128,
        dtype=np.uint8,
    )

    for x_position in range(40, 360, 8):
        cv2.line(
            demo_image,
            (x_position, 40),
            (x_position, 360),
            45,
            thickness=3,
        )

    demo_mask = np.zeros_like(
        demo_image
    )

    cv2.ellipse(
        demo_mask,
        center=(200, 200),
        axes=(140, 175),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    analysis = check_ridge_clarity(
        image=demo_image,
        roi_mask=demo_mask,
    )

    print("RidgeLens Phase 6 ridge demonstration")
    print(
        "Ridge score:",
        analysis.result.ridge_score,
    )
    print(
        "Normalized score:",
        analysis.result.normalized_score,
    )
    print(
        "Passed:",
        analysis.result.passed,
    )
    print(
        "Orientations:",
        analysis.result.details[
            "orientation_count"
        ],
    )
    print(
        "Processing time:",
        analysis.result.processing_time_ms,
        "ms",
    )