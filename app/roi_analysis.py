"""
Finger region detection and ROI-completeness assessment for RidgeLens.

The module detects the most likely finger region in a mobile-camera image,
generates a cleaned binary mask, measures its coverage, and determines whether
the finger occupies enough of the frame for downstream biometric processing.

The implementation combines:

1. Colour-based skin-region detection when a BGR image is available.
2. Grayscale foreground estimation as a fallback.
3. Morphological cleanup.
4. Largest-contour filtering.
5. Geometric plausibility checks.
6. ROI coverage scoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from app.config import get_default_config


class ROIAnalysisError(RuntimeError):
    """Raised when ROI detection receives invalid input or configuration."""


@dataclass(frozen=True)
class ROIResult:
    """Standard result of finger-region completeness assessment."""

    name: str
    roi_fraction: float
    normalized_score: float
    passed: bool
    threshold: dict[str, float]
    processing_time_ms: float
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable ROI result."""
        return asdict(self)


@dataclass(frozen=True)
class ROIAnalysis:
    """Full ROI analysis including diagnostic masks and bounding box."""

    result: ROIResult
    finger_mask: np.ndarray
    candidate_mask: np.ndarray
    bounding_box: tuple[int, int, int, int] | None
    contour: np.ndarray | None

    def to_summary(self) -> dict[str, Any]:
        """
        Return a serializable summary without raw NumPy arrays.
        """
        return {
            "result": self.result.to_dict(),
            "mask_shape": list(self.finger_mask.shape),
            "foreground_pixels": int(
                np.count_nonzero(self.finger_mask)
            ),
            "bounding_box": (
                list(self.bounding_box)
                if self.bounding_box is not None
                else None
            ),
            "contour_found": self.contour is not None,
        }


def check_roi_completeness(
    image: np.ndarray,
    minimum_fraction: float | None = None,
    config: dict[str, Any] | None = None,
) -> ROIAnalysis:
    """
    Detect the finger region and evaluate how much of the frame it occupies.

    Args:
        image:
            Grayscale uint8 image or three-channel BGR image.
        minimum_fraction:
            Optional minimum acceptable finger-area fraction. When omitted,
            the value from config.yaml is used.
        config:
            Optional RidgeLens configuration dictionary.

    Returns:
        ROIAnalysis containing the metric result, cleaned mask, candidate mask,
        largest contour, and bounding box.

    Raises:
        ROIAnalysisError:
            If image input or threshold settings are invalid.
    """
    start_time = perf_counter()
    active_config = config or get_default_config()

    threshold = (
        float(
            active_config["thresholds"]["roi"]["minimum_fraction"]
        )
        if minimum_fraction is None
        else float(minimum_fraction)
    )

    _validate_minimum_fraction(threshold)

    bgr_image, grayscale = _prepare_input_images(image)

    if bgr_image is not None:
        colour_mask = create_skin_candidate_mask(bgr_image)
        grayscale_mask = create_grayscale_candidate_mask(grayscale)

        candidate_mask = combine_candidate_masks(
            colour_mask,
            grayscale_mask,
        )
    else:
        candidate_mask = create_grayscale_candidate_mask(
            grayscale
        )

    cleaned_mask = clean_candidate_mask(candidate_mask)

    finger_mask, contour, bounding_box = retain_largest_contour(
        cleaned_mask
    )

    total_pixel_count = int(grayscale.size)
    roi_pixel_count = int(np.count_nonzero(finger_mask))

    roi_fraction = (
        roi_pixel_count / total_pixel_count
        if total_pixel_count > 0
        else 0.0
    )

    normalized_score = normalize_roi_score(
        roi_fraction=roi_fraction,
        minimum_fraction=threshold,
    )

    passed = (
        contour is not None
        and roi_fraction >= threshold
    )

    if contour is None:
        message = (
            "No reliable finger region was detected. Place one fingertip "
            "against a plain background and capture again."
        )
    elif roi_fraction < threshold:
        message = (
            "Finger coverage is too small. Move the finger closer to the "
            "camera and keep the fingertip fully inside the frame."
        )
    else:
        message = (
            "Finger coverage is sufficient for biometric processing."
        )

    elapsed_ms = (perf_counter() - start_time) * 1000.0

    contour_area = (
        float(cv2.contourArea(contour))
        if contour is not None
        else 0.0
    )

    bounding_box_fraction = _calculate_bounding_box_fraction(
        bounding_box=bounding_box,
        image_shape=grayscale.shape,
    )

    result = ROIResult(
        name="roi",
        roi_fraction=round(roi_fraction, 6),
        normalized_score=round(normalized_score, 4),
        passed=passed,
        threshold={
            "minimum_fraction": threshold,
        },
        processing_time_ms=round(elapsed_ms, 4),
        message=message,
        details={
            "roi_percentage": round(
                roi_fraction * 100.0,
                4,
            ),
            "roi_pixel_count": roi_pixel_count,
            "total_pixel_count": total_pixel_count,
            "contour_area": round(contour_area, 4),
            "bounding_box_fraction": round(
                bounding_box_fraction,
                6,
            ),
            "bounding_box": (
                list(bounding_box)
                if bounding_box is not None
                else None
            ),
            "method": (
                "combined_skin_and_grayscale_largest_contour"
                if bgr_image is not None
                else "grayscale_largest_contour"
            ),
        },
    )

    return ROIAnalysis(
        result=result,
        finger_mask=finger_mask,
        candidate_mask=candidate_mask,
        bounding_box=bounding_box,
        contour=contour,
    )


def create_skin_candidate_mask(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Create a skin-colour candidate mask using YCrCb and HSV colour spaces.

    Multiple colour spaces are combined because mobile-camera lighting can
    shift RGB values significantly. YCrCb separates luminance from chrominance,
    while HSV provides an additional hue-and-saturation constraint.
    """
    _validate_bgr_image(image_bgr)

    blurred = cv2.GaussianBlur(
        image_bgr,
        (5, 5),
        sigmaX=0,
    )

    ycrcb = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2YCrCb,
    )

    ycrcb_lower = np.array(
        [0, 128, 70],
        dtype=np.uint8,
    )
    ycrcb_upper = np.array(
        [255, 180, 140],
        dtype=np.uint8,
    )

    ycrcb_mask = cv2.inRange(
        ycrcb,
        ycrcb_lower,
        ycrcb_upper,
    )

    hsv = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2HSV,
    )

    hsv_lower = np.array(
        [0, 20, 30],
        dtype=np.uint8,
    )
    hsv_upper = np.array(
        [35, 255, 255],
        dtype=np.uint8,
    )

    hsv_mask = cv2.inRange(
        hsv,
        hsv_lower,
        hsv_upper,
    )

    combined_mask = cv2.bitwise_and(
        ycrcb_mask,
        hsv_mask,
    )

    return combined_mask


def create_grayscale_candidate_mask(
    grayscale: np.ndarray,
) -> np.ndarray:
    """
    Estimate foreground using blurred grayscale and Otsu thresholding.

    Both normal and inverse masks are evaluated. The mask with a plausible
    foreground fraction closer to the expected finger-area range is selected.
    """
    _validate_grayscale_image(grayscale)

    blurred = cv2.GaussianBlur(
        grayscale,
        (7, 7),
        sigmaX=0,
    )

    _, normal_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    inverse_mask = cv2.bitwise_not(normal_mask)

    normal_score = _foreground_plausibility_score(
        normal_mask
    )
    inverse_score = _foreground_plausibility_score(
        inverse_mask
    )

    if inverse_score > normal_score:
        return inverse_mask

    return normal_mask


def combine_candidate_masks(
    colour_mask: np.ndarray,
    grayscale_mask: np.ndarray,
) -> np.ndarray:
    """
    Combine colour and grayscale candidates.

    The colour mask is preferred when it contains a plausible amount of
    foreground. Grayscale foreground is used to recover weakly detected finger
    regions around the colour candidate.
    """
    _validate_binary_mask(colour_mask)
    _validate_binary_mask(grayscale_mask)

    if colour_mask.shape != grayscale_mask.shape:
        raise ROIAnalysisError(
            "Candidate masks must have identical dimensions."
        )

    colour_fraction = (
        np.count_nonzero(colour_mask) / colour_mask.size
    )

    if 0.03 <= colour_fraction <= 0.85:
        expanded_colour = cv2.dilate(
            colour_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9),
            ),
            iterations=1,
        )

        supported_grayscale = cv2.bitwise_and(
            grayscale_mask,
            expanded_colour,
        )

        return cv2.bitwise_or(
            colour_mask,
            supported_grayscale,
        )

    return grayscale_mask.copy()


def clean_candidate_mask(
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """
    Remove isolated noise and fill small gaps using morphology.
    """
    _validate_binary_mask(candidate_mask)

    height, width = candidate_mask.shape

    minimum_dimension = min(height, width)
    kernel_size = max(
        3,
        int(round(minimum_dimension * 0.015)),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel_size = min(kernel_size, 21)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    opened = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return closed


def retain_largest_contour(
    cleaned_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    tuple[int, int, int, int] | None,
]:
    """
    Retain the largest plausible connected foreground component.
    """
    _validate_binary_mask(cleaned_mask)

    contours, _ = cv2.findContours(
        cleaned_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    empty_mask = np.zeros_like(cleaned_mask)

    if not contours:
        return empty_mask, None, None

    image_area = float(cleaned_mask.size)

    plausible_contours: list[np.ndarray] = []

    for contour in contours:
        contour_area = float(
            cv2.contourArea(contour)
        )

        if contour_area <= 0:
            continue

        area_fraction = contour_area / image_area

        x, y, width, height = cv2.boundingRect(contour)

        if width <= 1 or height <= 1:
            continue

        aspect_ratio = max(
            width / height,
            height / width,
        )

        if (
            0.01 <= area_fraction <= 0.90
            and aspect_ratio <= 8.0
        ):
            plausible_contours.append(contour)

    if not plausible_contours:
        return empty_mask, None, None

    largest_contour = max(
        plausible_contours,
        key=cv2.contourArea,
    )

    cv2.drawContours(
        empty_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED,
    )

    bounding_box = cv2.boundingRect(
        largest_contour
    )

    return (
        empty_mask,
        largest_contour,
        bounding_box,
    )


def normalize_roi_score(
    roi_fraction: float,
    minimum_fraction: float,
) -> float:
    """
    Convert ROI coverage into a score between 0.0 and 1.0.

    Half the minimum required coverage maps to 0.0.

    The minimum accepted coverage maps to 0.5.

    Twice the minimum accepted coverage maps to 1.0.
    """
    roi_fraction = float(roi_fraction)
    minimum_fraction = float(minimum_fraction)

    if not 0 <= roi_fraction <= 1:
        raise ROIAnalysisError(
            "ROI fraction must be between 0.0 and 1.0."
        )

    _validate_minimum_fraction(minimum_fraction)

    lower_reference = minimum_fraction * 0.5
    upper_reference = min(
        minimum_fraction * 2.0,
        1.0,
    )

    if roi_fraction <= lower_reference:
        return 0.0

    if roi_fraction >= upper_reference:
        return 1.0

    if roi_fraction <= minimum_fraction:
        normalized = 0.5 * (
            (roi_fraction - lower_reference)
            / (minimum_fraction - lower_reference)
        )
    else:
        normalized = 0.5 + 0.5 * (
            (roi_fraction - minimum_fraction)
            / (upper_reference - minimum_fraction)
        )

    return _clamp(normalized)


def create_roi_overlay(
    image_bgr: np.ndarray,
    finger_mask: np.ndarray,
    bounding_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """
    Create a diagnostic overlay highlighting the detected finger region.

    The result is intended for the Streamlit diagnostic interface.
    """
    _validate_bgr_image(image_bgr)
    _validate_binary_mask(finger_mask)

    if image_bgr.shape[:2] != finger_mask.shape:
        raise ROIAnalysisError(
            "Image and ROI mask must have matching dimensions."
        )

    overlay = image_bgr.copy()

    highlight = np.zeros_like(image_bgr)
    highlight[:, :, 1] = finger_mask

    overlay = cv2.addWeighted(
        overlay,
        0.75,
        highlight,
        0.25,
        0,
    )

    if bounding_box is not None:
        x, y, width, height = bounding_box

        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            thickness=2,
        )

    return overlay


def _prepare_input_images(
    image: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Return optional BGR image and required grayscale representation."""
    if not isinstance(image, np.ndarray):
        raise ROIAnalysisError(
            "ROI input must be a NumPy array."
        )

    if image.size == 0:
        raise ROIAnalysisError(
            "ROI input image is empty."
        )

    if image.ndim == 2:
        grayscale = _convert_to_uint8(image)
        _validate_grayscale_image(grayscale)

        return None, grayscale

    if image.ndim == 3 and image.shape[2] == 3:
        bgr_image = _convert_to_uint8(image)
        _validate_bgr_image(bgr_image)

        grayscale = cv2.cvtColor(
            bgr_image,
            cv2.COLOR_BGR2GRAY,
        )

        return bgr_image, grayscale

    raise ROIAnalysisError(
        "Expected a grayscale image or a three-channel BGR image."
    )


def _foreground_plausibility_score(
    mask: np.ndarray,
) -> float:
    """
    Score how plausible a foreground mask is for a finger capture.
    """
    foreground_fraction = (
        np.count_nonzero(mask) / mask.size
    )

    ideal_fraction = 0.35
    fraction_score = 1.0 - min(
        abs(foreground_fraction - ideal_fraction)
        / ideal_fraction,
        1.0,
    )

    border_pixels = np.concatenate(
        [
            mask[0, :],
            mask[-1, :],
            mask[:, 0],
            mask[:, -1],
        ]
    )

    border_fraction = (
        np.count_nonzero(border_pixels)
        / border_pixels.size
    )

    border_penalty = min(
        border_fraction,
        1.0,
    )

    return fraction_score - (0.35 * border_penalty)


def _calculate_bounding_box_fraction(
    bounding_box: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
) -> float:
    """Calculate bounding-box area relative to full image area."""
    if bounding_box is None:
        return 0.0

    _, _, width, height = bounding_box
    image_height, image_width = image_shape

    image_area = image_width * image_height

    if image_area <= 0:
        return 0.0

    return (width * height) / image_area


def _validate_minimum_fraction(
    minimum_fraction: float,
) -> None:
    """Validate ROI acceptance threshold."""
    if not 0 < minimum_fraction <= 1:
        raise ROIAnalysisError(
            "Minimum ROI fraction must be greater than 0 and at most 1."
        )


def _validate_bgr_image(
    image_bgr: np.ndarray,
) -> None:
    """Validate an OpenCV BGR image."""
    if not isinstance(image_bgr, np.ndarray):
        raise ROIAnalysisError(
            "Expected BGR image as a NumPy array."
        )

    if image_bgr.size == 0:
        raise ROIAnalysisError(
            "BGR image is empty."
        )

    if (
        image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
    ):
        raise ROIAnalysisError(
            "Expected a three-channel BGR image."
        )

    if (
        image_bgr.shape[0] < 2
        or image_bgr.shape[1] < 2
    ):
        raise ROIAnalysisError(
            "Image dimensions are too small for ROI analysis."
        )


def _validate_grayscale_image(
    grayscale: np.ndarray,
) -> None:
    """Validate an 8-bit grayscale image."""
    if not isinstance(grayscale, np.ndarray):
        raise ROIAnalysisError(
            "Expected grayscale image as a NumPy array."
        )

    if grayscale.size == 0:
        raise ROIAnalysisError(
            "Grayscale image is empty."
        )

    if grayscale.ndim != 2:
        raise ROIAnalysisError(
            "Expected a single-channel grayscale image."
        )

    if (
        grayscale.shape[0] < 2
        or grayscale.shape[1] < 2
    ):
        raise ROIAnalysisError(
            "Image dimensions are too small for ROI analysis."
        )


def _validate_binary_mask(
    mask: np.ndarray,
) -> None:
    """Validate a single-channel binary mask."""
    _validate_grayscale_image(mask)

    unique_values = set(
        np.unique(mask).tolist()
    )

    if not unique_values.issubset({0, 255}):
        raise ROIAnalysisError(
            "Expected a binary mask containing only 0 and 255."
        )


def _convert_to_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """Convert supported numeric image arrays into uint8."""
    if image.dtype == np.uint8:
        return image.copy()

    return cv2.convertScaleAbs(image)


def _clamp(value: float) -> float:
    """Restrict a score to the inclusive range 0.0–1.0."""
    return max(
        0.0,
        min(1.0, float(value)),
    )


if __name__ == "__main__":
    demo_image = np.full(
        (480, 640, 3),
        (35, 35, 35),
        dtype=np.uint8,
    )

    cv2.ellipse(
        demo_image,
        center=(320, 260),
        axes=(95, 180),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(90, 145, 205),
        thickness=-1,
    )

    analysis = check_roi_completeness(
        demo_image
    )

    print("RidgeLens Phase 5 ROI demonstration")
    print(
        "ROI fraction:",
        analysis.result.roi_fraction,
    )
    print(
        "ROI percentage:",
        analysis.result.details[
            "roi_percentage"
        ],
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
        "Bounding box:",
        analysis.bounding_box,
    )
    print(
        "Processing time:",
        analysis.result.processing_time_ms,
        "ms",
    )