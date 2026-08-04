"""
Visualization utilities for the RidgeLens Streamlit dashboard.

This module converts OpenCV images, masks, and quality results into
display-ready diagnostic visualizations.

The functions remain independent from Streamlit so they can be tested and
reused by future report-generation or batch-evaluation modules.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Mapping

import cv2
import numpy as np

from app.ridge_analysis import create_ridge_overlay
from app.roi_analysis import create_roi_overlay


class VisualizationError(RuntimeError):
    """Raised when a diagnostic visualization cannot be created."""


METRIC_LABELS: dict[str, str] = {
    "blur": "Sharpness",
    "brightness": "Brightness",
    "glare": "Glare",
    "roi": "Finger Coverage",
    "ridge": "Ridge Clarity",
}


def convert_bgr_to_rgb(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """Convert an OpenCV BGR image into RGB format."""
    _validate_bgr_image(image_bgr)

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )


def convert_grayscale_to_rgb(
    grayscale: np.ndarray,
) -> np.ndarray:
    """Convert a grayscale image into three-channel RGB format."""
    _validate_grayscale_image(grayscale)

    return cv2.cvtColor(
        grayscale,
        cv2.COLOR_GRAY2RGB,
    )


def create_glare_overlay(
    image_bgr: np.ndarray,
    glare_mask: np.ndarray,
) -> np.ndarray:
    """
    Highlight glare-affected pixels using a translucent red overlay.
    """
    _validate_bgr_image(image_bgr)
    _validate_binary_mask(glare_mask)

    if image_bgr.shape[:2] != glare_mask.shape:
        raise VisualizationError(
            "Image and glare mask dimensions must match."
        )

    overlay = image_bgr.copy()

    glare_layer = np.zeros_like(
        image_bgr
    )
    glare_layer[:, :, 2] = glare_mask

    overlay = cv2.addWeighted(
        overlay,
        0.72,
        glare_layer,
        0.28,
        0,
    )

    contours, _ = cv2.findContours(
        glare_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if contours:
        cv2.drawContours(
            overlay,
            contours,
            contourIdx=-1,
            color=(0, 0, 255),
            thickness=2,
        )

    return overlay


def create_roi_diagnostic(
    image_bgr: np.ndarray,
    finger_mask: np.ndarray,
    bounding_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Create the ROI mask and bounding-box overlay."""
    return create_roi_overlay(
        image_bgr=image_bgr,
        finger_mask=finger_mask,
        bounding_box=bounding_box,
    )


def create_ridge_diagnostic(
    image_bgr: np.ndarray,
    response_visualization: np.ndarray,
    finger_mask: np.ndarray,
) -> np.ndarray:
    """Create a ridge-response heatmap restricted to the finger ROI."""
    return create_ridge_overlay(
        image_bgr=image_bgr,
        response_visualization=response_visualization,
        analysis_mask=finger_mask,
    )


def create_mask_preview(
    mask: np.ndarray,
) -> np.ndarray:
    """Convert a binary mask into an RGB preview."""
    _validate_binary_mask(mask)

    return cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2RGB,
    )


def build_metric_rows(
    metric_results: Mapping[str, Any],
    weights: Mapping[str, float],
    contributions: Mapping[str, float],
) -> list[dict[str, Any]]:
    """
    Convert metric results into table-ready dictionaries.
    """
    rows: list[dict[str, Any]] = []

    metric_order = (
        "blur",
        "brightness",
        "glare",
        "roi",
        "ridge",
    )

    for metric_name in metric_order:
        if metric_name not in metric_results:
            raise VisualizationError(
                f"Missing metric result: {metric_name}"
            )

        if metric_name not in weights:
            raise VisualizationError(
                f"Missing metric weight: {metric_name}"
            )

        if metric_name not in contributions:
            raise VisualizationError(
                f"Missing metric contribution: {metric_name}"
            )

        result = metric_results[
            metric_name
        ]

        raw_value = _get_raw_metric_value(
            metric_name,
            result,
        )

        rows.append(
            {
                "Metric": METRIC_LABELS.get(
                    metric_name,
                    metric_name.title(),
                ),
                "Status": (
                    "PASS"
                    if bool(result.passed)
                    else "FAIL"
                ),
                "Raw value": round(
                    float(raw_value),
                    4,
                ),
                "Quality score": round(
                    float(
                        result.normalized_score
                    )
                    * 100.0,
                    2,
                ),
                "Weight": round(
                    float(
                        weights[metric_name]
                    )
                    * 100.0,
                    2,
                ),
                "Contribution": round(
                    float(
                        contributions[
                            metric_name
                        ]
                    ),
                    2,
                ),
                "Time (ms)": round(
                    float(
                        result.processing_time_ms
                    ),
                    3,
                ),
            }
        )

    return rows


def create_score_ring_svg(
    score: float,
    passed: bool,
) -> str:
    """
    Return an HTML/SVG circular composite-score visualization.

    dedent().strip() ensures Streamlit treats the output as HTML instead of
    rendering the indented tags as a Markdown code block.
    """
    score_value = max(
        0.0,
        min(100.0, float(score)),
    )

    radius = 74.0
    circumference = (
        2.0
        * np.pi
        * radius
    )

    offset = (
        circumference
        * (1.0 - score_value / 100.0)
    )

    state_class = (
        "score-pass"
        if passed
        else "score-fail"
    )

    status = (
        "READY"
        if passed
        else "RETAKE"
    )

    html = f"""
    <div class="score-panel">
        <div class="score-ring {state_class}">
            <svg
                width="190"
                height="190"
                viewBox="0 0 190 190"
                role="img"
                aria-label="Composite quality score {score_value:.0f} out of 100"
            >
                <circle
                    class="score-track"
                    cx="95"
                    cy="95"
                    r="{radius}"
                ></circle>

                <circle
                    class="score-progress"
                    cx="95"
                    cy="95"
                    r="{radius}"
                    style="
                        stroke-dasharray: {circumference:.4f};
                        stroke-dashoffset: {offset:.4f};
                    "
                ></circle>
            </svg>

            <div class="score-content">
                <div class="score-number">
                    {score_value:.0f}
                </div>

                <div class="score-denominator">
                    / 100
                </div>
            </div>
        </div>

        <div class="score-status {state_class}">
            {status}
        </div>

        <div class="score-caption">
            Composite quality score
        </div>
    </div>
    """

    return dedent(html).strip()


def create_metric_card_html(
    metric_name: str,
    result: Any,
) -> str:
    """
    Build a premium metric result card.

    The returned HTML is dedented so Streamlit renders it as HTML rather than
    displaying the tags as a Markdown code block.
    """
    if metric_name not in METRIC_LABELS:
        raise VisualizationError(
            f"Unsupported metric: {metric_name}"
        )

    _validate_metric_result(
        metric_name=metric_name,
        result=result,
    )

    label = METRIC_LABELS[
        metric_name
    ]

    passed = bool(
        result.passed
    )

    state_class = (
        "metric-pass"
        if passed
        else "metric-fail"
    )

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    quality_percentage = max(
        0.0,
        min(
            100.0,
            float(
                result.normalized_score
            )
            * 100.0,
        ),
    )

    raw_value = _get_raw_metric_value(
        metric_name,
        result,
    )

    formatted_value = _format_metric_value(
        metric_name=metric_name,
        raw_value=raw_value,
    )

    message = _escape_basic_html(
        str(
            result.message
        )
    )

    html = f"""
    <div class="metric-card {state_class}">
        <div class="metric-card-top">
            <span class="metric-name">
                {label}
            </span>

            <span class="metric-badge">
                {status}
            </span>
        </div>

        <div class="metric-value">
            {formatted_value}
        </div>

        <div class="metric-score-label">
            Quality score: {quality_percentage:.1f}%
        </div>

        <div class="metric-progress">
            <div
                class="metric-progress-fill"
                style="width: {quality_percentage:.1f}%;"
            ></div>
        </div>

        <div class="metric-message">
            {message}
        </div>
    </div>
    """

    return dedent(html).strip()


def _get_raw_metric_value(
    metric_name: str,
    result: Any,
) -> float:
    """Read the raw measurement from supported metric result types."""
    if metric_name in {
        "blur",
        "brightness",
        "glare",
    }:
        if not hasattr(
            result,
            "raw_value",
        ):
            raise VisualizationError(
                f"Metric '{metric_name}' does not expose raw_value."
            )

        return float(
            result.raw_value
        )

    if metric_name == "roi":
        if not hasattr(
            result,
            "roi_fraction",
        ):
            raise VisualizationError(
                "ROI metric does not expose roi_fraction."
            )

        return float(
            result.roi_fraction
            * 100.0
        )

    if metric_name == "ridge":
        if not hasattr(
            result,
            "ridge_score",
        ):
            raise VisualizationError(
                "Ridge metric does not expose ridge_score."
            )

        return float(
            result.ridge_score
        )

    raise VisualizationError(
        f"Unsupported metric: {metric_name}"
    )


def _format_metric_value(
    metric_name: str,
    raw_value: float,
) -> str:
    """Format metric values using suitable user-facing units."""
    if metric_name == "roi":
        return f"{raw_value:.2f}%"

    if metric_name == "glare":
        return f"{raw_value * 100.0:.2f}%"

    if metric_name == "brightness":
        return f"{raw_value:.2f}"

    if metric_name == "blur":
        return f"{raw_value:.2f}"

    if metric_name == "ridge":
        return f"{raw_value:.2f}"

    raise VisualizationError(
        f"Unsupported metric: {metric_name}"
    )


def _validate_metric_result(
    metric_name: str,
    result: Any,
) -> None:
    """Validate the minimum interface needed by a metric card."""
    required_attributes = (
        "passed",
        "normalized_score",
        "processing_time_ms",
        "message",
    )

    for attribute in required_attributes:
        if not hasattr(
            result,
            attribute,
        ):
            raise VisualizationError(
                f"Metric '{metric_name}' does not expose {attribute}."
            )

    normalized_score = float(
        result.normalized_score
    )

    if not 0.0 <= normalized_score <= 1.0:
        raise VisualizationError(
            f"Metric '{metric_name}' normalized score must be between 0 and 1."
        )


def _escape_basic_html(
    value: str,
) -> str:
    """Escape basic HTML characters in dynamic user-facing text."""
    return (
        value
        .replace(
            "&",
            "&amp;",
        )
        .replace(
            "<",
            "&lt;",
        )
        .replace(
            ">",
            "&gt;",
        )
        .replace(
            '"',
            "&quot;",
        )
        .replace(
            "'",
            "&#x27;",
        )
    )


def _validate_bgr_image(
    image: np.ndarray,
) -> None:
    """Validate an OpenCV BGR image."""
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise VisualizationError(
            "Expected a NumPy image."
        )

    if image.size == 0:
        raise VisualizationError(
            "Image is empty."
        )

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise VisualizationError(
            "Expected a three-channel BGR image."
        )

    if (
        image.shape[0] < 1
        or image.shape[1] < 1
    ):
        raise VisualizationError(
            "Image dimensions are invalid."
        )


def _validate_grayscale_image(
    image: np.ndarray,
) -> None:
    """Validate a grayscale image."""
    if not isinstance(
        image,
        np.ndarray,
    ):
        raise VisualizationError(
            "Expected a NumPy image."
        )

    if image.size == 0:
        raise VisualizationError(
            "Image is empty."
        )

    if image.ndim != 2:
        raise VisualizationError(
            "Expected a single-channel image."
        )

    if (
        image.shape[0] < 1
        or image.shape[1] < 1
    ):
        raise VisualizationError(
            "Image dimensions are invalid."
        )


def _validate_binary_mask(
    mask: np.ndarray,
) -> None:
    """Validate a binary mask."""
    _validate_grayscale_image(
        mask
    )

    unique_values = set(
        np.unique(
            mask
        ).tolist()
    )

    if not unique_values.issubset(
        {0, 255}
    ):
        raise VisualizationError(
            "Expected a binary mask containing only 0 and 255."
        )


if __name__ == "__main__":
    print(
        "RidgeLens visualization utilities are ready."
    )