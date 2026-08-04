"""
Automated tests for the complete RidgeLens quality pipeline.

The tests connect preprocessing, all five quality metrics, composite scoring,
guidance generation, diagnostics, serialization, and performance reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from app.config import get_default_config
from app.guidance import (
    GuidanceError,
    build_guidance_report,
    get_failed_metric_names,
)
from app.image_processing import load_image_from_bytes
from app.quality_pipeline import (
    CompositeScore,
    PerformanceReport,
    QualityAssessment,
    QualityPipelineError,
    assess_image_bytes,
    assess_preprocessed_image,
    calculate_composite_score,
)


@dataclass(frozen=True)
class SyntheticMetric:
    """Minimal metric contract for composite-score tests."""

    normalized_score: float
    passed: bool


def create_synthetic_fingerprint_image(
    width: int = 640,
    height: int = 480,
    blurred: bool = False,
    brightness_offset: int = 0,
    glare_fraction: float = 0.0,
) -> np.ndarray:
    """Create a synthetic finger region containing ridge-like lines."""
    image = np.full(
        (height, width, 3),
        (30, 30, 30),
        dtype=np.uint8,
    )

    finger_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    cv2.ellipse(
        finger_mask,
        center=(width // 2, height // 2),
        axes=(115, 190),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    finger_colour = np.array(
        [95, 145, 205],
        dtype=np.int16,
    )

    finger_colour = np.clip(
        finger_colour + brightness_offset,
        0,
        255,
    ).astype(np.uint8)

    image[
        finger_mask > 0
    ] = finger_colour

    for x_position in range(
        width // 2 - 95,
        width // 2 + 96,
        8,
    ):
        cv2.line(
            image,
            (x_position, height // 2 - 155),
            (x_position, height // 2 + 155),
            (55, 75, 95),
            thickness=2,
        )

    image[
        finger_mask == 0
    ] = (30, 30, 30)

    if glare_fraction > 0:
        glare_pixel_count = int(
            image.shape[0]
            * image.shape[1]
            * glare_fraction
        )

        flattened = image.reshape(-1, 3)
        flattened[
            :glare_pixel_count
        ] = (255, 255, 255)

    if blurred:
        image = cv2.GaussianBlur(
            image,
            (31, 31),
            sigmaX=10,
            sigmaY=10,
        )

    return image


def encode_jpeg(
    image_bgr: np.ndarray,
) -> bytes:
    """Encode a synthetic BGR image as JPEG bytes."""
    success, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
    )

    assert success is True

    return encoded.tobytes()


def create_metric_results(
    score: float = 1.0,
    passed: bool = True,
) -> dict[str, SyntheticMetric]:
    """Create all five synthetic metric results."""
    return {
        "blur": SyntheticMetric(
            normalized_score=score,
            passed=passed,
        ),
        "brightness": SyntheticMetric(
            normalized_score=score,
            passed=passed,
        ),
        "glare": SyntheticMetric(
            normalized_score=score,
            passed=passed,
        ),
        "roi": SyntheticMetric(
            normalized_score=score,
            passed=passed,
        ),
        "ridge": SyntheticMetric(
            normalized_score=score,
            passed=passed,
        ),
    }


def test_composite_score_reaches_100_for_perfect_metrics() -> None:
    """Perfect normalized scores should produce 100."""
    composite = calculate_composite_score(
        create_metric_results(
            score=1.0,
            passed=True,
        )
    )

    assert isinstance(
        composite,
        CompositeScore,
    )
    assert composite.score == pytest.approx(
        100.0
    )
    assert composite.passed is True


def test_composite_score_uses_configured_weights() -> None:
    """Contributions must match the configured metric weights."""
    metric_results = {
        "blur": SyntheticMetric(1.0, True),
        "brightness": SyntheticMetric(0.0, False),
        "glare": SyntheticMetric(0.0, False),
        "roi": SyntheticMetric(0.0, False),
        "ridge": SyntheticMetric(0.0, False),
    }

    composite = calculate_composite_score(
        metric_results
    )

    assert composite.score == pytest.approx(
        25.0
    )
    assert composite.contributions[
        "blur"
    ] == pytest.approx(25.0)


def test_individual_metric_failure_blocks_final_pass() -> None:
    """
    A high composite score must not hide a critical metric failure.
    """
    metric_results = create_metric_results(
        score=1.0,
        passed=True,
    )

    metric_results["glare"] = SyntheticMetric(
        normalized_score=0.8,
        passed=False,
    )

    composite = calculate_composite_score(
        metric_results
    )

    assert composite.score >= 60
    assert composite.passed is False


def test_missing_metric_is_rejected() -> None:
    """Every configured metric is required."""
    metric_results = create_metric_results()
    del metric_results["ridge"]

    with pytest.raises(
        QualityPipelineError,
        match="Missing metric results",
    ):
        calculate_composite_score(
            metric_results
        )


def test_out_of_range_normalized_score_is_rejected() -> None:
    """Metric normalized scores must remain between zero and one."""
    metric_results = create_metric_results()
    metric_results["blur"] = SyntheticMetric(
        normalized_score=1.5,
        passed=True,
    )

    with pytest.raises(
        QualityPipelineError,
        match="between 0 and 1",
    ):
        calculate_composite_score(
            metric_results
        )


def test_guidance_prioritizes_roi_before_other_failures() -> None:
    """Finger-positioning issues should be presented first."""
    metric_results = create_metric_results()

    metric_results["blur"] = SyntheticMetric(
        0.1,
        False,
    )
    metric_results["roi"] = SyntheticMetric(
        0.1,
        False,
    )
    metric_results["glare"] = SyntheticMetric(
        0.1,
        False,
    )

    report = build_guidance_report(
        metric_results=metric_results,
        final_passed=False,
        composite_score=20.0,
    )

    assert report.items[0].metric == "roi"
    assert report.items[1].metric == "blur"
    assert report.items[2].metric == "glare"


def test_successful_guidance_has_no_corrective_items() -> None:
    """Accepted captures should receive concise positive guidance."""
    report = build_guidance_report(
        metric_results=create_metric_results(),
        final_passed=True,
        composite_score=90.0,
    )

    assert report.status_label == "READY"
    assert report.items == ()
    assert "ready" in (
        report.primary_message.lower()
    )


def test_failed_metric_names_follow_priority() -> None:
    """Failed names should use guidance priority order."""
    metric_results = create_metric_results()

    metric_results["ridge"] = SyntheticMetric(
        0.0,
        False,
    )
    metric_results["roi"] = SyntheticMetric(
        0.0,
        False,
    )
    metric_results["brightness"] = SyntheticMetric(
        0.0,
        False,
    )

    failed_names = get_failed_metric_names(
        metric_results
    )

    assert failed_names == (
        "roi",
        "brightness",
        "ridge",
    )


@pytest.mark.parametrize(
    "invalid_score",
    [-1.0, 101.0],
)
def test_invalid_guidance_score_is_rejected(
    invalid_score: float,
) -> None:
    """Guidance score must remain from 0 to 100."""
    with pytest.raises(
        GuidanceError,
        match="between 0 and 100",
    ):
        build_guidance_report(
            metric_results=create_metric_results(),
            final_passed=False,
            composite_score=invalid_score,
        )


def test_complete_pipeline_returns_all_five_metrics() -> None:
    """End-to-end assessment should return every required metric."""
    image_bytes = encode_jpeg(
        create_synthetic_fingerprint_image()
    )

    assessment = assess_image_bytes(
        image_bytes=image_bytes,
        source_name="synthetic.jpg",
    )

    assert isinstance(
        assessment,
        QualityAssessment,
    )

    assert set(
        assessment.metrics.keys()
    ) == {
        "blur",
        "brightness",
        "glare",
        "roi",
        "ridge",
    }


def test_pipeline_returns_composite_and_guidance() -> None:
    """Complete output should contain final decision components."""
    assessment = assess_image_bytes(
        image_bytes=encode_jpeg(
            create_synthetic_fingerprint_image()
        ),
        source_name="synthetic.jpg",
    )

    assert isinstance(
        assessment.composite,
        CompositeScore,
    )
    assert 0 <= assessment.score <= 100
    assert assessment.guidance.status_label
    assert assessment.guidance.primary_message


def test_pipeline_returns_performance_report() -> None:
    """Each complete assessment must report timing information."""
    assessment = assess_image_bytes(
        image_bytes=encode_jpeg(
            create_synthetic_fingerprint_image()
        ),
        source_name="synthetic.jpg",
    )

    assert isinstance(
        assessment.performance,
        PerformanceReport,
    )
    assert assessment.performance.total_ms >= 0

    assert set(
        assessment.performance.within_budget.keys()
    ) == {
        "blur",
        "brightness",
        "glare",
        "roi",
        "ridge",
        "total",
    }


def test_pipeline_provides_all_diagnostic_outputs() -> None:
    """UI-required masks and response images should be available."""
    assessment = assess_image_bytes(
        image_bytes=encode_jpeg(
            create_synthetic_fingerprint_image()
        ),
        source_name="synthetic.jpg",
    )

    expected_diagnostics = {
        "original_bgr",
        "resized_bgr",
        "grayscale",
        "enhanced_grayscale",
        "glare_mask",
        "roi_candidate_mask",
        "finger_mask",
        "roi_bounding_box",
        "ridge_response",
        "ridge_combined_response",
    }

    assert expected_diagnostics.issubset(
        assessment.diagnostics.keys()
    )


def test_pipeline_result_is_serializable() -> None:
    """Raw image arrays must be excluded from the public dictionary."""
    assessment = assess_image_bytes(
        image_bytes=encode_jpeg(
            create_synthetic_fingerprint_image()
        ),
        source_name="synthetic.jpg",
    )

    serialized = assessment.to_dict()

    assert "composite" in serialized
    assert "metrics" in serialized
    assert "guidance" in serialized
    assert "performance" in serialized
    assert "metadata" in serialized

    assert "available" in (
        serialized["diagnostics"]
    )

    assert "original_bgr" not in (
        serialized["diagnostics"]
    )


def test_preprocessed_image_can_be_assessed_directly() -> None:
    """Pipeline should support reuse of existing preprocessing output."""
    image_bytes = encode_jpeg(
        create_synthetic_fingerprint_image()
    )

    preprocessed = load_image_from_bytes(
        image_bytes=image_bytes,
        source_name="synthetic.jpg",
    )

    assessment = assess_preprocessed_image(
        preprocessed
    )

    assert isinstance(
        assessment,
        QualityAssessment,
    )
    assert assessment.metadata[
        "source_name"
    ] == "synthetic.jpg"


def test_dark_capture_reports_brightness_failure() -> None:
    """Strong negative brightness adjustment should trigger exposure guidance."""
    dark_image = create_synthetic_fingerprint_image(
        brightness_offset=-90,
    )

    assessment = assess_image_bytes(
        image_bytes=encode_jpeg(dark_image),
        source_name="dark.jpg",
    )

    assert (
        assessment.metrics[
            "brightness"
        ].passed
        is False
    )


def test_glare_capture_reports_glare_failure() -> None:
    """Excessive white coverage should fail glare assessment."""
    glare_image = create_synthetic_fingerprint_image(
        glare_fraction=0.10,
    )

    assessment = assess_image_bytes(
        image_bytes=encode_jpeg(glare_image),
        source_name="glare.jpg",
    )

    assert (
        assessment.metrics[
            "glare"
        ].passed
        is False
    )


def test_configuration_weights_total_one() -> None:
    """Pipeline scoring depends on normalized total weights."""
    config = get_default_config()

    assert sum(
        config["weights"].values()
    ) == pytest.approx(1.0)