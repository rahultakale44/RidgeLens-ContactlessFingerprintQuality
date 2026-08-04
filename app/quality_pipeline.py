"""
Unified RidgeLens fingerprint-quality assessment pipeline.

This module connects:

1. Image preprocessing
2. Blur assessment
3. Brightness assessment
4. Glare detection
5. Finger ROI completeness
6. Gabor-based ridge clarity
7. Weighted composite scoring
8. Final quality decision
9. Prioritized capture guidance
10. Performance-budget tracking
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO, Mapping

from app.config import get_default_config
from app.guidance import (
    GuidanceReport,
    build_guidance_report,
    get_failed_metric_names,
)
from app.image_processing import (
    PreprocessedImage,
    load_image_from_bytes,
    load_image_from_file_object,
    load_image_from_path,
)
from app.quality_metrics import (
    GlareAnalysis,
    MetricResult,
    check_blur,
    check_brightness,
    check_glare,
)
from app.ridge_analysis import (
    RidgeAnalysis,
    check_ridge_clarity,
)
from app.roi_analysis import (
    ROIAnalysis,
    check_roi_completeness,
)


class QualityPipelineError(RuntimeError):
    """Raised when the complete quality pipeline cannot be executed."""


@dataclass(frozen=True)
class PerformanceReport:
    """Timing measurements and configured performance-budget checks."""

    preprocessing_ms: float
    blur_ms: float
    brightness_ms: float
    glare_ms: float
    roi_ms: float
    ridge_ms: float
    total_ms: float
    budget_ms: dict[str, float]
    within_budget: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable performance report."""
        return asdict(self)


@dataclass(frozen=True)
class CompositeScore:
    """Weighted composite-quality score information."""

    score: float
    threshold: float
    passed: bool
    weights: dict[str, float]
    contributions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable score report."""
        return asdict(self)


@dataclass(frozen=True)
class QualityAssessment:
    """Complete RidgeLens assessment output."""

    composite: CompositeScore
    metrics: dict[str, Any]
    guidance: GuidanceReport
    performance: PerformanceReport
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return the final quality decision."""
        return self.composite.passed

    @property
    def score(self) -> float:
        """Return the composite score."""
        return self.composite.score

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-serializable result without raw diagnostic arrays.
        """
        return {
            "passed": self.passed,
            "score": self.score,
            "composite": self.composite.to_dict(),
            "metrics": {
                name: _serialize_metric_result(result)
                for name, result in self.metrics.items()
            },
            "guidance": self.guidance.to_dict(),
            "performance": self.performance.to_dict(),
            "metadata": self.metadata,
            "diagnostics": {
                "available": sorted(
                    self.diagnostics.keys()
                ),
            },
        }


def assess_preprocessed_image(
    preprocessed: PreprocessedImage,
    config: dict[str, Any] | None = None,
) -> QualityAssessment:
    """
    Run all five quality metrics on a preprocessed image.

    Metric inputs:

    - Blur: original grayscale
    - Brightness: original grayscale
    - Glare: original grayscale
    - ROI: resized BGR image
    - Ridge: CLAHE-enhanced grayscale restricted to the finger ROI

    Args:
        preprocessed:
            Output produced by the RidgeLens preprocessing module.
        config:
            Optional project configuration.

    Returns:
        Complete QualityAssessment object.
    """
    if not isinstance(preprocessed, PreprocessedImage):
        raise QualityPipelineError(
            "Expected a PreprocessedImage instance."
        )

    pipeline_start = perf_counter()
    active_config = config or get_default_config()

    blur_result = check_blur(
        image=preprocessed.grayscale,
        config=active_config,
    )

    brightness_result = check_brightness(
        image=preprocessed.grayscale,
        config=active_config,
    )

    glare_analysis = check_glare(
        image=preprocessed.grayscale,
        config=active_config,
    )

    roi_analysis = check_roi_completeness(
        image=preprocessed.resized_bgr,
        config=active_config,
    )

    ridge_mask = (
        roi_analysis.finger_mask
        if roi_analysis.contour is not None
        else None
    )

    ridge_analysis = check_ridge_clarity(
        image=preprocessed.enhanced_grayscale,
        roi_mask=ridge_mask,
        config=active_config,
    )

    metric_results: dict[str, Any] = {
        "blur": blur_result,
        "brightness": brightness_result,
        "glare": glare_analysis.result,
        "roi": roi_analysis.result,
        "ridge": ridge_analysis.result,
    }

    composite = calculate_composite_score(
        metric_results=metric_results,
        config=active_config,
    )

    guidance = build_guidance_report(
        metric_results=metric_results,
        final_passed=composite.passed,
        composite_score=composite.score,
    )

    total_ms = (
        perf_counter() - pipeline_start
    ) * 1000.0

    performance = build_performance_report(
        preprocessed=preprocessed,
        blur_result=blur_result,
        brightness_result=brightness_result,
        glare_analysis=glare_analysis,
        roi_analysis=roi_analysis,
        ridge_analysis=ridge_analysis,
        total_ms=total_ms,
        config=active_config,
    )

    diagnostics = {
        "original_bgr": preprocessed.original_bgr,
        "resized_bgr": preprocessed.resized_bgr,
        "grayscale": preprocessed.grayscale,
        "enhanced_grayscale": (
            preprocessed.enhanced_grayscale
        ),
        "glare_mask": glare_analysis.glare_mask,
        "roi_candidate_mask": (
            roi_analysis.candidate_mask
        ),
        "finger_mask": roi_analysis.finger_mask,
        "roi_bounding_box": roi_analysis.bounding_box,
        "ridge_response": (
            ridge_analysis.response_visualization
        ),
        "ridge_combined_response": (
            ridge_analysis.combined_response
        ),
    }

    metadata = {
        **preprocessed.metadata.to_dict(),
        "failed_metrics": list(
            get_failed_metric_names(metric_results)
        ),
        "metric_count": len(metric_results),
    }

    return QualityAssessment(
        composite=composite,
        metrics=metric_results,
        guidance=guidance,
        performance=performance,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def assess_image_path(
    image_path: str | Path,
    config: dict[str, Any] | None = None,
) -> QualityAssessment:
    """Load and assess an image from a local path."""
    active_config = config or get_default_config()

    preprocessed = load_image_from_path(
        image_path=image_path,
        config=active_config,
    )

    return assess_preprocessed_image(
        preprocessed=preprocessed,
        config=active_config,
    )


def assess_image_bytes(
    image_bytes: bytes,
    source_name: str = "uploaded-image.jpg",
    config: dict[str, Any] | None = None,
) -> QualityAssessment:
    """Load and assess raw uploaded image bytes."""
    active_config = config or get_default_config()

    preprocessed = load_image_from_bytes(
        image_bytes=image_bytes,
        source_name=source_name,
        config=active_config,
    )

    return assess_preprocessed_image(
        preprocessed=preprocessed,
        config=active_config,
    )


def assess_uploaded_file(
    file_object: BinaryIO,
    source_name: str,
    config: dict[str, Any] | None = None,
) -> QualityAssessment:
    """Load and assess a Streamlit-compatible uploaded file."""
    active_config = config or get_default_config()

    preprocessed = load_image_from_file_object(
        file_object=file_object,
        source_name=source_name,
        config=active_config,
    )

    return assess_preprocessed_image(
        preprocessed=preprocessed,
        config=active_config,
    )


def calculate_composite_score(
    metric_results: Mapping[str, Any],
    config: dict[str, Any] | None = None,
) -> CompositeScore:
    """
    Calculate weighted composite quality score from normalized metrics.

    Final score:

        sum(normalized metric score × configured weight) × 100

    The final decision requires:

    1. Composite score at or above configured threshold.
    2. Every individual quality metric to pass.

    This quality-gate rule prevents a strong metric from hiding a critical
    failure in another stage.
    """
    active_config = config or get_default_config()

    weights = {
        name: float(value)
        for name, value in active_config["weights"].items()
    }

    expected_metrics = set(weights.keys())
    supplied_metrics = set(metric_results.keys())

    missing_metrics = expected_metrics.difference(
        supplied_metrics
    )

    if missing_metrics:
        missing = ", ".join(
            sorted(missing_metrics)
        )
        raise QualityPipelineError(
            f"Missing metric results: {missing}"
        )

    contributions: dict[str, float] = {}
    all_metrics_passed = True

    for metric_name, weight in weights.items():
        result = metric_results[metric_name]

        if not hasattr(result, "normalized_score"):
            raise QualityPipelineError(
                f"Metric '{metric_name}' has no normalized score."
            )

        if not hasattr(result, "passed"):
            raise QualityPipelineError(
                f"Metric '{metric_name}' has no pass/fail decision."
            )

        normalized_score = float(
            result.normalized_score
        )

        if not 0 <= normalized_score <= 1:
            raise QualityPipelineError(
                f"Metric '{metric_name}' normalized score "
                "must be between 0 and 1."
            )

        contribution = (
            normalized_score
            * weight
            * 100.0
        )

        contributions[metric_name] = round(
            contribution,
            4,
        )

        all_metrics_passed = (
            all_metrics_passed
            and bool(result.passed)
        )

    score = round(
        sum(contributions.values()),
        2,
    )

    threshold = float(
        active_config[
            "thresholds"
        ]["composite"]["minimum_score"]
    )

    passed = (
        score >= threshold
        and all_metrics_passed
    )

    return CompositeScore(
        score=score,
        threshold=threshold,
        passed=passed,
        weights=weights,
        contributions=contributions,
    )


def build_performance_report(
    preprocessed: PreprocessedImage,
    blur_result: MetricResult,
    brightness_result: MetricResult,
    glare_analysis: GlareAnalysis,
    roi_analysis: ROIAnalysis,
    ridge_analysis: RidgeAnalysis,
    total_ms: float,
    config: dict[str, Any] | None = None,
) -> PerformanceReport:
    """Compare observed timings with configured performance budgets."""
    active_config = config or get_default_config()

    configured_budget = {
        name: float(value)
        for name, value in active_config[
            "performance_budget_ms"
        ].items()
    }

    timings = {
        "preprocessing": float(
            preprocessed.metadata.processing_time_ms
        ),
        "blur": float(
            blur_result.processing_time_ms
        ),
        "brightness": float(
            brightness_result.processing_time_ms
        ),
        "glare": float(
            glare_analysis.result.processing_time_ms
        ),
        "roi": float(
            roi_analysis.result.processing_time_ms
        ),
        "ridge": float(
            ridge_analysis.result.processing_time_ms
        ),
        "total": float(total_ms),
    }

    within_budget = {
        "blur": (
            timings["blur"]
            <= configured_budget["blur"]
        ),
        "brightness": (
            timings["brightness"]
            <= configured_budget["brightness"]
        ),
        "glare": (
            timings["glare"]
            <= configured_budget["glare"]
        ),
        "roi": (
            timings["roi"]
            <= configured_budget["roi"]
        ),
        "ridge": (
            timings["ridge"]
            <= configured_budget["ridge"]
        ),
        "total": (
            timings["total"]
            <= configured_budget["total"]
        ),
    }

    return PerformanceReport(
        preprocessing_ms=round(
            timings["preprocessing"],
            4,
        ),
        blur_ms=round(
            timings["blur"],
            4,
        ),
        brightness_ms=round(
            timings["brightness"],
            4,
        ),
        glare_ms=round(
            timings["glare"],
            4,
        ),
        roi_ms=round(
            timings["roi"],
            4,
        ),
        ridge_ms=round(
            timings["ridge"],
            4,
        ),
        total_ms=round(
            timings["total"],
            4,
        ),
        budget_ms=configured_budget,
        within_budget=within_budget,
    )


def _serialize_metric_result(
    result: Any,
) -> dict[str, Any]:
    """Serialize supported metric result objects."""
    if hasattr(result, "to_dict"):
        serialized = result.to_dict()

        if isinstance(serialized, dict):
            return serialized

    raise QualityPipelineError(
        "Metric result cannot be serialized."
    )


if __name__ == "__main__":
    print(
        "RidgeLens unified quality pipeline is ready. "
        "Provide a fingerprint image to generate a complete assessment."
    )