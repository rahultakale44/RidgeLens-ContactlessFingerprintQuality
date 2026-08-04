"""
Capture-guidance engine for RidgeLens.

The guidance engine converts metric failures into clear, prioritized,
user-facing corrective instructions.

Guidance is intentionally separated from quality algorithms so that:

- Metric calculations remain focused on measurement.
- User-facing messages can evolve independently.
- The Streamlit UI receives consistent recommendations.
- Batch reports can store machine-readable issue codes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


class GuidanceError(RuntimeError):
    """Raised when guidance generation receives invalid metric data."""


@dataclass(frozen=True)
class GuidanceItem:
    """One actionable capture recommendation."""

    metric: str
    priority: int
    title: str
    message: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable guidance item."""
        return asdict(self)


@dataclass(frozen=True)
class GuidanceReport:
    """Complete prioritized guidance output."""

    primary_message: str
    status_label: str
    items: tuple[GuidanceItem, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable guidance report."""
        return {
            "primary_message": self.primary_message,
            "status_label": self.status_label,
            "items": [
                item.to_dict()
                for item in self.items
            ],
        }


METRIC_PRIORITIES: dict[str, int] = {
    "roi": 1,
    "blur": 2,
    "brightness": 3,
    "glare": 4,
    "ridge": 5,
}


GUIDANCE_LIBRARY: dict[str, GuidanceItem] = {
    "roi": GuidanceItem(
        metric="roi",
        priority=METRIC_PRIORITIES["roi"],
        title="Position the fingertip correctly",
        message=(
            "The finger region is missing, incomplete, or too small "
            "for reliable biometric processing."
        ),
        action=(
            "Move the fingertip closer, keep it fully visible, and use "
            "a plain contrasting background."
        ),
    ),
    "blur": GuidanceItem(
        metric="blur",
        priority=METRIC_PRIORITIES["blur"],
        title="Improve focus and stability",
        message=(
            "The capture does not contain sufficient sharp edge detail."
        ),
        action=(
            "Hold the phone and finger steady, wait for autofocus, "
            "and capture again."
        ),
    ),
    "brightness": GuidanceItem(
        metric="brightness",
        priority=METRIC_PRIORITIES["brightness"],
        title="Correct the lighting level",
        message=(
            "The overall exposure is outside the acceptable brightness range."
        ),
        action=(
            "Use soft indirect lighting. Avoid very dark rooms and "
            "strong direct light."
        ),
    ),
    "glare": GuidanceItem(
        metric="glare",
        priority=METRIC_PRIORITIES["glare"],
        title="Remove reflective glare",
        message=(
            "Bright reflections may be hiding fingerprint ridge information."
        ),
        action=(
            "Change the finger or camera angle and avoid direct torchlight."
        ),
    ),
    "ridge": GuidanceItem(
        metric="ridge",
        priority=METRIC_PRIORITIES["ridge"],
        title="Improve ridge visibility",
        message=(
            "The ridge-and-valley pattern is not sufficiently clear."
        ),
        action=(
            "Move closer, improve focus, use soft side lighting, and "
            "keep the fingertip surface clean."
        ),
    ),
}


def build_guidance_report(
    metric_results: Mapping[str, Any],
    final_passed: bool,
    composite_score: float,
) -> GuidanceReport:
    """
    Build prioritized capture guidance from metric results.

    Each supplied metric result must expose a boolean `passed` attribute.

    Args:
        metric_results:
            Mapping containing blur, brightness, glare, ROI, and ridge results.
        final_passed:
            Final pipeline decision.
        composite_score:
            Composite quality score from 0 to 100.

    Returns:
        GuidanceReport with status, primary message, and ordered actions.

    Raises:
        GuidanceError:
            If metric results or composite score are invalid.
    """
    _validate_composite_score(composite_score)

    failed_metrics: list[str] = []

    for metric_name, metric_result in metric_results.items():
        if not hasattr(metric_result, "passed"):
            raise GuidanceError(
                f"Metric result '{metric_name}' does not expose a passed value."
            )

        if not bool(metric_result.passed):
            failed_metrics.append(metric_name)

    guidance_items = tuple(
        sorted(
            (
                GUIDANCE_LIBRARY[metric_name]
                for metric_name in failed_metrics
                if metric_name in GUIDANCE_LIBRARY
            ),
            key=lambda item: item.priority,
        )
    )

    if final_passed and not guidance_items:
        return GuidanceReport(
            primary_message=(
                "Good capture — ready for biometric processing."
            ),
            status_label="READY",
            items=(),
        )

    if guidance_items:
        first_issue = guidance_items[0]

        primary_message = (
            f"{first_issue.title}. {first_issue.action}"
        )
    else:
        primary_message = (
            "The composite score is below the acceptance threshold. "
            "Capture the fingerprint again under more stable conditions."
        )

    if composite_score < 30:
        status_label = "RETAKE REQUIRED"
    elif composite_score < 60:
        status_label = "QUALITY TOO LOW"
    else:
        status_label = "REVIEW CAPTURE"

    return GuidanceReport(
        primary_message=primary_message,
        status_label=status_label,
        items=guidance_items,
    )


def get_failed_metric_names(
    metric_results: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return failed metric names ordered by guidance priority."""
    failed_names: list[str] = []

    for metric_name, result in metric_results.items():
        if not hasattr(result, "passed"):
            raise GuidanceError(
                f"Metric result '{metric_name}' does not expose a passed value."
            )

        if not bool(result.passed):
            failed_names.append(metric_name)

    return tuple(
        sorted(
            failed_names,
            key=lambda name: METRIC_PRIORITIES.get(name, 999),
        )
    )


def _validate_composite_score(
    composite_score: float,
) -> None:
    """Validate final score range."""
    if not isinstance(composite_score, (int, float)):
        raise GuidanceError(
            "Composite score must be numeric."
        )

    if not 0 <= float(composite_score) <= 100:
        raise GuidanceError(
            "Composite score must be between 0 and 100."
        )


if __name__ == "__main__":
    print(
        "RidgeLens guidance engine is ready for pipeline integration."
    )