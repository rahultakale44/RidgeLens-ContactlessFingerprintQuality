"""
Command-line fingerprint-quality assessment for RidgeLens.

Usage:

    python quality_assessment.py path/to/fingerprint.jpg

The command prints:

- Composite score
- Final decision
- Individual metrics
- Guidance
- Performance information
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from sys import exit as system_exit

from app.config import ConfigurationError
from app.image_processing import ImageProcessingError
from app.quality_pipeline import (
    QualityPipelineError,
    assess_image_path,
)
from app.ridge_analysis import RidgeAnalysisError
from app.roi_analysis import ROIAnalysisError
from app.quality_metrics import QualityMetricError


def build_argument_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Assess contactless fingerprint image quality "
            "using the RidgeLens pipeline."
        )
    )

    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to a JPG, JPEG, PNG, or BMP fingerprint image.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete result as formatted JSON.",
    )

    return parser


def print_human_readable_result(
    result: object,
) -> None:
    """Print concise terminal assessment."""
    print()
    print("=" * 62)
    print("RIDGELENS CONTACTLESS FINGERPRINT QUALITY ASSESSMENT")
    print("=" * 62)

    print(
        f"Decision       : "
        f"{'PASS' if result.passed else 'RETAKE'}"
    )
    print(
        f"Composite score: "
        f"{result.score:.2f} / 100"
    )
    print(
        f"Threshold      : "
        f"{result.composite.threshold:.2f}"
    )
    print(
        f"Status         : "
        f"{result.guidance.status_label}"
    )

    print()
    print("METRICS")
    print("-" * 62)

    for metric_name, metric in result.metrics.items():
        raw_value = getattr(
            metric,
            "raw_value",
            getattr(metric, "roi_fraction", None),
        )

        if raw_value is None:
            raw_value = getattr(
                metric,
                "ridge_score",
                0.0,
            )

        print(
            f"{metric_name.title():12} "
            f"{'PASS' if metric.passed else 'FAIL':5} "
            f"raw={float(raw_value):10.4f} "
            f"score={metric.normalized_score:6.4f}"
        )

    print()
    print("GUIDANCE")
    print("-" * 62)
    print(result.guidance.primary_message)

    for item in result.guidance.items:
        print(
            f"- {item.metric.upper()}: {item.action}"
        )

    print()
    print("PERFORMANCE")
    print("-" * 62)
    print(
        f"Total processing time: "
        f"{result.performance.total_ms:.3f} ms"
    )
    print(
        f"Within total budget  : "
        f"{result.performance.within_budget['total']}"
    )
    print("=" * 62)


def main() -> None:
    """Run the command-line application."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    try:
        assessment = assess_image_path(
            arguments.image_path
        )

        if arguments.json:
            print(
                json.dumps(
                    assessment.to_dict(),
                    indent=2,
                )
            )
        else:
            print_human_readable_result(
                assessment
            )

    except (
        ConfigurationError,
        ImageProcessingError,
        QualityMetricError,
        ROIAnalysisError,
        RidgeAnalysisError,
        QualityPipelineError,
    ) as error:
        print(
            f"RidgeLens assessment failed: {error}"
        )
        system_exit(1)


if __name__ == "__main__":
    main()