"""
RidgeLens Streamlit dashboard.

Run with:

    python -m streamlit run quality_app.py
"""

from __future__ import annotations

import json
from copy import deepcopy
from html import escape
from typing import Any

import pandas as pd
import streamlit as st

from app.config import (
    ConfigurationError,
    get_default_config,
)
from app.image_processing import ImageProcessingError
from app.quality_metrics import QualityMetricError
from app.quality_pipeline import (
    QualityAssessment,
    QualityPipelineError,
    assess_uploaded_file,
)
from app.ridge_analysis import RidgeAnalysisError
from app.roi_analysis import ROIAnalysisError
from app.visualizations import (
    build_metric_rows,
    convert_bgr_to_rgb,
    convert_grayscale_to_rgb,
    create_glare_overlay,
    create_mask_preview,
    create_ridge_diagnostic,
    create_roi_diagnostic,
)


st.set_page_config(
    page_title="RidgeLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --surface: rgba(16, 24, 40, 0.78);
        --surface-soft: rgba(255, 255, 255, 0.045);
        --border: rgba(148, 163, 184, 0.20);
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --cyan: #22d3ee;
        --violet: #8b5cf6;
        --green: #22c55e;
        --red: #f43f5e;
        --amber: #f59e0b;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(34, 211, 238, 0.11),
                transparent 30%
            ),
            radial-gradient(
                circle at 85% 15%,
                rgba(139, 92, 246, 0.14),
                transparent 32%
            ),
            linear-gradient(
                145deg,
                #050816 0%,
                #081020 52%,
                #050713 100%
            );
        color: var(--text-main);
    }

    .block-container {
        max-width: 1480px;
        padding-top: 1.4rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                rgba(8, 15, 31, 0.98),
                rgba(6, 10, 23, 0.98)
            );
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.4rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.3rem;
        border-radius: 28px;
        background:
            linear-gradient(
                135deg,
                rgba(34, 211, 238, 0.12),
                rgba(139, 92, 246, 0.13)
            ),
            rgba(10, 18, 35, 0.86);
        border: 1px solid rgba(125, 211, 252, 0.20);
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.32);
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        padding: 0.42rem 0.85rem;
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.09);
        border: 1px solid rgba(34, 211, 238, 0.22);
        color: #a5f3fc;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
    }

    .hero-title {
        margin-top: 1rem;
        font-size: clamp(2.5rem, 5vw, 4.6rem);
        line-height: 1;
        font-weight: 900;
        letter-spacing: -0.055em;
        background:
            linear-gradient(
                90deg,
                #f8fafc 0%,
                #a5f3fc 48%,
                #c4b5fd 100%
            );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        max-width: 900px;
        margin-top: 1rem;
        color: #cbd5e1;
        font-size: 1.03rem;
        line-height: 1.75;
    }

    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1.3rem;
    }

    .hero-chip {
        padding: 0.55rem 0.85rem;
        border-radius: 14px;
        color: #dbeafe;
        background: rgba(255, 255, 255, 0.055);
        border: 1px solid var(--border);
        font-size: 0.82rem;
    }

    .section-heading {
        margin: 1.5rem 0 0.8rem 0;
        font-size: 1.3rem;
        font-weight: 850;
        letter-spacing: -0.025em;
    }

    .native-panel {
        padding: 1rem;
        border-radius: 20px;
        background: var(--surface);
        border: 1px solid var(--border);
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.16);
    }

    .status-ready {
        padding: 0.95rem 1rem;
        border-radius: 17px;
        background: rgba(34, 197, 94, 0.10);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: #bbf7d0;
        font-weight: 750;
    }

    .status-retake {
        padding: 0.95rem 1rem;
        border-radius: 17px;
        background: rgba(244, 63, 94, 0.10);
        border: 1px solid rgba(244, 63, 94, 0.25);
        color: #fecdd3;
        font-weight: 750;
    }

    .metric-pass-box {
        padding: 0.85rem;
        border-radius: 17px;
        background: rgba(34, 197, 94, 0.055);
        border: 1px solid rgba(34, 197, 94, 0.22);
        min-height: 260px;
    }

    .metric-fail-box {
        padding: 0.85rem;
        border-radius: 17px;
        background: rgba(244, 63, 94, 0.055);
        border: 1px solid rgba(244, 63, 94, 0.24);
        min-height: 260px;
    }

    .metric-title {
        font-size: 1rem;
        font-weight: 850;
        margin-bottom: 0.3rem;
    }

    .metric-badge-pass {
        display: inline-block;
        margin-bottom: 0.8rem;
        padding: 0.23rem 0.55rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.16);
        color: #86efac;
        font-size: 0.70rem;
        font-weight: 900;
        letter-spacing: 0.08em;
    }

    .metric-badge-fail {
        display: inline-block;
        margin-bottom: 0.8rem;
        padding: 0.23rem 0.55rem;
        border-radius: 999px;
        background: rgba(244, 63, 94, 0.16);
        color: #fda4af;
        font-size: 0.70rem;
        font-weight: 900;
        letter-spacing: 0.08em;
    }

    .metric-raw-label {
        color: var(--text-muted);
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .metric-raw-value {
        margin: 0.2rem 0 0.65rem 0;
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: -0.04em;
    }

    .metric-message {
        margin-top: 0.75rem;
        color: #cbd5e1;
        font-size: 0.78rem;
        line-height: 1.5;
    }

    .score-heading {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
    }

    .score-value-pass {
        margin-top: 0.35rem;
        text-align: center;
        color: #86efac;
        font-size: 4.2rem;
        font-weight: 950;
        line-height: 1;
        letter-spacing: -0.07em;
    }

    .score-value-fail {
        margin-top: 0.35rem;
        text-align: center;
        color: #fb7185;
        font-size: 4.2rem;
        font-weight: 950;
        line-height: 1;
        letter-spacing: -0.07em;
    }

    .score-denominator {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.85rem;
    }

    .score-decision-pass {
        width: fit-content;
        margin: 1rem auto 0 auto;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.14);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.25);
        font-size: 0.74rem;
        font-weight: 900;
        letter-spacing: 0.10em;
    }

    .score-decision-fail {
        width: fit-content;
        margin: 1rem auto 0 auto;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        background: rgba(244, 63, 94, 0.14);
        color: #fda4af;
        border: 1px solid rgba(244, 63, 94, 0.25);
        font-size: 0.74rem;
        font-weight: 900;
        letter-spacing: 0.10em;
    }

    .metadata-grid {
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.8rem;
    }

    .metadata-card {
        padding: 0.9rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid var(--border);
    }

    .metadata-label {
        color: var(--text-muted);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metadata-value {
        margin-top: 0.3rem;
        font-weight: 800;
        font-size: 0.91rem;
        word-break: break-word;
    }

    div[data-testid="stFileUploader"] {
        padding: 0.7rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.035);
        border: 1px dashed rgba(125, 211, 252, 0.32);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        padding: 0.35rem;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.035);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
    }

    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
        color: var(--text-muted);
        font-size: 0.75rem;
        text-align: center;
    }

    @media (max-width: 900px) {
        .hero {
            padding: 1.4rem;
        }

        .metric-pass-box,
        .metric-fail-box {
            min-height: auto;
        }
    }
</style>
"""


METRIC_NAMES: dict[str, str] = {
    "blur": "Sharpness",
    "brightness": "Brightness",
    "glare": "Glare",
    "roi": "Finger Coverage",
    "ridge": "Ridge Clarity",
}


def render_header() -> None:
    """Render the RidgeLens product header."""
    st.markdown(
        """
<div class="hero">
<div class="hero-kicker">◉ Contactless biometric quality gate</div>
<div class="hero-title">RidgeLens</div>
<div class="hero-description">
Evaluate mobile-camera fingerprint captures before downstream biometric
processing. RidgeLens measures sharpness, exposure, glare, finger coverage,
and ridge clarity, then returns an explainable quality decision with
actionable recapture guidance.
</div>
<div class="hero-meta">
<span class="hero-chip">5 quality metrics</span>
<span class="hero-chip">Composite score</span>
<span class="hero-chip">Diagnostic masks</span>
<span class="hero-chip">Performance tracking</span>
<span class="hero-chip">Privacy-first local processing</span>
</div>
</div>
""",
        unsafe_allow_html=True,
    )


def build_runtime_config(
    base_config: dict[str, Any],
) -> dict[str, Any]:
    """Create runtime configuration from sidebar controls."""
    runtime_config = deepcopy(base_config)

    st.sidebar.markdown("## RidgeLens Controls")
    st.sidebar.caption(
        "Tune quality thresholds before analysing the capture."
    )

    st.sidebar.markdown("### Quality thresholds")

    runtime_config[
        "thresholds"
    ]["blur"]["minimum_score"] = st.sidebar.slider(
        "Minimum sharpness",
        min_value=0.0,
        max_value=200.0,
        value=float(
            base_config["thresholds"]["blur"]["minimum_score"]
        ),
        step=1.0,
    )

    minimum_brightness = st.sidebar.slider(
        "Minimum brightness",
        min_value=0.0,
        max_value=150.0,
        value=float(
            base_config[
                "thresholds"
            ]["brightness"]["minimum_value"]
        ),
        step=1.0,
    )

    maximum_brightness = st.sidebar.slider(
        "Maximum brightness",
        min_value=151.0,
        max_value=255.0,
        value=float(
            base_config[
                "thresholds"
            ]["brightness"]["maximum_value"]
        ),
        step=1.0,
    )

    runtime_config[
        "thresholds"
    ]["brightness"]["minimum_value"] = minimum_brightness

    runtime_config[
        "thresholds"
    ]["brightness"]["maximum_value"] = maximum_brightness

    runtime_config[
        "thresholds"
    ]["glare"]["pixel_threshold"] = st.sidebar.slider(
        "Glare pixel threshold",
        min_value=180,
        max_value=254,
        value=int(
            base_config[
                "thresholds"
            ]["glare"]["pixel_threshold"]
        ),
        step=1,
    )

    glare_percentage = st.sidebar.slider(
        "Maximum glare coverage (%)",
        min_value=0.5,
        max_value=30.0,
        value=float(
            base_config[
                "thresholds"
            ]["glare"]["maximum_fraction"]
        )
        * 100.0,
        step=0.5,
    )

    runtime_config[
        "thresholds"
    ]["glare"]["maximum_fraction"] = glare_percentage / 100.0

    roi_percentage = st.sidebar.slider(
        "Minimum finger coverage (%)",
        min_value=1.0,
        max_value=70.0,
        value=float(
            base_config[
                "thresholds"
            ]["roi"]["minimum_fraction"]
        )
        * 100.0,
        step=1.0,
    )

    runtime_config[
        "thresholds"
    ]["roi"]["minimum_fraction"] = roi_percentage / 100.0

    runtime_config[
        "thresholds"
    ]["ridge"]["minimum_score"] = st.sidebar.slider(
        "Minimum ridge clarity",
        min_value=1.0,
        max_value=200.0,
        value=float(
            base_config[
                "thresholds"
            ]["ridge"]["minimum_score"]
        ),
        step=1.0,
    )

    runtime_config[
        "thresholds"
    ]["composite"]["minimum_score"] = st.sidebar.slider(
        "Composite pass score",
        min_value=1.0,
        max_value=100.0,
        value=float(
            base_config[
                "thresholds"
            ]["composite"]["minimum_score"]
        ),
        step=1.0,
    )

    st.sidebar.markdown("### Processing")

    runtime_config[
        "processing"
    ]["apply_clahe"] = st.sidebar.toggle(
        "Enable CLAHE enhancement",
        value=bool(
            base_config[
                "processing"
            ]["apply_clahe"]
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Threshold changes affect only the current analysis."
    )

    return runtime_config


def get_capture_input() -> Any | None:
    """Render upload and camera controls."""
    st.markdown(
        '<div class="section-heading">Capture input</div>',
        unsafe_allow_html=True,
    )

    upload_tab, camera_tab = st.tabs(
        [
            "Upload fingerprint",
            "Use camera",
        ]
    )

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a contactless fingerprint image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "bmp",
            ],
            help=(
                "Use a focused close-up fingertip image with a plain "
                "background and soft indirect lighting."
            ),
        )

    with camera_tab:
        st.info(
            "Bring only the fingertip close to the camera. "
            "Avoid including the face or room background."
        )

        camera_file = st.camera_input(
            "Capture fingertip photograph"
        )

    if camera_file is not None:
        return camera_file

    return uploaded_file


def render_score_panel(
    assessment: QualityAssessment,
) -> None:
    """Render a stable score panel without SVG."""
    score_class = (
        "score-value-pass"
        if assessment.passed
        else "score-value-fail"
    )

    decision_class = (
        "score-decision-pass"
        if assessment.passed
        else "score-decision-fail"
    )

    decision = (
        "READY"
        if assessment.passed
        else "RETAKE"
    )

    with st.container(border=True):
        st.markdown(
            '<div class="score-heading">Composite quality score</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="{score_class}">{assessment.score:.0f}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="score-denominator">out of 100</div>',
            unsafe_allow_html=True,
        )

        st.progress(
            min(
                max(
                    float(assessment.score) / 100.0,
                    0.0,
                ),
                1.0,
            )
        )

        st.markdown(
            f'<div class="{decision_class}">{decision}</div>',
            unsafe_allow_html=True,
        )

        st.caption(
            f"Pass threshold: {assessment.composite.threshold:.0f}"
        )


def render_metric_block(
    metric_name: str,
    result: Any,
) -> None:
    """Render one metric using native Streamlit controls."""
    passed = bool(result.passed)

    box_class = (
        "metric-pass-box"
        if passed
        else "metric-fail-box"
    )

    badge_class = (
        "metric-badge-pass"
        if passed
        else "metric-badge-fail"
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
            float(result.normalized_score) * 100.0,
        ),
    )

    raw_value = format_metric_value(
        metric_name,
        result,
    )

    title = METRIC_NAMES[metric_name]
    safe_message = escape(str(result.message))

    st.markdown(
        f"""
<div class="{box_class}">
<div class="metric-title">{title}</div>
<div class="{badge_class}">{status}</div>
<div class="metric-raw-label">Measured value</div>
<div class="metric-raw-value">{raw_value}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.progress(
        quality_percentage / 100.0,
        text=f"Quality score: {quality_percentage:.1f}%",
    )

    st.caption(
        f"{result.processing_time_ms:.2f} ms"
    )

    st.markdown(
        f'<div class="metric-message">{safe_message}</div>',
        unsafe_allow_html=True,
    )


def format_metric_value(
    metric_name: str,
    result: Any,
) -> str:
    """Return a readable raw metric value."""
    if metric_name == "roi":
        return f"{float(result.roi_fraction) * 100.0:.2f}%"

    if metric_name == "ridge":
        return f"{float(result.ridge_score):.2f}"

    if metric_name == "glare":
        return f"{float(result.raw_value) * 100.0:.2f}%"

    return f"{float(result.raw_value):.2f}"


def render_assessment(
    assessment: QualityAssessment,
) -> None:
    """Render complete assessment."""
    score_column, preview_column = st.columns(
        [0.31, 0.69],
        gap="large",
    )

    with score_column:
        render_score_panel(
            assessment
        )

    with preview_column:
        st.markdown(
            '<div class="section-heading">Analysed capture</div>',
            unsafe_allow_html=True,
        )

        st.image(
            convert_bgr_to_rgb(
                assessment.diagnostics["resized_bgr"]
            ),
            caption=assessment.metadata["source_name"],
            use_container_width=True,
        )

        status_class = (
            "status-ready"
            if assessment.passed
            else "status-retake"
        )

        st.markdown(
            f"""
<div class="{status_class}">
<strong>{escape(assessment.guidance.status_label)}</strong><br>
{escape(assessment.guidance.primary_message)}
</div>
""",
            unsafe_allow_html=True,
        )

    render_metric_grid(
        assessment
    )

    if assessment.guidance.items:
        render_guidance(
            assessment
        )

    render_diagnostics(
        assessment
    )
    render_metric_table(
        assessment
    )
    render_performance(
        assessment
    )
    render_metadata(
        assessment
    )
    render_downloads(
        assessment
    )


def render_metric_grid(
    assessment: QualityAssessment,
) -> None:
    """Render metrics in responsive rows instead of one five-card row."""
    st.markdown(
        '<div class="section-heading">Quality metrics</div>',
        unsafe_allow_html=True,
    )

    first_row = st.columns(
        3,
        gap="medium",
    )

    first_metrics = (
        "blur",
        "brightness",
        "glare",
    )

    for column, metric_name in zip(
        first_row,
        first_metrics,
    ):
        with column:
            render_metric_block(
                metric_name,
                assessment.metrics[metric_name],
            )

    second_row = st.columns(
        2,
        gap="medium",
    )

    second_metrics = (
        "roi",
        "ridge",
    )

    for column, metric_name in zip(
        second_row,
        second_metrics,
    ):
        with column:
            render_metric_block(
                metric_name,
                assessment.metrics[metric_name],
            )


def render_guidance(
    assessment: QualityAssessment,
) -> None:
    """Render prioritized recapture instructions."""
    st.markdown(
        '<div class="section-heading">Retake guidance</div>',
        unsafe_allow_html=True,
    )

    for index, item in enumerate(
        assessment.guidance.items
    ):
        with st.expander(
            f"{item.metric.upper()} — {item.title}",
            expanded=index == 0,
        ):
            st.write(
                item.message
            )
            st.info(
                item.action
            )


def render_diagnostics(
    assessment: QualityAssessment,
) -> None:
    """Render diagnostic image views."""
    st.markdown(
        '<div class="section-heading">Diagnostic views</div>',
        unsafe_allow_html=True,
    )

    (
        original_tab,
        grayscale_tab,
        enhanced_tab,
        glare_tab,
        roi_tab,
        ridge_tab,
        masks_tab,
    ) = st.tabs(
        [
            "Original",
            "Grayscale",
            "Enhanced",
            "Glare",
            "Finger ROI",
            "Ridge response",
            "Masks",
        ]
    )

    resized_bgr = assessment.diagnostics["resized_bgr"]
    finger_mask = assessment.diagnostics["finger_mask"]

    with original_tab:
        st.image(
            convert_bgr_to_rgb(
                resized_bgr
            ),
            use_container_width=True,
        )

    with grayscale_tab:
        st.image(
            convert_grayscale_to_rgb(
                assessment.diagnostics["grayscale"]
            ),
            use_container_width=True,
        )

    with enhanced_tab:
        st.image(
            convert_grayscale_to_rgb(
                assessment.diagnostics[
                    "enhanced_grayscale"
                ]
            ),
            use_container_width=True,
        )

    with glare_tab:
        glare_overlay = create_glare_overlay(
            image_bgr=resized_bgr,
            glare_mask=assessment.diagnostics[
                "glare_mask"
            ],
        )

        st.image(
            convert_bgr_to_rgb(
                glare_overlay
            ),
            caption=(
                "Red regions represent pixels above the glare threshold."
            ),
            use_container_width=True,
        )

    with roi_tab:
        roi_overlay = create_roi_diagnostic(
            image_bgr=resized_bgr,
            finger_mask=finger_mask,
            bounding_box=assessment.diagnostics[
                "roi_bounding_box"
            ],
        )

        st.image(
            convert_bgr_to_rgb(
                roi_overlay
            ),
            caption=(
                "Detected candidate region and ROI bounding box."
            ),
            use_container_width=True,
        )

    with ridge_tab:
        ridge_overlay = create_ridge_diagnostic(
            image_bgr=resized_bgr,
            response_visualization=assessment.diagnostics[
                "ridge_response"
            ],
            finger_mask=finger_mask,
        )

        st.image(
            convert_bgr_to_rgb(
                ridge_overlay
            ),
            caption=(
                "Gabor response inside the selected analysis region."
            ),
            use_container_width=True,
        )

    with masks_tab:
        candidate_column, final_column = st.columns(
            2
        )

        with candidate_column:
            st.image(
                create_mask_preview(
                    assessment.diagnostics[
                        "roi_candidate_mask"
                    ]
                ),
                caption="Initial ROI candidate mask",
                use_container_width=True,
            )

        with final_column:
            st.image(
                create_mask_preview(
                    finger_mask
                ),
                caption="Cleaned final finger mask",
                use_container_width=True,
            )


def render_metric_table(
    assessment: QualityAssessment,
) -> None:
    """Render detailed metric table."""
    st.markdown(
        '<div class="section-heading">Metric breakdown</div>',
        unsafe_allow_html=True,
    )

    rows = build_metric_rows(
        metric_results=assessment.metrics,
        weights=assessment.composite.weights,
        contributions=assessment.composite.contributions,
    )

    dataframe = pd.DataFrame(
        rows
    )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


def render_performance(
    assessment: QualityAssessment,
) -> None:
    """Render performance diagnostics in two rows."""
    st.markdown(
        '<div class="section-heading">Performance diagnostics</div>',
        unsafe_allow_html=True,
    )

    timing_values = [
        (
            "Blur",
            assessment.performance.blur_ms,
            assessment.performance.within_budget["blur"],
        ),
        (
            "Brightness",
            assessment.performance.brightness_ms,
            assessment.performance.within_budget["brightness"],
        ),
        (
            "Glare",
            assessment.performance.glare_ms,
            assessment.performance.within_budget["glare"],
        ),
        (
            "ROI",
            assessment.performance.roi_ms,
            assessment.performance.within_budget["roi"],
        ),
        (
            "Ridge",
            assessment.performance.ridge_ms,
            assessment.performance.within_budget["ridge"],
        ),
        (
            "Total",
            assessment.performance.total_ms,
            assessment.performance.within_budget["total"],
        ),
    ]

    first_row = st.columns(
        3
    )

    for column, item in zip(
        first_row,
        timing_values[:3],
    ):
        render_timing_metric(
            column,
            *item,
        )

    second_row = st.columns(
        3
    )

    for column, item in zip(
        second_row,
        timing_values[3:],
    ):
        render_timing_metric(
            column,
            *item,
        )


def render_timing_metric(
    column: Any,
    label: str,
    value: float,
    within_budget: bool,
) -> None:
    """Render one performance value."""
    with column:
        st.metric(
            label=label,
            value=f"{value:.2f} ms",
            delta=(
                "Within budget"
                if within_budget
                else "Over budget"
            ),
            delta_color=(
                "normal"
                if within_budget
                else "inverse"
            ),
        )


def render_metadata(
    assessment: QualityAssessment,
) -> None:
    """Render file and capture metadata."""
    st.markdown(
        '<div class="section-heading">Capture metadata</div>',
        unsafe_allow_html=True,
    )

    metadata = assessment.metadata

    file_size_bytes = metadata.get(
        "file_size_bytes"
    )

    metadata_items = [
        (
            "Source",
            metadata["source_name"],
        ),
        (
            "Original size",
            (
                f"{metadata['original_width']} × "
                f"{metadata['original_height']}"
            ),
        ),
        (
            "Processed size",
            (
                f"{metadata['processed_width']} × "
                f"{metadata['processed_height']}"
            ),
        ),
        (
            "File size",
            (
                f"{file_size_bytes / 1024:.1f} KB"
                if file_size_bytes is not None
                else "Unknown"
            ),
        ),
        (
            "Resized",
            (
                "Yes"
                if metadata["was_resized"]
                else "No"
            ),
        ),
        (
            "Failed metrics",
            (
                ", ".join(
                    metadata["failed_metrics"]
                )
                if metadata["failed_metrics"]
                else "None"
            ),
        ),
    ]

    metadata_html = '<div class="metadata-grid">'

    for label, value in metadata_items:
        metadata_html += (
            '<div class="metadata-card">'
            f'<div class="metadata-label">{escape(str(label))}</div>'
            f'<div class="metadata-value">{escape(str(value))}</div>'
            "</div>"
        )

    metadata_html += "</div>"

    st.markdown(
        metadata_html,
        unsafe_allow_html=True,
    )


def render_downloads(
    assessment: QualityAssessment,
) -> None:
    """Render JSON download."""
    st.markdown(
        '<div class="section-heading">Export assessment</div>',
        unsafe_allow_html=True,
    )

    json_content = json.dumps(
        assessment.to_dict(),
        indent=2,
    )

    st.download_button(
        label="Download JSON assessment",
        data=json_content,
        file_name="ridgelens-assessment.json",
        mime="application/json",
        use_container_width=True,
    )


def render_empty_state() -> None:
    """Render capture instructions before input."""
    st.markdown(
        '<div class="section-heading">Capture checklist</div>',
        unsafe_allow_html=True,
    )

    checklist_columns = st.columns(
        2
    )

    checklist_items = [
        (
            "Distance",
            "Keep only the fingertip close and clearly visible.",
        ),
        (
            "Lighting",
            "Use soft indirect light without strong reflections.",
        ),
        (
            "Focus",
            "Hold the camera and finger completely steady.",
        ),
        (
            "Background",
            "Use a plain background with strong contrast.",
        ),
    ]

    for index, (
        label,
        description,
    ) in enumerate(
        checklist_items
    ):
        with checklist_columns[index % 2]:
            with st.container(border=True):
                st.caption(
                    label.upper()
                )
                st.write(
                    f"**{description}**"
                )


def main() -> None:
    """Run RidgeLens."""
    st.markdown(
        CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    render_header()

    try:
        base_config = get_default_config()

        runtime_config = build_runtime_config(
            base_config
        )

        capture = get_capture_input()

        if capture is None:
            render_empty_state()
        else:
            with st.spinner(
                "Analysing fingerprint quality..."
            ):
                assessment = assess_uploaded_file(
                    file_object=capture,
                    source_name=capture.name,
                    config=runtime_config,
                )

            render_assessment(
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
        st.error(
            f"RidgeLens could not analyse this capture: {error}"
        )

        st.info(
            "Upload a supported JPG, JPEG, PNG, or BMP image "
            "with one clearly visible fingertip."
        )

    st.markdown(
        """
<div class="footer">
RidgeLens processes captures locally in the active session.
Fingerprint photographs are not permanently stored by the application.
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()