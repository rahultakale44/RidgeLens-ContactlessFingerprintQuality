# RidgeLens — Development and Learning Notes

This document tracks the engineering decisions, implementation details,
experiments, limitations, debugging lessons, and interview explanations
developed throughout the RidgeLens project.

---

## Project Objective

RidgeLens is a contactless fingerprint quality-assessment system for images
captured using a mobile camera.

The application evaluates whether a fingerprint image is suitable for
downstream biometric processing and provides clear recapture guidance when the
image does not meet quality requirements.

The planned quality pipeline checks:

1. Blur and sharpness
2. Brightness
3. Glare
4. Finger region completeness
5. Fingerprint ridge clarity
6. Composite quality score
7. Final pass or reject decision
8. User-facing recapture guidance

---

## Phase 1 — Project Foundation and Configuration

### Work completed

- Created a modular Python project structure.
- Created separate folders for application modules, tests, datasets, outputs,
  screenshots, plots, processed images, reports, and results.
- Created a Python virtual environment.
- Added project dependencies.
- Added an exact dependency lock file.
- Added centralized YAML configuration.
- Added configuration loading and validation.
- Added automated configuration tests.
- Protected confidential documents and private fingerprint photographs through
  `.gitignore`.
- Removed the generated virtual environment from Git tracking.

### Project structure

```text
RidgeLens-ContactlessFingerprintQuality/
|
|-- app/
|   |-- __init__.py
|   |-- config.py
|   |-- image_processing.py
|   |-- quality_metrics.py
|   |-- quality_pipeline.py
|   |-- guidance.py
|   `-- visualizations.py
|
|-- data/
|   |-- good/
|   |-- blurry/
|   |-- dark/
|   `-- glare/
|
|-- outputs/
|   |-- plots/
|   |-- processed/
|   |-- reports/
|   `-- results/
|
|-- screenshots/
|-- tests/
|
|-- config.yaml
|-- quality_assessment.py
|-- quality_app.py
|-- test_quality.py
|-- requirements.txt
|-- requirements-lock.txt
|-- PROJECT_NOTES.md
`-- README.md


---

## Phase 3 — Blur and Brightness Quality Metrics

### Work completed

- Added a reusable `MetricResult` data structure.
- Added Laplacian-variance blur detection.
- Added grayscale mean brightness assessment.
- Added normalized quality scores between 0.0 and 1.0.
- Added metric-specific PASS and FAIL decisions.
- Added user-facing corrective guidance.
- Added per-metric processing-time measurement.
- Added BGR and grayscale input support.
- Added metric serialization for Streamlit, JSON, and CSV output.
- Added automated tests using deterministic synthetic images.

### Blur detection

Blur is measured using variance of the Laplacian response.

```python
laplacian_response = cv2.Laplacian(
    grayscale,
    cv2.CV_64F,
)
blur_score = laplacian_response.var()