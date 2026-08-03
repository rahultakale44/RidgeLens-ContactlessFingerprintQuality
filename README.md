
# RidgeLens

<p align="center">
  <strong>Contactless Fingerprint Quality Assessment and Capture Guidance</strong>
</p>

<p align="center">
  A configurable image-quality pipeline for determining whether a mobile-camera
  fingerprint capture is suitable for biometric processing.
</p>

---

## Overview

RidgeLens evaluates contactless fingerprint photographs captured using a mobile
camera. It detects common image-quality problems before the image enters
downstream biometric stages such as segmentation, enhancement, feature
extraction, template generation, and matching.

Instead of returning only a pass or fail decision, RidgeLens is designed to
produce:

- Individual quality measurements
- Metric-level PASS or FAIL decisions
- A composite quality score from 0 to 100
- Actionable recapture guidance
- Processing-time measurements
- Diagnostic image visualizations

## Problem Statement

Contactless fingerprint captures are affected by challenges that are uncommon
in traditional contact scanners:

- Camera motion and focus errors
- Uneven or insufficient illumination
- Bright reflections and glare
- Incomplete finger coverage
- Weak or worn ridge patterns
- Background interference
- Variable camera distance
- Perspective distortion

A low-quality image can cause downstream biometric algorithms to fail or produce
unreliable features. RidgeLens acts as an early quality gate that rejects
unsuitable captures and tells the user how to improve them.

## Pipeline Architecture

```text
Input Fingerprint Image
          |
          v
Image Validation and Safe Decoding
          |
          v
EXIF Orientation Correction
          |
          v
Aspect-Ratio Preserving Resize
          |
          v
Grayscale and CLAHE Preprocessing
          |
          v
+--------------------------------------+
| Blur / Sharpness Assessment          |
| Brightness Assessment                |
| Glare Detection                      |
| Finger ROI Completeness              |
| Ridge Clarity Assessment             |
+--------------------------------------+
          |
          v
Metric Normalization
          |
          v
Weighted Composite Score
          |
          v
Quality Decision and Capture Guidance