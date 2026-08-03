"""
Image loading, validation, resizing, and preprocessing utilities for RidgeLens.

This module provides a safe and reusable preprocessing layer for all downstream
fingerprint quality metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import get_default_config


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be safely loaded or processed."""


@dataclass(frozen=True)
class ImageMetadata:
    """Descriptive information about a loaded image."""

    source_name: str
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    channels: int
    original_megapixels: float
    processed_megapixels: float
    was_resized: bool
    file_size_bytes: int | None
    processing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return metadata in JSON-serializable dictionary form."""
        return asdict(self)


@dataclass(frozen=True)
class PreprocessedImage:
    """Container holding all reusable preprocessing outputs."""

    original_bgr: np.ndarray
    resized_bgr: np.ndarray
    grayscale: np.ndarray
    enhanced_grayscale: np.ndarray
    metadata: ImageMetadata

    def to_summary(self) -> dict[str, Any]:
        """Return a lightweight summary without raw image arrays."""
        return {
            "metadata": self.metadata.to_dict(),
            "shapes": {
                "original_bgr": list(self.original_bgr.shape),
                "resized_bgr": list(self.resized_bgr.shape),
                "grayscale": list(self.grayscale.shape),
                "enhanced_grayscale": list(
                    self.enhanced_grayscale.shape
                ),
            },
        }


def load_image_from_path(
    image_path: str | Path,
    config: dict[str, Any] | None = None,
) -> PreprocessedImage:
    """
    Load and preprocess an image from the local file system.

    Args:
        image_path:
            Path to a supported image file.
        config:
            Optional RidgeLens configuration. Default configuration is loaded
            when not supplied.

    Returns:
        A PreprocessedImage object containing original, resized, grayscale,
        enhanced grayscale, and metadata outputs.

    Raises:
        ImageProcessingError:
            If the path is missing, unsupported, unreadable, or invalid.
    """
    path = Path(image_path)

    if not path.exists():
        raise ImageProcessingError(
            f"Image file was not found: {path.resolve()}"
        )

    if not path.is_file():
        raise ImageProcessingError(
            f"Expected an image file but received a directory: {path}"
        )

    active_config = config or get_default_config()
    _validate_extension(path.suffix, active_config)

    try:
        image_bytes = path.read_bytes()
    except OSError as error:
        raise ImageProcessingError(
            f"Unable to read image file '{path}': {error}"
        ) from error

    return load_image_from_bytes(
        image_bytes=image_bytes,
        source_name=path.name,
        config=active_config,
    )


def load_image_from_file_object(
    file_object: BinaryIO,
    source_name: str,
    config: dict[str, Any] | None = None,
) -> PreprocessedImage:
    """
    Load and preprocess an image from a binary file-like object.

    This function is useful for Streamlit uploads.
    """
    if not hasattr(file_object, "read"):
        raise ImageProcessingError(
            "The supplied object does not provide a read() method."
        )

    try:
        image_bytes = file_object.read()
    except OSError as error:
        raise ImageProcessingError(
            f"Unable to read uploaded image '{source_name}': {error}"
        ) from error

    if hasattr(file_object, "seek"):
        try:
            file_object.seek(0)
        except OSError:
            pass

    return load_image_from_bytes(
        image_bytes=image_bytes,
        source_name=source_name,
        config=config,
    )


def load_image_from_bytes(
    image_bytes: bytes,
    source_name: str = "uploaded-image",
    config: dict[str, Any] | None = None,
) -> PreprocessedImage:
    """
    Decode and preprocess an image from raw bytes.

    PIL is used first to safely decode the image and respect EXIF orientation.
    The image is then converted into OpenCV BGR format.
    """
    start_time = perf_counter()
    active_config = config or get_default_config()

    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ImageProcessingError(
            "Image input must be provided as bytes."
        )

    if len(image_bytes) == 0:
        raise ImageProcessingError(
            f"Image '{source_name}' is empty."
        )

    suffix = Path(source_name).suffix

    if suffix:
        _validate_extension(suffix, active_config)

    try:
        with Image.open(BytesIO(image_bytes)) as pil_image:
            pil_image.verify()

        with Image.open(BytesIO(image_bytes)) as pil_image:
            oriented_image = ImageOps.exif_transpose(pil_image)
            rgb_image = oriented_image.convert("RGB")
            rgb_array = np.asarray(rgb_image)

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as error:
        raise ImageProcessingError(
            f"Unable to decode image '{source_name}'. "
            "The file may be corrupted or unsupported."
        ) from error

    if rgb_array.size == 0:
        raise ImageProcessingError(
            f"Decoded image '{source_name}' contains no pixel data."
        )

    original_bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    _validate_image_array(original_bgr)

    maximum_width = int(
        active_config["image"]["maximum_width"]
    )
    maximum_height = int(
        active_config["image"]["maximum_height"]
    )

    resized_bgr, was_resized = resize_preserving_aspect_ratio(
        original_bgr,
        maximum_width=maximum_width,
        maximum_height=maximum_height,
    )

    grayscale = convert_to_grayscale(resized_bgr)

    processing_config = active_config["processing"]
    apply_clahe = bool(
        processing_config.get("apply_clahe", True)
    )

    if apply_clahe:
        enhanced_grayscale = apply_clahe_enhancement(
            grayscale,
            clip_limit=float(
                processing_config["clahe_clip_limit"]
            ),
            grid_size=int(
                processing_config["clahe_grid_size"]
            ),
        )
    else:
        enhanced_grayscale = grayscale.copy()

    elapsed_ms = (perf_counter() - start_time) * 1000.0

    original_height, original_width = original_bgr.shape[:2]
    processed_height, processed_width = resized_bgr.shape[:2]

    metadata = ImageMetadata(
        source_name=source_name,
        original_width=original_width,
        original_height=original_height,
        processed_width=processed_width,
        processed_height=processed_height,
        channels=int(resized_bgr.shape[2]),
        original_megapixels=round(
            (original_width * original_height) / 1_000_000,
            4,
        ),
        processed_megapixels=round(
            (processed_width * processed_height) / 1_000_000,
            4,
        ),
        was_resized=was_resized,
        file_size_bytes=len(image_bytes),
        processing_time_ms=round(elapsed_ms, 3),
    )

    return PreprocessedImage(
        original_bgr=original_bgr,
        resized_bgr=resized_bgr,
        grayscale=grayscale,
        enhanced_grayscale=enhanced_grayscale,
        metadata=metadata,
    )


def resize_preserving_aspect_ratio(
    image_bgr: np.ndarray,
    maximum_width: int,
    maximum_height: int,
) -> tuple[np.ndarray, bool]:
    """
    Resize an image only when it exceeds configured dimensions.

    The original aspect ratio is preserved. Images smaller than the limits are
    copied without upscaling.
    """
    _validate_image_array(image_bgr)

    if maximum_width <= 0 or maximum_height <= 0:
        raise ImageProcessingError(
            "Maximum image dimensions must be positive."
        )

    height, width = image_bgr.shape[:2]

    scale = min(
        maximum_width / width,
        maximum_height / height,
        1.0,
    )

    if scale == 1.0:
        return image_bgr.copy(), False

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image_bgr,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return resized, True


def convert_to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a valid BGR image into an 8-bit grayscale image."""
    _validate_image_array(image_bgr)

    grayscale = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    if grayscale.dtype != np.uint8:
        grayscale = cv2.convertScaleAbs(grayscale)

    return grayscale


def apply_clahe_enhancement(
    grayscale: np.ndarray,
    clip_limit: float = 2.0,
    grid_size: int = 8,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    CLAHE improves local contrast while limiting excessive amplification of
    noise. This is useful for fingerprint ridges captured under uneven
    illumination.
    """
    _validate_grayscale_array(grayscale)

    if clip_limit <= 0:
        raise ImageProcessingError(
            "CLAHE clip limit must be greater than zero."
        )

    if grid_size <= 0:
        raise ImageProcessingError(
            "CLAHE grid size must be greater than zero."
        )

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(grid_size), int(grid_size)),
    )

    return clahe.apply(grayscale)


def create_preview_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image into RGB format for Streamlit or Matplotlib display.
    """
    _validate_image_array(image_bgr)

    return cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )


def _validate_extension(
    extension: str,
    config: dict[str, Any],
) -> None:
    """Validate an image extension against the configured allow-list."""
    normalized_extension = extension.lower()

    supported_extensions = {
        str(item).lower()
        for item in config["image"]["supported_extensions"]
    }

    if normalized_extension not in supported_extensions:
        supported = ", ".join(sorted(supported_extensions))

        raise ImageProcessingError(
            f"Unsupported image extension '{extension}'. "
            f"Supported extensions: {supported}"
        )


def _validate_image_array(image: np.ndarray) -> None:
    """Validate that an array represents a usable OpenCV BGR image."""
    if not isinstance(image, np.ndarray):
        raise ImageProcessingError(
            "Expected an image represented as a NumPy array."
        )

    if image.size == 0:
        raise ImageProcessingError(
            "Image array is empty."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageProcessingError(
            "Expected a three-channel BGR image."
        )

    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ImageProcessingError(
            "Image dimensions are too small for processing."
        )


def _validate_grayscale_array(grayscale: np.ndarray) -> None:
    """Validate a single-channel grayscale image."""
    if not isinstance(grayscale, np.ndarray):
        raise ImageProcessingError(
            "Expected grayscale data as a NumPy array."
        )

    if grayscale.size == 0:
        raise ImageProcessingError(
            "Grayscale image is empty."
        )

    if grayscale.ndim != 2:
        raise ImageProcessingError(
            "Expected a single-channel grayscale image."
        )

    if grayscale.dtype != np.uint8:
        raise ImageProcessingError(
            "Expected grayscale image with uint8 pixel values."
        )


if __name__ == "__main__":
    print(
        "RidgeLens image-processing module is ready. "
        "Provide an image through the pipeline to inspect metadata."
    )