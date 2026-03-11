"""Image I/O utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as BGR numpy array.

    Parameters
    ----------
    path : str | Path
        Path to the image file.

    Returns
    -------
    np.ndarray
        BGR image.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If cv2 fails to decode the image.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    return image


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Save an image to disk, creating parent directories as needed.

    Parameters
    ----------
    image : np.ndarray
        Image to save.
    path : str | Path
        Destination file path.

    Returns
    -------
    Path
        Resolved output path.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path
