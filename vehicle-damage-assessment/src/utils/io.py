"""Image I/O utilities.

Thin wrappers around cv2 for consistent error handling and path resolution.
"""

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
        If the image file does not exist.
    ValueError
        If cv2 fails to decode the image (corrupt or unsupported format).
    """
    # TODO:
    #   1. resolve and validate path exists
    #   2. cv2.imread
    #   3. check result is not None
    raise NotImplementedError


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Save an image to disk.

    Creates parent directories if they don't exist.

    Parameters
    ----------
    image : np.ndarray
        Image to save (BGR or grayscale).
    path : str | Path
        Destination file path (extension determines format).

    Returns
    -------
    Path
        Resolved path where the image was saved.
    """
    # TODO:
    #   1. resolve path, mkdir parents
    #   2. cv2.imwrite
    #   3. return resolved path
    raise NotImplementedError
