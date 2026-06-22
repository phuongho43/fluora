from pathlib import Path

import numpy as np
import tifffile


def load_images(imgs_dir: str | Path) -> tuple[list[float], list[np.ndarray]]:
    paths = sorted(Path(imgs_dir).glob("*.tiff"), key=lambda p: float(p.stem))
    if not paths:
        raise FileNotFoundError(f"No .tiff files found in {imgs_dir}")
    timestamps = [float(p.stem) for p in paths]
    images = [tifffile.imread(p) for p in paths]
    return timestamps, images
