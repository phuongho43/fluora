import os

import numpy as np
from cellSAM import cellsam_pipeline


def segment_frames(images: list[np.ndarray], device: str = "cpu") -> list[np.ndarray]:
    # cellsam_pipeline auto-selects the device via torch.cuda.is_available()
    # and exposes no device argument. Force CPU by hiding CUDA devices; leave
    # auto-detection in place when a GPU is requested.
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    masks = []
    for i, img in enumerate(images):
        print(f"  Segmenting frame {i + 1}/{len(images)}...", flush=True)
        mask = cellsam_pipeline(img, use_wsi=False, low_contrast_enhancement=False)
        masks.append(mask)
    return masks
