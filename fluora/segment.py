import numpy as np
from cellSAM import cellsam_pipeline


def segment_frames(images: list[np.ndarray], device: str = "cpu") -> list[np.ndarray]:
    masks = []
    for i, img in enumerate(images):
        print(f"  Segmenting frame {i + 1}/{len(images)}...", flush=True)
        mask = cellsam_pipeline(img, device=device, use_wsi=False, low_contrast_enhancement=False)
        masks.append(mask)
    return masks
