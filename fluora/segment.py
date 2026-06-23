import numpy as np
from cellpose import models


def segment_frames(images: list[np.ndarray], device: str = "cpu") -> list[np.ndarray]:
    # Cellpose downloads the cpsam weights from a public host (no access token)
    # and caches them under ~/.cellpose. The model is loaded once and reused
    # across frames. eval() returns (masks, flows, styles); we keep the masks.
    model = models.CellposeModel(gpu=(device == "cuda"))
    masks = []
    for i, img in enumerate(images):
        print(f"  Segmenting frame {i + 1}/{len(images)}...", flush=True)
        mask = model.eval(img)[0]
        masks.append(mask.astype(np.int32))
    return masks
