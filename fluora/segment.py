import numpy as np
from cellpose import models
from skimage.exposure import rescale_intensity


def normalize_frame(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Per-frame percentile contrast stretch to [0, 1].

    Rescues low-SNR fluorescence frames (e.g. dim ddFP) whose signal occupies a
    tiny slice of the uint16 range, so segmentation sees consistent contrast
    frame-to-frame. Segmentation only -- intensities are still measured on the
    raw images downstream (see extract.py).
    """
    p_lo, p_hi = np.percentile(img, [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    return rescale_intensity(img, in_range=(p_lo, p_hi), out_range=(0.0, 1.0)).astype(np.float32)


def segment_frames(
    images: list[np.ndarray],
    device: str = "cpu",
    diameter: float = 150.0,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    norm_lo: float = 1.0,
    norm_hi: float = 99.0,
) -> list[np.ndarray]:
    """Per-frame instance masks via Cellpose-SAM (cpsam).

    Defaults are tuned for the 100x ddFP timelapses (large, confluent, often dim
    HEK293T): a fixed ``diameter`` (not auto) both matches the cell size and keeps
    segmentation consistent across frames for tracking. For the 10x transcription-
    reporter images the cells are ~10x smaller -- pass a smaller ``diameter``
    (~15-25 px). ``diameter=0`` falls back to Cellpose's auto estimate.
    """
    # Cellpose downloads the cpsam weights from a public host (no access token)
    # and caches them under ~/.cellpose. The model is loaded once and reused
    # across frames. eval() returns (masks, flows, styles); we keep the masks.
    model = models.CellposeModel(gpu=(device == "cuda"))
    diam = diameter or None
    masks = []
    for i, img in enumerate(images):
        print(f"  Segmenting frame {i + 1}/{len(images)}...", flush=True)
        disp = normalize_frame(img, norm_lo, norm_hi)
        mask = model.eval(
            disp,
            diameter=diam,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )[0]
        masks.append(mask.astype(np.int32))
    return masks
