import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def estimate_background(img_f: np.ndarray, fg_mask: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth low-frequency background, interpolated under the cells.

    Uses the Cellpose foreground mask (we know exactly where cells are) instead of
    guessing foreground by thresholding. Background is estimated by *normalized
    convolution*: Gaussian-smooth the background-only signal and divide by the
    Gaussian-smoothed background weight. This fills the holes left by cells by
    borrowing from nearby background, and degrades gracefully in confluent fields
    where background pixels are scarce.
    """
    w = (~fg_mask).astype(np.float32)          # 1 on background, 0 on cells
    num = gaussian_filter(img_f * w, sigma)
    den = gaussian_filter(w, sigma)
    return num / np.maximum(den, 1e-6)


def subtract_background(img: np.ndarray, fg_mask: np.ndarray, sigma: float = 50.0) -> np.ndarray:
    """Return the frame with its mask-aware background surface subtracted (clipped >=0)."""
    img_f = img.astype(np.float32)
    sub = img_f - estimate_background(img_f, fg_mask, sigma)
    np.clip(sub, 0.0, None, out=sub)
    return sub


def extract_intensities(
    timestamps: list[float],
    images: list[np.ndarray],
    masks: list[np.ndarray],
    track_df: pd.DataFrame,
    bgd_sigma: float = 50.0,
    f0_n: int = 5,
    min_frames: int = 1,
) -> pd.DataFrame:
    """Per-tracked-cell fluorescence traces.

    For each frame, a mask-aware background surface (from non-cell pixels) is
    subtracted before measuring each cell's mean intensity. Output columns:
      cell_id, time_seconds, mean_intensity (raw), mean_intensity_sub
      (background-subtracted), dF_F0 (per-cell fold-change over its first
      ``f0_n`` frames -- the biosensor readout).

    ``min_frames`` drops cells tracked for fewer than that many frames (short
    fragments that add noise to the averaged trace); set to e.g. the movie length
    to keep only fully-tracked cells.
    """
    # Background-subtract each frame once, using that frame's union of cell masks.
    subs = [subtract_background(images[t], masks[t] > 0, bgd_sigma) for t in range(len(images))]

    # predict_overlap_dataframe returns frame/label as a MultiIndex, with
    # tree_id/track_id as columns. Promote the index to columns so we can
    # iterate rows uniformly.
    track_df = track_df.reset_index()
    records = []
    for _, row in track_df.iterrows():
        frame = int(row["frame"])
        label = int(row["label"])
        cell_id = int(row["track_id"])
        sel = masks[frame] == label
        if sel.any():
            ys, xs = np.nonzero(sel)
            records.append({
                "cell_id": cell_id,
                "time_seconds": timestamps[frame],
                "mean_intensity": float(images[frame][sel].mean()),
                "mean_intensity_sub": float(subs[frame][sel].mean()),
                "area": int(sel.sum()),
                "centroid_x": float(xs.mean()),
                "centroid_y": float(ys.mean()),
            })

    traces = (
        pd.DataFrame(records)
        .sort_values(["cell_id", "time_seconds"])
        .reset_index(drop=True)
    )

    if min_frames > 1:
        keep = traces.groupby("cell_id")["cell_id"].transform("size") >= min_frames
        traces = traces[keep].reset_index(drop=True)

    # dF/F0 per cell on the background-subtracted signal (F0 = mean of the cell's
    # first f0_n frames = pre-stimulus baseline). transform keeps every row/column
    # aligned (groupby.apply would drop the cell_id grouping column).
    f0 = traces.groupby("cell_id")["mean_intensity_sub"].transform(
        lambda s: s.iloc[:f0_n].mean()
    )
    traces["dF_F0"] = np.where(f0 > 0, (traces["mean_intensity_sub"] - f0) / f0, np.nan)
    return traces
