import numpy as np
import pandas as pd


def extract_intensities(
    timestamps: list[float],
    images: list[np.ndarray],
    masks: list[np.ndarray],
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    # predict_overlap_dataframe returns frame/label as a MultiIndex, with
    # tree_id/track_id as columns. Promote the index to columns so we can
    # iterate rows uniformly.
    track_df = track_df.reset_index()
    records = []
    for _, row in track_df.iterrows():
        frame = int(row["frame"])
        label = int(row["label"])
        cell_id = int(row["track_id"])
        pixels = images[frame][masks[frame] == label]
        if pixels.size > 0:
            records.append({
                "cell_id": cell_id,
                "time_seconds": timestamps[frame],
                "mean_intensity": float(pixels.mean()),
            })
    return (
        pd.DataFrame(records)
        .sort_values(["cell_id", "time_seconds"])
        .reset_index(drop=True)
    )
