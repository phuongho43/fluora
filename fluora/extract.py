import numpy as np
import pandas as pd


def extract_intensities(
    timestamps: list[float],
    images: list[np.ndarray],
    masks: list[np.ndarray],
    track_df: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    for _, row in track_df.iterrows():
        frame = int(row["frame"])
        label = int(row["label"])
        cell_id = int(row["tree_id"])
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
