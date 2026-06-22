import numpy as np
import pandas as pd
from laptrack import OverLapTrack


def track_cells(masks: list[np.ndarray]) -> pd.DataFrame:
    labels = np.stack(masks)  # (T, H, W)
    olt = OverLapTrack(gap_closing_max_frame_count=2)
    track_df, _, _ = olt.predict_overlap_dataframe(labels)
    return track_df
