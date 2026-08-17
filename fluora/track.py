import numpy as np
import pandas as pd
from laptrack import OverLapTrack


def track_cells(
    masks: list[np.ndarray],
    gap_closing_max_frame_count: int = 8,
    cutoff: float = 0.9,
    gap_closing_cutoff: float = 0.9,
) -> pd.DataFrame:
    """Link per-frame instance masks into tracks by mask overlap.

    ``OverLapTrack`` scores a candidate link by ``distance = 1 - ratio_2`` (the
    fraction of the later mask covered by the overlap), so ``distance`` lies in
    ``[0, 1]`` and is exactly ``1.0`` for *non-overlapping* labels. laptrack's
    default ``cutoff``/``gap_closing_cutoff`` of 225 (a squared-euclidean default)
    therefore never rejects anything: during gap closing a dropped track could
    link to a completely non-overlapping, distant cell, producing catastrophic
    identity switches (a track whose centroid teleports hundreds of px between
    frames). We set both cutoffs **below 1.0** so a link *requires* real overlap
    (``ratio_2 > 1 - cutoff``); this is the root-cause fix for the switch
    artifacts seen in confluent ddFP fields.

    ``gap_closing_max_frame_count`` bridges frames where a cell's mask briefly
    drops out (e.g. a dim ddFP frame). The default is generous (8) because these
    adherent HEK293T barely move -- and now that gap closing *requires overlap*,
    a long gap only reconnects a cell that reappears where it vanished, never a
    distant one -- so it fixes fragmentation without reintroducing switches.
    """
    labels = np.stack(masks)  # (T, H, W)
    olt = OverLapTrack(
        cutoff=cutoff,
        gap_closing_cutoff=gap_closing_cutoff,
        gap_closing_max_frame_count=gap_closing_max_frame_count,
    )
    track_df, _, _ = olt.predict_overlap_dataframe(labels)
    return track_df
