"""Per-track quality control for single-cell traces.

Tracking and segmentation fail in characteristic ways in confluent fields, and
those failures masquerade as large ΔF/F₀ excursions even in a flat negative
control. Two dominant modes, both diagnosed on the ddFP Plain-ddFP data:

* **Identity switch** — gap-closing reconnects a lost track to a *different,
  distant* cell, so the track's centroid teleports across the field between
  consecutive frames (hundreds of px, vs the <~50 px an adherent HEK actually
  moves). The stitched trace is a chimera of several cells.
* **Segmentation merge** — several confluent cells fuse into one mask, giving an
  implausibly large ROI whose membership (and mean intensity) drifts as the merge
  shifts frame to frame.

``qc_tracks`` measures centroid displacement and mask area per track from the
instance masks and flags tracks that fail stability/plausibility thresholds, so
they can be dropped before averaging. Well-tracked cells with genuine intensity
changes are *kept* — QC targets the artifacts, not the biology.
"""
import numpy as np
import pandas as pd


def _median_cell_area(masks, stride: int = 10) -> float:
    areas = []
    for f in range(0, len(masks), stride):
        labs = np.unique(masks[f])
        for l in labs[labs > 0]:
            areas.append(int((masks[f] == l).sum()))
    return float(np.median(areas)) if areas else 0.0


def track_metrics(masks, track_df: pd.DataFrame) -> pd.DataFrame:
    """Per-track stability metrics from instance masks.

    Columns: cell_id, n_frames, max_disp (largest single-frame centroid move, px),
    med_area, max_area, max_area_jump (largest frame-to-frame fractional area change).
    """
    tdf = track_df.reset_index()
    recs = []
    for tid, g in tdf.groupby("track_id"):
        g = g.sort_values("frame")
        areas, cents = [], []
        for _, r in g.iterrows():
            m = masks[int(r["frame"])] == int(r["label"])
            a = int(m.sum())
            if a == 0:
                continue
            ys, xs = np.where(m)
            areas.append(a)
            cents.append((xs.mean(), ys.mean()))
        if len(areas) < 2:
            continue
        areas = np.asarray(areas, float)
        cents = np.asarray(cents, float)
        disp = np.hypot(*np.diff(cents, axis=0).T)
        arel = np.abs(np.diff(areas)) / np.maximum(areas[:-1], 1.0)
        recs.append(dict(
            cell_id=int(tid), n_frames=len(areas),
            max_disp=float(disp.max()), med_area=float(np.median(areas)),
            max_area=float(areas.max()), max_area_jump=float(arel.max()),
        ))
    return pd.DataFrame(recs)


def qc_from_traces(
    traces: pd.DataFrame,
    reloc_px: float = 100.0, reloc_frac_max: float = 0.2, area_cv_max: float = 0.30,
    faint_frac: float | None = None,
) -> pd.DataFrame:
    """Per-cell QC from a trace CSV carrying ``area``/``centroid``/``mean_intensity_sub``.

    Mask-free and reproducible straight from the extracted CSV (plotting never
    needs the mask arrays). The metrics are **robust to single-frame glitches**,
    which is what reviewer inspection of the overlay movies showed matters: a
    momentary segmentation wobble (a one-frame centroid blip or mask collapse
    that immediately recovers) leaves the ΔF/F₀ trace clean and must NOT drop the
    cell. Only *sustained* problems fail a track (ANY of):

    * **identity switch (sustained)** -- the centroid sits > ``reloc_px`` px from
      the track's own median centroid for more than ``reloc_frac_max`` of frames
      (a true switch relocates and stays; a blip returns immediately). Replaces
      the old single-frame ``max_disp``, which over-punished transient wobbles.
    * **flicker / unstable outline** -- area coefficient of variation
      (std/median of area over the track) > ``area_cv_max`` (a stably segmented
      cell's area barely varies; a merge that drifts membership does not).
    * **faint / non-expressing** (opt-in, ``faint_frac`` not None) -- median
      background-subtracted intensity < ``faint_frac`` x the field median. OFF by
      default: an *expression/inclusion* criterion, kept separate from the
      segmentation/tracking-quality filters and out of the overlay movies.

    The overlap-required linking in :func:`fluora.track.track_cells` is the
    primary artifact fix; this QC is a light backstop. Apply after selecting
    full-length tracks (short fragments are handled by the min-frames filter).

    Returns one row per cell_id with the metrics plus ``pass_qc``.
    """
    need = {"cell_id", "time_seconds", "area", "centroid_x", "centroid_y"}
    missing = need - set(traces.columns)
    if missing:
        raise ValueError(f"traces missing columns for QC: {sorted(missing)}")
    have_sub = "mean_intensity_sub" in traces.columns
    recs = []
    for cid, g in traces.groupby("cell_id"):
        g = g.sort_values("time_seconds")
        areas = g["area"].to_numpy(float)
        cx = g["centroid_x"].to_numpy(float)
        cy = g["centroid_y"].to_numpy(float)
        if len(areas) < 2:
            continue
        dfm = np.hypot(cx - np.median(cx), cy - np.median(cy))  # dist from median centroid
        med_area = float(np.median(areas))
        recs.append(dict(
            cell_id=int(cid), n_frames=len(areas), med_area=med_area,
            reloc_frac=float(np.mean(dfm > reloc_px)),
            area_cv=float(np.std(areas) / max(med_area, 1.0)),
            med_sub=float(np.median(g["mean_intensity_sub"])) if have_sub else np.nan,
        ))
    m = pd.DataFrame(recs)
    if m.empty:
        m["pass_qc"] = []
        return m
    m["fail_switch"] = m.reloc_frac > reloc_frac_max
    m["fail_flicker"] = m.area_cv > area_cv_max
    fails = ["fail_switch", "fail_flicker"]
    if faint_frac is not None and have_sub:
        m["fail_faint"] = m.med_sub < faint_frac * float(m.med_sub.median())
        fails.append("fail_faint")
    m["pass_qc"] = ~m[fails].any(axis=1)
    return m


def qc_tracks(
    masks, track_df: pd.DataFrame,
    disp_max: float = 60.0, area_mult: float = 2.5, area_jump_max: float = 0.4,
) -> pd.DataFrame:
    """Flag tracks failing stability/plausibility QC.

    Returns ``track_metrics`` plus boolean columns ``fail_disp`` (identity switch),
    ``fail_area`` (merged blob: area > ``area_mult`` x median single-cell area),
    ``fail_jump`` (merge/split: fractional area jump > ``area_jump_max``), and
    ``pass_qc``. Adherent HEK move << ``disp_max`` px/frame, so a large centroid
    displacement is a mislink, not motion.
    """
    m = track_metrics(masks, track_df)
    if m.empty:
        m["pass_qc"] = []
        return m
    area_max = area_mult * _median_cell_area(masks)
    m["fail_disp"] = m.max_disp > disp_max
    m["fail_area"] = m.max_area > area_max
    m["fail_jump"] = m.max_area_jump > area_jump_max
    m["pass_qc"] = ~m[["fail_disp", "fail_area", "fail_jump"]].any(axis=1)
    return m
