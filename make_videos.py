"""Render segmentation+tracking overlay movies for a representative Plain/Dense/
Sparse decoder timelapse (fixed tracking + QC; per-track stable colours).

    uv run python make_videos.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from fluora.io import load_images
from fluora.track import track_cells
from fluora.qc import qc_from_traces
from fluora.overlay import render_tracking_video

# cached segmentation masks (fetch once with:
#   modal volume get fluora-data /output ./masks --prefix <name>_masks )
MASKS = Path("masks")
DATA = Path("/home/phuong/projects/csc-revisions-2026/data/1--biosensor/7--decoder-new")
RES = Path("results")
OUT = Path("results/videos")
MIN_FRAMES = 55

MOVIES = [
    ("Plain-ddFP",  "plainnew-4",  DATA / "0--plain-ddFP/4/imgs"),
    ("Dense-ddFP",  "densenew-2",  DATA / "1--dense-ddFP/2/imgs"),
    ("Sparse-ddFP", "sparsenew-3", DATA / "2--sparse-ddFP/3/imgs"),
]


def main():
    for title, name, imgs_dir in MOVIES:
        z = np.load(MASKS / f"{name}_masks.npz")
        masks = [m for m in z["masks"]]
        timestamps, images = load_images(imgs_dir)
        track_df = track_cells(masks)

        # draw & score exactly the cells the figure uses: full-length tracks,
        # QC from the extracted CSV (same qc_from_traces as the figure).
        csv = pd.read_csv(RES / f"{name}.csv")
        full = csv.groupby("cell_id").filter(lambda g: len(g) >= MIN_FRAMES)
        n_full = full["cell_id"].nunique()
        q = qc_from_traces(full)
        pass_ids = set(q.loc[q.pass_qc, "cell_id"].astype(int))
        # draw ONLY the included (QC-pass) cells -- no red QC-fail outlines
        out = OUT / f"{name}_tracking.mp4"
        render_tracking_video(masks, images, track_df, out, qc_pass_ids=pass_ids,
                              draw_ids=pass_ids, timestamps=timestamps, title=title, fps=8)
        print(f"saved {out}  ({len(pass_ids)} included cells drawn, "
              f"{n_full - len(pass_ids)} excluded of {n_full} full-length)", flush=True)


if __name__ == "__main__":
    main()
