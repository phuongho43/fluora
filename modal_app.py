"""Run the fluora pipeline on cloud GPUs over a batch of timelapses, via Modal.

Modal is a deploy/run tool you invoke from your laptop; it is NOT a runtime
dependency of the fluora package. The container's dependencies are defined by
`image` below, so it is reproducible and independent of the local environment.

One-time setup
--------------
    uv tool install modal      # or: pip install modal
    modal setup                # authenticate (opens browser)

Data layout
-----------
Each timelapse is a folder of `<timestamp>.tiff` frames. Put all timelapses
under a single local directory, one subfolder per movie:

    my_timelapses/
      movie01/  0.0.tiff  30.0.tiff  60.0.tiff ...   (62 frames)
      movie02/  0.0.tiff  30.0.tiff ...
      ...

Upload it once to the persistent volume (re-runs then need no re-upload):

    modal volume put fluora-data ./my_timelapses /input

Run
---
    modal run modal_app.py                 # process every timelapse, in parallel
    modal run modal_app.py --name movie01  # just one

Fetch results (one CSV of single-cell traces per timelapse)
-----------------------------------------------------------
    modal volume get fluora-data /output ./results

Notes
-----
- Containers run in parallel, one T4 GPU per timelapse, so a batch finishes in
  roughly the time of a single movie (~1.5 min for 62 frames) plus cold start.
- cpsam weights (~1.2 GB) download on first use and are cached on a second
  volume, so later runs skip the download.
- Bump `gpu="T4"` to `"A10G"`/`"A100"` for more speed (all are sm_75+).
"""
import modal

app = modal.App("fluora")

# Pinned container image: deps installed here, fluora source shipped from local.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "cellpose==4.2.1.1",
        "laptrack>=0.17",
        "tifffile",
        "scikit-image",
        "scipy",
        "numpy",
        "pandas",
    )
    .add_local_python_source("fluora")
)

# Persistent storage: input/output data, and a separate cache for model weights.
data_vol = modal.Volume.from_name("fluora-data", create_if_missing=True)
cache_vol = modal.Volume.from_name("fluora-cellpose-cache", create_if_missing=True)

INPUT_DIR = "/data/input"
OUTPUT_DIR = "/data/output"


@app.function(image=image, volumes={"/data": data_vol})
def list_timelapses() -> list[str]:
    """List timelapse subfolders on the volume (reads the mounted filesystem)."""
    import os

    if not os.path.isdir(INPUT_DIR):
        return []
    return sorted(
        d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))
    )


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": data_vol, "/root/.cellpose": cache_vol},
    timeout=30 * 60,
)
def process_timelapse(name: str, diameter: float = 150.0) -> dict:
    """Segment, track, and extract one timelapse; write its traces CSV + masks.

    ``diameter`` is the Cellpose cell-size hint in px: 150 for the 100x ddFP
    timelapses (default), ~20 for the 10x transcription-reporter images.

    The instance masks are also cached to ``<name>_masks.npz`` (with timestamps)
    so tracking/extraction can be re-tuned locally without re-running the GPU
    segmentation -- segmentation is the only expensive step.
    """
    import time
    from pathlib import Path

    import numpy as np

    from fluora.io import load_images
    from fluora.segment import segment_frames
    from fluora.track import track_cells
    from fluora.extract import extract_intensities

    t0 = time.time()
    timestamps, images = load_images(Path(INPUT_DIR) / name)
    masks = segment_frames(images, device="cuda", diameter=diameter)
    track_df = track_cells(masks)
    traces = extract_intensities(timestamps, images, masks, track_df)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(OUTPUT_DIR) / f"{name}.csv"
    traces.to_csv(out, index=False)
    np.savez_compressed(
        Path(OUTPUT_DIR) / f"{name}_masks.npz",
        masks=np.stack(masks).astype(np.int32),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )
    data_vol.commit()   # persist the output CSV + masks
    cache_vol.commit()  # persist downloaded cpsam weights for next run

    return {
        "name": name,
        "frames": len(images),
        "cells": int(traces["cell_id"].nunique()),
        "seconds": round(time.time() - t0, 1),
    }


@app.function(image=image, volumes={"/data": data_vol}, timeout=30 * 60)
def retrack_timelapse(name: str) -> dict:
    """Re-track + re-extract one timelapse from its CACHED masks (no GPU).

    Loads ``<name>_masks.npz`` and the raw frames, re-runs the (fixed) overlap
    tracking and intensity extraction, and rewrites ``<name>.csv``. Use this to
    propagate a tracking/extraction change without paying for re-segmentation --
    segmentation is deterministic, so the cached masks are still valid.
    """
    import time
    from pathlib import Path

    import numpy as np

    from fluora.io import load_images
    from fluora.track import track_cells
    from fluora.extract import extract_intensities

    t0 = time.time()
    z = np.load(Path(OUTPUT_DIR) / f"{name}_masks.npz")
    masks = [m for m in z["masks"]]
    timestamps, images = load_images(Path(INPUT_DIR) / name)
    track_df = track_cells(masks)
    traces = extract_intensities(timestamps, images, masks, track_df)

    out = Path(OUTPUT_DIR) / f"{name}.csv"
    traces.to_csv(out, index=False)
    data_vol.commit()
    return {
        "name": name,
        "frames": len(images),
        "cells": int(traces["cell_id"].nunique()),
        "seconds": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def retrack(prefix: str = ""):
    """Re-track+extract all cached timelapses (optionally filtered by ``prefix``).

        modal run modal_app.py::retrack --prefix plainnew
    """
    names = list_timelapses.remote()
    if prefix:
        names = [n for n in names if n.startswith(prefix)]
    if not names:
        print("No timelapses found under /input.")
        return
    print(f"Re-tracking {len(names)} timelapse(s) from cached masks: {', '.join(names)}")
    total = 0
    for r in retrack_timelapse.map(names):
        total += r["cells"]
        print(f"  {r['name']}: {r['cells']} cells from {r['frames']} frames in {r['seconds']}s")
    print(f"Done. {total} total cell traces. Fetch with: modal volume get fluora-data /output ./results")


@app.local_entrypoint()
def main(name: str = "", diameter: float = 150.0, prefix: str = ""):
    names = [name] if name else list_timelapses.remote()
    if prefix:
        names = [n for n in names if n.startswith(prefix)]
    if not names:
        print("No timelapses found under /input on the 'fluora-data' volume.")
        print("Upload first, e.g.:  modal volume put fluora-data ./my_timelapses /input")
        return

    print(f"Processing {len(names)} timelapse(s) at diameter={diameter}: {', '.join(names)}")
    total_cells = 0
    for r in process_timelapse.map(names, kwargs={"diameter": diameter}):
        total_cells += r["cells"]
        print(f"  {r['name']}: {r['cells']} cells from {r['frames']} frames in {r['seconds']}s")
    print(f"Done. {total_cells} total cell traces across {len(names)} timelapse(s).")
    print("Fetch results with:  modal volume get fluora-data /output ./results")
