"""Run the fluora pipeline on cloud GPUs over a batch of timelapses, via Modal.

Modal is a deploy/run tool you invoke from your laptop; it is NOT a runtime
dependency of the fluora package. The container's dependencies are defined by
`image` below, so this sidesteps the local Python>=3.14 / Colab version churn.

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
def process_timelapse(name: str) -> dict:
    """Segment, track, and extract one timelapse; write its traces CSV."""
    import time
    from pathlib import Path

    from fluora.io import load_images
    from fluora.segment import segment_frames
    from fluora.track import track_cells
    from fluora.extract import extract_intensities

    t0 = time.time()
    timestamps, images = load_images(Path(INPUT_DIR) / name)
    masks = segment_frames(images, device="cuda")
    track_df = track_cells(masks)
    traces = extract_intensities(timestamps, images, masks, track_df)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out = Path(OUTPUT_DIR) / f"{name}.csv"
    traces.to_csv(out, index=False)
    data_vol.commit()   # persist the output CSV
    cache_vol.commit()  # persist downloaded cpsam weights for next run

    return {
        "name": name,
        "frames": len(images),
        "cells": int(traces["cell_id"].nunique()),
        "seconds": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main(name: str = ""):
    names = [name] if name else list_timelapses.remote()
    if not names:
        print("No timelapses found under /input on the 'fluora-data' volume.")
        print("Upload first, e.g.:  modal volume put fluora-data ./my_timelapses /input")
        return

    print(f"Processing {len(names)} timelapse(s): {', '.join(names)}")
    total_cells = 0
    for r in process_timelapse.map(names):
        total_cells += r["cells"]
        print(f"  {r['name']}: {r['cells']} cells from {r['frames']} frames in {r['seconds']}s")
    print(f"Done. {total_cells} total cell traces across {len(names)} timelapse(s).")
    print("Fetch results with:  modal volume get fluora-data /output ./results")
