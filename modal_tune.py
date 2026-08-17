"""Tune Cellpose segmentation params on real frames, on a cloud GPU, via Modal.

This is a *tuning* companion to modal_app.py -- it does not replace the pipeline.
It segments a small set of representative frames under several parameter sets (in
parallel, one T4 per set) and writes side-by-side overlay PNGs (normalized input |
red mask outlines) so you can eyeball which params segment your cells cleanly.
It reuses modal_app's named volumes, so data and the cpsam weight cache are shared.

Workflow
--------
1) Put a handful of representative full-frame .tiff under a local folder, e.g.
   ./tuning/  (mix conditions: dark/bright, dense/sparse, early/late timepoints)

2) Upload once to the shared volume:
       modal volume put fluora-data ./tuning /tuning

3) Sweep params (edit GRID below, then):
       modal run modal_tune.py

4) Fetch overlays and look:
       modal volume get fluora-data /output/spotcheck ./spotcheck
   Filenames encode the param tag + cell count, e.g. d150_f0.4_<frame>_n23.png

Iterate GRID -> run -> fetch -> view until the masks look right, then bake the
winning params into fluora/segment.py for the production runs.
"""
import modal

app = modal.App("fluora-tune")

# Same container recipe as the pipeline (kept in sync with modal_app.py).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "cellpose==4.2.1.1",
        "tifffile",
        "scikit-image",
        "numpy",
    )
    .add_local_python_source("fluora")
)

data_vol = modal.Volume.from_name("fluora-data", create_if_missing=True)
cache_vol = modal.Volume.from_name("fluora-cellpose-cache", create_if_missing=True)

TUNING_DIR = "/data/tuning"
SPOTCHECK_OUT = "/data/output/spotcheck"


@app.function(image=image, volumes={"/data": data_vol})
def list_tuning_frames() -> list[str]:
    import os

    if not os.path.isdir(TUNING_DIR):
        return []
    return sorted(f for f in os.listdir(TUNING_DIR) if f.lower().endswith((".tif", ".tiff")))


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": data_vol, "/root/.cellpose": cache_vol},
    timeout=20 * 60,
)
def spotcheck(params: dict) -> dict:
    """Segment every frame under /data/tuning with one param set; write overlays."""
    import os
    import numpy as np
    import tifffile
    from skimage.exposure import rescale_intensity
    from skimage.segmentation import find_boundaries
    from skimage.io import imsave
    from cellpose import models

    tag = params["tag"]
    diam = params.get("diameter") or None      # 0/None => model auto
    flow = params.get("flow", 0.4)
    cellprob = params.get("cellprob", 0.0)
    lo = params.get("norm_lo", 1.0)
    hi = params.get("norm_hi", 99.0)

    os.makedirs(SPOTCHECK_OUT, exist_ok=True)
    model = models.CellposeModel(gpu=True)

    frames = sorted(
        f for f in os.listdir(TUNING_DIR) if f.lower().endswith((".tif", ".tiff"))
    )
    results = []
    for fname in frames:
        img = tifffile.imread(os.path.join(TUNING_DIR, fname))
        p_lo, p_hi = np.percentile(img, [lo, hi])
        disp = rescale_intensity(
            img, in_range=(p_lo, p_hi), out_range=(0.0, 1.0)
        ).astype(np.float32)

        masks = model.eval(
            disp, diameter=diam, flow_threshold=flow, cellprob_threshold=cellprob
        )[0]
        ncells = int(masks.max())
        areas = np.bincount(masks.ravel())[1:] if ncells else np.array([])
        med_area = int(np.median(areas)) if areas.size else 0

        g = (disp * 255).astype(np.uint8)
        left = np.stack([g] * 3, axis=-1)
        right = left.copy()
        right[find_boundaries(masks, mode="outer")] = [255, 0, 0]
        gap = np.zeros((g.shape[0], 6, 3), dtype=np.uint8)
        panel = np.concatenate([left, gap, right], axis=1)

        stem = os.path.splitext(fname)[0]
        out = os.path.join(SPOTCHECK_OUT, f"{tag}_{stem}_n{ncells}.png")
        imsave(out, panel, check_contrast=False)
        results.append({"frame": stem, "ncells": ncells, "med_area": med_area})

    data_vol.commit()
    cache_vol.commit()
    return {"tag": tag, "frames": results}


# ---- Parameter grid to sweep. Edit this, re-run, re-fetch, compare. -----------
# diameter in px (0 = cpsam auto); flow_threshold (shape strictness);
# cellprob_threshold (lower catches dimmer cells); norm_lo/hi = percentile stretch.
GRID = [
    {"tag": "d150cp0",  "diameter": 150, "flow": 0.4, "cellprob": 0.0},
    {"tag": "d150cp-2", "diameter": 150, "flow": 0.4, "cellprob": -2.0},
    {"tag": "d150cp-4", "diameter": 150, "flow": 0.4, "cellprob": -4.0},
    {"tag": "d150cp-2w","diameter": 150, "flow": 0.4, "cellprob": -2.0, "norm_lo": 0.3, "norm_hi": 99.7},
]


@app.local_entrypoint()
def main():
    frames = list_tuning_frames.remote()
    if not frames:
        print("No .tiff found under /tuning on the 'fluora-data' volume.")
        print("Upload first:  modal volume put fluora-data ./tuning /tuning")
        return
    print(f"Tuning on {len(frames)} frame(s): {', '.join(frames)}")
    print(f"Sweeping {len(GRID)} param set(s): {', '.join(g['tag'] for g in GRID)}")
    for r in spotcheck.map(GRID):
        counts = ", ".join(f"{f['frame']}={f['ncells']}(area~{f['med_area']})" for f in r["frames"])
        print(f"  [{r['tag']}] {counts}")
    print("Fetch overlays:  modal volume get fluora-data /output/spotcheck ./spotcheck")
