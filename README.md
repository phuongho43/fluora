# fluora

Single-cell fluorescence trace analysis from timelapse microscopy.

Given a timelapse of fluorescence images, fluora segments individual cells in
each frame, links them across frames, and extracts a per-cell intensity trace
over time. Built for the CSC-revision ddFP / FM-decoder experiments; segmentation
runs on cloud GPUs via [Modal](https://modal.com).

## Pipeline

1. **Load** — read a folder of `<timestamp>.tiff` frames (`fluora/io.py`).
2. **Segment** — per-frame cell masks via [Cellpose](https://github.com/MouseLand/cellpose)'s
   `cpsam` model, with a fixed diameter hint + per-frame 1–99% contrast
   normalization (`fluora/segment.py`; default `diameter=150` tuned for 100× ddFP).
3. **Track** — link cells across frames by mask overlap via
   [laptrack](https://github.com/yfukai/laptrack). Links require real overlap
   (`cutoff=0.9`) and short dropouts are bridged (`gap_closing=8`), avoiding the
   identity switches and fragmentation seen in confluent fields (`fluora/track.py`).
4. **Extract** — **mask-aware background subtraction** (background estimated from
   non-cell pixels by normalized Gaussian convolution) then mean intensity per
   tracked cell per frame, plus per-cell **ΔF/F₀**. Also records per-cell area and
   centroid for QC (`fluora/extract.py`).
5. **QC** — drop tracks with sustained identity switches or unstable outlines
   (`fluora/qc.py`).

Output is a `traces.csv` per timelapse with columns `cell_id, time_seconds,
mean_intensity, mean_intensity_sub, area, centroid_x, centroid_y, dF_F0`.

Analysis / figures are driven by the top-level scripts, all using the ezplot house
style and the reviewer-compliant stats in `fluora/stats.py` (ANOVA / mixed models,
multiple-comparison correction, exact p-values):

- `analyze_decoders.py` — ddFP decoder single-cell traces + per-regime summary (Fig 2i, 2j).
- `analyze_expression.py` — transcriptional frequency sweep + dual decoder (Fig 4c, 4g).
- `analyze_antigen.py` — CAR-T cytotoxicity + in-vivo BLI (Fig 5c, 5f).
- `make_videos.py` — segmentation/tracking overlay movies.

## Input format

A directory of single-channel `.tiff` frames, each filename being its timestamp
in seconds; one subfolder per movie:

```
my_timelapses/
  movie01/  0.0.tiff  30.0.tiff  60.0.tiff ...
  movie02/  0.0.tiff  30.0.tiff ...
```

## Run (Modal cloud GPUs)

Segmentation is the bottleneck; it runs on cloud GPUs from a pinned, reproducible
image — one container per timelapse, in parallel.

```bash
uv sync
uv tool install modal && modal setup            # one-time auth

# Upload timelapses once (re-runs then need no re-upload):
modal volume put fluora-data ./my_timelapses /input

modal run modal_app.py                            # segment+track+extract, all movies
# modal run modal_app.py --name movie01           # or just one
# modal run modal_app.py::retrack --prefix movie  # re-track/extract from cached masks (no GPU)

modal volume get fluora-data /output ./results    # fetch one CSV (+ cached masks) per timelapse
```

Instance masks are cached to `<name>_masks.npz` so tracking/extraction/QC can be
re-tuned locally in seconds without re-segmenting. Segmentation params can be
swept with `modal_tune.py` (writes mask-overlay PNGs). See `modal_app.py` for details.
