# fluora

Single-cell fluorescence trace analysis from timelapse microscopy.

Given a timelapse of fluorescence images, fluora segments individual cells in
each frame, links them across frames, and extracts a mean-intensity trace per
cell over time.

## Pipeline

1. **Load** — read a folder of `<timestamp>.tiff` frames (`fluora/io.py`).
2. **Segment** — per-frame cell masks via [Cellpose](https://github.com/MouseLand/cellpose)'s
   `cpsam` model (`fluora/segment.py`).
3. **Track** — link cells across frames by mask overlap via
   [laptrack](https://github.com/yfukai/laptrack) (`fluora/track.py`).
4. **Extract** — mean fluorescence intensity per tracked cell per frame, written
   to a tidy CSV (`fluora/extract.py`).

Output is a `traces.csv` with columns `cell_id, time_seconds, mean_intensity`.

## Input format

A directory of single-channel `.tiff` frames, each filename being its timestamp
in seconds:

```
imgs/
  0.0.tiff
  30.0.tiff
  60.0.tiff
  ...
```

## Run locally

```bash
uv sync
# put your frames in ./imgs, then:
uv run python main.py
```

By default `main.py` runs on CPU (`DEVICE = "cpu"`). Segmentation is the
bottleneck at ~157s/frame on CPU. Set `DEVICE = "cuda"` to use a local GPU —
note this requires a GPU of compute capability **sm_75 or newer** (Turing /
RTX 20-series and later); older cards (e.g. GTX 10-series, sm_61) are not
supported by the installed PyTorch build.

## Run on a GPU in the cloud

Segmentation is ~125× faster on a GPU (~1.3s/frame on a T4 vs ~157s on CPU).

### Google Colab (interactive, free GPU)

Open the notebook and `Run all` (set `Runtime > Change runtime type > T4 GPU`):

https://colab.research.google.com/github/phuongho43/fluora/blob/main/notebooks/fluora_colab.ipynb

Good for spot-checking segmentation and small runs.

### Modal (batch, recurring runs)

For processing many timelapses repeatedly, [Modal](https://modal.com) runs the
pipeline on cloud GPUs from a pinned, reproducible image — one container per
timelapse, in parallel.

```bash
uv tool install modal && modal setup            # one-time auth

# Upload timelapses (one subfolder of <timestamp>.tiff frames per movie):
#   my_timelapses/movie01/0.0.tiff ...   my_timelapses/movie02/ ...
modal volume put fluora-data ./my_timelapses /input

modal run modal_app.py                            # process all, in parallel
# modal run modal_app.py --name movie01           # or just one

modal volume get fluora-data /output ./results    # fetch one CSV per timelapse
```

See `modal_app.py` for details.
