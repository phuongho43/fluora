"""Render segmentation + tracking overlay movies.

Draws each tracked cell's outline on the raw frames, coloured by a *stable*
per-track colour so a cell keeps its colour across the whole movie -- that is
what makes tracking (not just per-frame segmentation) visible. Tracks that fail
stability QC are drawn thicker in a fixed warning colour so the filtering step
is visible too.
"""
import subprocess
from pathlib import Path

import numpy as np
import cv2

from .segment import normalize_frame


def _ffmpeg_writer(out, w, h, fps):
    """Open an ffmpeg subprocess that encodes raw rgb24 frames from stdin to h264."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-movflags", "+faststart", str(out),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# a fixed, high-contrast qualitative palette for QC-passing tracks (RGB 0-255).
# Pure reds are deliberately excluded so no included cell reads as "flagged".
_BASE = np.array([
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (23, 190, 207),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (44, 200, 160), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (120, 200, 255), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
], dtype=np.uint8)
QC_FAIL_COLOR = (255, 40, 40)   # bright red, thick -- only used if failures are drawn


def _track_colors(track_ids, qc_pass_ids):
    """Assign a stable RGB to each track id (fail ids -> red)."""
    colors = {}
    k = 0
    for tid in sorted(track_ids):
        if qc_pass_ids is not None and tid not in qc_pass_ids:
            colors[tid] = QC_FAIL_COLOR
        else:
            colors[tid] = tuple(int(c) for c in _BASE[k % len(_BASE)])
            k += 1
    return colors


def render_tracking_video(
    masks, images, track_df, out, qc_pass_ids=None, draw_ids=None, timestamps=None,
    fps=8, title=None, norm_lo=1.0, norm_hi=99.0, thickness=2, fail_thickness=5,
    label_ids=False,
):
    """Write an mp4 of raw frames with per-track coloured cell outlines.

    masks : list of (H,W) int label images (one per frame).
    images : list of raw frames (same order).
    track_df : laptrack output (MultiIndex frame,label; column track_id).
    qc_pass_ids : optional set of track ids that pass QC; others drawn as failures.
    draw_ids : optional set of track ids to draw at all (e.g. the full-length cells
        actually used in the analysis); short fragment tracks are skipped so the
        movie shows exactly the cells behind the figure.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tdf = track_df.reset_index()
    # (frame,label) -> track_id  and the full set of track ids
    key2tid = {(int(r.frame), int(r.label)): int(r.track_id) for r in tdf.itertuples()}
    colors = _track_colors(set(tdf.track_id.astype(int)), qc_pass_ids)

    h, w = images[0].shape[:2]
    proc = _ffmpeg_writer(out, w, h, fps)
    try:
        for f, (mask, img) in enumerate(zip(masks, images)):
            g = (normalize_frame(img, norm_lo, norm_hi) * 255).astype(np.uint8)
            rgb = np.ascontiguousarray(np.dstack([g, g, g]))
            labs = np.unique(mask); labs = labs[labs > 0]
            for lab in labs:
                tid = key2tid.get((f, int(lab)))
                if tid is None or (draw_ids is not None and tid not in draw_ids):
                    continue
                color = colors[tid]
                fail = qc_pass_ids is not None and tid not in qc_pass_ids
                cnts, _ = cv2.findContours((mask == lab).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, cnts, -1, color,
                                 fail_thickness if fail else thickness)
                if label_ids and cnts:
                    c = max(cnts, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M["m00"]:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        cv2.putText(rgb, str(tid), (cx - 10, cy + 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            # header text
            parts = []
            if title:
                parts.append(title)
            if timestamps is not None:
                parts.append(f"t = {timestamps[f]:.0f} s")
            if parts:
                cv2.putText(rgb, "   ".join(parts), (24, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
            proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    return out
