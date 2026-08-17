"""Reviewer-compliant stats for the ddFP biosensor kinetics panels (Fig S2b, S2d).

Whole-field ddFP ΔF/F₀ per movie (F₀ = mean of the first 5 frames), then the peak
response per movie: the **minimum** for the LOV dissociation panels (fluorescence
drops) and the **maximum** for the iLID binding panels (fluorescence rises). Each
movie is one replicate.

Two 2-group comparisons (currently Student's t + mean ± 1.96×SEM in the manuscript):
  * Fig S2b -- LOVfast at 20 vs 200 µW/mm² (min ΔF/F₀; n = 4/group).
  * Fig S2d -- iLIDslow with 13 vs 20 aa linker (max ΔF/F₀; n = 7/group).
Reanalyzed with Welch's t-test, exact p, and mean ± SEM (not 1.96×SEM).

    uv run python analyze_ddfp_kinetics.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt

from fluora.plotting import STYLE, INK, errorbar_halfwidth
from fluora.stats import format_p

D = Path("/home/phuong/projects/csc-revisions-2026/data/1--biosensor")
RESULTS = Path("results")


def peak_dff(base, which):
    """Per-movie peak ΔF/F₀ (``which`` = 'min' or 'max') for a condition folder."""
    vals = []
    for m in sorted((D / base).glob("*/results/y.csv")):
        y = pd.read_csv(m).y.to_numpy()
        f0 = y[:5].mean()
        dff = (y - f0) / f0
        vals.append(dff.min() if which == "min" else dff.max())
    return np.array(vals)


def two_group_strip(groups, out, ylabel, colors, title=None):
    """Two-group strip: per-movie points + mean ± SEM + Welch-t bracket.

    groups : list of (label, values). colors : {label: hex}.
    """
    (la, va), (lb, vb) = groups
    p = float(st.ttest_ind(va, vb, equal_var=False).pvalue)
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(18, 15))
        rng = np.random.default_rng(0)
        for x, (lab, v) in enumerate(groups):
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=600, color=colors[lab],
                       alpha=0.8, linewidths=2, edgecolors=INK, zorder=3)
            ax.errorbar(x, v.mean(), yerr=errorbar_halfwidth(v, "sem"), fmt="o", ms=24,
                        color=INK, ecolor=INK, elinewidth=8, capsize=18, capthick=8, zorder=4)
        # significance bracket
        lo = min(va.min(), vb.min())
        hi = max(va.max(), vb.max())
        pad = 0.12 * (hi - lo)
        y = hi + pad
        ax.plot([0, 0, 1, 1], [y - pad * 0.3, y, y, y - pad * 0.3], color=INK, lw=5)
        ax.text(0.5, y + pad * 0.1, format_p(p), ha="center", va="bottom", fontsize=48, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([la, lb])
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(top=y + pad * 0.9)
        if title:
            ax.set_title(title, fontsize=52)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out, p


def main():
    # Fig S2b -- LOVfast intensity: min ΔF/F0 (dissociation -> fluorescence drops)
    lo = peak_dff("2--intensity/0--LOVfast-BL20uW", "min")
    hi = peak_dff("2--intensity/1--LOVfast-BL200uW", "min")
    out, p = two_group_strip(
        [(r"20 $\mathbf{\mu}$W", lo), (r"200 $\mathbf{\mu}$W", hi)],
        RESULTS / "fig_S2b_LOV_intensity.png",
        ylabel=r"Min $\mathbf{\Delta F/F_{0}}$",
        colors={r"20 $\mathbf{\mu}$W": "#2ECC71", r"200 $\mathbf{\mu}$W": "#EA822C"})
    print(f"S2b LOVfast 20 vs 200 uW (min dF/F0): {lo.mean():.3f} vs {hi.mean():.3f}  "
          f"Welch {format_p(p)}  (n={len(lo)}/{len(hi)})")
    print("saved", out)

    # Fig S2d -- iLIDslow linker: max ΔF/F0 (binding -> fluorescence rises)
    aa13 = peak_dff("4--linker/0--iLIDslow-13AA", "max")
    aa20 = peak_dff("4--linker/1--iLIDslow-20AA", "max")
    out, p = two_group_strip(
        [("13 aa", aa13), ("20 aa", aa20)],
        RESULTS / "fig_S2d_linker.png",
        ylabel=r"Max $\mathbf{\Delta F/F_{0}}$",
        colors={"13 aa": "#2ECC71", "20 aa": "#EA822C"})
    print(f"S2d linker 13 vs 20 aa (max dF/F0): {aa13.mean():.3f} vs {aa20.mean():.3f}  "
          f"Welch {format_p(p)}  (n={len(aa13)}/{len(aa20)})")
    print("saved", out)


if __name__ == "__main__":
    main()
