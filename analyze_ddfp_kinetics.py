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


STIM = "#648FFF"
VAR_COLORS = {"ddFP": "#34495E", "LOVfast": "#2ECC71", "LOVslow": "#D143A4",
              "iLIDfast": "#2ECC71", "iLIDslow": "#D143A4",
              "20 uW": "#2ECC71", "200 uW": "#EA822C", "13 aa": "#2ECC71", "20 aa": "#EA822C"}


def dff_traces(base):
    """Per-movie ΔF/F₀ traces + shared time grid for a condition folder."""
    traces, tgrid = [], None
    for m in sorted((D / base).glob("*/results/y.csv")):
        d = pd.read_csv(m)
        y = d.y.to_numpy()
        f0 = y[:5].mean()
        traces.append((y - f0) / f0)
        if tgrid is None:
            tgrid = d.t.to_numpy()
    return np.vstack(traces), tgrid


def stim_time(base):
    """Onset of the (single) light pulse from any movie's u.csv."""
    u = pd.read_csv(next((D / base).glob("*/u.csv")))
    return float(u.ta.iloc[0]), float(u.tb.iloc[0])


def plot_variant_timeseries(variants, out, stim, ylabel=r"$\mathbf{\Delta F/F_{0}}$",
                            lsizes=None, ylim=None):
    """Mean ± SEM ΔF/F₀ over time per variant (Fig 2c/2f/S2a/S2c style)."""
    lsizes = lsizes or [16, 12, 8]
    ta, tb = stim
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(22, 16))
        ax.axvspan(ta, tb, color=STIM, lw=0, alpha=0.8, zorder=0)
        ax.axhline(0, color=INK, lw=4, ls=(0, (1, 2)), alpha=0.6, zorder=1)
        for i, (label, M, tgrid) in enumerate(variants):
            mean = M.mean(0)
            sem = M.std(0, ddof=1) / np.sqrt(M.shape[0])
            lw = lsizes[i] if i < len(lsizes) else lsizes[-1]
            color = VAR_COLORS.get(label, "#888")
            ax.fill_between(tgrid, mean - sem, mean + sem, color=color, alpha=0.25, lw=0, zorder=2)
            ax.plot(tgrid, mean, color=color, lw=lw, label=f"{label}  (n={M.shape[0]})", zorder=3)
        ax.legend(loc="best", framealpha=0.9, fontsize=44)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(tgrid[0], tgrid[-1])
        if ylim:
            ax.set_ylim(ylim)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out


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
    # --- Fig 2c: LOV reversion variants (ddFP control, LOVfast, LOVslow) ---
    lov = [("ddFP", *dff_traces("0--ddFP")),
           ("LOVfast", *dff_traces("1--LOV/0--I427V")),
           ("LOVslow", *dff_traces("1--LOV/1--V416I"))]
    out = plot_variant_timeseries(lov, RESULTS / "fig_2c_LOV_variants.png",
                                  stim_time("1--LOV/0--I427V"))
    print("saved", out)

    # --- Fig 2f: iLID reversion variants (ddFP control, iLIDfast, iLIDslow) ---
    ilid = [("ddFP", *dff_traces("0--ddFP")),
            ("iLIDfast", *dff_traces("3--iLID/0--I427V")),
            ("iLIDslow", *dff_traces("3--iLID/1--V416I"))]
    out = plot_variant_timeseries(ilid, RESULTS / "fig_2f_iLID_variants.png",
                                  stim_time("3--iLID/0--I427V"))
    print("saved", out)
    # 2f claim: iLIDslow has lower induction -> one-way ANOVA + Tukey on max ΔF/F0
    peaks = {lab: M.max(1) for lab, M, _ in ilid}
    fval, pval = st.f_oneway(*peaks.values())
    print(f"2f max ΔF/F0 one-way ANOVA across {list(peaks)}: F={fval:.1f}, {format_p(pval)}")
    for a, b in [("iLIDfast", "iLIDslow"), ("ddFP", "iLIDslow")]:
        p = float(st.ttest_ind(peaks[a], peaks[b], equal_var=False).pvalue)
        print(f"    {a} vs {b} (max, Welch): {peaks[a].mean():.3f} vs {peaks[b].mean():.3f}  {format_p(p)}")

    # --- Fig S2a / S2c: intensity & linker timeseries ---
    inten = [("20 uW", *dff_traces("2--intensity/0--LOVfast-BL20uW")),
             ("200 uW", *dff_traces("2--intensity/1--LOVfast-BL200uW"))]
    out = plot_variant_timeseries(inten, RESULTS / "fig_S2a_LOV_intensity_ts.png",
                                  stim_time("2--intensity/0--LOVfast-BL20uW"), lsizes=[16, 12])
    print("saved", out)
    link = [("13 aa", *dff_traces("4--linker/0--iLIDslow-13AA")),
            ("20 aa", *dff_traces("4--linker/1--iLIDslow-20AA"))]
    out = plot_variant_timeseries(link, RESULTS / "fig_S2c_linker_ts.png",
                                  stim_time("4--linker/0--iLIDslow-13AA"), lsizes=[16, 12])
    print("saved", out)

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
