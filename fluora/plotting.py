"""Publication plots for fluora single-cell traces.

Conventions follow flowsmith/ASSUMPTIONS.md (no pseudoreplication; error at the
replicate level; SD / SEM / t-based CI; individual replicates shown at small n)
and the ezplot house style (`/home/phuong/projects/ezplot`, palette + STYLE rc).
Self-contained (matplotlib only, no seaborn/ezplot import) so it carries no
cross-repo dependency, but is visually consistent with the published figures
(e.g. fig_2i) so the revised panels drop straight into the manuscript.

Replicate model for microscopy: cell -> movie (technical) -> experiment (biological).
The *movie* is the replicate unit for the error band; report n cells / n movies /
n experiments in the caption (see section3-image-analysis-statistics.md). This is
the key rigor improvement over the original fig_2i, whose seaborn band pooled all
cells (pseudoreplication) — here the band is the spread across movies.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

# ezplot house palette (Dense, Sparse, green, magenta, yellow, slate, blue)
PALETTE = ["#8069EC", "#EA822C", "#2ECC71", "#D143A4", "#F1C40F", "#34495E", "#648FFF"]
# semantic colors for the decoder conditions
DECODER_COLORS = {"plain": "#34495E", "dense": "#8069EC", "sparse": "#EA822C"}
STIM_COLOR = "#648FFF"
INK = "#212121"

# ezplot STYLE (mirrored from ezplot/style.py) — large fonts, thick lines, 24x16,
# dpi 300 — so fluora figures match the published panels one-to-one.
STYLE = {
    "figure.figsize": (24, 16),
    "lines.linewidth": 16,
    "lines.markersize": 24,
    "lines.markeredgecolor": INK,
    "lines.markeredgewidth": 2,
    "text.color": INK,
    "font.size": 64,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 96,
    "axes.labelpad": 18,
    "axes.labelcolor": INK,
    "axes.labelweight": 600,
    "axes.linewidth": 12,
    "axes.edgecolor": INK,
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d5d5d5",
    "grid.linewidth": 6,
    "xtick.bottom": True,
    "ytick.left": True,
    "xtick.major.pad": 18,
    "ytick.major.pad": 18,
    "xtick.labelsize": 72,
    "ytick.labelsize": 72,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.size": 36,
    "ytick.major.size": 36,
    "xtick.major.width": 12,
    "ytick.major.width": 12,
    "legend.fontsize": 64,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.4,
    "legend.handlelength": 1,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def errorbar_halfwidth(vals: np.ndarray, kind: str = "sem") -> float:
    """Half-width of an error bar for 1D ``vals`` (flowsmith conventions).

    ``sd`` = std (spread), ``sem`` = SD/sqrt(n) (precision of the mean),
    ``ci`` = proper 95% CI = t(0.975, n-1) * SEM (NOT 1.96*SEM at small n). 0 for n<2.
    """
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    if n < 2:
        return 0.0
    sd = float(np.std(vals, ddof=1))
    if kind == "sd":
        return sd
    sem = sd / np.sqrt(n)
    if kind == "sem":
        return sem
    if kind == "ci":
        return float(sp_stats.t.ppf(0.975, n - 1)) * sem
    raise ValueError(f"unknown errorbar kind {kind!r}; use 'sd', 'sem' or 'ci'")


def _errorbar_label(kind: str) -> str:
    return {"sd": "SD", "sem": "SEM", "ci": "95% CI"}.get(kind, kind)


def movie_trace(csv_path, tgrid, min_frames=55, value="dF_F0"):
    """One per-cell-mean trace for a movie, interpolated onto ``tgrid``.

    Averages ``value`` over full-length cells (tracked >= ``min_frames`` frames)
    at each timepoint. Returns (trace, n_cells) or (None, 0) if no full-length cells.
    """
    d = pd.read_csv(csv_path)
    good = d.groupby("cell_id").filter(lambda g: len(g) >= min_frames)
    if good.empty:
        return None, 0
    m = good.groupby("time_seconds")[value].mean()
    return np.interp(tgrid, m.index.values, m.values), int(good["cell_id"].nunique())


def cell_traces(csv_path, tgrid, min_frames=55, value="dF_F0", apply_qc=False):
    """Every full-length cell's trace in a movie, interpolated onto ``tgrid``.

    With ``apply_qc`` (needs area/centroid columns), tracks failing stability QC
    (identity switches / merged blobs) are dropped first -- see ``fluora.qc``.
    Returns (traces (n_cells, len(tgrid)), n_kept, n_total_full_length).
    """
    d = pd.read_csv(csv_path)
    good = d.groupby("cell_id").filter(lambda g: len(g) >= min_frames)
    n_full = int(good["cell_id"].nunique()) if not good.empty else 0
    if apply_qc and not good.empty:
        from fluora.qc import qc_from_traces
        q = qc_from_traces(good)
        keep_ids = set(q.loc[q.pass_qc, "cell_id"].astype(int))
        good = good[good["cell_id"].astype(int).isin(keep_ids)]
    rows = []
    for _, g in good.groupby("cell_id"):
        g = g.sort_values("time_seconds")
        rows.append(np.interp(tgrid, g["time_seconds"].values, g[value].values))
    arr = np.vstack(rows) if rows else np.empty((0, len(tgrid)))
    return arr, arr.shape[0], n_full


def stim_spans(u_csv_path) -> np.ndarray:
    """Light-pulse (on, off) intervals from a u.csv (columns ta, tb).

    Returns an (n, 2) array of [ta, tb] rows. Drawn as blue axvspans so a
    high-frequency (dense) train reads as a solid block and a low-frequency
    (sparse) train reads as separated bars — exactly as in the published fig_2i.
    """
    u = pd.read_csv(u_csv_path)
    return u[["ta", "tb"]].to_numpy()


# back-compat alias (older callers imported stim_pulses)
def stim_pulses(u_csv_path) -> np.ndarray:
    return stim_spans(u_csv_path)


def plot_decoder_timeseries(
    conditions, out, tgrid, pulses=None, errorbar="sem",
    show_replicates=False, xlabel="Time (s)", ylabel=r"$\mathbf{\Delta F/F_{0}}$",
    title=None, caption=None, ylim=(-0.3, 0.35), yticks=None,
    lsizes=None, legend_loc="lower left",
):
    """Overlay decoder condition traces in the ezplot house style (fig_2i look).

    Per-cell mean trace per movie, then the mean across movies with a
    replicate-level (movie) error band — the statistically honest version of the
    published fig_2i (whose band pooled all cells).

    Parameters
    ----------
    conditions : list of dict, each
        {"label": str, "color": hex, "movies": [csv paths]}
        Each movie csv is one replicate (per-cell traces from fluora.extract).
    pulses : (n, 2) array of [ta, tb] light-pulse intervals (from ``stim_spans``),
        drawn as blue axvspans behind the traces.
    errorbar : "sd" | "sem" | "ci"   error band across MOVIES (replicate level).
    show_replicates : also draw each movie's trace faintly (recommended at small n).
    lsizes : per-condition line widths (defaults to ezplot's [16, 12, 8] emphasis).
    title / caption : off by default — the published panels carry neither; the
        n cells / movies / experiments belong in the manuscript caption instead.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if lsizes is None:
        base = STYLE["lines.linewidth"]
        lsizes = [base, int(base * 0.75), int(base * 0.5)]  # 16, 12, 8

    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots()
        # blue stimulus bands behind everything
        if pulses is not None and len(pulses):
            spans = np.atleast_2d(pulses)
            for ta, tb in spans:
                ax.axvspan(ta, tb, color=STIM_COLOR, lw=0, alpha=0.8, zorder=0)
        for i, cond in enumerate(conditions):
            traces, ncells = [], 0
            for csv in cond["movies"]:
                y, nc = movie_trace(csv, tgrid)
                if y is not None:
                    traces.append(y)
                    ncells += nc
            if not traces:
                continue
            M = np.vstack(traces)
            nmov = M.shape[0]
            mean = M.mean(0)
            err = np.array([errorbar_halfwidth(M[:, j], errorbar) for j in range(M.shape[1])])
            color = cond["color"]
            lw = lsizes[i] if i < len(lsizes) else lsizes[-1]
            if show_replicates:
                for y in M:
                    ax.plot(tgrid, y, color=color, lw=lw * 0.2, alpha=0.35, zorder=2)
            ax.fill_between(tgrid, mean - err, mean + err, color=color,
                            alpha=0.28, lw=0, zorder=3)
            ax.plot(tgrid, mean, color=color, lw=lw, solid_capstyle="round",
                    solid_joinstyle="round", label=cond["label"], zorder=4)
        ax.axhline(0, color=INK, lw=4, ls=(0, (1, 2)), alpha=0.6, zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(tgrid[0], tgrid[-1])
        if ylim:
            ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.locator_params(axis="x", nbins=7)
        leg = ax.legend(loc=legend_loc, framealpha=0.9, fontsize=44)
        leg.set_zorder(6)
        if title:
            ax.set_title(title)
        if caption:
            fig.text(0.99, 0.005, caption, ha="right", fontsize=28, color="#555")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out


def regime_movie_means(csv_path, regimes, min_frames=55, value="dF_F0", apply_qc=True):
    """Per-movie mean ``value`` within each input-regime time window.

    For each (name, t_lo, t_hi) in ``regimes``, averages ``value`` over the
    full-length (QC-passed) cells and over the timepoints in [t_lo, t_hi).
    Returns {regime_name: mean or nan}.
    """
    d = pd.read_csv(csv_path)
    good = d.groupby("cell_id").filter(lambda g: len(g) >= min_frames)
    if apply_qc and not good.empty:
        from fluora.qc import qc_from_traces
        keep = set(qc_from_traces(good).query("pass_qc").cell_id.astype(int))
        good = good[good["cell_id"].astype(int).isin(keep)]
    out = {}
    for name, t_lo, t_hi in regimes:
        w = good[(good.time_seconds >= t_lo) & (good.time_seconds < t_hi)]
        out[name] = float(w[value].mean()) if len(w) else np.nan
    return out


def plot_input_regime_summary(
    conditions, regimes, out, errorbar="sem", ylabel=r"Mean $\mathbf{\Delta F/F_{0}}$",
    ylim=None, dodge=0.18, jitter=0.05, compare=True, min_frames=55, pvalues=None,
    interaction_p=None,
):
    """fig_2j-style summary: mean ΔF/F₀ per input regime, per decoder condition.

    x categories = input regimes (e.g. None / Sparse / Dense Input); within each,
    the decoder conditions are dodged side by side. Each point is one movie
    (replicate); the black marker is the mean with an error bar (``errorbar`` =
    sd/sem/ci across movies). With ``compare`` and exactly two conditions, a
    Welch t-test between them is drawn per regime (*/**/***/ns).

    conditions : list of {"label", "color", "movies": [csv paths]}.
    regimes    : list of (name, t_lo, t_hi).
    Returns (out_path, table_df) where table_df holds every per-movie value.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # gather per-movie values: rows = (regime, condition, movie_idx, value)
    rows = []
    for cond in conditions:
        for mi, csv in enumerate(cond["movies"]):
            vals = regime_movie_means(csv, regimes, min_frames=min_frames)
            for rname, *_ in regimes:
                rows.append(dict(regime=rname, condition=cond["label"],
                                 color=cond["color"], movie=mi, value=vals[rname]))
    table = pd.DataFrame(rows)
    reg_names = [r[0] for r in regimes]
    conds = [c["label"] for c in conditions]
    ncond = len(conds)
    offsets = np.linspace(-dodge, dodge, ncond) if ncond > 1 else [0.0]

    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(20, 16))
        ax.axhline(0, color=INK, lw=4, ls=(0, (1, 2)), alpha=0.6, zorder=1)
        for ci, cond in enumerate(conditions):
            color = cond["color"]
            for ri, rname in enumerate(reg_names):
                x0 = ri + offsets[ci]
                v = table[(table.regime == rname) & (table.condition == cond["label"])].value.to_numpy()
                v = v[~np.isnan(v)]
                if not len(v):
                    continue
                xs = x0 + rng.uniform(-jitter, jitter, len(v))
                ax.scatter(xs, v, s=600, color=color, alpha=0.75, linewidths=2,
                           edgecolors=INK, zorder=3)
                m = v.mean(); e = errorbar_halfwidth(v, errorbar)
                ax.errorbar(x0, m, yerr=e, fmt="o", ms=22, color=INK, ecolor=INK,
                            elinewidth=8, capsize=18, capthick=8, zorder=4)
        # significance between the two conditions within each regime
        if compare and ncond == 2:
            tops = []
            for rname in reg_names:
                vv = table[table.regime == rname].value.dropna()
                tops.append(vv.max() if len(vv) else 0.0)
            tick = 0.012
            for ri, rname in enumerate(reg_names):
                a = table[(table.regime == rname) & (table.condition == conds[0])].value.dropna()
                b = table[(table.regime == rname) & (table.condition == conds[1])].value.dropna()
                if len(a) > 1 and len(b) > 1:
                    # exact Holm-corrected p from the omnibus model if supplied,
                    # else a plain Welch t-test
                    if pvalues is not None and rname in pvalues:
                        from fluora.stats import format_p
                        lab = format_p(pvalues[rname])
                    else:
                        p = float(sp_stats.ttest_ind(a, b, equal_var=False).pvalue)
                        lab = ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns")
                    y = tops[ri] + 0.03
                    x1, x2 = ri + offsets[0], ri + offsets[1]
                    ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y], color=INK, lw=5)
                    ax.text((x1 + x2) / 2, y + tick, lab, ha="center", va="bottom",
                            fontsize=44, color=INK)
        # legend (condition colours) -- lower-left corner is clear of the data
        handles = [mpl.lines.Line2D([], [], marker="o", ls="none", ms=28,
                   markerfacecolor=c["color"], markeredgecolor=INK, markeredgewidth=2,
                   label=c["label"]) for c in conditions]
        ax.legend(handles=handles, loc="lower left", framealpha=0.9, fontsize=44)
        ax.set_xticks(range(len(reg_names)))
        ax.set_xticklabels([f"{n}\nInput" for n in reg_names])
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, len(reg_names) - 0.4)
        if ylim:
            ax.set_ylim(ylim)
        if interaction_p is not None:
            from fluora.stats import format_p
            ax.set_title(f"decoder x input interaction  {format_p(interaction_p)}", fontsize=48)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out, table


def plot_decoder_tuning(
    conditions, regimes, out, compare=("Sparse", "Dense"),
    regime_colors=None, errorbar="sem", ylabel=r"Mean $\mathbf{\Delta F/F_{0}}$",
    ylim=None, dodge=0.2, min_frames=55,
):
    """Within-decoder tuning: each decoder's response to its matched vs mismatched
    input, paired across movies.

    x categories = decoders; within each, the two ``compare`` regimes are dodged
    and the same movie's two points are joined by a faint line (paired design).
    A **paired** t-test between the two regimes is drawn per decoder. This is the
    natural, higher-power test of the decoder claim (each decoder prefers its own
    input frequency), vs the between-decoder test in ``plot_input_regime_summary``.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if regime_colors is None:  # light->dark blue = low->high input frequency
        regime_colors = {"Sparse": "#9DC3E6", "Dense": "#2E5984"}
    rng = np.random.default_rng(0)
    conds = [c["label"] for c in conditions]
    offs = np.linspace(-dodge, dodge, len(compare))

    with mpl.rc_context({**STYLE, "xtick.labelsize": 64}):
        fig, ax = plt.subplots(figsize=(24, 16))
        ax.axhline(0, color=INK, lw=4, ls=(0, (1, 2)), alpha=0.6, zorder=1)
        for ci, cond in enumerate(conditions):
            # per-movie value in each compared regime (aligned by movie index)
            vals = {r: [] for r in compare}
            for csv in cond["movies"]:
                m = regime_movie_means(csv, regimes, min_frames=min_frames)
                for r in compare:
                    vals[r].append(m[r])
            V = {r: np.array(v) for r, v in vals.items()}
            xj = {r: ci + offs[k] + rng.uniform(-0.03, 0.03, len(V[r]))
                  for k, r in enumerate(compare)}
            # paired connector lines (same movie across the two regimes)
            for mi in range(len(cond["movies"])):
                ys = [V[r][mi] for r in compare]
                if not any(np.isnan(ys)):
                    ax.plot([xj[compare[0]][mi], xj[compare[1]][mi]], ys,
                            color=INK, lw=2, alpha=0.3, zorder=2)
            for k, r in enumerate(compare):
                v = V[r][~np.isnan(V[r])]
                ax.scatter(xj[r], V[r], s=550, color=regime_colors.get(r, "#888"),
                           alpha=0.85, linewidths=2, edgecolors=INK, zorder=3)
                ax.errorbar(ci + offs[k], v.mean(), yerr=errorbar_halfwidth(v, errorbar),
                            fmt="o", ms=20, color=INK, ecolor=INK, elinewidth=8,
                            capsize=16, capthick=8, zorder=4)
            # paired significance between the two regimes for this decoder
            a = V[compare[0]]; b = V[compare[1]]
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() > 1:
                p = float(sp_stats.ttest_rel(a[ok], b[ok]).pvalue)
                star = ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns")
                y = np.nanmax(np.concatenate([a, b])) + 0.03
                x1, x2 = ci + offs[0], ci + offs[-1]
                ax.plot([x1, x1, x2, x2], [y, y + 0.012, y + 0.012, y], color=INK, lw=5)
                ax.text(ci, y + 0.012, star, ha="center", va="bottom", fontsize=56, color=INK)
        handles = [mpl.lines.Line2D([], [], marker="o", ls="none", ms=26,
                   markerfacecolor=regime_colors.get(r, "#888"), markeredgecolor=INK,
                   markeredgewidth=2, label=f"{r} input") for r in compare]
        ax.legend(handles=handles, loc="lower left", framealpha=0.9, fontsize=44)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        if ylim:
            ax.set_ylim(ylim)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out


def plot_single_cell_traces(
    conditions, out, tgrid, pulses=None, xlabel="Time (s)",
    ylabel=r"$\mathbf{\Delta F/F_{0}}$", ylim=(-0.6, 0.9), yticks=None,
    cell_alpha=0.3, cell_lw=1.5, cell_color=None, mean_lw=4.0,
    min_frames=55, pulse_alpha=0.35, apply_qc=True,
):
    """One panel per condition: every single-cell ΔF/F₀ trajectory + the mean.

    Answers the reviewer's single-cell focus directly — each thin translucent
    line is one tracked cell; the bold white-haloed line is the mean across all
    of that condition's cells (the flowsmith individuals-plus-haloed-mean idiom).
    The spread of the thin lines *is* the error display, so no band is drawn.

    conditions : list of dict {"label", "color", "movies": [csv paths]}.
    pulses : (n,2) [ta,tb] light spans (from ``stim_spans``) drawn as blue bands.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(conditions)
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, n, figsize=(12 * n, 16), sharey=True)
        axes = np.atleast_1d(axes)
        spans = np.atleast_2d(pulses) if pulses is not None and len(pulses) else None
        for ax, cond in zip(axes, conditions):
            cells, n_full = [], 0
            for csv in cond["movies"]:
                arr, n_kept, nf = cell_traces(csv, tgrid, min_frames, apply_qc=apply_qc)
                n_full += nf
                if arr.size:
                    cells.append(arr)
            C = np.vstack(cells) if cells else np.empty((0, len(tgrid)))
            ncells = C.shape[0]
            n_dropped = n_full - ncells
            if spans is not None:
                for ta, tb in spans:
                    ax.axvspan(ta, tb, color=STIM_COLOR, lw=0, alpha=pulse_alpha, zorder=0)
            # single cells: thin, only slightly transparent, in the condition
            # colour (matching the mean) unless a fixed cell_color is given
            ccolor = cell_color if cell_color is not None else cond["color"]
            for y in C:
                ax.plot(tgrid, y, color=ccolor, lw=cell_lw, alpha=cell_alpha, zorder=2)
            if ncells:
                mean = C.mean(0)
                # mean: condition-coloured line, a white halo to lift it off the
                # same-coloured cells, and a thin dark core so it pops
                ax.plot(tgrid, mean, color="white", lw=mean_lw + 3.0, zorder=3.8,
                        solid_capstyle="round", solid_joinstyle="round")
                ax.plot(tgrid, mean, color=cond["color"], lw=mean_lw, zorder=4,
                        solid_capstyle="round", solid_joinstyle="round")
                ax.plot(tgrid, mean, color=INK, lw=max(mean_lw * 0.35, 1.2), zorder=4.1,
                        solid_capstyle="round", solid_joinstyle="round")
            ax.axhline(0, color=INK, lw=4, ls=(0, (1, 2)), alpha=0.6, zorder=1)
            # cell counts go in the caption (not on-figure), matching the other panels
            ax.set_title(cond["label"], fontsize=64)
            if apply_qc:
                print(f"  {cond['label']}: {ncells} cells kept, {n_dropped} QC-dropped "
                      f"of {n_full} full-length")
            ax.set_xlabel(xlabel)
            ax.set_xlim(tgrid[0], tgrid[-1])
            ax.locator_params(axis="x", nbins=6)
            if ylim:
                ax.set_ylim(ylim)
            if yticks is not None:
                ax.set_yticks(yticks)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout(w_pad=0.6)
        fig.savefig(out)
        plt.close(fig)
    return out
