"""Reviewer-compliant statistics for the FM expression assays (Fig 4c, 4g).

Two two-factor (reporter x input) categorical designs:
  * FM-dual  (Fig 4g): dual decoder, Dense-YFP + Sparse-RFP under None / Sparse
    (1s-4s) / Dense (1s-1s) input. Reviewer #115 asks whether the Dense-YFP vs
    Sparse-RFP difference at dense input survives multiple-comparison correction.
  * FM-single (Fig 4c frequency sweep): Dense-RFP + Sparse-RFP across 6 input
    periods. Reviewer #44 wants ANOVA/LMM + correction for the sweep.

Readout, matching the manuscript Methods: per replicate, the mean microscopy
intensity (averaged over fields) is expressed as a **log2 fold-ratio to the
None/dark control** (per reporter) -- this makes the YFP and RFP channels
comparable. Each condition has 3 replicates (independent wells = the unit of
analysis, n = 3); several fields per replicate are technical observations.

    uv run python analyze_expression.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pingouin as pg

import matplotlib as mpl
import matplotlib.pyplot as plt
from fluora.stats import two_factor_stats, blocked_interaction, format_p
from fluora.plotting import STYLE, INK, errorbar_halfwidth

warnings.filterwarnings("ignore")
RESULTS = Path("results")
RCOLOR = {"Dense-YFP": "#8069EC", "Dense-RFP": "#8069EC", "Sparse-RFP": "#EA822C",
          "Reporter only": "#34495E", "TetR-LOVfast + Zdk-VP64": "#2ECC71",
          "TetR-iLIDslow + sspB-VP64": "#D143A4"}
D = Path("/home/phuong/projects/csc-revisions-2026/data/3--expression")
INPUT = {"0--dark": "None", "1--1s-20s": "1/20s", "2--1s-10s": "1/10s",
         "3--1s-4s": "1/4s", "4--1s-2s": "1/2s", "5--1s-1s": "1/1s",
         "1--1s-4s": "Sparse", "2--1s-1s": "Dense"}
REPORTER = {"0--dense-YFP": "Dense-YFP", "1--sparse-RFP": "Sparse-RFP",
            "0--dense-RFP": "Dense-RFP"}
FLOOR = 1e-5  # detection floor for the log2 fold-ratio (None means are ~1e-4)


def load_log2fc(base, reporter_map=None, input_map=None, baseline="None"):
    """Per-replicate log2 fold-ratio to ``baseline`` (per reporter).

    ``reporter_map`` / ``input_map`` translate the folder names at the reporter
    (path[-5]) and input (path[-4]) levels; both default to the FM maps. The
    per-reporter ``baseline`` condition is the normalization reference.
    """
    reporter_map = REPORTER if reporter_map is None else reporter_map
    input_map = INPUT if input_map is None else input_map
    rows = []
    for rd in (D / base).rglob("results/y.csv"):
        p = rd.parts
        rows.append(dict(reporter=reporter_map.get(p[-5], p[-5]),
                         input=input_map.get(p[-4], p[-4]), rep=p[-3],
                         raw=float(pd.read_csv(rd).y.mean())))
    df = pd.DataFrame(rows)
    base_mean = df[df.input == baseline].groupby("reporter").raw.mean()  # per-reporter baseline
    df["value"] = np.log2((df.raw + FLOOR) / (df.reporter.map(base_mean) + FLOOR))
    return df


def contrast_reporters_within_input(df, at_input):
    """Holm-corrected reporter-vs-reporter test within one input level."""
    ph = pg.pairwise_tests(df, dv="value", between=["input", "reporter"], padjust="holm")
    sub = ph[ph.Contrast == "input * reporter"]
    r = sub[sub["input"] == at_input]
    return None if r.empty else r.iloc[0]


def report(name, df, order):
    df = df.copy()
    df["input"] = pd.Categorical(df["input"], order, ordered=True)
    S = two_factor_stats(df, dv="value", factor_a="reporter", factor_b="input")
    print("\n" + "=" * 74)
    print(f"{name}   (log2 fold-ratio to None; n=3 replicates/group)")
    print("=" * 74)
    print(df.groupby(["input", "reporter"], observed=True).value
            .agg(["mean", "std", "size"]).round(3).to_string())
    # replicate-blocked interaction (reps are paired experimental batches)
    B = blocked_interaction(df, dv="value", factor_a="reporter", factor_b="input", subject="rep")
    print(f"\nBLOCKED (rep as random block) reporter x input INTERACTION p = "
          f"{B['interaction_p']:.2e}   <- decoding signature (accounts for batch effects)")
    if B["anova"] is not None:
        print(B["anova"].round(4).to_string(index=False))
    S["blocked_interaction_p"] = B["interaction_p"]; S["blocked_anova"] = B["anova"]
    if S["normality"] is not None:
        print("\nNormality (Shapiro by input), homoscedasticity (Levene):")
        print("  normal by group:", S["normality"]["normal"].to_dict())
    if S["homoscedasticity"] is not None:
        print("  equal_var:", bool(S["homoscedasticity"]["equal_var"].iloc[0]),
              "(Levene p=%.3f)" % S["homoscedasticity"]["pval"].iloc[0])
    print("\nPost-hoc: input-within-reporter (Holm, exact p) -- does each reporter"
          " track its matched input?")
    ph = S["posthoc"]
    inter = ph[ph.Contrast == "reporter * input"]
    print(inter[["reporter", "A", "B", "p_corr", "hedges"]].round(4).to_string(index=False))
    return df, S


def plot_grouped_strip(df, order, out, reporters, pvals=None, ylabel=None,
                       interaction_p=None, legend_loc="upper left"):
    """Fig 4g-style: reporters dodged within each input; per-rep points + mean±SEM."""
    rng = np.random.default_rng(0)
    offs = np.linspace(-0.18, 0.18, len(reporters))
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(20, 16))
        ax.axhline(0, color=INK, lw=4, ls=(0, (1, 2)), alpha=0.6, zorder=1)
        for ci, rep in enumerate(reporters):
            for ri, inp in enumerate(order):
                v = df[(df.reporter == rep) & (df.input == inp)].value.to_numpy()
                if not len(v):
                    continue
                x0 = ri + offs[ci]
                ax.scatter(x0 + rng.uniform(-0.05, 0.05, len(v)), v, s=600,
                           color=RCOLOR[rep], alpha=0.8, linewidths=2, edgecolors=INK, zorder=3)
                ax.errorbar(x0, v.mean(), yerr=errorbar_halfwidth(v, "sem"), fmt="o", ms=22,
                            color=INK, ecolor=INK, elinewidth=8, capsize=18, capthick=8, zorder=4)
        if pvals:
            for ri, inp in enumerate(order):
                if inp in pvals:
                    y = df[df.input == inp].value.max() + 0.6  # above this input's own data
                    ax.plot([ri + offs[0], ri + offs[0], ri + offs[-1], ri + offs[-1]],
                            [y, y + 0.25, y + 0.25, y], color=INK, lw=5)
                    ax.text(ri, y + 0.3, format_p(pvals[inp]), ha="center", va="bottom",
                            fontsize=44, color=INK)
        handles = [mpl.lines.Line2D([], [], marker="o", ls="none", ms=26,
                   markerfacecolor=RCOLOR[r], markeredgecolor=INK, markeredgewidth=2,
                   label=r) for r in reporters]
        ax.legend(handles=handles, loc=legend_loc, framealpha=0.9, fontsize=44)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{o}\nInput" for o in order])
        ax.set_ylabel(ylabel or r"$\mathbf{Log_2}$ Norm. Output")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(top=df.value.max() + 1.4)
        if interaction_p is not None:
            ax.set_title(f"reporter x input interaction  {format_p(interaction_p)}", fontsize=48)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out


def within_reporter_tukey(df, reporter, ia, ib):
    """Tukey-HSD p for input ``ia`` vs ``ib`` within one reporter/component.

    Post-hoc for the randomized-block panels: the block (rep) enters the omnibus
    RM-ANOVA; the pairwise uses Tukey's pooled error (homoscedastic per Levene),
    which keeps power at n=3 instead of collapsing to the 2-df paired test.
    """
    sub = df[df.reporter == reporter]
    tuk = pg.pairwise_tukey(sub, dv="value", between="input")
    pcol = "p-tukey" if "p-tukey" in tuk.columns else "p_tukey"
    r = tuk[((tuk.A == ia) & (tuk.B == ib)) | ((tuk.A == ib) & (tuk.B == ia))]
    return float(r[pcol].iloc[0]) if len(r) else float("nan")


def plot_4e_components(df, order, out, reporters, brackets):
    """Fig 4e: the two Sparse-decoder components overlaid, with *within-component*
    significance brackets (matching the original fig 4e).

    ``brackets`` : list of (reporter, input_a, input_b, y_level) — a within-component
    comparison drawn from column ``input_a`` to ``input_b`` at height ``y_level``.
    """
    rng = np.random.default_rng(0)
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(20, 16))
        ax.axhline(0, color=INK, lw=4, ls=(0, (1, 2)), alpha=0.6, zorder=1)
        for rep in reporters:
            for ri, inp in enumerate(order):
                v = df[(df.reporter == rep) & (df.input == inp)].value.to_numpy()
                if not len(v):
                    continue
                ax.scatter(ri + rng.uniform(-0.07, 0.07, len(v)), v, s=600,
                           color=RCOLOR[rep], alpha=0.85, linewidths=2, edgecolors=INK, zorder=3)
                ax.errorbar(ri, v.mean(), yerr=errorbar_halfwidth(v, "sem"), fmt="o", ms=22,
                            color=INK, ecolor=INK, elinewidth=8, capsize=18, capthick=8, zorder=4)
        for rep, ia, ib, y in brackets:
            p = within_reporter_tukey(df, rep, ia, ib)
            x1, x2 = order.index(ia), order.index(ib)
            ax.plot([x1, x1, x2, x2], [y - 0.4, y, y, y - 0.4], color=INK, lw=5, zorder=5)
            ax.text((x1 + x2) / 2, y + 0.1, format_p(p), ha="center", va="bottom",
                    fontsize=40, color=INK)
        handles = [mpl.lines.Line2D([], [], marker="o", ls="none", ms=26,
                   markerfacecolor=RCOLOR[r], markeredgecolor=INK, markeredgewidth=2,
                   label=r) for r in reporters]
        ax.legend(handles=handles, loc="lower left", framealpha=0.95, fontsize=44)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{o}\nInput" for o in order])
        ax.set_ylabel(r"$\mathbf{Log_2}$ Norm. Output")
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(top=max(y for *_, y in brackets) + 1.5)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out


def freq_reporter_pvals(df, inputs, reporters):
    """Tukey-HSD p for reporter-vs-reporter within each given input (pooled error)."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    d = df.copy()
    d["_cell"] = d.reporter + "|" + d.input.astype(str)
    tuk = pairwise_tukeyhsd(d.value.to_numpy(), d["_cell"].to_numpy())
    tk = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
    tk["p"] = tk["p-adj"].astype(float)
    out = {}
    for inp in inputs:
        g1, g2 = f"{reporters[0]}|{inp}", f"{reporters[1]}|{inp}"
        r = tk[((tk.group1 == g1) & (tk.group2 == g2)) | ((tk.group1 == g2) & (tk.group2 == g1))]
        if len(r):
            out[inp] = float(r["p"].iloc[0])
    return out


HZ_TICKS = {"None": "0", "1/20s": "0.05", "1/10s": "0.1",
            "1/4s": "0.25", "1/2s": "0.5", "1/1s": "1"}


def plot_freq_response(df, order, out, reporters, interaction_p=None, pvals=None,
                       xlabel="Input Pulse Freq. (Hz)", xtick_map=None,
                       ylabel=r"$\mathbf{Log_2}$ Norm. Output"):
    """Fig 4c-style response curve: log2fc vs an ordered input axis, per reporter."""
    xtick_map = HZ_TICKS if xtick_map is None else xtick_map
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(22, 16))
        ax.axhline(0, color=INK, lw=4, ls=(0, (1, 2)), alpha=0.6, zorder=1)
        x = np.arange(len(order))
        means = {}
        for rep in reporters:
            m = [df[(df.reporter == rep) & (df.input == o)].value for o in order]
            mean = np.array([v.mean() for v in m])
            means[rep] = mean
            err = np.array([errorbar_halfwidth(v.to_numpy(), "sem") for v in m])
            for xi, v in zip(x, m):
                ax.scatter([xi] * len(v), v, s=350, color=RCOLOR[rep], alpha=0.5,
                           linewidths=2, edgecolors=INK, zorder=3)
            ax.plot(x, mean, color=RCOLOR[rep], lw=12, marker="o", ms=26,
                    markeredgecolor=INK, markeredgewidth=2, label=rep, zorder=4)
            ax.fill_between(x, mean - err, mean + err, color=RCOLOR[rep], alpha=0.2, zorder=2)
        # Dense-vs-Sparse significance: a short bracket centered above the data
        # points at that frequency, p-value above it.
        if pvals:
            for inp, p in pvals.items():
                if inp not in order:
                    continue
                xi = order.index(inp)
                ytop = df[df.input == inp].value.max()
                y = ytop + 0.6
                ax.plot([xi - 0.18, xi - 0.18, xi + 0.18, xi + 0.18],
                        [y - 0.25, y, y, y - 0.25], color=INK, lw=5, zorder=5)
                ax.text(xi, y + 0.15, format_p(p), ha="center", va="bottom",
                        fontsize=40, color=INK)
        ax.set_ylim(top=df.value.max() + 1.4)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=44)
        ax.set_xticks(x)
        ax.set_xticklabels([xtick_map.get(o, o) for o in order])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if interaction_p is not None:
            ax.set_title(f"reporter x frequency interaction  {format_p(interaction_p)}", fontsize=52)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out


def main():
    dorder = ["None", "Sparse", "Dense"]
    fd, Sd = report("FM-DUAL  (Fig 4g)", load_log2fc("3--293T-FM-dual"), dorder)
    # Dense-YFP vs Sparse-RFP within each input, Tukey (pooled error)
    pv = freq_reporter_pvals(fd, ["Sparse", "Dense"], ["Dense-YFP", "Sparse-RFP"])
    print(f"\n>>> Reviewer #115: Dense-YFP vs Sparse-RFP at DENSE input "
          f"(Tukey): {format_p(pv.get('Dense'))}")
    print(f"    (at SPARSE input: {format_p(pv.get('Sparse'))})")
    out = plot_grouped_strip(fd, dorder, RESULTS / "fig_4g_dual_decoder.png",
                             ["Dense-YFP", "Sparse-RFP"], pvals=pv)  # interaction p -> caption
    print("saved", out)

    # --- Fig 4c + Fig S4: frequency sweep (293T and K562) ---
    sorder = ["None", "1/20s", "1/10s", "1/4s", "1/2s", "1/1s"]
    reps = ["Dense-RFP", "Sparse-RFP"]
    for base, out_name, label in [
        ("1--293T-FM-single", "fig_4c_freq_sweep.png", "FM-SINGLE (Fig 4c, 293T)"),
        ("4--K562-FM-single", "fig_S4_K562_freq_sweep.png", "FM-SINGLE (Fig S4, K562)"),
    ]:
        fs, Ss = report(label, load_log2fc(base), sorder)
        pv = freq_reporter_pvals(fs, ["1/4s", "1/1s"], reps)
        for inp, p in pv.items():
            print(f"  Dense-RFP vs Sparse-RFP at {inp}: {format_p(p)}")
        out = plot_freq_response(fs, sorder, RESULTS / out_name, reps, pvals=pv)
        print("saved", out)

    # --- Fig 4b: blue-light intensity dose-response (reporter-only vs Dense-RFP) ---
    imap = {"0--BL00uW": "0", "1--BL01uW": "1", "2--BL05uW": "5",
            "3--BL10uW": "10", "4--BL50uW": "50"}
    rmap = {"0--reporter-only": "Reporter only", "1--dense-RFP": "Dense-RFP"}
    iorder = ["0", "1", "5", "10", "50"]
    fb = load_log2fc("0--293T-intensity", reporter_map=rmap, input_map=imap, baseline="0")
    _, _ = report("INTENSITY (Fig 4b)", fb, iorder)
    ireps = ["Reporter only", "Dense-RFP"]
    pv = freq_reporter_pvals(fb, ["10", "50"], ireps)  # sig at higher intensities
    for inp, p in pv.items():
        print(f"  Reporter-only vs Dense-RFP at {inp} uW: {format_p(p)}")
    out = plot_freq_response(
        fb, iorder, RESULTS / "fig_4b_intensity.png", ireps, pvals=pv,
        xlabel=r"Blue Light Intensity ($\mathbf{\mu}$W/mm$^2$)",
        xtick_map={i: i for i in iorder})
    print("saved", out)

    # --- Fig 4e: individual Sparse-RFP components under None/Sparse/Dense ---
    cmap = {"0--12TetO-YB-mScI_TetR-mNLS-LOV27V_Zdk1-mNLS-VP64": "TetR-LOVfast + Zdk-VP64",
            "1--12TetO-YB-mScI_TetR-mNLS-iLID16I_sspBn-mNLS-VP64": "TetR-iLIDslow + sspB-VP64"}
    fe = load_log2fc("2--293T-iLID-vs-LOV", reporter_map=cmap)
    _, _ = report("COMPONENTS (Fig 4e)", fe, dorder)
    lov, ilid = "TetR-LOVfast + Zdk-VP64", "TetR-iLIDslow + sspB-VP64"
    # Within-component comparisons (matching original fig 4e):
    #   iLIDslow activates at both sparse AND dense (None-vs-Sparse, None-vs-Dense);
    #   LOVfast represses only at dense (Sparse-vs-Dense).
    brackets = [(ilid, "None", "Sparse", 11.0), (ilid, "None", "Dense", 12.6),
                (lov, "Sparse", "Dense", 1.6)]
    for rep, ia, ib, _ in brackets:
        print(f"  {rep.split(' + ')[0]}: {ia} vs {ib}  "
              f"{format_p(within_reporter_tukey(fe, rep, ia, ib))}")
    out = plot_4e_components(fe, dorder, RESULTS / "fig_4e_components.png", [lov, ilid], brackets)
    print("saved", out)


if __name__ == "__main__":
    main()
