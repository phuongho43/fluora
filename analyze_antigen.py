"""Reviewer-compliant statistics for the CAR-T antigen experiments (Fig 5c, 5f).

Fig 5c -- CAR-T cytotoxicity (in vitro): % killing of Dense-CD19 or Sparse-PSMA
K562 targets under None / Sparse / Dense input. Two-factor design (decoder x
input), n = 3 experimental replicates. Reviewers ask for ANOVA + correction +
exact p (#115/#116), the missing Dense-CD19 None-vs-Sparse test (#116), and a
quantification of the no-light background / leakiness (#48/#49) and cross-talk
(#95).

Fig 5f -- bilateral-tumor BLI (in vivo): tumor total flux over days 3-18 for
Dense-CD19 and Sparse-PSMA tumors under Dense or Sparse input (2x2), 5 mice/group.
Reviewer #117: plot on a log scale (BLI is log-normal), show individual mice, and
give a statistic at the last complete timepoint (day 18).

    uv run python analyze_antigen.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats as _st
import matplotlib as mpl
import matplotlib.pyplot as plt

from fluora.stats import two_factor_stats, format_p
from fluora.plotting import STYLE, INK, errorbar_halfwidth

warnings.filterwarnings("ignore")
D = Path("/home/phuong/projects/csc-revisions-2026/data/4--antigen")
RESULTS = Path("results")
DEC_COLOR = {"Dense-CD19": "#8069EC", "Sparse-PSMA": "#EA822C"}
INPUT_ORDER = ["None", "Sparse", "Dense"]


# ----------------------------- Fig 5c: cytotoxicity -----------------------------
def cytotox():
    d = pd.read_csv(D / "1--CAR-killing-assay/y.csv")
    d["decoder"] = d["class"].map({0: "Dense-CD19", 1: "Sparse-PSMA"})
    d["input"] = d["group"].map({0: "None", 1: "Sparse", 2: "Dense"})
    d = d.rename(columns={"response": "value"})
    d["input"] = pd.Categorical(d["input"], INPUT_ORDER, ordered=True)

    S = two_factor_stats(d, dv="value", factor_a="decoder", factor_b="input")
    print("\n" + "=" * 74)
    print("FIG 5c  CAR-T cytotoxicity (% killing)  n=3 replicates/group")
    print("=" * 74)
    print(d.groupby(["decoder", "input"], observed=True).value
            .agg(["mean", "std", "size"]).round(1).to_string())
    a = S["anova"]
    print("\nTwo-way ANOVA (decoder x input):")
    print(a[["Source", "DF", "F", "p_unc", "np2"]].round(4).to_string(index=False))
    print(f"decoder x input INTERACTION p = {S['lmm_interaction_p']:.2e}  <- antigen-specificity signature")
    if S["homoscedasticity"] is not None:
        print("equal_var:", bool(S["homoscedasticity"]["equal_var"].iloc[0]),
              "(Levene p=%.3f)" % S["homoscedasticity"]["pval"].iloc[0])

    # Post-hoc = the crossover test the claim rests on: within each input, do the
    # two decoders differ? (Dense-CD19 vs Sparse-PSMA, both % cytotoxicity). Tukey
    # HSD across all 6 cells (pooled error from the ANOVA; variances homogeneous,
    # Levene p=0.71) -- the standard, correctly-corrected post-hoc for a 2-way design.
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    d["_cell"] = d.decoder + "|" + d.input.astype(str)
    tuk = pairwise_tukeyhsd(d.value.to_numpy(), d["_cell"].to_numpy())
    tk = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
    tk["p"] = tk["p-adj"].astype(float)
    padj = {}
    print("\nBetween-decoder within each input (Dense-CD19 vs Sparse-PSMA, Tukey HSD):")
    for inp in INPUT_ORDER:
        g1, g2 = f"Dense-CD19|{inp}", f"Sparse-PSMA|{inp}"
        row = tk[((tk.group1 == g1) & (tk.group2 == g2)) | ((tk.group1 == g2) & (tk.group2 == g1))]
        p = float(row["p"].iloc[0]) if len(row) else np.nan
        padj[inp] = p
        dd = d[(d.decoder == "Dense-CD19") & (d.input == inp)].value.mean()
        sp = d[(d.decoder == "Sparse-PSMA") & (d.input == inp)].value.mean()
        print(f"    {inp:>6} input: Dense-CD19 {dd:.0f}% vs Sparse-PSMA {sp:.0f}%  {format_p(p)}")
    # reviewer #116 within-decoder Dense-CD19 None-vs-Sparse (also from the Tukey table)
    row = tk[((tk.group1 == "Dense-CD19|None") & (tk.group2 == "Dense-CD19|Sparse")) |
             ((tk.group1 == "Dense-CD19|Sparse") & (tk.group2 == "Dense-CD19|None"))]
    print(f"  (#116 Dense-CD19 None vs Sparse: {format_p(float(row['p'].iloc[0]))})")

    # leakiness / crosstalk quantification (#48/#49/#95)
    print("\nLeakiness (None input) & cross-talk (mismatched input):")
    for dec in ("Dense-CD19", "Sparse-PSMA"):
        none = d[(d.decoder == dec) & (d.input == "None")].value.mean()
        mism = "Sparse" if dec == "Dense-CD19" else "Dense"
        cx = d[(d.decoder == dec) & (d.input == mism)].value.mean()
        on = d[(d.decoder == dec) & (d.input == ("Dense" if dec == "Dense-CD19" else "Sparse"))].value.mean()
        print(f"  {dec}: None(leak)={none:.0f}%  {mism}(crosstalk)={cx:.0f}%  matched(on)={on:.0f}%")

    _plot_cytotox(d, planned_padj=padj)  # interaction p reported in the caption
    return d


def _plot_cytotox(d, planned_padj, interaction_p=None):
    rng = np.random.default_rng(0)
    decs = ["Dense-CD19", "Sparse-PSMA"]
    offs = np.linspace(-0.18, 0.18, len(decs))
    xpos = {(dec, inp): ri + offs[ci]
            for ci, dec in enumerate(decs) for ri, inp in enumerate(INPUT_ORDER)}
    with mpl.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(20, 16))
        for ci, dec in enumerate(decs):
            for ri, inp in enumerate(INPUT_ORDER):
                v = d[(d.decoder == dec) & (d.input == inp)].value.to_numpy()
                x0 = xpos[(dec, inp)]
                ax.scatter(x0 + rng.uniform(-0.05, 0.05, len(v)), v, s=600,
                           color=DEC_COLOR[dec], alpha=0.8, linewidths=2, edgecolors=INK, zorder=3)
                ax.errorbar(x0, v.mean(), yerr=errorbar_halfwidth(v, "sem"), fmt="o", ms=22,
                            color=INK, ecolor=INK, elinewidth=8, capsize=18, capthick=8, zorder=4)
        # crossover test: within each input, Dense-CD19 vs Sparse-PSMA (short bracket)
        for ri, inp in enumerate(INPUT_ORDER):
            p = planned_padj.get(inp)
            if p is None:
                continue
            x1, x2 = xpos[("Dense-CD19", inp)], xpos[("Sparse-PSMA", inp)]
            top = max(d[d.input == inp].value.max(), 0)
            y = top + 6
            ax.plot([x1, x1, x2, x2], [y - 2, y, y, y - 2], color=INK, lw=5)
            ax.text((x1 + x2) / 2, y, format_p(p), ha="center", va="bottom",
                    fontsize=38, color=INK)
        handles = [mpl.lines.Line2D([], [], marker="o", ls="none", ms=22,
                   markerfacecolor=DEC_COLOR[dc], markeredgecolor=INK, markeredgewidth=2,
                   label=dc) for dc in decs]
        ax.legend(handles=handles, loc="lower right", framealpha=0.95, fontsize=44)
        ax.set_xticks(range(len(INPUT_ORDER)))
        ax.set_xticklabels([f"{o}\nInput" for o in INPUT_ORDER])
        ax.set_ylabel("% Cytotoxicity")
        if interaction_p is not None:
            ax.set_title(f"decoder x input interaction  {format_p(interaction_p)}", fontsize=48)
        ax.set_xlim(-0.6, len(INPUT_ORDER) - 0.4)
        ax.set_ylim(0, 112)
        fig.tight_layout()
        fig.savefig(RESULTS / "fig_5c_cytotoxicity.png")
        plt.close(fig)
    print("saved", RESULTS / "fig_5c_cytotoxicity.png")


# ----------------------------- Fig 5f: in-vivo BLI ------------------------------
BLI_GROUP = {0: ("Dense-CD19", "Dense"), 1: ("Sparse-PSMA", "Dense"),
             2: ("Dense-CD19", "Sparse"), 3: ("Sparse-PSMA", "Sparse")}


def bli():
    d = pd.read_csv(D / "2--CAR-bilateral-tumor/y.csv")
    d["tumor"] = d.c.map(lambda c: BLI_GROUP[c][0])
    d["input"] = d.c.map(lambda c: BLI_GROUP[c][1])
    d["log10_flux"] = np.log10(d.y)
    # index mice within each group/timepoint (5 mice, in file order)
    d["mouse"] = d.groupby(["c", "t"]).cumcount()

    print("\n" + "=" * 74)
    print("FIG 5f  bilateral-tumor BLI  (log10 total flux; 5 mice/group)")
    print("=" * 74)
    last = d[d.t == 18].copy()
    print("Day-18 log10(flux) by group:")
    print(last.groupby(["input", "tumor"]).log10_flux.agg(["mean", "std", "size"]).round(2).to_string())

    # 2x2 ANOVA at day 18 on log10 flux; interaction = antigen-selective control
    S = two_factor_stats(last, dv="log10_flux", factor_a="tumor", factor_b="input")
    a = S["anova"]
    print("\nDay-18 two-way ANOVA (tumor x input) on log10(flux):")
    print(a[["Source", "DF", "F", "p_unc", "np2"]].round(4).to_string(index=False))
    print(f"tumor x input INTERACTION p = {S['lmm_interaction_p']:.2e}  <- antigen-selective tumor control")

    # planned matched-vs-mismatched at day 18 (the biological claim), Holm
    def cmp(inp, ta, tb):
        a_ = last[(last.input == inp) & (last.tumor == ta)].log10_flux
        b_ = last[(last.input == inp) & (last.tumor == tb)].log10_flux
        return float(_st.ttest_ind(a_, b_, equal_var=False).pvalue)
    planned = {
        "Dense input: Dense-CD19 (target) vs Sparse-PSMA (bystander)": cmp("Dense", "Dense-CD19", "Sparse-PSMA"),
        "Sparse input: Sparse-PSMA (target) vs Dense-CD19 (bystander)": cmp("Sparse", "Sparse-PSMA", "Dense-CD19"),
    }
    from statsmodels.stats.multitest import multipletests
    keys = list(planned)
    padj = multipletests([planned[k] for k in keys], method="holm")[1]
    print("\nDay-18 planned contrasts (Welch t on log10 flux, Holm, exact p):")
    for k, pc in zip(keys, padj):
        print(f"  {k:60s} {format_p(pc)}")

    day18_p = {"Dense": padj[0], "Sparse": padj[1]}
    _plot_bli(d, day18_p=day18_p)  # interaction p reported in the caption
    return d


def _plot_bli(d, interaction_p=None, day18_p=None):
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(30, 15), sharey=True)
        for ax, inp in zip(axes, ["Dense", "Sparse"]):
            ymax = 0
            for tumor in ["Dense-CD19", "Sparse-PSMA"]:
                g = d[(d.input == inp) & (d.tumor == tumor)]
                color = DEC_COLOR[tumor]
                for m, gm in g.groupby("mouse"):
                    gm = gm.sort_values("t")
                    ax.plot(gm.t, gm.y, color=color, lw=3, alpha=0.35, zorder=2)
                    ax.scatter(gm.t, gm.y, s=180, color=color, alpha=0.5,
                               edgecolors=INK, linewidths=1.5, zorder=3)
                med = g.groupby("t").y.median()
                ax.plot(med.index, med.values, color=color, lw=12, zorder=4,
                        marker="o", ms=22, markeredgecolor=INK, markeredgewidth=2,
                        label=tumor)
                ymax = max(ymax, g.y.max())
            # day-18 matched-vs-bystander significance: short bracket centered on
            # day 18, above the data points, p-value above it (log-scale spacing).
            if day18_p is not None and inp in day18_p:
                y = ymax * 1.5
                ax.plot([17.4, 17.4, 18.6, 18.6], [y / 1.25, y, y, y / 1.25],
                        color=INK, lw=5, zorder=5)
                ax.text(18, y * 1.35, format_p(day18_p[inp]), ha="center", va="bottom",
                        fontsize=40, color=INK)
                ax.set_ylim(top=ymax * 4)
            ax.set_yscale("log")
            ax.set_title(f"{inp} Input", fontsize=64)
            ax.set_xlabel("Day")
            ax.set_xticks([3, 6, 9, 12, 15, 18])
            ax.set_xlim(2, 19.5)
            ax.legend(loc="upper left", framealpha=0.9, fontsize=44)
        axes[0].set_ylabel("Tumor flux (p/s)")
        if interaction_p is not None:
            fig.suptitle(f"Day 18 tumor x input interaction  {format_p(interaction_p)}",
                         fontsize=52, y=0.915)
        fig.tight_layout(w_pad=1.0)
        fig.savefig(RESULTS / "fig_5f_bli.png")
        plt.close(fig)
    print("saved", RESULTS / "fig_5f_bli.png")


if __name__ == "__main__":
    cytotox()
    bli()
