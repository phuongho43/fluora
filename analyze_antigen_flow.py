"""Reviewer-compliant stats for the antigen flow cytometry (Fig 5b).

K562 decoders stained for CD19 or PSMA antigen; antibody signal is on **FL4_A**
(235x separation between the plain-K562 negative and the constitutive-antigen
positive). Per (marker, group, replicate) we compute the **% positive** = fraction
of events above a gate set from the plain-K562 negative control (99th percentile).
Each replicate is one experiment; % is computed **per replicate** (n = 3), not by
pooling cells across replicates (the manuscript aggregated cells -> pseudoreplication).

Reviewer asks (Sec 3/4): ANOVA on per-replicate %, report cells + n. Here: per-marker
one-way RM-ANOVA over None/Sparse/Dense input (replicate as block) + Tukey HSD.

    uv run python analyze_antigen_flow.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import flowkit as fk
import pingouin as pg
import matplotlib as mpl
import matplotlib.pyplot as plt

from fluora.plotting import STYLE, INK, errorbar_halfwidth
from fluora.stats import blocked_interaction, format_p, write_ledger

warnings.filterwarnings("ignore")
D = Path("/home/phuong/projects/csc-revisions-2026/data/4--antigen/0--K562-fc-staining")
RESULTS = Path("results")
# Figures live in one place for the whole project; results/ keeps only the data.
FIGURES = Path("/home/phuong/projects/csc-revisions-2026/figures/regenerated")
CH = "FL4_A"
GROUPS = {"0--plain-K562": "Plain", "1--const-antigen": "Const",
          "2--decoder-none-input": "None", "3--decoder-sparse-input": "Sparse",
          "4--decoder-dense-input": "Dense"}
ORDER = ["Plain", "Const", "None", "Sparse", "Dense"]
GCOLOR = {"Plain": "#B0B0B0", "Const": "#34495E", "None": "#7F8C8D",
          "Sparse": "#EA822C", "Dense": "#8069EC"}


def fl4(fcs_path):
    df = fk.Sample(str(fcs_path)).as_dataframe(source="raw")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[CH].to_numpy()


def load_marker(marker_dir):
    """Long df: group, rep, pct_pos, n_cells -- gate from pooled plain-K562."""
    plain = np.concatenate([fl4(p) for p in (marker_dir / "0--plain-K562").glob("*/*.fcs")])
    thresh = np.percentile(plain, 99.0)  # 1% false-positive on the negative control
    rows = []
    for gdir, glabel in GROUPS.items():
        for rep_dir in sorted((marker_dir / gdir).glob("*")):
            fcs = next(rep_dir.glob("*.fcs"), None)
            if fcs is None:
                continue
            v = fl4(fcs)
            rows.append(dict(group=glabel, rep=rep_dir.name,
                             pct_pos=100.0 * np.mean(v > thresh), n_cells=len(v)))
    return pd.DataFrame(rows), thresh


def analyze(marker, title):
    df, thresh = load_marker(D / marker)
    df["group"] = pd.Categorical(df["group"], ORDER, ordered=True)
    print("\n" + "=" * 68)
    print(f"FIG 5b {title}  (% {CH}-positive; gate = 99th pct of plain-K562)")
    print("=" * 68)
    print(df.groupby("group", observed=True).agg(
        pct_mean=("pct_pos", "mean"), pct_sd=("pct_pos", "std"),
        n_rep=("pct_pos", "size"), cells=("n_cells", "sum")).round(1).to_string())

    # decoder inputs only -> one-way RM-ANOVA (rep as block) + Tukey
    dec = df[df.group.isin(["None", "Sparse", "Dense"])].copy()
    dec["group"] = dec.group.astype(str)
    rm = pg.rm_anova(dec, dv="pct_pos", within="group", subject="rep")
    pcol = "p-unc" if "p-unc" in rm.columns else "p_unc"
    print(f"\nRM-ANOVA over inputs (rep block): F={rm['F'].iloc[0]:.1f}, "
          f"{format_p(float(rm[pcol].iloc[0]))}")
    tuk = pg.pairwise_tukey(dec, dv="pct_pos", between="group")
    tpc = "p-tukey" if "p-tukey" in tuk.columns else "p_tukey"
    pvals = {}
    for _, r in tuk.iterrows():
        pvals[frozenset((r.A, r.B))] = float(r[tpc])
        print(f"    {r.A:>6} vs {r.B:<6}  {format_p(float(r[tpc]))}")
    return df, pvals


def plot(panels, out):
    """panels: list of (df, title, pvals) -- draws the Dense-vs-Sparse input
    bracket per panel (the decoder's frequency selectivity)."""
    rng = np.random.default_rng(0)
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(30, 15), sharey=True)
        for ax, (df, title, pvals) in zip(axes, panels):
            for x, g in enumerate(ORDER):
                v = df[df.group == g].pct_pos.to_numpy()
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=550,
                           color=GCOLOR[g], alpha=0.85, linewidths=2, edgecolors=INK, zorder=3)
                ax.errorbar(x, v.mean(), yerr=errorbar_halfwidth(v, "sem"), fmt="o", ms=22,
                            color=INK, ecolor=INK, elinewidth=8, capsize=16, capthick=8, zorder=4)
            # bracket: Dense vs Sparse input (frequency selectivity)
            p = pvals.get(frozenset(("Dense", "Sparse")))
            if p is not None:
                x1, x2 = ORDER.index("Sparse"), ORDER.index("Dense")
                y = df.pct_pos.max() + 10
                ax.plot([x1, x1, x2, x2], [y - 4, y, y, y - 4], color=INK, lw=5)
                ax.text((x1 + x2) / 2, y + 1, format_p(p), ha="center", va="bottom",
                        fontsize=62, color=INK)
            ax.set_title(title, fontsize=64)
            ax.set_xticks(range(len(ORDER)))
            ax.set_xticklabels(ORDER, rotation=35, ha="right")
            ax.set_xlim(-0.6, len(ORDER) - 0.4)
        axes[0].set_ylabel("% Antigen+")
        axes[0].set_ylim(top=max(cd.pct_pos.max() for cd, *_ in panels) + 22)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out


def main():
    cd19, cd_p = analyze("0--CD19", "CD19 (Dense-CD19 decoder)")
    psma, ps_p = analyze("1--PSMA", "PSMA (Sparse-PSMA decoder)")
    out = plot([(cd19, "Dense-CD19", cd_p), (psma, "Sparse-PSMA", ps_p)],
               FIGURES / "fig_S7b_antigen_flow.png")
    print("\nsaved", out)
    # plot() draws ONE bracket per sub-panel -- Dense vs Sparse, the decoder's frequency
    # selectivity. The other Tukey pairs are computed but never appear on the figure.
    drawn, computed = {}, {}
    for label, pv in (("Dense-CD19", cd_p), ("Sparse-PSMA", ps_p)):
        for pair, p in pv.items():
            a, b = sorted(pair)
            target = drawn if {"Dense", "Sparse"} == set(pair) else computed
            target[f"{label}, {a} vs {b}"] = p
    write_ledger(FIGURES / "fig_S7b_antigen_flow.png", "analyze_antigen_flow.py",
                 "one-way RM-ANOVA over inputs with replicate as block, Tukey HSD pairwise",
                 drawn, computed_only=computed)


if __name__ == "__main__":
    main()
