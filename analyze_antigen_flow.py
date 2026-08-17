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
from fluora.stats import blocked_interaction, format_p

warnings.filterwarnings("ignore")
D = Path("/home/phuong/projects/csc-revisions-2026/data/4--antigen/0--K562-fc-staining")
RESULTS = Path("results")
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


def plot(cd19, psma, out):
    rng = np.random.default_rng(0)
    with mpl.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(30, 15), sharey=True)
        for ax, (df, title) in zip(axes, [(cd19, "CD19"), (psma, "PSMA")]):
            for x, g in enumerate(ORDER):
                v = df[df.group == g].pct_pos.to_numpy()
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=550,
                           color=GCOLOR[g], alpha=0.85, linewidths=2, edgecolors=INK, zorder=3)
                ax.errorbar(x, v.mean(), yerr=errorbar_halfwidth(v, "sem"), fmt="o", ms=22,
                            color=INK, ecolor=INK, elinewidth=8, capsize=16, capthick=8, zorder=4)
            ax.set_title(f"{title} antigen", fontsize=64)
            ax.set_xticks(range(len(ORDER)))
            ax.set_xticklabels(ORDER, rotation=35, ha="right")
            ax.set_xlim(-0.6, len(ORDER) - 0.4)
        axes[0].set_ylabel("% Antigen+")
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
    return out


def main():
    cd19, _ = analyze("0--CD19", "CD19 (Dense-CD19 decoder)")
    psma, _ = analyze("1--PSMA", "PSMA (Sparse-PSMA decoder)")
    out = plot(cd19, psma, RESULTS / "fig_5b_antigen_flow.png")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
