"""Regenerate the ddFP decoder figure from fluora per-cell traces.

Reproducible driver on top of fluora.plotting. Expects the per-cell trace CSVs
(from the Modal pipeline) under RESULTS as ``<prefix>-<i>.csv``. Replicate model:
each movie csv = one technical replicate; biological n = number of experiment dates
(reported in the caption). See section3-image-analysis-statistics.md.

    uv run python analyze_decoders.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from fluora.plotting import (
    plot_single_cell_traces, plot_input_regime_summary, plot_decoder_tuning,
    stim_spans, DECODER_COLORS,
)

# input-regime windows from the FM pulse train: baseline before the first pulse,
# then low-freq (10 s-period) "sparse" pulses, then high-freq (5.6 s) "dense" pulses.
REGIMES = [("None", 0.0, 60.0), ("Sparse", 60.0, 186.0), ("Dense", 186.0, 1e9)]

RESULTS = Path("results")
SRC = Path("/home/phuong/projects/csc-revisions-2026/data/1--biosensor/7--decoder-new")

CONDITIONS = [
    {"label": "Plain-ddFP",  "color": DECODER_COLORS["plain"],  "prefix": "plainnew"},
    {"label": "Dense-ddFP",  "color": DECODER_COLORS["dense"],  "prefix": "densenew"},
    {"label": "Sparse-ddFP", "color": DECODER_COLORS["sparse"], "prefix": "sparsenew"},
]
N_MOVIES = 7
N_EXPERIMENTS = 2  # from image-file acquisition dates (2024-08-07, 2024-10-02)


def main():
    tgrid = np.sort(pd.read_csv(RESULTS / "sparsenew-3.csv").time_seconds.unique())
    for c in CONDITIONS:
        c["movies"] = [RESULTS / f"{c['prefix']}-{i}.csv" for i in range(N_MOVIES)]
    # single FM input protocol (same for all decoders); read spans from one u.csv
    pulses = stim_spans(next((SRC / "2--sparse-ddFP").glob("*/u.csv")))
    # Single-cell figure: every cell trajectory + understated mean (reviewer's focus).
    out_sc = plot_single_cell_traces(
        CONDITIONS, RESULTS / "fig_2i_singlecell.png", tgrid, pulses=pulses,
        ylim=(-0.6, 0.9), yticks=np.arange(-0.6, 0.91, 0.3),
    )
    # fig_2j-style per-regime summary + reviewer-compliant stats (ANOVA/LMM +
    # Holm correction + assumption tests + exact p) -- see fluora.stats.
    summ = [c for c in CONDITIONS if c["label"] in ("Dense-ddFP", "Sparse-ddFP")]
    from fluora.stats import decoder_regime_stats
    S = decoder_regime_stats(summ, REGIMES)
    print("\n=== Normality (Shapiro by regime) ===\n", S["normality"].round(3).to_string())
    print("\n=== Homoscedasticity (Levene) ===\n", S["homoscedasticity"].round(4).to_string())
    print("\n=== Mixed ANOVA (within=regime, between=decoder) ===\n", S["anova"].round(4).to_string(index=False))
    print(f"\n=== LMM decoder x regime interaction: p = {S['lmm_interaction_p']:.2e} ===")
    print("\n=== Post-hoc (Holm-corrected, exact p) ===\n", S["posthoc"].round(4).to_string(index=False))
    out_2j, table = plot_input_regime_summary(
        summ, REGIMES, RESULTS / "fig_2j_regime_means.png",
        errorbar="sem", ylim=(-0.35, 0.52), pvalues=S["between_p"],
    )  # interaction p reported in the caption, not on-figure
    piv = table.pivot_table(index=["regime", "condition"], values="value",
                            aggfunc=["mean", "std", "count"]).round(3)
    print(piv)
    # within-decoder tuning: matched vs mismatched input (paired across movies)
    out_tune = plot_decoder_tuning(
        summ, REGIMES, RESULTS / "fig_2j_decoder_tuning.png",
        compare=("Sparse", "Dense"), ylim=(-0.35, 0.52),
    )
    print("saved", out_tune)
    print(f"saved {out_sc} and {out_2j}  (per-cell traces; n={N_MOVIES} movies/condition; "
          f"{N_EXPERIMENTS} independent experiments 2024-08-07, 2024-10-02)")


if __name__ == "__main__":
    main()
