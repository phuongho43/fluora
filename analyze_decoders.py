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
# Figures live in one place for the whole project; results/ keeps only the data.
FIGURES = Path("/home/phuong/projects/csc-revisions-2026/figures/regenerated")
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
    # Three sub-plots side by side made this the widest source in the paper (~34 in) for a
    # 3.5 in slot, so its type printed at a third the size of its neighbours'. The canvas is now
    # half the size, which raises printed type without touching the aspect the layout depends on.
    out_sc = plot_single_cell_traces(
        CONDITIONS, FIGURES / "fig_2h_singlecell.png", tgrid, pulses=pulses,
        ylim=(-0.6, 0.9), yticks=np.arange(-0.6, 0.91, 0.3),
        # The mean is the point of this panel -- the single-cell cloud is context -- and it was
        # getting lost. Linewidths are in points, so widening the canvas thins every line relative
        # to the plot: moving this panel to its own 9.3 cm slot on 2026-09-08 took the canvas from
        # 25.2 to 37.8 inches and made the mean 1.5x thinner. 4.0 -> 9.0 restores that and then
        # some. The cells stay at cell_lw=1.5 so the contrast between them and the mean widens.
        mean_lw=9.0,
    )
    # fig_2j-style per-regime summary + reviewer-compliant stats (ANOVA/LMM +
    # Holm correction + assumption tests + exact p) -- see fluora.stats.
    summ = [c for c in CONDITIONS if c["label"] in ("Dense-ddFP", "Sparse-ddFP")]
    from fluora.stats import decoder_regime_stats
    # Stats over ALL THREE conditions, plot only the two decoders. The manuscript's Table S1
    # reports the Plain-ddFP contrasts too ("dense decoder versus plain control under sparse
    # input p = 0.006"), so those tests belong in the correction family: Holm over the 9
    # decoder-pair contrasts, not the 3 that remain when the control is dropped. Correcting
    # over 3 while reporting 9 is anti-conservative, and it made the on-figure sparse contrast
    # read p = 0.16 where Table S1 says 0.33. Same tests either way, different family.
    S = decoder_regime_stats(CONDITIONS, REGIMES)
    print("\n=== Normality (Shapiro by regime) ===\n", S["normality"].round(3).to_string())
    print("\n=== Homoscedasticity (Levene) ===\n", S["homoscedasticity"].round(4).to_string())
    print("\n=== Mixed ANOVA (within=regime, between=decoder) ===\n", S["anova"].round(4).to_string(index=False))
    print(f"\n=== LMM decoder x regime interaction: p = {S['lmm_interaction_p']:.2e} ===")
    print("\n=== Post-hoc (Holm-corrected, exact p) ===\n", S["posthoc"].round(4).to_string(index=False))
    # ---- statistics ledger -------------------------------------------------------------
    from fluora.stats import write_ledger
    inter = S["anova"][S["anova"].Source == "Interaction"].iloc[0]
    write_ledger(
        FIGURES / "fig_2i_regime_means.png", "analyze_decoders.py",
        "Holm over the 9 regime x decoder-pair contrasts, i.e. computed across ALL THREE "
        "conditions including the Plain-ddFP control, because the manuscript reports the Plain "
        "contrasts too. Correcting over only the 2 decoders (3 contrasts) while reporting 9 is "
        "anti-conservative.",
        {"decoder x regime interaction F": (float(inter.F), "F"),
         **{f"{regime} input, Dense-ddFP vs Sparse-ddFP": S["between_p"][regime]
            for regime, _, _ in REGIMES if regime in S["between_p"]}},
        notes=f"{SRC.name}; n={N_MOVIES} movies/condition, {N_EXPERIMENTS} experiments")

    out_2j, table = plot_input_regime_summary(
        summ, REGIMES, FIGURES / "fig_2i_regime_means.png",
        # Headroom for the three significance brackets. At the old square aspect 0.52 was enough;
        # flattening the panel on 2026-09-08 left the Dense bracket clipped against the top spine,
        # because the brackets are placed a fixed distance above the highest point in data units
        # and there is far less axis to absorb it.
        errorbar="sem", ylim=(-0.35, 0.62), pvalues=S["between_p"],
    )  # interaction p reported in the caption, not on-figure
    piv = table.pivot_table(index=["regime", "condition"], values="value",
                            aggfunc=["mean", "std", "count"]).round(3)
    print(piv)
    # within-decoder tuning: matched vs mismatched input (paired across movies)
    out_tune = plot_decoder_tuning(
        summ, REGIMES, FIGURES / "UNPLACED_decoder_tuning.png",
        compare=("Sparse", "Dense"), ylim=(-0.35, 0.52),
    )
    print("saved", out_tune)
    print(f"saved {out_sc} and {out_2j}  (per-cell traces; n={N_MOVIES} movies/condition; "
          f"{N_EXPERIMENTS} independent experiments 2024-08-07, 2024-10-02)")


if __name__ == "__main__":
    main()
