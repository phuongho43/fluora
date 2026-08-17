"""Reviewer-compliant statistics for the decoder / input-regime design.

Reviewers asked (critiques #44, #115) that multi-group designs with a repeated
factor (none/sparse/dense) be analysed with **ANOVA / linear mixed models and
multiple-testing correction**, with exact p-values, biological-vs-technical
replicates specified, and normality/homoscedasticity assumptions tested. This
module provides exactly that for the per-movie regime-mean table.

Design: per-movie mean ΔF/F₀ in each input regime. ``regime`` (None/Sparse/Dense)
is a **within-movie repeated** factor; ``decoder`` (Dense/Sparse) is a
**between-movie** factor; each movie is the subject (technical replicate). The
**decoder × regime interaction** is the frequency-decoding signature.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def regime_table(conditions, regimes, min_frames=55, apply_qc=True):
    """Long-format per-movie regime means: columns value, decoder, regime, movie."""
    from fluora.plotting import regime_movie_means
    rows = []
    for cond in conditions:
        for i, csv in enumerate(cond["movies"]):
            m = regime_movie_means(csv, regimes, min_frames=min_frames, apply_qc=apply_qc)
            for rname, *_ in regimes:
                rows.append(dict(value=m[rname], decoder=cond["label"],
                                 regime=rname, movie=f"{cond['label']}{i}"))
    df = pd.DataFrame(rows)
    df["regime"] = pd.Categorical(df["regime"], [r[0] for r in regimes])
    return df


def decoder_regime_stats(conditions, regimes, min_frames=55, apply_qc=True):
    """Full reviewer-compliant analysis of the decoder × regime design.

    Returns a dict with: ``table`` (long df), ``normality``, ``homoscedasticity``,
    ``anova`` (mixed ANOVA, Greenhouse-Geisser corrected), ``posthoc`` (Holm-
    corrected pairwise, exact p), ``lmm_interaction_p`` (LRT for the interaction
    from a linear mixed model with a per-movie random intercept), and
    ``between_p`` = {regime: Holm-corrected Dense-vs-Sparse p} for figure labels.
    """
    import pingouin as pg
    import statsmodels.formula.api as smf
    from scipy import stats as st

    df = regime_table(conditions, regimes, min_frames, apply_qc)
    grp = df.assign(grp=df.decoder + "_" + df.regime.astype(str))

    normality = pg.normality(df, dv="value", group="regime")
    homoscedasticity = pg.homoscedasticity(grp, dv="value", group="grp")
    anova = pg.mixed_anova(df, dv="value", within="regime", between="decoder",
                           subject="movie")
    posthoc = pg.pairwise_tests(df, dv="value", within="regime", between="decoder",
                                subject="movie", padjust="holm")

    # between-decoder Holm-corrected p per regime (for the fig_2j annotations)
    inter = posthoc[posthoc.Contrast == "regime * decoder"]
    between_p = {row.regime: float(row["p_corr"]) for _, row in inter.iterrows()}

    # LMM: value ~ decoder*regime + (1|movie); LRT for the interaction
    f_full = "value ~ C(decoder)*C(regime)"
    f_add = "value ~ C(decoder)+C(regime)"
    m1 = smf.mixedlm(f_full, df, groups=df.movie).fit(reml=False)
    m0 = smf.mixedlm(f_add, df, groups=df.movie).fit(reml=False)
    lmm_p = float(st.chi2.sf(2 * (m1.llf - m0.llf), len(regimes) - 1))

    return dict(table=df, normality=normality, homoscedasticity=homoscedasticity,
                anova=anova, posthoc=posthoc, between_p=between_p,
                lmm_interaction_p=lmm_p)


def two_factor_stats(df, dv, factor_a, factor_b, subject=None, padjust="holm"):
    """Reviewer-compliant analysis of a two-factor categorical design.

    Handles the between-subjects expression assays (reporter x input frequency),
    exactly the designs reviewers flagged for ANOVA + correction (critiques #44,
    #115). One value per replicate (aggregate fields first -- no pseudoreplication).

    Returns dict: ``anova`` (two-way, with the interaction = frequency-decoding
    signature), ``posthoc`` (pairwise ``factor_b`` within each ``factor_a`` and
    vice-versa, Holm-corrected, exact p), ``normality`` / ``homoscedasticity``
    (assumption tests; robust to zero-variance groups), and ``lmm_interaction_p``
    (LRT for the interaction from an OLS/LM with the interaction term).
    """
    import pingouin as pg
    import statsmodels.formula.api as smf
    from scipy import stats as st

    d = df.dropna(subset=[dv]).copy()
    # assumption tests (guard against all-constant groups where they're undefined)
    try:
        normality = pg.normality(d, dv=dv, group=factor_b)
    except Exception:
        normality = None
    d["_grp"] = d[factor_a].astype(str) + "|" + d[factor_b].astype(str)
    nonconst = d.groupby("_grp")[dv].transform("std").fillna(0) > 0
    try:
        homoscedasticity = pg.homoscedasticity(d[nonconst], dv=dv, group="_grp")
    except Exception:
        homoscedasticity = None

    anova = pg.anova(d, dv=dv, between=[factor_a, factor_b], detailed=True)
    posthoc = pg.pairwise_tests(d, dv=dv, between=[factor_a, factor_b],
                                padjust=padjust)
    # interaction via nested-model F (OLS): full vs additive
    m1 = smf.ols(f"{dv} ~ C({factor_a})*C({factor_b})", d).fit()
    m0 = smf.ols(f"{dv} ~ C({factor_a})+C({factor_b})", d).fit()
    from statsmodels.stats.anova import anova_lm
    lrt = anova_lm(m0, m1)
    inter_p = float(lrt["Pr(>F)"].iloc[-1])

    return dict(anova=anova, posthoc=posthoc, normality=normality,
                homoscedasticity=homoscedasticity, lmm_interaction_p=inter_p)


def format_p(p):
    """Compact exact-p label (reviewers asked for exact p-values)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 1e-3:
        return f"p={p:.0e}".replace("e-0", "e-")
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"
