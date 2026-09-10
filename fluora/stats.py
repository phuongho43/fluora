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
    # Key on the decoder PAIR as well as the regime. Keying on regime alone silently
    # kept whichever pair sorted last (Plain-vs-Sparse), not the Dense-vs-Sparse
    # contrast the name implies. Only reached when a caller annotates a 2-condition
    # plot, but it was wrong for the 3-condition call.
    between_p = {(row.regime, row.A, row.B): float(row["p_corr"])
                 for _, row in inter.iterrows()}
    for _, row in inter.iterrows():
        if {row.A, row.B} == {"Dense-ddFP", "Sparse-ddFP"}:
            between_p[row.regime] = float(row["p_corr"])

    # LMM: value ~ decoder*regime + (1|movie); LRT for the interaction
    f_full = "value ~ C(decoder)*C(regime)"
    f_add = "value ~ C(decoder)+C(regime)"
    m1 = smf.mixedlm(f_full, df, groups=df.movie).fit(reml=False)
    m0 = smf.mixedlm(f_add, df, groups=df.movie).fit(reml=False)
    # df = (n_decoders - 1) * (n_regimes - 1), not n_regimes - 1. The old form used 2
    # for the 3x3 design, where the interaction actually costs 4 parameters, and was
    # therefore anti-conservative. Take df from the fitted models so it cannot drift.
    lmm_df = len(m1.params) - len(m0.params)
    lmm_p = float(st.chi2.sf(2 * (m1.llf - m0.llf), lmm_df))
    lmm_converged = bool(m1.converged and m0.converged)

    return dict(table=df, normality=normality, homoscedasticity=homoscedasticity,
                anova=anova, posthoc=posthoc, between_p=between_p,
                lmm_interaction_p=lmm_p, lmm_interaction_df=lmm_df,
                lmm_converged=lmm_converged)


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


def blocked_interaction(df, dv, factor_a, factor_b, subject):
    """Interaction of a two-factor design that shares a paired ``subject`` block.

    The replicates/blocks are crossed with the conditions (each block contains all
    condition combinations -- randomized-block / repeated-measures design), so a
    plain between-subjects ANOVA ignores real structure and inflates the error.
    Here the block enters as a random intercept: ``dv ~ C(a)*C(b) + (1|subject)``.

    Returns dict with ``interaction_p`` (likelihood-ratio test of the interaction),
    plus ``anova`` = the repeated-measures / mixed ANOVA table (pingouin) for F/df.
    """
    import pingouin as pg
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm  # noqa: F401  (parity with two_factor)
    from scipy import stats as st

    d = df.dropna(subset=[dv]).copy()
    # Finite-sample F-test (the valid test at small n): fully-crossed (both within)
    # -> two-way RM-ANOVA; one within + one between -> mixed-design ANOVA. Small-n
    # asymptotics (LMM likelihood-ratio / Wald) are wildly anticonservative here.
    counts = d.groupby(subject)[[factor_a, factor_b]].nunique()
    both_within = (counts[factor_a] > 1).all() and (counts[factor_b] > 1).all()
    if both_within:
        anova = pg.rm_anova(d, dv=dv, within=[factor_a, factor_b], subject=subject)
    else:
        within, between = (factor_b, factor_a) if (counts[factor_a] == 1).any() else (factor_a, factor_b)
        anova = pg.mixed_anova(d, dv=dv, within=within, between=between, subject=subject)
    src = anova["Source"].astype(str)
    row = anova[src.str.contains(r"\*") | (src == "Interaction")].iloc[0]
    pcol = "p-unc" if "p-unc" in anova.columns else "p_unc"
    return dict(interaction_p=float(row[pcol]),
                interaction_F=float(row["F"]), anova=anova)


def mixed_contrast_p(df, dv, subject, group_col, a, b, cond_col=None, cond_val=None):
    """p for group ``a`` vs ``b`` from a replicate-blocked linear mixed model.

    Fits a cell-means model ``dv ~ 0 + cell + (1|subject)`` over the whole design
    (cell = ``group_col`` or ``group_col``x``cond_col``) and t-tests the contrast
    between the two cells. This blocks on ``subject`` (like a paired test, removing
    batch offsets) while pooling the residual variance across all cells, so it
    keeps power at small n instead of collapsing to n-1 df. Assumes homogeneous
    variances (check Levene). Returns the two-sided p.
    """
    import numpy as np
    import statsmodels.formula.api as smf

    d = df.dropna(subset=[dv]).copy()
    if cond_col is not None:
        d["_cell"] = d[group_col].astype(str) + "|" + d[cond_col].astype(str)
        la, lb = f"{a}|{cond_val}", f"{b}|{cond_val}"
    else:
        d["_cell"] = d[group_col].astype(str)
        la, lb = str(a), str(b)
    m = smf.mixedlm(f"{dv} ~ 0 + C(_cell)", d, groups=d[subject]).fit(reml=False)
    beta = m.fe_params
    names = list(beta.index)

    def _idx(lbl):
        hits = [i for i, n in enumerate(names) if f"[{lbl}]" in n]
        return hits[0] if hits else None

    ia, ib = _idx(la), _idx(lb)
    if ia is None or ib is None:
        return float("nan")
    k = len(names)
    c = np.zeros(k)
    c[ia], c[ib] = 1.0, -1.0
    cov = np.asarray(m.cov_params())[:k, :k]  # fixed-effects block
    est = float(c @ beta.values)
    se = float(np.sqrt(c @ cov @ c))
    from scipy import stats as st
    return float(2 * st.norm.sf(abs(est / se)))


def paired_p(df, dv, subject, group_col, a, b, cond_col=None, cond_val=None):
    """Paired t-test p for group ``a`` vs ``b`` (matched by ``subject``).

    Optionally restricted to ``cond_col == cond_val`` first. Pairs the two groups
    on the shared ``subject`` (block/mouse), removing between-block variance.
    """
    from scipy import stats as st
    d = df if cond_col is None else df[df[cond_col] == cond_val]
    pa = d[d[group_col] == a].groupby(subject)[dv].mean()
    pb = d[d[group_col] == b].groupby(subject)[dv].mean()
    common = pa.index.intersection(pb.index)
    if len(common) < 2:
        return float("nan")
    return float(st.ttest_rel(pa.loc[common], pb.loc[common]).pvalue)


def holm(pdict):
    """Holm-correct a {label: p} dict; returns {label: p_adj}."""
    from statsmodels.stats.multitest import multipletests
    keys = [k for k in pdict if pdict[k] == pdict[k]]  # drop nan
    if not keys:
        return dict(pdict)
    padj = multipletests([pdict[k] for k in keys], method="holm")[1]
    out = dict(pdict)
    out.update(dict(zip(keys, padj)))
    return out


def format_p(p):
    """Compact exact-p label (reviewers asked for exact p-values)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p <= 0:  # numeric underflow (e.g. Tukey studentized range) -> honest bound
        return "p<1e-4"
    if p < 1e-3:
        return f"p={p:.0e}".replace("e-0", "e-")
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def ledger_forms(value, kind="p"):
    """Every way the manuscript might legitimately write this number.

    A checker comparing a panel's annotation against the text has to allow for the fact that the
    same p-value is written "p = 0.33", "p=0.33", "0.33", and that 6e-6 appears as "6 x 10-6" with
    the exponent in a superscript run. Centralised so the four analysis drivers cannot each invent
    a slightly different set.
    """
    import numpy as _np
    if value is None or (isinstance(value, float) and _np.isnan(value)):
        return []
    if kind == "F":
        return [f"F = {value:.1f}", f"F={value:.1f}", f"{value:.1f}"]
    lab = format_p(value)
    out = [lab, lab.replace("p=", "p = "), lab.replace("p=", "")]
    if lab.startswith("p<"):
        # format_p reports an underflowed p as the bound "p<1e-4"; the manuscript writes the
        # same thing as "p < 10-4" with the exponent in a superscript run.
        out += ["p < 10-4", "p<10-4", "< 10-4", "p < 1e-4", "p < 0.0001"]
    if 0 < value < 1e-3:
        mant, exp = f"{value:.0e}".split("e")
        e = int(exp)
        out += [f"{mant} x 10{e}", f"{mant} x 10-{abs(e)}", f"{mant}e{e}", f"{mant}e-{abs(e)}",
                f"{mant} × 10-{abs(e)}"]
    elif value >= 1e-3:
        # "%.2f" of a small p is "0.00", which matches almost any sentence and would turn the
        # check into a rubber stamp. Only emit a rounded form that still carries information.
        two = f"{value:.2f}"
        if float(two) > 0:
            out.append(two)
        out.append(f"{value:.3f}".rstrip("0").rstrip("."))
    # Drop anything too short or too generic to be evidence -- "1." would match almost any
    # sentence and turn the check into a rubber stamp.
    return sorted({o for o in out if o and len(o) >= 3 and not o.endswith(".")})


def write_ledger(panel_path, driver, correction, values, notes=None, computed_only=None):
    """Record what a panel is annotated with, so a checker can hold the text to it.

    Written beside the panel as <panel>.stats.json and read by
    csc-revisions-2026/analysis/check_manuscript.py. Exists because Figure 2I carried p = 0.16
    while Table S1 said p = 0.33 -- the same contrast under a 3- and a 9-comparison Holm family --
    and no check compared the figure to the text.

        write_ledger(FIGURES / "fig_S7c_cytotoxicity.png", "analyze_antigen.py",
                     "Tukey HSD across the 3 between-decoder contrasts",
                     {"Sparse input, Dense-CD19 vs Sparse-PSMA": p_sparse, ...})

    `values` maps a human-readable label to a p-value (or to (value, "F") for an F statistic).
    Put a number under `computed_only` instead when the driver computes it but does NOT draw it on
    the panel -- a supporting analysis, or a panel Table S1 declares descriptive. Those are
    recorded for provenance and are not required to appear in the text; requiring them would make
    the check cry wolf, and a check that cries wolf gets ignored.
    """
    import json
    import datetime
    from pathlib import Path as _Path
    panel = _Path(panel_path)
    quoted = {}
    for label, v in values.items():
        kind = "p"
        if isinstance(v, (tuple, list)):
            v, kind = v
        forms = ledger_forms(float(v), kind)
        if forms:
            quoted[label] = forms
    extra = {}
    for label, v in (computed_only or {}).items():
        kind = "p"
        if isinstance(v, (tuple, list)):
            v, kind = v
        forms = ledger_forms(float(v), kind)
        if forms:
            extra[label] = forms
    rec = {"panel": panel.name, "driver": driver,
           "generated": datetime.datetime.now().isoformat(timespec="seconds"),
           "correction": correction, "quoted_in_text": quoted,
           "computed_not_annotated": extra}
    if notes:
        rec["notes"] = notes
    out = panel.with_suffix(".stats.json")
    out.write_text(json.dumps(rec, indent=2))
    print(f"  ledger {out.name}: {len(quoted)} value(s)")
    return out
