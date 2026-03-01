"""OLS regresjonsmotor med diagnostikk."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_single_regression(stock_returns: pd.Series, factor_data: pd.DataFrame) -> dict | None:
    """Kjør OLS-regresjon for én aksje mot alle faktorer.

    Args:
        stock_returns: Daglige log-avkastninger for aksjen.
        factor_data: DataFrame med faktorer som kolonner.

    Returns:
        Dict med betaer, t-stats, p-verdier, R², diagnostikk, osv.
        None dersom regresjonen feiler.
    """
    # Align på dato
    aligned = pd.concat([stock_returns, factor_data], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return None

    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return None

    # Betaer (ekskl. konstant)
    betas = model.params.drop("const")
    t_stats = model.tvalues.drop("const")
    p_values = model.pvalues.drop("const")

    # Diagnostikk
    dw = durbin_watson(model.resid)

    try:
        _, bp_pvalue, _, _ = het_breuschpagan(model.resid, X)
    except Exception:
        bp_pvalue = None

    return {
        "betas": betas.to_dict(),
        "t_stats": t_stats.to_dict(),
        "p_values": p_values.to_dict(),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": int(model.nobs),
        "start_date": aligned.index.min().strftime("%Y-%m-%d"),
        "durbin_watson": dw,
        "bp_pvalue": bp_pvalue,
        "model": model,
    }


def run_all_regressions(stock_returns: pd.DataFrame, factor_data: pd.DataFrame,
                        tickers: list[str]) -> dict:
    """Kjør regresjoner for alle aksjer + SPY benchmark.

    Returns:
        Dict med ticker -> regresjonsresultat.
    """
    results = {}
    for ticker in tickers:
        if ticker not in stock_returns.columns:
            continue
        ret = stock_returns[ticker].dropna()
        result = run_single_regression(ret, factor_data)
        if result is not None:
            results[ticker] = result
    return results


def calc_vif(factor_data: pd.DataFrame) -> pd.Series:
    """Beregn Variance Inflation Factor for alle faktorer."""
    clean = factor_data.dropna()
    if len(clean) < 10:
        return pd.Series(dtype=float)

    X = sm.add_constant(clean)
    vif_data = {}
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif_data[col] = variance_inflation_factor(X.values, i)
        except Exception:
            vif_data[col] = np.nan

    return pd.Series(vif_data)


def portfolio_betas(regression_results: dict, weights: dict,
                    factor_names: list[str]) -> dict:
    """Beregn vektede gjennomsnittlige faktor-betaer for porteføljen.

    Args:
        regression_results: Dict med ticker -> regresjonsresultat.
        weights: Dict med ticker -> normalisert vekt (0-1).
        factor_names: Liste med faktornavn.

    Returns:
        Dict med faktornavn -> vektet beta.
    """
    port_betas = {f: 0.0 for f in factor_names}
    total_weight = 0.0

    for ticker, w in weights.items():
        if ticker not in regression_results:
            continue
        betas = regression_results[ticker]["betas"]
        for f in factor_names:
            if f in betas:
                port_betas[f] += w * betas[f]
        total_weight += w

    # Renormaliser hvis noen aksjer mangler
    if total_weight > 0 and total_weight < 0.99:
        for f in factor_names:
            port_betas[f] /= total_weight

    return port_betas
