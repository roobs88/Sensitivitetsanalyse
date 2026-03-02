"""OLS regresjonsmotor med diagnostikk."""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


def run_single_regression(stock_returns: pd.Series, factor_data: pd.DataFrame,
                          window: int | None = None) -> dict | None:
    """Kjør OLS-regresjon for én aksje mot alle faktorer.

    Args:
        stock_returns: Daglige log-avkastninger for aksjen.
        factor_data: DataFrame med faktorer som kolonner.
        window: Antall handelsdager å bruke (None = all data).

    Returns:
        Dict med betaer, t-stats, p-verdier, R², diagnostikk, osv.
        None dersom regresjonen feiler.
    """
    # Align på dato
    aligned = pd.concat([stock_returns, factor_data], axis=1, join="inner").dropna()

    # Kutter til rullende vindu hvis spesifisert
    if window is not None and len(aligned) > window:
        aligned = aligned.iloc[-window:]

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
                        tickers: list[str], window: int | None = None) -> dict:
    """Kjør regresjoner for alle aksjer + SPY benchmark.

    Args:
        window: Antall handelsdager å bruke (None = all data).

    Returns:
        Dict med ticker -> regresjonsresultat.
    """
    results = {}
    for ticker in tickers:
        if ticker not in stock_returns.columns:
            continue
        ret = stock_returns[ticker].dropna()
        result = run_single_regression(ret, factor_data, window=window)
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


def _fit_ols(y, X):
    """Hjelpefunksjon: kjør OLS og returner betaer-dict eller None."""
    try:
        X_c = sm.add_constant(X)
        model = sm.OLS(y, X_c).fit()
        return model.params.drop("const").to_dict()
    except Exception:
        return None


def run_regime_regression(stock_returns: pd.Series, factor_data: pd.DataFrame,
                          vix_levels: pd.Series, threshold: float = 25,
                          window: int | None = None) -> dict | None:
    """Kjør separat OLS for normal- og stress-regime basert på VIX-nivå.

    Returns:
        Dict med betas (kombinert), betas_normal, betas_stress, regime_counts.
        None dersom regresjonen feiler.
    """
    aligned = pd.concat([stock_returns, factor_data], axis=1, join="inner").dropna()

    if window is not None and len(aligned) > window:
        aligned = aligned.iloc[-window:]

    if len(aligned) < 30:
        return None

    # Align VIX-nivåer
    vix_aligned = vix_levels.reindex(aligned.index).ffill()
    vix_aligned = vix_aligned.loc[aligned.index].dropna()
    common_idx = aligned.index.intersection(vix_aligned.index)
    aligned = aligned.loc[common_idx]
    vix_aligned = vix_aligned.loc[common_idx]

    if len(aligned) < 30:
        return None

    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]

    # Del i regimer
    normal_mask = vix_aligned <= threshold
    stress_mask = vix_aligned > threshold
    n_normal = normal_mask.sum()
    n_stress = stress_mask.sum()

    # Kjør kombinert regresjon først (fallback + diagnostikk)
    try:
        X_c = sm.add_constant(X)
        combined_model = sm.OLS(y, X_c).fit()
        combined_betas = combined_model.params.drop("const").to_dict()
    except Exception:
        return None

    # Regime-spesifikke regresjoner
    min_obs = 20
    if n_normal >= min_obs:
        betas_normal = _fit_ols(y[normal_mask], X[normal_mask])
    else:
        betas_normal = combined_betas

    if n_stress >= min_obs:
        betas_stress = _fit_ols(y[stress_mask], X[stress_mask])
    else:
        betas_stress = combined_betas

    if betas_normal is None:
        betas_normal = combined_betas
    if betas_stress is None:
        betas_stress = combined_betas

    # Diagnostikk fra kombinert modell
    t_stats = combined_model.tvalues.drop("const").to_dict()
    p_values = combined_model.pvalues.drop("const").to_dict()
    dw = durbin_watson(combined_model.resid)
    try:
        _, bp_pvalue, _, _ = het_breuschpagan(combined_model.resid, X_c)
    except Exception:
        bp_pvalue = None

    return {
        "betas": combined_betas,
        "betas_normal": betas_normal,
        "betas_stress": betas_stress,
        "regime_counts": {"normal": int(n_normal), "stress": int(n_stress)},
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": combined_model.rsquared,
        "adj_r_squared": combined_model.rsquared_adj,
        "n_obs": len(aligned),
        "start_date": aligned.index.min().strftime("%Y-%m-%d"),
        "durbin_watson": dw,
        "bp_pvalue": bp_pvalue,
        "model": combined_model,
    }


def run_all_regime_regressions(stock_returns: pd.DataFrame, factor_data: pd.DataFrame,
                               tickers: list[str], vix_levels: pd.Series,
                               threshold: float = 25, window: int | None = None) -> dict:
    """Kjør regime-regresjoner for alle aksjer."""
    results = {}
    for ticker in tickers:
        if ticker not in stock_returns.columns:
            continue
        ret = stock_returns[ticker].dropna()
        result = run_regime_regression(ret, factor_data, vix_levels,
                                       threshold=threshold, window=window)
        if result is not None:
            results[ticker] = result
    return results


def portfolio_betas_regime(regression_results: dict, weights: dict,
                           factor_names: list[str], beta_key: str = "betas") -> dict:
    """Beregn vektede faktor-betaer med valg av regime (betas/betas_normal/betas_stress)."""
    port_betas = {f: 0.0 for f in factor_names}
    total_weight = 0.0

    for ticker, w in weights.items():
        if ticker not in regression_results:
            continue
        betas = regression_results[ticker].get(beta_key, regression_results[ticker]["betas"])
        for f in factor_names:
            if f in betas:
                port_betas[f] += w * betas[f]
        total_weight += w

    if total_weight > 0 and total_weight < 0.99:
        for f in factor_names:
            port_betas[f] /= total_weight

    return port_betas


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
