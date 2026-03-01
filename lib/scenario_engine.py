"""Scenarioberegninger og historisk backtesting."""

import pandas as pd
import numpy as np
from config.factors import FACTOR_NAMES

# ── Capping-grenser for estimert aksjeeffekt ──
CAP_FLOOR = -0.75  # Maks drawdown per aksje: -75%
CAP_CEIL = 0.50    # Maks oppside per aksje: +50%


def _cap_impact(impact: float) -> float:
    """Cap estimert aksjeeffekt til realistisk intervall."""
    return max(CAP_FLOOR, min(CAP_CEIL, impact))


def calc_stock_scenario_impact(betas: dict, scenario: dict) -> float:
    """Beregn estimert avkastning for én aksje i ett scenario.

    estimert = Σ(β_j × sjokk_j)
    """
    impact = 0.0
    for factor in FACTOR_NAMES:
        beta = betas.get(factor, 0.0)
        shock = scenario.get(factor, 0.0)
        impact += beta * shock
    return impact


def calc_all_stock_impacts(regression_results: dict, scenario: dict) -> dict:
    """Beregn cappet estimert avkastning for alle aksjer i ett scenario."""
    impacts = {}
    for ticker, result in regression_results.items():
        raw = calc_stock_scenario_impact(result["betas"], scenario)
        impacts[ticker] = _cap_impact(raw)
    return impacts


def calc_all_stock_impacts_detailed(regression_results: dict, scenario: dict) -> dict:
    """Beregn estimert avkastning med both rå og cappet verdi + flagg.

    Returns:
        Dict[ticker -> {"raw": float, "capped": float, "is_capped": bool}]
    """
    details = {}
    for ticker, result in regression_results.items():
        raw = calc_stock_scenario_impact(result["betas"], scenario)
        capped = _cap_impact(raw)
        details[ticker] = {
            "raw": raw,
            "capped": capped,
            "is_capped": raw != capped,
        }
    return details


def calc_portfolio_impact(regression_results: dict, weights: dict,
                          scenario: dict) -> float:
    """Beregn estimert porteføljeavkastning i ett scenario.

    Inkluderer CASH med 0 effekt.
    """
    stock_impacts = calc_all_stock_impacts(regression_results, scenario)
    portfolio_return = 0.0
    for ticker, w in weights.items():
        if ticker == "CASH":
            continue  # CASH har null sensitivitet
        impact = stock_impacts.get(ticker, 0.0)
        portfolio_return += (w / 100.0) * impact
    # CASH-bidraget er 0
    return portfolio_return


def calc_benchmark_impact(spy_result: dict, scenario: dict) -> float:
    """Beregn estimert S&P 500-avkastning i ett scenario."""
    if spy_result is None:
        return 0.0
    return calc_stock_scenario_impact(spy_result["betas"], scenario)


def calc_all_scenarios(regression_results: dict, spy_result: dict,
                       weights: dict, scenarios: dict) -> pd.DataFrame:
    """Beregn alle scenarioer og returner sammendrag.

    Returns:
        DataFrame med kolonner: Scenario, Portefølje, S&P 500, Differanse
    """
    rows = []
    for name, scenario in scenarios.items():
        port = calc_portfolio_impact(regression_results, weights, scenario)
        bench = calc_benchmark_impact(spy_result, scenario)
        rows.append({
            "Scenario": name,
            "Portefølje": port * 100,  # til prosent
            "S&P 500": bench * 100,
            "Differanse": (port - bench) * 100,
        })
    return pd.DataFrame(rows)


def calc_factor_decomposition(port_betas: dict, spy_betas: dict,
                              scenario: dict) -> pd.DataFrame:
    """Dekomponerer scenarioeffekten per faktor for portefølje og benchmark.

    Viser: faktor, sjokk, portefølje-beta, SPY-beta, portefølje-bidrag,
    SPY-bidrag, og differanse (aktivt bidrag).
    """
    rows = []
    for factor in FACTOR_NAMES:
        shock = scenario.get(factor, 0.0)
        p_beta = port_betas.get(factor, 0.0)
        s_beta = spy_betas.get(factor, 0.0)
        p_impact = p_beta * shock * 100  # til prosent
        s_impact = s_beta * shock * 100
        rows.append({
            "Faktor": factor,
            "Sjokk": shock,
            "Port. beta": p_beta,
            "SPY beta": s_beta,
            "Port. bidrag (%)": p_impact,
            "SPY bidrag (%)": s_impact,
            "Aktivt bidrag (%)": p_impact - s_impact,
        })
    return pd.DataFrame(rows)


def calc_stock_contributions(regression_results: dict, weights: dict,
                             scenario: dict) -> pd.DataFrame:
    """Beregn bidrag fra hver aksje til porteføljeeffekten (med capping).

    Returns:
        DataFrame med kolonner: Ticker, Vekt, Estimert effekt, Bidrag,
        Uklippet effekt, Cappet (flagg)
    """
    details = calc_all_stock_impacts_detailed(regression_results, scenario)
    rows = []
    for ticker, w in weights.items():
        if ticker == "CASH":
            rows.append({
                "Ticker": ticker,
                "Vekt (%)": w,
                "Estimert effekt (%)": 0.0,
                "Bidrag (pp)": 0.0,
                "Uklippet effekt (%)": 0.0,
                "Uklippet bidrag (pp)": 0.0,
                "Cappet": False,
            })
            continue
        d = details.get(ticker, {"raw": 0.0, "capped": 0.0, "is_capped": False})
        contribution = (w / 100.0) * d["capped"] * 100
        raw_contribution = (w / 100.0) * d["raw"] * 100
        rows.append({
            "Ticker": ticker,
            "Vekt (%)": w,
            "Estimert effekt (%)": d["capped"] * 100,
            "Bidrag (pp)": contribution,
            "Uklippet effekt (%)": d["raw"] * 100,
            "Uklippet bidrag (pp)": raw_contribution,
            "Cappet": d["is_capped"],
        })
    df = pd.DataFrame(rows)
    return df.sort_values("Bidrag (pp)", ascending=True)


def calc_heatmap_data(regression_results: dict, scenarios: dict,
                      weights: dict) -> pd.DataFrame:
    """Beregn heatmap-data: aksjer × scenarioer.

    Returns:
        DataFrame med aksjer som rader (sortert etter vekt), scenarioer som kolonner.
        Verdier i prosent.
    """
    sorted_tickers = sorted(weights.keys(), key=lambda t: weights[t], reverse=True)
    sorted_tickers = [t for t in sorted_tickers if t != "CASH"]

    data = {}
    for scenario_name, scenario in scenarios.items():
        stock_impacts = calc_all_stock_impacts(regression_results, scenario)
        col = []
        for ticker in sorted_tickers:
            col.append(stock_impacts.get(ticker, 0.0) * 100)
        data[scenario_name] = col

    return pd.DataFrame(data, index=sorted_tickers)


def calc_backtest(factor_data: pd.DataFrame, stock_returns: pd.DataFrame,
                  regression_results: dict, spy_result: dict,
                  weights: dict, start_date: str, end_date: str) -> dict:
    """Kjør historisk backtesting for en gitt periode.

    Sammenligner modellens estimat med faktisk avkastning.
    """
    mask = (factor_data.index >= start_date) & (factor_data.index <= end_date)
    period_factors = factor_data[mask]

    if period_factors.empty:
        return None

    # Kumulativ faktorendring i perioden
    cumulative_shocks = {}
    for col in period_factors.columns:
        cumulative_shocks[col] = period_factors[col].sum()

    # Modellens estimat
    est_portfolio = calc_portfolio_impact(regression_results, weights, cumulative_shocks)
    est_benchmark = calc_benchmark_impact(spy_result, cumulative_shocks)

    # Faktisk avkastning i perioden
    ret_mask = (stock_returns.index >= start_date) & (stock_returns.index <= end_date)
    period_returns = stock_returns[ret_mask]

    actual_portfolio = 0.0
    for ticker, w in weights.items():
        if ticker == "CASH" or ticker not in period_returns.columns:
            continue
        actual_return = period_returns[ticker].sum()  # sum av log-returns ≈ log-return
        actual_portfolio += (w / 100.0) * actual_return

    actual_benchmark = 0.0
    if "SPY" in period_returns.columns:
        actual_benchmark = period_returns["SPY"].sum()

    return {
        "est_portfolio": est_portfolio * 100,
        "est_benchmark": est_benchmark * 100,
        "est_diff": (est_portfolio - est_benchmark) * 100,
        "actual_portfolio": actual_portfolio * 100,
        "actual_benchmark": actual_benchmark * 100,
        "actual_diff": (actual_portfolio - actual_benchmark) * 100,
        "tracking_error_port": (est_portfolio - actual_portfolio) * 100,
        "tracking_error_bench": (est_benchmark - actual_benchmark) * 100,
        "cumulative_shocks": cumulative_shocks,
    }


def calc_vulnerability_analysis(regression_results: dict, spy_result: dict,
                                 weights: dict, scenarios: dict,
                                 port_betas: dict) -> dict:
    """Analyser porteføljens sårbarhet og generer forbedringsforslag.

    Returns:
        Dict med:
        - worst_scenarios: Scenarioer der porteføljen taper mest vs SPY
        - best_scenarios: Scenarioer der porteføljen vinner mest vs SPY
        - stock_drags: Per scenario, hvilke aksjer som drar mest ned
        - factor_tilts: Aktive faktoreksponeringer med størst effekt
        - suggestions: Konkrete forbedringsforslag
    """
    spy_betas = spy_result["betas"] if spy_result else {}

    # Beregn alle scenarioer
    scenario_results = []
    for name, scenario in scenarios.items():
        port = calc_portfolio_impact(regression_results, weights, scenario) * 100
        bench = calc_benchmark_impact(spy_result, scenario) * 100
        diff = port - bench

        # Stock contributions for this scenario
        stock_impacts = calc_all_stock_impacts(regression_results, scenario)
        stock_contribs = []
        for ticker, w in weights.items():
            if ticker == "CASH":
                continue
            impact = stock_impacts.get(ticker, 0.0)
            contrib = (w / 100.0) * impact * 100
            stock_contribs.append({
                "ticker": ticker, "weight": w,
                "impact_pct": impact * 100, "contrib_pp": contrib,
            })

        # Factor decomposition
        factor_contribs = []
        for factor in FACTOR_NAMES:
            shock = scenario.get(factor, 0.0)
            p_beta = port_betas.get(factor, 0.0)
            s_beta = spy_betas.get(factor, 0.0)
            active_contrib = (p_beta - s_beta) * shock * 100
            factor_contribs.append({
                "factor": factor,
                "active_beta": p_beta - s_beta,
                "shock": shock,
                "active_contrib_pct": active_contrib,
            })

        scenario_results.append({
            "name": name, "port": port, "bench": bench, "diff": diff,
            "stock_contribs": sorted(stock_contribs, key=lambda x: x["contrib_pp"]),
            "factor_contribs": sorted(factor_contribs, key=lambda x: x["active_contrib_pct"]),
        })

    # Sorter etter differanse (verst først)
    scenario_results.sort(key=lambda x: x["diff"])

    # Identifiser sårbare faktor-tilts
    factor_vulnerability = {f: 0.0 for f in FACTOR_NAMES}
    for sr in scenario_results:
        if sr["diff"] < 0:  # kun scenarioer der vi taper
            for fc in sr["factor_contribs"]:
                factor_vulnerability[fc["factor"]] += fc["active_contrib_pct"]

    # Generer forslag
    suggestions = []

    # 1. Faktorbaserte forslag
    active_betas = {f: port_betas.get(f, 0) - spy_betas.get(f, 0) for f in FACTOR_NAMES}
    sorted_vuln = sorted(factor_vulnerability.items(), key=lambda x: x[1])

    for factor, total_drag in sorted_vuln[:3]:
        if total_drag < -1.0:  # kun vesentlig drag
            ab = active_betas[factor]
            direction = "overeksponert" if ab > 0 else "undereksponert"
            suggestions.append({
                "type": "faktor",
                "factor": factor,
                "active_beta": ab,
                "total_drag": total_drag,
                "direction": direction,
                "text": f"Porteføljen er {direction} mot {factor} "
                        f"(aktiv beta: {ab:.5f}). "
                        f"Dette koster totalt {total_drag:.1f}pp på tvers av tapsscenariene.",
            })

    # 2. Aksjebaserte forslag — finn gjengangere blant de verste bidragsyterne
    stock_drag_count = {}
    for sr in scenario_results[:5]:  # topp 5 verste scenarioer
        for sc in sr["stock_contribs"][:5]:  # topp 5 verste aksjer per scenario
            if sc["contrib_pp"] < 0:
                t = sc["ticker"]
                if t not in stock_drag_count:
                    stock_drag_count[t] = {"count": 0, "total_drag": 0.0,
                                           "weight": sc["weight"]}
                stock_drag_count[t]["count"] += 1
                stock_drag_count[t]["total_drag"] += sc["contrib_pp"]

    repeat_drags = sorted(stock_drag_count.items(),
                          key=lambda x: x[1]["total_drag"])
    for ticker, info in repeat_drags[:5]:
        if info["count"] >= 2 and info["total_drag"] < -0.5:
            suggestions.append({
                "type": "aksje",
                "ticker": ticker,
                "weight": info["weight"],
                "count": info["count"],
                "total_drag": info["total_drag"],
                "text": f"{ticker} ({info['weight']:.1f}%) drar ned porteføljen "
                        f"i {info['count']} av de 5 verste scenarioene "
                        f"(totalt {info['total_drag']:.2f}pp). "
                        f"Vurder å redusere vekten.",
            })

    return {
        "scenario_results": scenario_results,
        "worst": scenario_results[:5],
        "best": scenario_results[-5:][::-1],
        "factor_vulnerability": factor_vulnerability,
        "active_betas": active_betas,
        "suggestions": suggestions,
    }
