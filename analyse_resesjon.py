"""Analyse: Hvorfor faller porteføljen mer enn SPY i dyp resesjon?

Kjør: python analyse_resesjon.py
"""

import sys
import os
import getpass
import tomllib
from pathlib import Path

# Prøv secrets.toml først, deretter miljøvariabel, til slutt spør interaktivt
api_key = ""
secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
if secrets_path.exists():
    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)
    api_key = secrets.get("FRED_API_KEY", "")
    if "LIMEN_INN" in api_key:
        api_key = ""

if not api_key:
    api_key = os.environ.get("FRED_API_KEY", "")

if not api_key:
    api_key = getpass.getpass("Skriv inn FRED API-nøkkel (input er skjult): ")

if not api_key:
    print("Ingen nøkkel oppgitt. Avslutter.")
    sys.exit(1)

print("Tester FRED API-nøkkel...")
from fredapi import Fred
try:
    fred_test = Fred(api_key=api_key)
    test_series = fred_test.get_series("DGS10", observation_start="2025-01-01")
    print(f"  OK! Fikk {len(test_series)} datapunkter for 10-års rente.")
except Exception as e:
    print(f"  FEIL: {e}")
    print("  Sjekk at API-nøkkelen er korrekt.")
    sys.exit(1)

print("Henter data... (kan ta 1-2 minutter)")

from config.portfolio import PORTFOLIO, STOCK_TICKERS, BENCHMARK_TICKER
from config.factors import FACTOR_NAMES, FACTOR_SHORT_NAMES
from config.scenarios import SCENARIOS
from lib.data_fetcher import fetch_stock_prices, compute_log_returns, get_all_factor_data, fetch_vix_levels
from lib.factor_model import (
    run_all_regressions, run_single_regression, calc_vif, portfolio_betas,
    run_all_regime_regressions, run_regime_regression,
)
from lib.scenario_engine import (
    calc_stock_contributions, calc_portfolio_impact, calc_benchmark_impact,
    calc_factor_decomposition,
)
from lib.utils import normalize_weights

# ── Hent data ──
prices = fetch_stock_prices(STOCK_TICKERS, force_refresh=True)
factor_data = get_all_factor_data(api_key, force_refresh=True)
returns = compute_log_returns(prices)
norm_weights = normalize_weights(PORTFOLIO)
vix_levels = fetch_vix_levels(force_refresh=True)

WINDOW = 3 * 252  # 3 år

# Sjekk hvilke faktorer som faktisk har data
print(f"\nFaktordata: {factor_data.shape[0]} rader, {factor_data.shape[1]} kolonner")
print(f"  Kolonner: {list(factor_data.columns)}")
missing = [f for f in FACTOR_NAMES if f not in factor_data.columns]
if missing:
    print(f"  ADVARSEL: Mangler data for: {missing}")

print(f"\nFaktorer: {FACTOR_NAMES}")
print(f"Antall aksjer: {len(STOCK_TICKERS)}")
print(f"Regresjonsvindu: 3 år ({WINDOW} handelsdager)")

# ── VIF ──
print("\n" + "=" * 60)
print("VIF (Variance Inflation Factor)")
print("=" * 60)
vif = calc_vif(factor_data)
for f, v in vif.items():
    flag = " ⚠️  HØYT" if v > 5 else ""
    print(f"  {FACTOR_SHORT_NAMES.get(f, f):25s} VIF = {v:.1f}{flag}")

# ── Standard regresjon (3-års vindu) ──
print("\n" + "=" * 60)
print("STANDARD REGRESJON (3 år)")
print("=" * 60)

reg_results = run_all_regressions(returns, factor_data, STOCK_TICKERS, window=WINDOW)
spy_result = run_single_regression(returns[BENCHMARK_TICKER].dropna(), factor_data, window=WINDOW)
port_betas_dict = portfolio_betas(reg_results, norm_weights, FACTOR_NAMES)

scenario = SCENARIOS["Dyp resesjon (2008-type)"]
port_impact = calc_portfolio_impact(reg_results, PORTFOLIO, scenario) * 100
bench_impact = calc_benchmark_impact(spy_result, scenario) * 100

print(f"\nDyp resesjon — Standard betaer:")
print(f"  Portefølje:  {port_impact:+.1f}%")
print(f"  S&P 500:     {bench_impact:+.1f}%")
print(f"  Differanse:  {port_impact - bench_impact:+.1f}pp")

# ── Faktordekomponering ──
print("\n" + "-" * 60)
print("FAKTORDEKOMPONERING: Hva driver forskjellen?")
print("-" * 60)
spy_betas = spy_result["betas"] if spy_result else {}
decomp = calc_factor_decomposition(port_betas_dict, spy_betas, scenario)
print(f"  {'Faktor':25s} {'Sjokk':>8s} {'Port.β':>10s} {'SPY β':>10s} {'Port.bid%':>10s} {'SPY bid%':>10s} {'Aktiv%':>8s}")
for _, row in decomp.iterrows():
    print(f"  {FACTOR_SHORT_NAMES.get(row['Faktor'], row['Faktor']):25s} "
          f"{row['Sjokk']:+8.2f} "
          f"{row['Port. beta']:10.6f} "
          f"{row['SPY beta']:10.6f} "
          f"{row['Port. bidrag (%)']:+10.2f} "
          f"{row['SPY bidrag (%)']:+10.2f} "
          f"{row['Aktivt bidrag (%)']:+8.2f}")

# ── Aksje-bidrag (topp tapere) ──
print("\n" + "-" * 60)
print("AKSJER SOM DRAR MEST NED (Dyp resesjon)")
print("-" * 60)
contrib = calc_stock_contributions(reg_results, PORTFOLIO, scenario)
contrib = contrib[contrib["Ticker"] != "CASH"]
print(f"  {'Ticker':8s} {'Vekt%':>6s} {'Effekt%':>10s} {'Bidrag pp':>10s} {'Cappet':>6s}")
for _, row in contrib.head(15).iterrows():
    cap = "*" if row["Cappet"] else ""
    print(f"  {row['Ticker']:8s} {row['Vekt (%)']:6.1f} {row['Estimert effekt (%)']:+10.1f} "
          f"{row['Bidrag (pp)']:+10.2f} {cap:>6s}")

# ── Regime-regresjon ──
print("\n" + "=" * 60)
print("REGIME-REGRESJON (VIX-basert, terskel=25)")
print("=" * 60)

reg_regime = run_all_regime_regressions(returns, factor_data, STOCK_TICKERS, vix_levels,
                                         threshold=25, window=WINDOW)
spy_regime = run_regime_regression(returns[BENCHMARK_TICKER].dropna(), factor_data, vix_levels,
                                    threshold=25, window=WINDOW)

if spy_regime:
    counts = spy_regime.get("regime_counts", {})
    print(f"  SPY: {counts.get('normal', 0)} normale dager, {counts.get('stress', 0)} stress-dager")

# Implied VIX i dyp resesjon: 20 + 40 = 60 → STRESS
current_vix = 20.0
vix_shock = scenario.get("VIX", 0)
implied_vix = current_vix + vix_shock
print(f"\n  Nåværende VIX: {current_vix}, sjokk: +{vix_shock} → implied: {implied_vix} → STRESS-betaer")

regime_port = calc_portfolio_impact(reg_regime, PORTFOLIO, scenario,
                                     regime_mode=True, vix_threshold=25, current_vix=current_vix) * 100
regime_bench = calc_benchmark_impact(spy_regime, scenario,
                                      regime_mode=True, vix_threshold=25, current_vix=current_vix) * 100

print(f"\n  Dyp resesjon — Regime-betaer:")
print(f"    Portefølje:  {regime_port:+.1f}%")
print(f"    S&P 500:     {regime_bench:+.1f}%")
print(f"    Differanse:  {regime_port - regime_bench:+.1f}pp")

# ── Regime aksje-bidrag ──
print("\n" + "-" * 60)
print("AKSJER SOM DRAR MEST NED (Regime-betaer, stress)")
print("-" * 60)
regime_contrib = calc_stock_contributions(reg_regime, PORTFOLIO, scenario,
                                           regime_mode=True, vix_threshold=25, current_vix=current_vix)
regime_contrib = regime_contrib[regime_contrib["Ticker"] != "CASH"]
print(f"  {'Ticker':8s} {'Vekt%':>6s} {'Effekt%':>10s} {'Bidrag pp':>10s}")
for _, row in regime_contrib.head(15).iterrows():
    print(f"  {row['Ticker']:8s} {row['Vekt (%)']:6.1f} {row['Estimert effekt (%)']:+10.1f} "
          f"{row['Bidrag (pp)']:+10.2f}")

# ── Sammenligning standard vs regime ──
print("\n" + "=" * 60)
print("SAMMENLIGNING: Standard vs Regime")
print("=" * 60)
print(f"  {'':20s} {'Standard':>12s} {'Regime':>12s} {'Forskjell':>12s}")
print(f"  {'Portefølje':20s} {port_impact:+12.1f}% {regime_port:+12.1f}% {regime_port - port_impact:+12.1f}pp")
print(f"  {'S&P 500':20s} {bench_impact:+12.1f}% {regime_bench:+12.1f}% {regime_bench - bench_impact:+12.1f}pp")
print(f"  {'Differanse':20s} {port_impact - bench_impact:+12.1f}pp {regime_port - regime_bench:+12.1f}pp")

# ── Historiske kriser — faktiske drawdowns ──
print("\n" + "=" * 60)
print("HISTORISKE KRISER — FAKTISKE DRAWDOWNS")
print("=" * 60)

from config.crises import CRISIS_PERIODS, SECTOR_PROXY
from lib.crisis_analyzer import calc_all_crises

print("Henter historiske krisedata fra Yahoo Finance...")
crisis_results = calc_all_crises(norm_weights, SECTOR_PROXY)

for crisis_name, data in crisis_results.items():
    p_dd = data["portefolje_drawdown"]
    s_dd = data["spy_drawdown"]
    n_proxy = sum(1 for v in data["proxy_info"].values() if v is not None)
    print(f"\n{'─' * 50}")
    print(f"  {crisis_name}  ({data['periode'][0]} → {data['periode'][1]})")
    print(f"{'─' * 50}")
    print(f"  Portefølje:  {p_dd:+.1f}%")
    print(f"  S&P 500:     {s_dd:+.1f}%")
    if p_dd is not None and s_dd is not None:
        print(f"  Differanse:  {p_dd - s_dd:+.1f}pp")
    print(f"  Proxyer brukt: {n_proxy} av {len(data['proxy_info'])} aksjer")

    # Topp 10 verste aksjer
    if data["aksjer"]:
        print(f"\n  {'Ticker':8s} {'Vekt%':>6s} {'Drawdown':>10s} {'Bidrag pp':>10s} {'Datakilde':>15s}")
        for stock in data["aksjer"][:10]:
            src = f"Proxy: {stock['Proxy']}" if stock["Proxy"] else "Faktisk"
            print(f"  {stock['Ticker']:8s} {stock['Vekt (%)']:6.1f} {stock['Drawdown (%)']:+10.1f}% "
                  f"{stock['Bidrag (pp)']:+10.2f} {src:>15s}")

print("\nFerdig! Kjør 'streamlit run app.py' for interaktiv analyse.")
