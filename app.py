"""Kvantitativ Portefølje Scenarioanalyse — Streamlit Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config.portfolio import PORTFOLIO, STOCK_TICKERS, BENCHMARK_TICKER
from config.factors import (
    FACTOR_NAMES, FACTOR_DESCRIPTIONS, FACTOR_SHORT_NAMES, FACTOR_UNITS,
    FRED_FACTORS, YAHOO_FACTORS,
)
from config.scenarios import SCENARIOS, BACKTEST_PERIODS
from config.crises import CRISIS_PERIODS, SECTOR_PROXY
from lib.crisis_analyzer import calc_all_crises

# Bump denne ved faktor-/modellendringer for å tvinge cache-invalidering
_MODEL_VERSION = 2  # v2: fjernet YIELD_CURVE
from lib.data_fetcher import (
    fetch_stock_prices, compute_log_returns, get_all_factor_data,
    fetch_single_ticker, fetch_vix_levels,
)
from lib.factor_model import (
    run_all_regressions, run_single_regression, calc_vif, portfolio_betas,
    run_all_regime_regressions, run_regime_regression, portfolio_betas_regime,
)
from lib.scenario_engine import (
    calc_all_scenarios, calc_stock_contributions, calc_heatmap_data,
    calc_portfolio_impact, calc_benchmark_impact, calc_all_stock_impacts,
    calc_all_stock_impacts_detailed, calc_backtest, calc_factor_decomposition,
    calc_vulnerability_analysis, calc_stock_scenario_impact, _cap_impact,
    CAP_FLOOR, CAP_CEIL,
)
from lib.utils import normalize_weights, cache_timestamp

st.set_page_config(page_title="Portefølje Scenarioanalyse", page_icon="📊", layout="wide")

# ═══════════════════════════════════════════════
# PASSORD
# ═══════════════════════════════════════════════

CORRECT_PASSWORD = st.secrets.get("APP_PASSWORD", "ODINUSA2026")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("📊 Portefølje Scenarioanalyse")
    password = st.text_input("Skriv inn passord for å få tilgang:", type="password")
    if password:
        if password == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Feil passord.")
    st.stop()

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════

st.sidebar.title("⚙️ Innstillinger")

# Les FRED API-nøkkel: prøv st.secrets først, fallback til manuell input
api_key = st.secrets.get("FRED_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("FRED API-nøkkel", type="password",
                                     help="Gratis fra https://fred.stlouisfed.org/")
else:
    st.sidebar.caption("🔑 FRED API-nøkkel lastet fra secrets")

if not api_key:
    st.title("📊 Portefølje Scenarioanalyse")
    st.info("Skriv inn FRED API-nøkkel i sidebar for å starte.\n\n"
            "Gratis nøkkel: https://fred.stlouisfed.org/ → My Account → API Keys")
    st.stop()

# ── Portefølje: opplasting eller default ──
st.sidebar.divider()
st.sidebar.subheader("Portefølje")
uploaded_file = st.sidebar.file_uploader(
    "Last opp portefølje (CSV/Excel)",
    type=["csv", "xlsx", "xls"],
    help="Filen må ha kolonnene 'Ticker' og 'Vekt' (i prosent). "
         "Eksempel: AAPL, 5.0. "
         "Legg til en rad med Ticker='CASH' for kontanter.",
)

def _parse_portfolio_file(file) -> dict | None:
    """Parser opplastet porteføljefil til dict {ticker: vekt%}."""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Normaliser kolonnenavn
        df.columns = [c.strip().lower() for c in df.columns]

        # Finn ticker- og vekt-kolonner (fleksibel matching)
        ticker_col = None
        weight_col = None
        for c in df.columns:
            if c in ("ticker", "symbol", "aksje", "holding"):
                ticker_col = c
            if c in ("vekt", "weight", "vekt (%)", "weight (%)", "%", "andel", "prosent"):
                weight_col = c

        if ticker_col is None or weight_col is None:
            return None

        df[ticker_col] = df[ticker_col].astype(str).str.strip().str.upper()
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
        df = df.dropna(subset=[ticker_col, weight_col])

        return dict(zip(df[ticker_col], df[weight_col]))
    except Exception:
        return None

if uploaded_file is not None:
    custom_portfolio = _parse_portfolio_file(uploaded_file)
    if custom_portfolio is None:
        st.sidebar.error("Kunne ikke lese filen. Sjekk at den har kolonnene 'Ticker' og 'Vekt'.")
        ACTIVE_PORTFOLIO = dict(PORTFOLIO)
    else:
        ACTIVE_PORTFOLIO = custom_portfolio
        st.sidebar.success(f"Lastet {len(custom_portfolio)} posisjoner "
                           f"({sum(custom_portfolio.values()):.1f}% total)")
else:
    ACTIVE_PORTFOLIO = dict(PORTFOLIO)

# Utled tickers fra aktiv portefølje
ACTIVE_TICKERS = [t for t in ACTIVE_PORTFOLIO if t != "CASH"]

st.sidebar.caption(f"Portefølje: {len(ACTIVE_TICKERS)} aksjer + "
                   f"{'CASH ' + str(ACTIVE_PORTFOLIO.get('CASH', 0)) + '%' if 'CASH' in ACTIVE_PORTFOLIO else 'ingen CASH'}")

with st.sidebar.expander("Vis portefølje"):
    port_display = pd.DataFrame([
        {"Ticker": t, "Vekt (%)": w}
        for t, w in sorted(ACTIVE_PORTFOLIO.items(), key=lambda x: -x[1])
    ])
    st.dataframe(port_display, use_container_width=True, hide_index=True, height=300)

# Last ned mal
st.sidebar.download_button(
    "📥 Last ned mal (CSV)",
    data="Ticker,Vekt\nAAPL,5.0\nMSFT,5.0\nGOOG,4.0\nCASH,1.0\n",
    file_name="portefolje_mal.csv",
    mime="text/csv",
)

st.sidebar.divider()
st.sidebar.subheader("Modellinnstillinger")
reg_years = st.sidebar.slider("Regresjonsvindu (år)", min_value=1, max_value=10, value=3,
                               help="Antall år med data for OLS-regresjon. Kortere = mer relevant, lengre = mer stabilt.")
reg_window = reg_years * 252

regime_mode = st.sidebar.toggle("Regime-betaer (VIX-basert)", value=False,
                                 help="Bruker separate betaer for rolige og stressede markeder basert på VIX-nivå.")
if regime_mode:
    vix_threshold = st.sidebar.slider("VIX-terskel (normal/stress)", min_value=15, max_value=40, value=25,
                                       help="VIX over denne verdien = stress-regime")
    current_vix = st.sidebar.number_input("Nåværende VIX-nivå", min_value=10.0, max_value=80.0, value=20.0, step=1.0,
                                           help="Brukes til å beregne implied VIX etter sjokk")
else:
    vix_threshold = 25
    current_vix = 20.0

st.sidebar.divider()
force_refresh = st.sidebar.button("🔄 Oppdater data")
sig_only = st.sidebar.toggle("Vis kun signifikante faktorer (p < 0.05)", value=False)

st.sidebar.divider()
st.sidebar.caption(f"📅 Sist oppdatert: {cache_timestamp('all_factors')}")

# ═══════════════════════════════════════════════
# DATAHENTING OG MODELLKJØRING
# ═══════════════════════════════════════════════

# Lag en stabil cache-nøkkel basert på porteføljen og modellinnstillinger
_portfolio_key = (tuple(sorted(ACTIVE_PORTFOLIO.items())), reg_window, regime_mode, vix_threshold, _MODEL_VERSION)

@st.cache_data(show_spinner="Henter aksjedata...", ttl=3600)
def _fetch_prices(_tickers, _force, _version=_MODEL_VERSION):
    return fetch_stock_prices(_tickers, force_refresh=_force)

@st.cache_data(show_spinner="Henter faktordata...", ttl=3600)
def _fetch_factors(_api_key, _force, _version=_MODEL_VERSION):
    return get_all_factor_data(_api_key, force_refresh=_force)

with st.spinner("Laster data og kjører regresjoner..."):
    # Tving oppdatering hvis modellversjon har endret seg
    if st.session_state.get("_model_version") != _MODEL_VERSION:
        st.cache_data.clear()
        force_refresh = True
        st.session_state["_model_version"] = _MODEL_VERSION

    prices = _fetch_prices(ACTIVE_TICKERS, force_refresh)
    factor_data = _fetch_factors(api_key, force_refresh)

    if factor_data.empty:
        st.error("Ingen faktordata tilgjengelig. Sjekk FRED API-nøkkel.")
        st.stop()

    returns = compute_log_returns(prices)
    norm_weights = normalize_weights(ACTIVE_PORTFOLIO)

    # Hent VIX-nivåer for regime-modus
    if regime_mode:
        vix_levels = fetch_vix_levels(force_refresh)
    else:
        vix_levels = pd.Series(dtype=float)

    # Kjør regresjoner (re-kjør hvis porteføljen endres)
    cached_key = st.session_state.get("_portfolio_key")
    if cached_key != _portfolio_key or force_refresh:
        if regime_mode and not vix_levels.empty:
            reg_results = run_all_regime_regressions(
                returns, factor_data, ACTIVE_TICKERS, vix_levels,
                threshold=vix_threshold, window=reg_window)
            spy_result = run_regime_regression(
                returns[BENCHMARK_TICKER].dropna(), factor_data, vix_levels,
                threshold=vix_threshold, window=reg_window
            ) if BENCHMARK_TICKER in returns.columns else None
        else:
            reg_results = run_all_regressions(returns, factor_data, ACTIVE_TICKERS, window=reg_window)
            spy_result = run_single_regression(
                returns[BENCHMARK_TICKER].dropna(), factor_data, window=reg_window
            ) if BENCHMARK_TICKER in returns.columns else None
        vif_scores = calc_vif(factor_data)
        port_betas = portfolio_betas(reg_results, norm_weights, FACTOR_NAMES)

        st.session_state["regression_results"] = reg_results
        st.session_state["spy_result"] = spy_result
        st.session_state["vif_scores"] = vif_scores
        st.session_state["port_betas"] = port_betas
        st.session_state["_portfolio_key"] = _portfolio_key

    reg_results = st.session_state["regression_results"]
    spy_result = st.session_state["spy_result"]
    vif_scores = st.session_state["vif_scores"]
    port_betas = st.session_state["port_betas"]

    # Regime-kwargs for scenario-funksjoner
    _regime_kw = dict(regime_mode=regime_mode, vix_threshold=vix_threshold, current_vix=current_vix)

# ═══════════════════════════════════════════════
# FANER
# ═══════════════════════════════════════════════

st.title("📊 Portefølje Scenarioanalyse")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Scenario Overview", "Scenario Deep Dive", "Faktoreksponering",
    "Regresjonsdiagnostikk", "Custom Scenario", "Historisk Backtest",
    "Historiske Kriser", "Porteføljeoptimering", "Aksjesøk",
])

# ───────────────────────────────────────────────
# FANE 1: SCENARIO OVERVIEW
# ───────────────────────────────────────────────

with tab1:
    scenario_df = calc_all_scenarios(reg_results, spy_result, ACTIVE_PORTFOLIO, SCENARIOS, **_regime_kw)

    # Oppsummering
    n_beat = (scenario_df["Differanse"] > 0).sum()
    worst_vs = scenario_df.loc[scenario_df["Differanse"].idxmin(), "Scenario"]
    best_vs = scenario_df.loc[scenario_df["Differanse"].idxmax(), "Scenario"]

    info_msg = (f"Porteføljen slår S&P 500 i **{n_beat} av {len(SCENARIOS)}** scenarioer. "
                f"Mest sårbar vs benchmark i **{worst_vs}**. "
                f"Best posisjonert vs benchmark i **{best_vs}**.")
    if regime_mode:
        n_stress = (scenario_df.get("Regime") == "Stress").sum() if "Regime" in scenario_df.columns else 0
        info_msg += f" | Regime-modus aktiv: **{n_stress}** scenarioer bruker stress-betaer."
    st.info(info_msg)

    # Grouped bar chart
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Portefølje", x=scenario_df["Scenario"], y=scenario_df["Portefølje"],
        marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in scenario_df["Portefølje"]],
    ))
    fig_bar.add_trace(go.Bar(
        name="S&P 500", x=scenario_df["Scenario"], y=scenario_df["S&P 500"],
        marker_color=["#4ade80" if v >= 0 else "#f87171" for v in scenario_df["S&P 500"]],
        opacity=0.7,
    ))
    fig_bar.add_trace(go.Bar(
        name="Differanse", x=scenario_df["Scenario"], y=scenario_df["Differanse"],
        marker_color=["#2563eb" if v >= 0 else "#dc2626" for v in scenario_df["Differanse"]],
    ))
    fig_bar.update_layout(
        barmode="group", title="Estimert avkastning per scenario (%)",
        xaxis_tickangle=-45, height=500,
        yaxis_title="Avkastning (%)", template="plotly_white",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Heatmap
    st.subheader("Aksje-heatmap")
    heatmap_df = calc_heatmap_data(reg_results, SCENARIOS, ACTIVE_PORTFOLIO, **_regime_kw)
    fig_heat = px.imshow(
        heatmap_df.values,
        x=heatmap_df.columns, y=heatmap_df.index,
        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        aspect="auto", text_auto=".1f",
    )
    fig_heat.update_layout(
        title="Estimert aksjeeffekt per scenario (%)",
        height=max(400, len(heatmap_df) * 22),
        xaxis_tickangle=-45, template="plotly_white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ───────────────────────────────────────────────
# FANE 2: SCENARIO DEEP DIVE
# ───────────────────────────────────────────────

with tab2:
    selected_scenario = st.selectbox("Velg scenario", list(SCENARIOS.keys()))
    scenario = SCENARIOS[selected_scenario]

    port_impact = calc_portfolio_impact(reg_results, ACTIVE_PORTFOLIO, scenario, **_regime_kw) * 100
    bench_impact = calc_benchmark_impact(spy_result, scenario, **_regime_kw) * 100
    diff_impact = port_impact - bench_impact

    # Regime-indikator
    if regime_mode:
        implied_vix = current_vix + scenario.get("VIX", 0.0)
        regime_label = "Stress" if implied_vix > vix_threshold else "Normal"
        st.caption(f"Regime: **{regime_label}** (implied VIX: {implied_vix:.0f})")

    # KPI-bokser
    c1, c2, c3 = st.columns(3)
    c1.metric("Portefølje", f"{port_impact:.1f}%",
              delta=f"{port_impact:.1f}%", delta_color="normal")
    c2.metric("S&P 500", f"{bench_impact:.1f}%",
              delta=f"{bench_impact:.1f}%", delta_color="normal")
    c3.metric("Differanse", f"{diff_impact:.1f}%",
              delta=f"{diff_impact:+.1f}pp", delta_color="normal")

    # Waterfall chart
    st.subheader("Bidrag per aksje (waterfall)")
    contrib_df = calc_stock_contributions(reg_results, ACTIVE_PORTFOLIO, scenario, **_regime_kw)
    contrib_df = contrib_df[contrib_df["Ticker"] != "CASH"]
    any_capped = contrib_df["Cappet"].any()

    # Marker cappede aksjer med * og bruk mørkere farge
    wf_labels = [
        f"{t}*" if c else t
        for t, c in zip(contrib_df["Ticker"], contrib_df["Cappet"])
    ]
    wf_colors = []
    for _, row in contrib_df.iterrows():
        if row["Cappet"]:
            wf_colors.append("#991b1b" if row["Bidrag (pp)"] < 0 else "#166534")
        else:
            wf_colors.append("#ef4444" if row["Bidrag (pp)"] < 0 else "#22c55e")

    # Hover med cappet/uklippet info
    wf_hover = []
    for _, row in contrib_df.iterrows():
        if row["Cappet"]:
            wf_hover.append(
                f"{row['Ticker']}*<br>"
                f"Bidrag: {row['Bidrag (pp)']:.2f} pp (cappet)<br>"
                f"Uklippet estimat: {row['Uklippet bidrag (pp)']:.2f} pp<br>"
                f"Estimert effekt: {row['Estimert effekt (%)']:.1f}% "
                f"(cappet fra {row['Uklippet effekt (%)']:.1f}%)<br>"
                f"Vekt: {row['Vekt (%)']:.2f}%"
            )
        else:
            wf_hover.append(
                f"{row['Ticker']}<br>"
                f"Bidrag: {row['Bidrag (pp)']:.2f} pp<br>"
                f"Estimert effekt: {row['Estimert effekt (%)']:.1f}%<br>"
                f"Vekt: {row['Vekt (%)']:.2f}%"
            )

    fig_wf = go.Figure(go.Waterfall(
        x=wf_labels,
        y=contrib_df["Bidrag (pp)"],
        textposition="outside",
        text=[f"{v:.2f}" for v in contrib_df["Bidrag (pp)"]],
        hovertext=wf_hover,
        hoverinfo="text",
        connector={"line": {"color": "rgba(0,0,0,0.1)"}},
        increasing={"marker": {"color": "#22c55e"}},
        decreasing={"marker": {"color": "#ef4444"}},
    ))
    # Cappede aksjer er allerede markert med * i labels
    fig_wf.update_layout(
        title=f"Bidrag til porteføljeeffekt — {selected_scenario}",
        yaxis_title="Bidrag (pp)", height=500, template="plotly_white",
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Disclaimer ved capping
    if any_capped:
        n_capped = contrib_df["Cappet"].sum()
        st.caption(
            f"⚠️ *Lineær faktormodell — estimater for ekstreme scenarioer kan overvurdere effekter. "
            f"{n_capped} aksje{'r' if n_capped > 1 else ''} merket med \\* er cappet til "
            f"[{CAP_FLOOR*100:.0f}%, +{CAP_CEIL*100:.0f}%]. Hover for å se uklippet estimat.*"
        )

    # Tabell
    st.subheader("Detaljtabell")
    display_df = contrib_df.copy()
    display_df["Estimert effekt (%)"] = display_df["Estimert effekt (%)"].round(2)
    display_df["Bidrag (pp)"] = display_df["Bidrag (pp)"].round(3)
    # Vis uklippet effekt kun for cappede aksjer
    display_df["Uklippet effekt (%)"] = display_df.apply(
        lambda r: f"{r['Uklippet effekt (%)']:.1f}%" if r["Cappet"] else "",
        axis=1,
    )

    # Legg til topp 3 signifikante faktorer per aksje
    top_factors = []
    for ticker in display_df["Ticker"]:
        if ticker in reg_results:
            betas = reg_results[ticker]["betas"]
            p_vals = reg_results[ticker]["p_values"]
            sig_factors = [(f, betas[f]) for f in betas
                           if p_vals.get(f, 1) < 0.05 and f in scenario and scenario[f] != 0]
            sig_factors.sort(key=lambda x: abs(x[1] * scenario.get(x[0], 0)), reverse=True)
            top3 = ", ".join(
                f"{FACTOR_SHORT_NAMES.get(f, f)} ({b:.4f})"
                for f, b in sig_factors[:3]
            )
            top_factors.append(top3 if top3 else "–")
        else:
            top_factors.append("–")
    display_df["Topp faktorer"] = top_factors
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Faktor-dekomponering ──
    st.divider()
    st.subheader("🔍 Faktor-dekomponering — Hva driver scenarioet?")
    st.caption("Viser bidraget fra hver faktor til porteføljens og benchmarkens estimerte avkastning, "
               "og det aktive bidraget (differansen).")

    spy_b = spy_result["betas"] if spy_result else {}
    decomp_df = calc_factor_decomposition(port_betas, spy_b, scenario)

    # Bruk lesbare navn
    decomp_df["Faktor-navn"] = decomp_df["Faktor"].map(
        lambda f: f"{FACTOR_SHORT_NAMES.get(f, f)}"
    )
    # Formater sjokk med enhet — prosent for det meste, poeng for VIX
    decomp_df["Sjokk (enhet)"] = decomp_df.apply(
        lambda r: f"{r['Sjokk']:+.0f} poeng" if r['Faktor'] == 'VIX'
        else (f"{r['Sjokk']:+.2f}%" if r['Sjokk'] != 0 else "0%"),
        axis=1,
    )

    # Grouped bar chart — faktor-bidrag
    fig_decomp = go.Figure()
    fig_decomp.add_trace(go.Bar(
        name="Portefølje-bidrag",
        x=decomp_df["Faktor-navn"], y=decomp_df["Port. bidrag (%)"],
        marker_color="#2563eb",
    ))
    fig_decomp.add_trace(go.Bar(
        name="S&P 500-bidrag",
        x=decomp_df["Faktor-navn"], y=decomp_df["SPY bidrag (%)"],
        marker_color="#9ca3af",
    ))
    fig_decomp.add_trace(go.Bar(
        name="Aktivt bidrag (differanse)",
        x=decomp_df["Faktor-navn"], y=decomp_df["Aktivt bidrag (%)"],
        marker_color=["#22c55e" if v >= 0 else "#ef4444"
                       for v in decomp_df["Aktivt bidrag (%)"]],
    ))
    fig_decomp.update_layout(
        barmode="group",
        title=f"Faktor-bidrag til scenarioeffekt — {selected_scenario}",
        yaxis_title="Bidrag (%)",
        height=450, template="plotly_white",
    )
    st.plotly_chart(fig_decomp, use_container_width=True)

    # Tabell med detaljer
    show_decomp = decomp_df[["Faktor-navn", "Sjokk (enhet)",
                              "Port. beta", "SPY beta",
                              "Port. bidrag (%)", "SPY bidrag (%)",
                              "Aktivt bidrag (%)"]].copy()
    show_decomp["Port. beta"] = show_decomp["Port. beta"].apply(lambda v: f"{v:.6f}")
    show_decomp["SPY beta"] = show_decomp["SPY beta"].apply(lambda v: f"{v:.6f}")
    show_decomp["Port. bidrag (%)"] = show_decomp["Port. bidrag (%)"].round(2)
    show_decomp["SPY bidrag (%)"] = show_decomp["SPY bidrag (%)"].round(2)
    show_decomp["Aktivt bidrag (%)"] = show_decomp["Aktivt bidrag (%)"].round(2)

    # Totalrad
    total_row = pd.DataFrame([{
        "Faktor-navn": "TOTAL",
        "Sjokk (enhet)": "",
        "Port. beta": "",
        "SPY beta": "",
        "Port. bidrag (%)": decomp_df["Port. bidrag (%)"].sum().round(2),
        "SPY bidrag (%)": decomp_df["SPY bidrag (%)"].sum().round(2),
        "Aktivt bidrag (%)": decomp_df["Aktivt bidrag (%)"].sum().round(2),
    }])
    show_decomp = pd.concat([show_decomp, total_row], ignore_index=True)
    st.dataframe(show_decomp, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────
# FANE 3: FAKTOREKSPONERING
# ───────────────────────────────────────────────

with tab3:
    spy_betas = spy_result["betas"] if spy_result else {}

    # Radar chart
    st.subheader("Faktoreksponering — Portefølje vs S&P 500")

    # Normaliser betaer til enhet-standardavvik for radar
    all_betas_vals = []
    for f in FACTOR_NAMES:
        all_betas_vals.extend([abs(port_betas.get(f, 0)), abs(spy_betas.get(f, 0))])
    max_beta = max(all_betas_vals) if all_betas_vals and max(all_betas_vals) > 0 else 1

    port_norm = [port_betas.get(f, 0) / max_beta for f in FACTOR_NAMES]
    spy_norm = [spy_betas.get(f, 0) / max_beta for f in FACTOR_NAMES]

    short_names = [FACTOR_SHORT_NAMES.get(f, f) for f in FACTOR_NAMES]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=port_norm + [port_norm[0]],
        theta=short_names + [short_names[0]],
        name="Portefølje", fill="toself", opacity=0.6,
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=spy_norm + [spy_norm[0]],
        theta=short_names + [short_names[0]],
        name="S&P 500", fill="toself", opacity=0.4,
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        height=500, template="plotly_white",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Grouped bar chart — betaer
    st.subheader("Faktor-betaer: Portefølje vs S&P 500 vs Aktiv")
    beta_comp = pd.DataFrame({
        "Faktor": [FACTOR_SHORT_NAMES.get(f, f) for f in FACTOR_NAMES],
        "Portefølje": [port_betas.get(f, 0) for f in FACTOR_NAMES],
        "S&P 500": [spy_betas.get(f, 0) for f in FACTOR_NAMES],
        "Aktiv": [port_betas.get(f, 0) - spy_betas.get(f, 0) for f in FACTOR_NAMES],
    })

    fig_beta = go.Figure()
    fig_beta.add_trace(go.Bar(name="Portefølje", x=beta_comp["Faktor"],
                              y=beta_comp["Portefølje"], marker_color="#2563eb"))
    fig_beta.add_trace(go.Bar(name="S&P 500", x=beta_comp["Faktor"],
                              y=beta_comp["S&P 500"], marker_color="#9ca3af"))
    fig_beta.add_trace(go.Bar(
        name="Aktiv", x=beta_comp["Faktor"], y=beta_comp["Aktiv"],
        marker_color=["#22c55e" if v >= 0 else "#ef4444" for v in beta_comp["Aktiv"]],
    ))
    fig_beta.update_layout(
        barmode="group", title="Faktor-betaer",
        yaxis_title="Beta", height=450, template="plotly_white",
    )
    st.plotly_chart(fig_beta, use_container_width=True)

    # Bar chart per faktor
    st.subheader("Faktor-beta per aksje")
    sel_factor = st.selectbox("Velg faktor", FACTOR_NAMES,
                              format_func=lambda f: f"{FACTOR_SHORT_NAMES.get(f, f)} — {FACTOR_DESCRIPTIONS[f]}")
    factor_betas = []
    for ticker in ACTIVE_TICKERS:
        if ticker in reg_results:
            b = reg_results[ticker]["betas"].get(sel_factor, 0)
            p = reg_results[ticker]["p_values"].get(sel_factor, 1)
            if sig_only and p >= 0.05:
                b = 0
            factor_betas.append({"Ticker": ticker, "Beta": b, "Vekt (%)": ACTIVE_PORTFOLIO[ticker]})
    fb_df = pd.DataFrame(factor_betas).sort_values("Beta")

    fig_fb = px.bar(fb_df, x="Ticker", y="Beta", color="Beta",
                    color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                    hover_data=["Vekt (%)"])
    fig_fb.update_layout(title=f"Beta for {sel_factor}", height=400, template="plotly_white")
    st.plotly_chart(fig_fb, use_container_width=True)

    # Korrelasjonsheatmap
    st.subheader("Faktorkorrelasjon")
    available_factors = [f for f in FACTOR_NAMES if f in factor_data.columns]
    corr = factor_data[available_factors].corr()
    corr_labels = [FACTOR_SHORT_NAMES.get(f, f) for f in available_factors]
    fig_corr = px.imshow(corr.values, x=corr_labels, y=corr_labels,
                         text_auto=".2f", color_continuous_scale="RdBu_r",
                         color_continuous_midpoint=0, aspect="auto")
    fig_corr.update_layout(title="Parvise korrelasjoner mellom faktorer",
                           height=500, template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

# ───────────────────────────────────────────────
# FANE 4: REGRESJONSDIAGNOSTIKK
# ───────────────────────────────────────────────

with tab4:
    # SPY benchmark først
    st.subheader("📌 Benchmark: S&P 500 (SPY)")
    if spy_result:
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("R²", f"{spy_result['r_squared']:.4f}")
        sc2.metric("Adj. R²", f"{spy_result['adj_r_squared']:.4f}")
        sc3.metric("Durbin-Watson", f"{spy_result['durbin_watson']:.2f}")
        sc4.metric("Observasjoner", spy_result["n_obs"])

        with st.expander("SPY — Full regresjonsdetalj"):
            spy_detail = pd.DataFrame({
                "Faktor": [FACTOR_SHORT_NAMES.get(f, f) for f in spy_result["betas"]],
                "Beta": list(spy_result["betas"].values()),
                "t-stat": [spy_result["t_stats"][f] for f in spy_result["betas"]],
                "p-verdi": [spy_result["p_values"][f] for f in spy_result["betas"]],
            })
            spy_detail["Signifikant"] = spy_detail["p-verdi"].apply(
                lambda p: "✅" if p < 0.05 else "")
            st.dataframe(spy_detail, use_container_width=True, hide_index=True)
    else:
        st.warning("SPY-regresjon ikke tilgjengelig")

    st.divider()

    # VIF-advarsler
    st.subheader("VIF (Variance Inflation Factor)")
    vif_df = vif_scores.reset_index()
    vif_df.columns = ["Faktor_key", "VIF"]
    vif_df["Faktor"] = vif_df["Faktor_key"].map(lambda f: FACTOR_SHORT_NAMES.get(f, f))
    vif_df = vif_df[["Faktor", "VIF"]]
    vif_df["Flagg"] = vif_df["VIF"].apply(lambda v: "⚠️ > 5" if v > 5 else "OK")
    st.dataframe(vif_df, use_container_width=True, hide_index=True)

    high_vif = vif_df[vif_df["VIF"] > 5]
    if not high_vif.empty:
        st.warning(f"Faktorer med høy VIF (>5): {', '.join(high_vif['Faktor'])}. "
                   "Dette kan indikere multikollinearitet.")

    st.divider()

    # Per-aksje tabell
    st.subheader("Regresjonsresultater per aksje")
    diag_rows = []
    for ticker in ACTIVE_TICKERS:
        if ticker not in reg_results:
            diag_rows.append({"Ticker": ticker, "R²": None, "Adj. R²": None,
                              "DW": None, "Obs": None, "Start": None, "Advarsel": "Ingen data"})
            continue
        r = reg_results[ticker]
        warnings = []
        if r["r_squared"] < 0.05:
            warnings.append("Lav R²")
        if r["durbin_watson"] < 1.5 or r["durbin_watson"] > 2.5:
            warnings.append("DW utenfor 1.5-2.5")
        if r["bp_pvalue"] is not None and r["bp_pvalue"] < 0.05:
            warnings.append("Heteroskedastisitet")
        diag_rows.append({
            "Ticker": ticker,
            "R²": round(r["r_squared"], 4),
            "Adj. R²": round(r["adj_r_squared"], 4),
            "DW": round(r["durbin_watson"], 2),
            "Obs": r["n_obs"],
            "Start": r["start_date"],
            "Advarsel": ", ".join(warnings) if warnings else "OK",
        })

    diag_df = pd.DataFrame(diag_rows)
    st.dataframe(diag_df, use_container_width=True, hide_index=True)

    # Expandable per aksje
    for ticker in ACTIVE_TICKERS:
        if ticker not in reg_results:
            continue
        r = reg_results[ticker]
        with st.expander(f"{ticker} — R²={r['r_squared']:.4f}, {r['n_obs']} obs"):
            detail = pd.DataFrame({
                "Faktor": [FACTOR_SHORT_NAMES.get(f, f) for f in r["betas"]],
                "Beta": [round(v, 6) for v in r["betas"].values()],
                "t-stat": [round(r["t_stats"][f], 2) for f in r["betas"]],
                "p-verdi": [round(r["p_values"][f], 4) for f in r["betas"]],
            })
            detail["Signifikant"] = detail["p-verdi"].apply(
                lambda p: "✅" if p < 0.05 else "")
            st.dataframe(detail, use_container_width=True, hide_index=True)

    # Regime-statistikk
    if regime_mode:
        st.divider()
        st.subheader("Regime-statistikk (VIX-basert)")
        st.caption(f"Terskel: VIX = {vix_threshold} | Nåværende VIX: {current_vix}")

        regime_rows = []
        for ticker in ACTIVE_TICKERS:
            if ticker not in reg_results:
                continue
            r = reg_results[ticker]
            counts = r.get("regime_counts", {})
            if counts:
                regime_rows.append({
                    "Ticker": ticker,
                    "Normal obs": counts.get("normal", "N/A"),
                    "Stress obs": counts.get("stress", "N/A"),
                    "Stress-andel (%)": round(counts.get("stress", 0) /
                                              max(1, counts.get("normal", 0) + counts.get("stress", 0)) * 100, 1),
                })
        if regime_rows:
            st.dataframe(pd.DataFrame(regime_rows), use_container_width=True, hide_index=True)

            # Vis eksempel på beta-forskjell for SPY
            if spy_result and "betas_normal" in spy_result and "betas_stress" in spy_result:
                st.markdown("**SPY beta-sammenligning: Normal vs Stress**")
                beta_comp = []
                for f in FACTOR_NAMES:
                    beta_comp.append({
                        "Faktor": FACTOR_SHORT_NAMES.get(f, f),
                        "Normal beta": round(spy_result["betas_normal"].get(f, 0), 6),
                        "Stress beta": round(spy_result["betas_stress"].get(f, 0), 6),
                        "Differanse": round(spy_result["betas_stress"].get(f, 0) -
                                            spy_result["betas_normal"].get(f, 0), 6),
                    })
                st.dataframe(pd.DataFrame(beta_comp), use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────
# FANE 5: CUSTOM SCENARIO BUILDER
# ───────────────────────────────────────────────

with tab5:
    st.subheader("Bygg ditt eget scenario")
    st.caption("Juster faktorsjokk med sliderne. Resultatene oppdateres i sanntid.")

    custom_shocks = {}
    cols = st.columns(3)
    for i, factor in enumerate(FACTOR_NAMES):
        cfg = FRED_FACTORS.get(factor, YAHOO_FACTORS.get(factor, {}))
        desc = FACTOR_DESCRIPTIONS[factor]
        transform = cfg.get("transform", "")

        short = FACTOR_SHORT_NAMES.get(factor, factor)
        unit = FACTOR_UNITS.get(factor, "")

        with cols[i % 3]:
            if factor == "VIX":
                val = st.slider(f"{short} (poeng)", -30, 50, 0, step=1,
                                help=desc, key=f"custom_{factor}")
            elif transform == "diff":
                val = st.slider(f"{short} (%)", -3.00, 3.00, 0.0, step=0.10,
                                help=desc, format="%.2f%%", key=f"custom_{factor}")
            else:
                val = st.slider(f"{short} (%)", -0.50, 0.50, 0.0, step=0.01,
                                help=desc, format="%.0f%%", key=f"custom_{factor}")
            custom_shocks[factor] = val

    # Resultater
    st.divider()
    custom_port = calc_portfolio_impact(reg_results, ACTIVE_PORTFOLIO, custom_shocks, **_regime_kw) * 100
    custom_bench = calc_benchmark_impact(spy_result, custom_shocks, **_regime_kw) * 100
    custom_diff = custom_port - custom_bench

    k1, k2, k3 = st.columns(3)
    k1.metric("Portefølje", f"{custom_port:.1f}%")
    k2.metric("S&P 500", f"{custom_bench:.1f}%")
    k3.metric("Differanse", f"{custom_diff:+.1f}pp",
              delta_color="normal" if custom_diff >= 0 else "inverse")

    # Topp 5 vinnere/tapere
    custom_impacts = calc_all_stock_impacts(reg_results, custom_shocks, **_regime_kw)
    impact_sorted = sorted(custom_impacts.items(), key=lambda x: x[1])

    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Topp 5 tapere**")
        for ticker, imp in impact_sorted[:5]:
            st.markdown(f"🔴 {ticker}: {imp*100:.2f}%")
    with w2:
        st.markdown("**Topp 5 vinnere**")
        for ticker, imp in impact_sorted[-5:][::-1]:
            st.markdown(f"🟢 {ticker}: {imp*100:.2f}%")

    # Mini waterfall
    contrib = calc_stock_contributions(reg_results, ACTIVE_PORTFOLIO, custom_shocks, **_regime_kw)
    contrib = contrib[contrib["Ticker"] != "CASH"]
    custom_capped = contrib["Cappet"].any()

    mini_labels = [f"{t}*" if c else t
                   for t, c in zip(contrib["Ticker"], contrib["Cappet"])]
    mini_colors = []
    for _, row in contrib.iterrows():
        if row["Cappet"]:
            mini_colors.append("#991b1b" if row["Bidrag (pp)"] < 0 else "#166534")
        else:
            mini_colors.append("#ef4444" if row["Bidrag (pp)"] < 0 else "#22c55e")

    fig_mini = go.Figure(go.Waterfall(
        x=mini_labels, y=contrib["Bidrag (pp)"],
        connector={"line": {"color": "rgba(0,0,0,0.1)"}},
        increasing={"marker": {"color": "#22c55e"}},
        decreasing={"marker": {"color": "#ef4444"}},
    ))
    if custom_capped:
        fig_mini.update_traces(marker=dict(color=mini_colors))
    fig_mini.update_layout(title="Bidrag per aksje", height=400, template="plotly_white")
    st.plotly_chart(fig_mini, use_container_width=True)

    if custom_capped:
        n_c = contrib["Cappet"].sum()
        st.caption(
            f"⚠️ *{n_c} aksje{'r' if n_c > 1 else ''} merket med \\* er cappet til "
            f"[{CAP_FLOOR*100:.0f}%, +{CAP_CEIL*100:.0f}%].*"
        )

# ───────────────────────────────────────────────
# FANE 6: HISTORISK BACKTESTING
# ───────────────────────────────────────────────

with tab6:
    st.subheader("Historisk backtesting")
    st.caption("Sammenlign modellens estimat med faktisk avkastning i historiske perioder.")

    period_name = st.selectbox("Velg periode", list(BACKTEST_PERIODS.keys()))
    start_dt, end_dt = BACKTEST_PERIODS[period_name]

    bt = calc_backtest(factor_data, returns, reg_results, spy_result,
                       ACTIVE_PORTFOLIO, start_dt, end_dt)

    if bt is None:
        st.warning("Ikke nok data for denne perioden.")
    else:
        st.subheader(period_name)

        # Estimat vs faktisk
        r1, r2, r3 = st.columns(3)
        r1.markdown("### Portefølje")
        r1.metric("Estimert", f"{bt['est_portfolio']:.1f}%")
        r1.metric("Faktisk", f"{bt['actual_portfolio']:.1f}%")
        r1.metric("Tracking error", f"{bt['tracking_error_port']:+.1f}pp")

        r2.markdown("### S&P 500")
        r2.metric("Estimert", f"{bt['est_benchmark']:.1f}%")
        r2.metric("Faktisk", f"{bt['actual_benchmark']:.1f}%")
        r2.metric("Tracking error", f"{bt['tracking_error_bench']:+.1f}pp")

        r3.markdown("### Differanse (aktiv)")
        r3.metric("Estimert", f"{bt['est_diff']:+.1f}pp")
        r3.metric("Faktisk", f"{bt['actual_diff']:+.1f}pp")

        # Faktorsjokk i perioden
        with st.expander("Faktorsjokk i perioden"):
            shock_df = pd.DataFrame([
                {"Faktor": FACTOR_SHORT_NAMES.get(f, f),
                 "Kumulativ endring": round(v, 4),
                 "Beskrivelse": FACTOR_DESCRIPTIONS.get(f, "")}
                for f, v in bt["cumulative_shocks"].items()
            ])
            st.dataframe(shock_df, use_container_width=True, hide_index=True)

        # Sammenligning bar chart
        comp_data = pd.DataFrame({
            "Kategori": ["Portefølje", "S&P 500", "Differanse"],
            "Estimert": [bt["est_portfolio"], bt["est_benchmark"], bt["est_diff"]],
            "Faktisk": [bt["actual_portfolio"], bt["actual_benchmark"], bt["actual_diff"]],
        })

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Bar(name="Estimert", x=comp_data["Kategori"],
                                y=comp_data["Estimert"], marker_color="#2563eb"))
        fig_bt.add_trace(go.Bar(name="Faktisk", x=comp_data["Kategori"],
                                y=comp_data["Faktisk"], marker_color="#22c55e"))
        fig_bt.update_layout(barmode="group", title="Estimert vs Faktisk (%)",
                             height=400, template="plotly_white")
        st.plotly_chart(fig_bt, use_container_width=True)

# ───────────────────────────────────────────────
# ───────────────────────────────────────────────
# FANE 7: HISTORISKE KRISER
# ───────────────────────────────────────────────

with tab7:
    st.subheader("Historiske kriser — faktiske drawdowns")
    st.caption("Viser faktisk peak-to-trough drawdown for porteføljen i historiske kriseperioder. "
               "Aksjer uten data erstattes med sektor-ETF proxy.")

    if st.button("Beregn krisedata", key="crisis_calc"):
        with st.spinner("Henter historiske priser fra Yahoo Finance..."):
            norm_w = normalize_weights(ACTIVE_PORTFOLIO)
            crisis_results = calc_all_crises(norm_w, SECTOR_PROXY)
            st.session_state["crisis_results"] = crisis_results

    if "crisis_results" in st.session_state:
        crisis_results = st.session_state["crisis_results"]

        # ── Oppsummeringstabell ──
        summary_rows = []
        for crisis_name, data in crisis_results.items():
            p_dd = data["portefolje_drawdown"]
            s_dd = data["spy_drawdown"]
            n_proxy = sum(1 for v in data["proxy_info"].values() if v is not None)
            summary_rows.append({
                "Krise": crisis_name,
                "Periode": f"{data['periode'][0]} → {data['periode'][1]}",
                "Portefølje (%)": p_dd,
                "S&P 500 (%)": s_dd,
                "Differanse (pp)": round(p_dd - s_dd, 2) if p_dd is not None and s_dd is not None else None,
                "Proxyer brukt": n_proxy,
            })
        summary_df = pd.DataFrame(summary_rows)

        st.markdown("### Oversikt")

        # KPI-er for verste krise
        worst = summary_df.loc[summary_df["Portefølje (%)"].idxmin()]
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Verste krise", worst["Krise"])
        col_k2.metric("Portefølje drawdown", f"{worst['Portefølje (%)']:.1f}%")
        col_k3.metric("vs S&P 500", f"{worst['Differanse (pp)']:+.1f}pp")

        st.dataframe(
            summary_df.style.format({
                "Portefølje (%)": "{:.1f}",
                "S&P 500 (%)": "{:.1f}",
                "Differanse (pp)": "{:+.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # ── Grouped bar chart: portefølje vs SPY per krise ──
        fig_crisis = go.Figure()
        fig_crisis.add_trace(go.Bar(
            name="Portefølje", x=summary_df["Krise"],
            y=summary_df["Portefølje (%)"],
            marker_color="#ef4444",
        ))
        fig_crisis.add_trace(go.Bar(
            name="S&P 500", x=summary_df["Krise"],
            y=summary_df["S&P 500 (%)"],
            marker_color="#9ca3af",
        ))
        fig_crisis.update_layout(
            barmode="group", title="Portefølje vs S&P 500 — historiske kriser",
            yaxis_title="Drawdown (%)", height=450, template="plotly_white",
        )
        st.plotly_chart(fig_crisis, use_container_width=True)

        # ── Heatmap: aksjer × kriser ──
        st.markdown("### Heatmap — drawdown per aksje og krise")
        heatmap_data = {}
        for crisis_name, data in crisis_results.items():
            for stock in data["aksjer"]:
                ticker = stock["Ticker"]
                if ticker not in heatmap_data:
                    heatmap_data[ticker] = {}
                label = f"{stock['Drawdown (%)']}%"
                if stock["Proxy"]:
                    label += " *"
                heatmap_data[ticker][crisis_name] = stock["Drawdown (%)"]

        if heatmap_data:
            hm_df = pd.DataFrame(heatmap_data).T
            hm_df = hm_df[list(CRISIS_PERIODS.keys())]  # riktig rekkefølge
            hm_df = hm_df.sort_values(hm_df.columns[0])  # sorter etter første krise

            fig_hm = px.imshow(
                hm_df.values, x=hm_df.columns.tolist(), y=hm_df.index.tolist(),
                color_continuous_scale="RdYlGn", aspect="auto",
                labels=dict(color="Drawdown (%)"),
                text_auto=".0f",
            )
            fig_hm.update_layout(
                title="Drawdown per aksje × krise",
                height=max(400, len(hm_df) * 22),
                template="plotly_white",
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption("* = sektor-ETF proxy brukt (aksjen eksisterte ikke i perioden)")

        # ── Per-krise deep dive ──
        st.markdown("### Detaljer per krise")
        sel_crisis = st.selectbox("Velg krise", list(CRISIS_PERIODS.keys()), key="crisis_select")
        c_data = crisis_results[sel_crisis]

        if c_data["aksjer"]:
            c_df = pd.DataFrame(c_data["aksjer"])
            # Marker proxyer
            c_df["Datakilde"] = c_df["Proxy"].apply(
                lambda x: f"Proxy: {x}" if x else "Faktisk"
            )

            col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.metric("Portefølje", f"{c_data['portefolje_drawdown']:.1f}%")
            col_c2.metric("S&P 500", f"{c_data['spy_drawdown']:.1f}%")
            diff = c_data['portefolje_drawdown'] - c_data['spy_drawdown']
            col_c3.metric("Differanse", f"{diff:+.1f}pp")

            # Waterfall-lignende bar chart
            fig_detail = go.Figure()
            colors = ["#f97316" if row["Proxy"] else "#3b82f6" for _, row in c_df.iterrows()]
            fig_detail.add_trace(go.Bar(
                x=c_df["Ticker"], y=c_df["Drawdown (%)"],
                marker_color=colors,
                text=c_df["Drawdown (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ))
            fig_detail.update_layout(
                title=f"{sel_crisis}: Drawdown per aksje",
                yaxis_title="Drawdown (%)", height=500,
                template="plotly_white", showlegend=False,
                xaxis_tickangle=-45,
            )
            # Legg til legend manuelt
            fig_detail.add_trace(go.Bar(x=[None], y=[None], marker_color="#3b82f6", name="Faktisk data", showlegend=True))
            fig_detail.add_trace(go.Bar(x=[None], y=[None], marker_color="#f97316", name="Sektor-proxy", showlegend=True))
            fig_detail.update_layout(showlegend=True)
            st.plotly_chart(fig_detail, use_container_width=True)

            # Tabell
            show_cols = ["Ticker", "Vekt (%)", "Drawdown (%)", "Bidrag (pp)", "Datakilde"]
            st.dataframe(
                c_df[show_cols].style.format({
                    "Vekt (%)": "{:.1f}",
                    "Drawdown (%)": "{:.1f}",
                    "Bidrag (pp)": "{:+.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Ingen data tilgjengelig for denne krisen.")
    else:
        st.info("Klikk «Beregn krisedata» for å hente historiske priser og beregne drawdowns.")

# ───────────────────────────────────────────────
# FANE 8: PORTEFØLJEOPTIMERING
# ───────────────────────────────────────────────

with tab8:
    st.subheader("Porteføljeoptimering")
    st.caption("Identifiser svakheter og simuler vektjusteringer for å forbedre porteføljen.")

    vuln = calc_vulnerability_analysis(
        reg_results, spy_result, ACTIVE_PORTFOLIO, SCENARIOS, port_betas, **_regime_kw
    )

    # ── Sårbarhetsoversikt ──
    st.markdown("### Scenarioscorecard")
    st.caption("Scenarioer rangert etter relativ avkastning vs S&P 500")

    scorecard_data = []
    for sr in vuln["scenario_results"]:
        scorecard_data.append({
            "Scenario": sr["name"],
            "Portefølje (%)": round(sr["port"], 2),
            "S&P 500 (%)": round(sr["bench"], 2),
            "Differanse (pp)": round(sr["diff"], 2),
            "Resultat": "✅ Slår" if sr["diff"] > 0 else "❌ Taper",
        })
    scorecard_df = pd.DataFrame(scorecard_data)

    # Fargekod differanse
    st.dataframe(
        scorecard_df.style.applymap(
            lambda v: "color: #22c55e" if isinstance(v, str) and "Slår" in v
            else "color: #ef4444" if isinstance(v, str) and "Taper" in v else "",
            subset=["Resultat"]
        ),
        use_container_width=True, hide_index=True,
    )

    n_beat = sum(1 for sr in vuln["scenario_results"] if sr["diff"] > 0)
    n_total = len(vuln["scenario_results"])
    avg_diff = np.mean([sr["diff"] for sr in vuln["scenario_results"]])
    st.info(f"Porteføljen slår S&P 500 i **{n_beat}/{n_total}** scenarioer. "
            f"Gjennomsnittlig differanse: **{avg_diff:+.2f}pp**.")

    st.divider()

    # ── Verste scenarioer: hva går galt? ──
    st.markdown("### De 5 verste scenarioene — hva går galt?")

    for sr in vuln["worst"][:3]:
        with st.expander(f"❌ {sr['name']} — differanse {sr['diff']:+.1f}pp"):
            # Faktorforklaring
            st.markdown("**Faktor-bidrag til underavkastning:**")
            fc_neg = [fc for fc in sr["factor_contribs"] if fc["active_contrib_pct"] < -0.1]
            if fc_neg:
                for fc in fc_neg[:5]:
                    fn = FACTOR_SHORT_NAMES.get(fc["factor"], fc["factor"])
                    st.markdown(
                        f"- **{fn}**: aktiv beta {fc['active_beta']:.5f} × "
                        f"sjokk {fc['shock']:+.1f} = **{fc['active_contrib_pct']:+.2f}pp**"
                    )
            else:
                st.caption("Ingen enkeltfaktor dominerer — spredt underavkastning.")

            # Aksjer som drar ned
            st.markdown("**Aksjer med størst negativt bidrag:**")
            worst_stocks = [sc for sc in sr["stock_contribs"] if sc["contrib_pp"] < -0.05][:5]
            for sc in worst_stocks:
                st.markdown(
                    f"- **{sc['ticker']}** ({sc['weight']:.1f}%): "
                    f"effekt {sc['impact_pct']:+.1f}%, bidrag **{sc['contrib_pp']:+.2f}pp**"
                )

    st.divider()

    # ── Aktive faktor-tilts ──
    st.markdown("### Aktive faktoreksponeringer")
    st.caption("Positiv = overeksponert vs SPY, negativ = undereksponert. "
               "Samlet drag = kumulativt bidrag til underavkastning i tapsscenariene.")

    tilt_data = []
    for f in FACTOR_NAMES:
        ab = vuln["active_betas"][f]
        drag = vuln["factor_vulnerability"][f]
        tilt_data.append({
            "Faktor": FACTOR_SHORT_NAMES.get(f, f),
            "Aktiv beta": ab,
            "Retning": "Overeksponert" if ab > 0 else "Undereksponert",
            "Samlet drag (pp)": round(drag, 2),
        })
    tilt_df = pd.DataFrame(tilt_data)

    fig_tilt = go.Figure()
    fig_tilt.add_trace(go.Bar(
        x=[FACTOR_SHORT_NAMES.get(f, f) for f in FACTOR_NAMES],
        y=[vuln["active_betas"][f] for f in FACTOR_NAMES],
        marker_color=["#2563eb" if vuln["active_betas"][f] >= 0 else "#f97316"
                       for f in FACTOR_NAMES],
        name="Aktiv beta",
    ))
    fig_tilt.update_layout(
        title="Aktive faktoreksponeringer (portefølje minus SPY)",
        yaxis_title="Aktiv beta", height=400, template="plotly_white",
    )
    st.plotly_chart(fig_tilt, use_container_width=True)

    st.dataframe(tilt_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Konkrete forbedringsforslag ──
    st.markdown("### Forbedringsforslag")

    if vuln["suggestions"]:
        for i, s in enumerate(vuln["suggestions"]):
            if s["type"] == "faktor":
                fn = FACTOR_SHORT_NAMES.get(s["factor"], s["factor"])
                icon = "📐"
                st.markdown(
                    f"{icon} **Faktor-tilt: {fn}**\n\n"
                    f"{s['text']}\n\n"
                    f"*Tiltak: Vurder aksjer som har motsatt eksponering mot denne faktoren.*"
                )
            elif s["type"] == "aksje":
                icon = "📉"
                st.markdown(
                    f"{icon} **Aksje: {s['ticker']}**\n\n"
                    f"{s['text']}"
                )
            if i < len(vuln["suggestions"]) - 1:
                st.markdown("---")
    else:
        st.success("Ingen vesentlige forbedringsforslag — porteføljen er godt balansert!")

    st.divider()

    # ── Vektsimulator ──
    st.markdown("### Vektsimulator")
    st.caption("Juster vekter og se hvordan det påvirker scenarioprofilen. "
               "Endringene beregnes relativt til nåværende portefølje.")

    # Velg aksjer å justere
    adjust_tickers = st.multiselect(
        "Velg aksjer å justere",
        ACTIVE_TICKERS,
        default=[s["ticker"] for s in vuln["suggestions"] if s["type"] == "aksje"][:3],
    )

    if adjust_tickers:
        sim_weights = dict(ACTIVE_PORTFOLIO)
        adjustments = {}

        cols_sim = st.columns(min(len(adjust_tickers), 4))
        for i, ticker in enumerate(adjust_tickers):
            with cols_sim[i % len(cols_sim)]:
                current = ACTIVE_PORTFOLIO.get(ticker, 0)
                new_w = st.slider(
                    f"{ticker} ({current:.1f}%)",
                    0.0, min(current * 3, 15.0), current,
                    step=0.1, key=f"sim_{ticker}",
                )
                adjustments[ticker] = new_w - current
                sim_weights[ticker] = new_w

        # Redistribuer differansen til CASH
        total_adj = sum(adjustments.values())
        sim_weights["CASH"] = max(0, ACTIVE_PORTFOLIO.get("CASH", 0) - total_adj)

        # Renormaliser og beregn
        sim_norm = normalize_weights(sim_weights)
        sim_port_betas = portfolio_betas(reg_results, sim_norm, FACTOR_NAMES)

        # Sammenlign scenarier
        st.markdown("#### Scenariopåvirkning av justeringene")
        sim_rows = []
        for name, scen in SCENARIOS.items():
            orig_port = calc_portfolio_impact(reg_results, ACTIVE_PORTFOLIO, scen, **_regime_kw) * 100
            bench = calc_benchmark_impact(spy_result, scen, **_regime_kw) * 100

            # Simulert portefølje
            sim_port = 0.0
            stock_impacts = calc_all_stock_impacts(reg_results, scen, **_regime_kw)
            for ticker, w in sim_weights.items():
                if ticker == "CASH":
                    continue
                sim_port += (w / 100.0) * stock_impacts.get(ticker, 0.0) * 100

            orig_diff = orig_port - bench
            sim_diff = sim_port - bench

            sim_rows.append({
                "Scenario": name,
                "Nåværende (pp)": round(orig_diff, 2),
                "Simulert (pp)": round(sim_diff, 2),
                "Endring (pp)": round(sim_diff - orig_diff, 2),
            })

        sim_df = pd.DataFrame(sim_rows)

        # Bar chart: endring
        improved = (sim_df["Endring (pp)"] > 0.01).sum()
        worsened = (sim_df["Endring (pp)"] < -0.01).sum()

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            x=sim_df["Scenario"], y=sim_df["Nåværende (pp)"],
            name="Nåværende", marker_color="#9ca3af", opacity=0.5,
        ))
        fig_sim.add_trace(go.Bar(
            x=sim_df["Scenario"], y=sim_df["Simulert (pp)"],
            name="Simulert",
            marker_color=["#22c55e" if v >= 0 else "#ef4444"
                           for v in sim_df["Simulert (pp)"]],
        ))
        fig_sim.update_layout(
            barmode="group",
            title="Differanse vs S&P 500 — nåværende vs simulert",
            yaxis_title="Differanse (pp)", height=500,
            xaxis_tickangle=-45, template="plotly_white",
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        st.markdown(
            f"Justeringen **forbedrer** {improved} scenarioer og "
            f"**forverrer** {worsened} scenarioer."
        )

        st.dataframe(sim_df, use_container_width=True, hide_index=True)

# ───────────────────────────────────────────────
# FANE 9: AKSJESØK
# ───────────────────────────────────────────────

with tab9:
    st.subheader("Aksjesøk & Sammenligning")
    st.caption("Søk opp en aksje og se hvordan den passer inn i porteføljen din.")

    # ── Input-seksjon ──
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_ticker = st.text_input(
            "Ticker", placeholder="F.eks. NVDA, JPM, AMZN",
            key="search_ticker_input",
        ).strip().upper()
    with search_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("Analyser", type="primary", key="search_btn")

    # Session state cache for søkt aksje
    if "search_cache" not in st.session_state:
        st.session_state["search_cache"] = {}

    if search_clicked and search_ticker:
        # Hent prisdata
        with st.spinner(f"Henter data for {search_ticker}..."):
            search_prices = fetch_single_ticker(search_ticker)

        if search_prices is None:
            st.error(f"Kunne ikke hente data for '{search_ticker}'. Sjekk at tickeren er gyldig.")
        else:
            # Beregn log-returns og kjør regresjon
            search_returns = compute_log_returns(search_prices.to_frame())
            search_ret_series = search_returns[search_ticker].dropna()
            search_reg = run_single_regression(search_ret_series, factor_data)

            if search_reg is None:
                st.error(f"Regresjonen feilet for {search_ticker}. For lite data.")
            else:
                st.session_state["search_cache"] = {
                    "ticker": search_ticker,
                    "result": search_reg,
                }

    # Vis resultater fra cache
    if st.session_state.get("search_cache", {}).get("ticker"):
        s_ticker = st.session_state["search_cache"]["ticker"]
        s_result = st.session_state["search_cache"]["result"]
        s_betas = s_result["betas"]
        spy_betas_search = spy_result["betas"] if spy_result else {}

        st.success(f"Viser analyse for **{s_ticker}**")

        # ══════════════════════════════════════
        # Seksjon A: Regresjonsresultat
        # ══════════════════════════════════════
        st.subheader(f"A) Regresjonsresultat — {s_ticker}")

        ka1, ka2, ka3, ka4 = st.columns(4)
        ka1.metric("R²", f"{s_result['r_squared']:.4f}")
        ka2.metric("Adj. R²", f"{s_result['adj_r_squared']:.4f}")
        ka3.metric("Durbin-Watson", f"{s_result['durbin_watson']:.2f}")
        ka4.metric("Observasjoner", s_result["n_obs"])

        search_detail = pd.DataFrame({
            "Faktor": [FACTOR_SHORT_NAMES.get(f, f) for f in s_result["betas"]],
            "Beta": list(s_result["betas"].values()),
            "t-stat": [round(s_result["t_stats"][f], 2) for f in s_result["betas"]],
            "p-verdi": [round(s_result["p_values"][f], 4) for f in s_result["betas"]],
        })
        search_detail["Signifikant"] = search_detail["p-verdi"].apply(
            lambda p: "✅" if p < 0.05 else "")
        st.dataframe(search_detail, use_container_width=True, hide_index=True)

        st.divider()

        # ══════════════════════════════════════
        # Seksjon B: Faktoreksponering — Radar
        # ══════════════════════════════════════
        st.subheader(f"B) Faktoreksponering — Radar")

        # Normaliser betaer for radar
        all_search_betas = []
        for f in FACTOR_NAMES:
            all_search_betas.extend([
                abs(s_betas.get(f, 0)),
                abs(port_betas.get(f, 0)),
                abs(spy_betas_search.get(f, 0)),
            ])
        max_beta_search = max(all_search_betas) if all_search_betas and max(all_search_betas) > 0 else 1

        s_norm = [s_betas.get(f, 0) / max_beta_search for f in FACTOR_NAMES]
        p_norm = [port_betas.get(f, 0) / max_beta_search for f in FACTOR_NAMES]
        spy_norm_s = [spy_betas_search.get(f, 0) / max_beta_search for f in FACTOR_NAMES]

        short_names_s = [FACTOR_SHORT_NAMES.get(f, f) for f in FACTOR_NAMES]

        fig_radar_s = go.Figure()
        fig_radar_s.add_trace(go.Scatterpolar(
            r=s_norm + [s_norm[0]],
            theta=short_names_s + [short_names_s[0]],
            name=s_ticker, fill="toself", opacity=0.6,
            line=dict(color="#f59e0b"),
        ))
        fig_radar_s.add_trace(go.Scatterpolar(
            r=p_norm + [p_norm[0]],
            theta=short_names_s + [short_names_s[0]],
            name="Portefølje", fill="toself", opacity=0.4,
            line=dict(color="#2563eb"),
        ))
        fig_radar_s.add_trace(go.Scatterpolar(
            r=spy_norm_s + [spy_norm_s[0]],
            theta=short_names_s + [short_names_s[0]],
            name="S&P 500", fill="toself", opacity=0.3,
            line=dict(color="#9ca3af"),
        ))
        fig_radar_s.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            height=500, template="plotly_white",
            title=f"Faktoreksponering — {s_ticker} vs Portefølje vs S&P 500",
        )
        st.plotly_chart(fig_radar_s, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════
        # Seksjon C: Faktor-beta sammenligning
        # ══════════════════════════════════════
        st.subheader(f"C) Faktor-beta sammenligning")

        beta_comp_s = pd.DataFrame({
            "Faktor": [FACTOR_SHORT_NAMES.get(f, f) for f in FACTOR_NAMES],
            s_ticker: [s_betas.get(f, 0) for f in FACTOR_NAMES],
            "Portefølje": [port_betas.get(f, 0) for f in FACTOR_NAMES],
            "S&P 500": [spy_betas_search.get(f, 0) for f in FACTOR_NAMES],
        })

        fig_beta_s = go.Figure()
        fig_beta_s.add_trace(go.Bar(
            name=s_ticker, x=beta_comp_s["Faktor"],
            y=beta_comp_s[s_ticker], marker_color="#f59e0b",
        ))
        fig_beta_s.add_trace(go.Bar(
            name="Portefølje", x=beta_comp_s["Faktor"],
            y=beta_comp_s["Portefølje"], marker_color="#2563eb",
        ))
        fig_beta_s.add_trace(go.Bar(
            name="S&P 500", x=beta_comp_s["Faktor"],
            y=beta_comp_s["S&P 500"], marker_color="#9ca3af",
        ))
        fig_beta_s.update_layout(
            barmode="group",
            title=f"Faktor-betaer — {s_ticker} vs Portefølje vs S&P 500",
            yaxis_title="Beta", height=450, template="plotly_white",
        )
        st.plotly_chart(fig_beta_s, use_container_width=True)

        # Differansetabell
        beta_comp_s["Diff vs Portefølje"] = beta_comp_s[s_ticker] - beta_comp_s["Portefølje"]
        beta_comp_s["Diff vs S&P 500"] = beta_comp_s[s_ticker] - beta_comp_s["S&P 500"]
        show_beta_s = beta_comp_s.copy()
        for col in [s_ticker, "Portefølje", "S&P 500", "Diff vs Portefølje", "Diff vs S&P 500"]:
            show_beta_s[col] = show_beta_s[col].apply(lambda v: f"{v:.6f}")
        st.dataframe(show_beta_s, use_container_width=True, hide_index=True)

        st.divider()

        # ══════════════════════════════════════
        # Seksjon D: Scenarioprofil
        # ══════════════════════════════════════
        st.subheader(f"D) Scenarioprofil — {s_ticker} vs Portefølje vs S&P 500")

        scenario_rows_s = []
        for sc_name, sc_shocks in SCENARIOS.items():
            s_impact = _cap_impact(calc_stock_scenario_impact(s_betas, sc_shocks)) * 100
            p_impact = calc_portfolio_impact(reg_results, ACTIVE_PORTFOLIO, sc_shocks, **_regime_kw) * 100
            b_impact = calc_benchmark_impact(spy_result, sc_shocks, **_regime_kw) * 100
            scenario_rows_s.append({
                "Scenario": sc_name,
                s_ticker: round(s_impact, 2),
                "Portefølje": round(p_impact, 2),
                "S&P 500": round(b_impact, 2),
            })
        scenario_df_s = pd.DataFrame(scenario_rows_s)

        # Grouped bar chart med fargekoding
        fig_scen_s = go.Figure()
        fig_scen_s.add_trace(go.Bar(
            name=s_ticker, x=scenario_df_s["Scenario"],
            y=scenario_df_s[s_ticker],
            marker_color=["#22c55e" if v >= 0 else "#ef4444"
                           for v in scenario_df_s[s_ticker]],
        ))
        fig_scen_s.add_trace(go.Bar(
            name="Portefølje", x=scenario_df_s["Scenario"],
            y=scenario_df_s["Portefølje"], marker_color="#2563eb", opacity=0.6,
        ))
        fig_scen_s.add_trace(go.Bar(
            name="S&P 500", x=scenario_df_s["Scenario"],
            y=scenario_df_s["S&P 500"], marker_color="#9ca3af", opacity=0.5,
        ))
        fig_scen_s.update_layout(
            barmode="group",
            title=f"Estimert effekt per scenario — {s_ticker} vs Portefølje vs S&P 500",
            yaxis_title="Estimert effekt (%)", height=550,
            xaxis_tickangle=-45, template="plotly_white",
        )
        st.plotly_chart(fig_scen_s, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════
        # Seksjon E: Sammenligning med porteføljeaksjer
        # ══════════════════════════════════════
        st.subheader(f"E) Sammenligning med porteføljeaksjer")

        sel_scenario_e = st.selectbox(
            "Velg scenario for sammenligning",
            list(SCENARIOS.keys()),
            key="search_scenario_select",
        )
        sel_shocks_e = SCENARIOS[sel_scenario_e]

        # Beregn effekt for alle porteføljeaksjer + søkt aksje
        comp_rows = []
        for ticker in ACTIVE_TICKERS:
            if ticker not in reg_results:
                continue
            t_betas = reg_results[ticker]["betas"]
            t_impact = _cap_impact(calc_stock_scenario_impact(t_betas, sel_shocks_e)) * 100
            comp_rows.append({
                "Ticker": ticker,
                "Vekt (%)": ACTIVE_PORTFOLIO.get(ticker, 0),
                "Estimert effekt (%)": round(t_impact, 2),
                "Kilde": "Portefølje",
            })

        # Legg til søkt aksje
        s_impact_e = _cap_impact(calc_stock_scenario_impact(s_betas, sel_shocks_e)) * 100
        comp_rows.append({
            "Ticker": f"➡ {s_ticker}",
            "Vekt (%)": 0.0,
            "Estimert effekt (%)": round(s_impact_e, 2),
            "Kilde": "Søk",
        })

        comp_df = pd.DataFrame(comp_rows).sort_values("Estimert effekt (%)", ascending=False)

        # Fargelegg raden for søkt aksje
        def _highlight_search(row):
            if row["Kilde"] == "Søk":
                return ["background-color: #fef3c7; font-weight: bold"] * len(row)
            return [""] * len(row)

        show_comp = comp_df[["Ticker", "Vekt (%)", "Estimert effekt (%)"]].copy()
        styled_comp = comp_df.style.apply(_highlight_search, axis=1).format({
            "Vekt (%)": "{:.1f}",
            "Estimert effekt (%)": "{:.2f}",
        })
        st.dataframe(styled_comp, use_container_width=True, hide_index=True)
