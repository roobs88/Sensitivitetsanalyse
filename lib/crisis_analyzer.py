"""Historisk kriseanalyse — faktiske drawdowns for porteføljen."""

import pandas as pd
import yfinance as yf

from config.crises import CRISIS_PERIODS, SECTOR_PROXY


def fetch_crisis_prices(tickers: list, period: tuple,
                        sector_map: dict = None) -> tuple[pd.DataFrame, dict]:
    """Hent close-priser for en kriseperiode. Bruker sektor-proxy ved manglende data.

    Args:
        tickers: Liste med aksjetickers
        period: (start_date, end_date) som strenger
        sector_map: Ticker → sektor-ETF mapping

    Returns:
        (prices_df, proxy_info) der proxy_info viser hvem som brukte proxy
    """
    if sector_map is None:
        sector_map = SECTOR_PROXY

    start, end = period
    # Hent litt ekstra margin for å sikre at start/slutt-dato dekkes
    all_tickers = set(tickers) | {"SPY"}
    etf_tickers = {sector_map.get(t) for t in tickers if t in sector_map}
    all_tickers |= {etf for etf in etf_tickers if etf is not None}

    # Last alt i én batch
    data = yf.download(
        list(all_tickers),
        start=start,
        end=pd.Timestamp(end) + pd.Timedelta(days=5),  # litt margin
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame(), {}

    # yfinance returnerer MultiIndex (metric, ticker) for flere tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        # Bare én ticker
        prices = data[["Close"]].rename(columns={"Close": list(all_tickers)[0]})

    # Trim til faktisk periode
    prices = prices.loc[start:end]

    # Bestem proxy-bruk per ticker
    proxy_info = {}
    result_prices = pd.DataFrame(index=prices.index)

    for ticker in tickers:
        if ticker in prices.columns and prices[ticker].notna().sum() >= 2:
            result_prices[ticker] = prices[ticker]
            proxy_info[ticker] = None  # bruker faktisk data
        elif ticker in sector_map and sector_map[ticker] in prices.columns:
            etf = sector_map[ticker]
            if prices[etf].notna().sum() >= 2:
                result_prices[ticker] = prices[etf]
                proxy_info[ticker] = etf
            elif "SPY" in prices.columns:
                result_prices[ticker] = prices["SPY"]
                proxy_info[ticker] = "SPY (fallback)"
            else:
                proxy_info[ticker] = "MANGLER"
        elif "SPY" in prices.columns:
            result_prices[ticker] = prices["SPY"]
            proxy_info[ticker] = "SPY (fallback)"
        else:
            proxy_info[ticker] = "MANGLER"

    # Legg til SPY
    if "SPY" in prices.columns:
        result_prices["SPY"] = prices["SPY"]

    return result_prices, proxy_info


def calc_crisis_drawdowns(prices: pd.DataFrame, weights: dict,
                          proxy_info: dict) -> dict:
    """Beregn peak-to-trough drawdown for porteføljen i en kriseperiode.

    Args:
        prices: DataFrame med close-priser (ticker-kolonner)
        weights: Normaliserte vekter (sum ~ 1.0)
        proxy_info: Dict med proxy-info fra fetch_crisis_prices

    Returns:
        Dict med per-aksje drawdowns, portefølje-total, og SPY-drawdown
    """
    if prices.empty or len(prices) < 2:
        return {
            "aksjer": [],
            "portefolje_drawdown": None,
            "spy_drawdown": None,
        }

    first_prices = prices.iloc[0]
    last_prices = prices.iloc[-1]

    stock_details = []
    portfolio_dd = 0.0
    total_weight = 0.0

    for ticker in weights:
        if ticker == "CASH" or ticker not in prices.columns:
            continue

        p0 = first_prices.get(ticker)
        p1 = last_prices.get(ticker)
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            continue

        dd = (p1 / p0 - 1) * 100  # prosent
        w = weights.get(ticker, 0)
        proxy = proxy_info.get(ticker)

        stock_details.append({
            "Ticker": ticker,
            "Vekt (%)": round(w * 100, 2),
            "Drawdown (%)": round(dd, 2),
            "Bidrag (pp)": round(w * dd, 2),
            "Proxy": proxy,
        })

        portfolio_dd += w * dd
        total_weight += w

    # SPY drawdown
    spy_dd = None
    if "SPY" in prices.columns:
        spy_p0 = first_prices.get("SPY")
        spy_p1 = last_prices.get("SPY")
        if not pd.isna(spy_p0) and not pd.isna(spy_p1) and spy_p0 != 0:
            spy_dd = round((spy_p1 / spy_p0 - 1) * 100, 2)

    # Sorter etter drawdown (mest negativ først)
    stock_details.sort(key=lambda x: x["Drawdown (%)"])

    return {
        "aksjer": stock_details,
        "portefolje_drawdown": round(portfolio_dd, 2),
        "spy_drawdown": spy_dd,
    }


def calc_all_crises(weights: dict, sector_map: dict = None) -> dict:
    """Beregn drawdowns for alle historiske kriseperioder.

    Args:
        weights: Normaliserte vekter (sum ~ 1.0)
        sector_map: Sektor-proxy mapping (default: SECTOR_PROXY)

    Returns:
        Dict med krise-navn → {aksjer, portefolje_drawdown, spy_drawdown, proxy_info, periode}
    """
    if sector_map is None:
        sector_map = SECTOR_PROXY

    tickers = [t for t in weights if t != "CASH"]
    results = {}

    for crisis_name, period in CRISIS_PERIODS.items():
        prices, proxy_info = fetch_crisis_prices(tickers, period, sector_map)
        dd = calc_crisis_drawdowns(prices, weights, proxy_info)
        dd["proxy_info"] = proxy_info
        dd["periode"] = period
        results[crisis_name] = dd

    return results
