"""Korrelasjonsanalyse: finn aksjer som er korrelert med en gitt ticker."""

import pandas as pd
import numpy as np
import yfinance as yf


def find_correlated_stocks(
    ticker: str,
    universe: list[str],
    period: str = "3y",
    min_corr: float = 0.50,
    return_ref_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Finn aksjer i universet som er korrelert med *ticker*.

    Returnerer DataFrame med kolonnene:
        Ticker, Korrelasjon, Årlig avk. (%), Vol. (%), Sharpe, Max Drawdown (%),

    Sortert etter Sharpe (fallende), filtrert på korrelasjon ≥ min_corr.

    Hvis return_ref_stats=True returneres (DataFrame, ref_stats_dict).
    """
    # Sørg for at søkt aksje er med i nedlastingen
    all_tickers = list(set(universe + [ticker]))

    # Batch-nedlasting
    prices = yf.download(all_tickers, period=period, auto_adjust=True, progress=False)

    if prices.empty:
        empty = pd.DataFrame()
        return (empty, {}) if return_ref_stats else empty

    # Håndter MultiIndex (flere tickers) og enkel Serie (én ticker)
    if isinstance(prices.columns, pd.MultiIndex):
        close = prices["Close"]
    else:
        close = prices[["Close"]]
        close.columns = all_tickers[:1]

    # Daglige returns
    returns = close.pct_change().dropna()

    if ticker not in returns.columns:
        empty = pd.DataFrame()
        return (empty, {}) if return_ref_stats else empty

    # Korrelasjon
    corr = returns.corrwith(returns[ticker])

    # Beregn nøkkeltall per aksje
    trading_days = 252
    rows = []

    for t in universe:
        if t == ticker or t not in returns.columns:
            continue
        c = corr.get(t, np.nan)
        if pd.isna(c) or c < min_corr:
            continue

        r = returns[t]
        ann_ret = r.mean() * trading_days * 100
        ann_vol = r.std() * np.sqrt(trading_days) * 100
        sharpe = (r.mean() / r.std()) * np.sqrt(trading_days) if r.std() > 0 else 0.0

        # Max drawdown
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = ((cum - peak) / peak).min() * 100  # negativ verdi

        rows.append({
            "Ticker": t,
            "Korrelasjon": round(c, 3),
            "Årlig avk. (%)": round(ann_ret, 1),
            "Vol. (%)": round(ann_vol, 1),
            "Sharpe": round(sharpe, 2),
            "Max Drawdown (%)": round(dd, 1),
        })

    if not rows:
        empty = pd.DataFrame()
        if return_ref_stats:
            # Beregn ref_stats selv om ingen korrelerte aksjer ble funnet
            if ticker in returns.columns:
                ref = get_ticker_stats(ticker, returns[ticker])
            else:
                ref = {}
            return empty, ref
        return empty

    df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)

    if return_ref_stats:
        ref = get_ticker_stats(ticker, returns[ticker])
        return df, ref
    return df


def get_ticker_stats(
    ticker: str,
    returns: pd.Series,
) -> dict:
    """Beregn nøkkeltall for en enkelt ticker fra returns-serie."""
    trading_days = 252
    ann_ret = returns.mean() * trading_days * 100
    ann_vol = returns.std() * np.sqrt(trading_days) * 100
    sharpe = (returns.mean() / returns.std()) * np.sqrt(trading_days) if returns.std() > 0 else 0.0
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = ((cum - peak) / peak).min() * 100
    return {
        "Årlig avk. (%)": round(ann_ret, 1),
        "Vol. (%)": round(ann_vol, 1),
        "Sharpe": round(sharpe, 2),
        "Max Drawdown (%)": round(dd, 1),
    }
