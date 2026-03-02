"""Datahenting fra yfinance og FRED med lokal pickle-cache."""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred

from config.factors import FRED_FACTORS, YAHOO_FACTORS, DATA_START_DATE
from config.portfolio import STOCK_TICKERS, BENCHMARK_TICKER
from lib.utils import save_cache, load_cache


def fetch_single_ticker(ticker: str) -> pd.Series | None:
    """Hent daglige close-priser for én ticker fra yfinance (on-demand, ingen cache).

    Returns:
        pd.Series med close-priser, eller None ved feil.
    """
    try:
        data = yf.download(ticker, start=DATA_START_DATE, auto_adjust=True, progress=False)
        if data.empty:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = ticker
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        return close.sort_index()
    except Exception:
        return None


def fetch_stock_prices(tickers: list[str], force_refresh: bool = False) -> pd.DataFrame:
    """Hent daglige close-priser for alle aksjer fra yfinance.

    Returnerer DataFrame med tickers som kolonner og datoer som indeks.
    """
    cached = None if force_refresh else load_cache("stock_prices")
    if cached is not None:
        return cached

    all_tickers = tickers + [BENCHMARK_TICKER]
    prices = pd.DataFrame()
    failed = []

    for ticker in all_tickers:
        try:
            data = yf.download(ticker, start=DATA_START_DATE, auto_adjust=True,
                               progress=False)
            if data.empty:
                failed.append(ticker)
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.name = ticker
            prices = prices.join(close, how="outer") if not prices.empty else close.to_frame()
        except Exception:
            failed.append(ticker)

    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index()

    save_cache(prices, "stock_prices")

    if failed:
        import streamlit as st
        st.warning(f"Kunne ikke hente data for: {', '.join(failed)}")

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Beregn daglige log-avkastninger fra prisnivåer."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def fetch_fred_factors(api_key: str, force_refresh: bool = False) -> pd.DataFrame:
    """Hent FRED-faktorer og appliser transformasjoner.

    Returnerer DataFrame med faktornavn som kolonner, datoer som indeks.
    """
    cached = None if force_refresh else load_cache("fred_factors")
    if cached is not None:
        return cached

    fred = Fred(api_key=api_key)
    raw = {}

    for name, cfg in FRED_FACTORS.items():
        try:
            series = fred.get_series(cfg["series"], observation_start=DATA_START_DATE)
            series = series.dropna()
            raw[name] = series
        except Exception as e:
            import streamlit as st
            st.warning(f"Kunne ikke hente FRED-serie {cfg['series']}: {e}")

    # Kombiner til DataFrame, reindekser til business days
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    bdays = pd.bdate_range(start=df.index.min(), end=df.index.max())
    df = df.reindex(bdays)
    df = df.ffill(limit=5)

    # Appliser transformasjoner
    result = pd.DataFrame(index=df.index)
    for name, cfg in FRED_FACTORS.items():
        if name not in df.columns:
            continue
        if cfg["transform"] == "diff":
            result[name] = df[name].diff()
        elif cfg["transform"] == "pct_change":
            result[name] = df[name].pct_change()

    result = result.dropna(how="all")
    save_cache(result, "fred_factors")
    return result


def fetch_yahoo_factors(force_refresh: bool = False) -> pd.DataFrame:
    """Hent Yahoo-baserte faktorer (VIX, relative ratioer)."""
    cached = None if force_refresh else load_cache("yahoo_factors")
    if cached is not None:
        return cached

    result = pd.DataFrame()

    # VIX
    vix_cfg = YAHOO_FACTORS["VIX"]
    try:
        vix_data = yf.download(vix_cfg["ticker"], start=DATA_START_DATE,
                               auto_adjust=True, progress=False)
        if not vix_data.empty:
            close = vix_data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            result["VIX"] = close.diff()
    except Exception:
        pass

    # Ratio-faktorer
    for name in ["TECH_REL", "CYCL_REL"]:
        cfg = YAHOO_FACTORS[name]
        try:
            tickers = cfg["tickers"]
            data = yf.download(tickers, start=DATA_START_DATE,
                               auto_adjust=True, progress=False)
            if not data.empty:
                close = data["Close"]
                if isinstance(close, pd.DataFrame) and len(close.columns) == 2:
                    ratio = close[tickers[0]] / close[tickers[1]]
                    result[name] = ratio.pct_change()
        except Exception:
            pass

    if not result.empty:
        result.index = pd.to_datetime(result.index)
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        result = result.sort_index()

    save_cache(result, "yahoo_factors")
    return result


def fetch_vix_levels(force_refresh: bool = False) -> pd.Series:
    """Hent VIX close-priser (nivå, ikke diff) for regime-klassifisering."""
    cached = None if force_refresh else load_cache("vix_levels")
    if cached is not None:
        return cached

    try:
        vix_data = yf.download("^VIX", start=DATA_START_DATE,
                               auto_adjust=True, progress=False)
        if vix_data.empty:
            return pd.Series(dtype=float)
        close = vix_data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = "VIX_LEVEL"
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        close = close.sort_index()
        save_cache(close, "vix_levels")
        return close
    except Exception:
        return pd.Series(dtype=float)


def get_all_factor_data(api_key: str, force_refresh: bool = False) -> pd.DataFrame:
    """Kombiner FRED- og Yahoo-faktorer til én aligned DataFrame."""
    cached = None if force_refresh else load_cache("all_factors")
    if cached is not None:
        return cached

    fred_df = fetch_fred_factors(api_key, force_refresh)
    yahoo_df = fetch_yahoo_factors(force_refresh)

    if fred_df.empty and yahoo_df.empty:
        return pd.DataFrame()

    if fred_df.empty:
        combined = yahoo_df
    elif yahoo_df.empty:
        combined = fred_df
    else:
        combined = fred_df.join(yahoo_df, how="inner")

    # Reindekser til business days og rens
    bdays = pd.bdate_range(start=combined.index.min(), end=combined.index.max())
    combined = combined.reindex(bdays)
    combined = combined.ffill(limit=5)
    combined = combined.dropna()

    save_cache(combined, "all_factors")
    return combined
