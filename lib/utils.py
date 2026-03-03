"""Hjelpefunksjoner for porteføljeanalyse."""

import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def ensure_cache_dir():
    """Opprett cache-mappe om den ikke finnes."""
    CACHE_DIR.mkdir(exist_ok=True)


def save_cache(data, name: str):
    """Lagre data som pickle med datostempel."""
    ensure_cache_dir()
    path = CACHE_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"data": data, "timestamp": datetime.now()}, f)


def load_cache(name: str, max_age_hours: int = 24):
    """Last pickle-cache. Returnerer None om den er for gammel eller mangler."""
    path = CACHE_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        age = datetime.now() - cached["timestamp"]
        if age > timedelta(hours=max_age_hours):
            return None
        return cached["data"]
    except Exception:
        return None


def cache_timestamp(name: str) -> str:
    """Returnerer tidspunkt for siste cache-oppdatering."""
    path = CACHE_DIR / f"{name}.pkl"
    if not path.exists():
        return "Ingen data cachet"
    try:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        return cached["timestamp"].strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "Ukjent"


def normalize_weights(portfolio: dict) -> dict:
    """Normaliser aksjevekter (ekskl. CASH) til å summere til 1.0."""
    stock_weights = {k: v for k, v in portfolio.items() if k != "CASH"}
    total = sum(stock_weights.values())
    return {k: v / total for k, v in stock_weights.items()}


def format_pct(value: float, decimals: int = 2) -> str:
    """Formater tall som prosent-streng."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


# Mapping fra yfinance-sektornavn til GICS-standard
_YF_TO_GICS = {
    "Technology": "Information Technology",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Healthcare": "Health Care",
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}


def resolve_gics_sectors(
    tickers: list[str],
    known_sectors: dict[str, str],
) -> tuple[dict[str, str], list[str], list[str]]:
    """Finn GICS-sektor for alle tickers. Bruker yfinance for ukjente.

    Args:
        tickers: Liste med ticker-symboler
        known_sectors: Eksisterende mapping ticker -> sektor

    Returns:
        (full_mapping, auto_resolved, failed)
        - full_mapping: komplett ticker->sektor dict
        - auto_resolved: tickers som ble hentet via yfinance
        - failed: tickers som ikke kunne klassifiseres
    """
    import yfinance as yf

    # Sjekk disk-cache for tidligere yfinance-oppslag
    cached = load_cache("gics_yfinance", max_age_hours=168) or {}  # 7 dager

    full = {}
    auto_resolved = []
    failed = []

    for t in tickers:
        if t == "CASH":
            continue
        if t in known_sectors:
            full[t] = known_sectors[t]
            continue
        # Prøv disk-cache
        if t in cached:
            full[t] = cached[t]
            auto_resolved.append(t)
            continue
        # Hent fra yfinance
        try:
            info = yf.Ticker(t).info
            raw = info.get("sector", "")
            gics = _YF_TO_GICS.get(raw, raw)
            if gics:
                full[t] = gics
                cached[t] = gics
                auto_resolved.append(t)
            else:
                failed.append(t)
        except Exception:
            failed.append(t)

    # Oppdater disk-cache
    if auto_resolved:
        save_cache(cached, "gics_yfinance")

    return full, auto_resolved, failed
