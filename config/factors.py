"""Faktordefinisjoner og datakilder for regresjonsmodellen."""

FRED_FACTORS = {
    "US_10Y": {
        "series": "DGS10",
        "desc": "10-års statsrente USA — langsiktig rentenivå, påvirker verdsettelse av vekstaksjer",
        "short": "10-års rente",
        "unit": "%",
        "transform": "diff",
    },
    "US_2Y": {
        "series": "DGS2",
        "desc": "2-års statsrente USA — reflekterer forventet pengepolitikk fra Fed",
        "short": "2-års rente",
        "unit": "%",
        "transform": "diff",
    },
    "YIELD_CURVE": {
        "series": "T10Y2Y",
        "desc": "Rentekurven (10Y minus 2Y) — bratt = vekstoptimisme, invertert = resesjonsvarsel",
        "short": "Rentekurve",
        "unit": "%",
        "transform": "diff",
    },
    "OIL_WTI": {
        "series": "DCOILWTICO",
        "desc": "Oljepris (WTI) — drivstoff for inflasjon og energikostnader",
        "short": "Oljepris",
        "unit": "%",
        "transform": "pct_change",
    },
    "HY_SPREAD": {
        "series": "BAMLH0A0HYM2",
        "desc": "Kredittspread (High Yield) — mål på markedets frykt for mislighold",
        "short": "Kredittspread",
        "unit": "%",
        "transform": "diff",
    },
    "USD_BROAD": {
        "series": "DTWEXBGS",
        "desc": "US-dollar styrke — sterk dollar rammer multinasjonale selskapers inntjening",
        "short": "USD-styrke",
        "unit": "%",
        "transform": "pct_change",
    },
}

YAHOO_FACTORS = {
    "VIX": {
        "ticker": "^VIX",
        "desc": "Volatilitetsindeks (VIX) — «fryktindeksen», stiger ved usikkerhet i markedet",
        "short": "VIX (frykt)",
        "unit": "poeng (%)",
        "transform": "diff",
    },
    "TECH_REL": {
        "tickers": ["QQQ", "SPY"],
        "desc": "Tech-momentum (QQQ vs SPY) — viser om teknologi leder eller sakker etter markedet",
        "short": "Tech vs marked",
        "unit": "%",
        "transform": "ratio_pct",
    },
    "CYCL_REL": {
        "tickers": ["XLI", "XLP"],
        "desc": "Syklisk vs defensiv (industri vs forbruk) — viser risikoappetitt i økonomien",
        "short": "Syklisk vs defensiv",
        "unit": "%",
        "transform": "ratio_pct",
    },
}

# Alle faktornavn i fast rekkefølge
FACTOR_NAMES = list(FRED_FACTORS.keys()) + list(YAHOO_FACTORS.keys())

# Beskrivelser samlet
FACTOR_DESCRIPTIONS = {}
FACTOR_SHORT_NAMES = {}
FACTOR_UNITS = {}
for k, v in FRED_FACTORS.items():
    FACTOR_DESCRIPTIONS[k] = v["desc"]
    FACTOR_SHORT_NAMES[k] = v["short"]
    FACTOR_UNITS[k] = v["unit"]
for k, v in YAHOO_FACTORS.items():
    FACTOR_DESCRIPTIONS[k] = v["desc"]
    FACTOR_SHORT_NAMES[k] = v["short"]
    FACTOR_UNITS[k] = v["unit"]

# Datahenting startdato
DATA_START_DATE = "1999-01-01"
