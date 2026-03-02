"""Historiske kriseperioder og sektor-ETF proxy-mapping."""

# Historiske perioder (kriser + rallyer)
CRISIS_PERIODS = {
    "Dotcom-krasjet": ("2000-03-24", "2002-10-09"),
    "Finanskrisen": ("2007-10-09", "2009-03-09"),
    "COVID-krasjet": ("2020-02-19", "2020-03-23"),
    "Bear-markedet 2022": ("2022-01-03", "2022-10-12"),
    "AI/Tech-rally 2023–2025": ("2023-01-01", "2025-02-19"),
    "Tollkaos 2025": ("2025-01-01", "2025-04-09"),
}

# Sektor-ETF proxy for aksjer som mangler data i eldre perioder
SECTOR_PROXY = {
    "MSFT": "XLK", "GOOG": "XLK", "AMZN": "XLY", "AAPL": "XLK",
    "META": "XLK", "SNPS": "XLK", "VEEV": "XLK", "CDW": "XLK", "VRSN": "XLK",
    "BRK-B": "XLF", "SPGI": "XLF", "MA": "XLF", "V": "XLF",
    "CPAY": "XLK", "MMC": "XLF", "CFR": "XLF",
    "TMO": "XLV", "UNH": "XLV", "MDT": "XLV", "EW": "XLV",
    "OTIS": "XLI", "WM": "XLI", "APH": "XLI", "CPRT": "XLI",
    "GWW": "XLI", "EME": "XLI", "CSL": "XLI", "ROP": "XLI",
    "CNM": "XLI", "SITE": "XLI",
    "PEP": "XLP", "HSY": "XLP", "AZO": "XLY", "BKNG": "XLY", "FND": "XLY",
    "TSM": "XLK", "EXP": "XLB", "OTCM": "XLF",
}
