"""Porteføljedefinisjoner med tickers og vekter (prosent)."""

PORTFOLIO = {
    "MSFT":  6.37,
    "GOOG":  6.29,
    "AMZN":  5.29,
    "BRK-B": 4.07,
    "AAPL":  3.97,
    "SPGI":  3.89,
    "CPRT":  3.52,
    "OTIS":  3.37,
    "WM":    3.20,
    "TMO":   2.90,
    "TSM":   2.83,
    "APH":   2.74,
    "CNM":   2.67,
    "MA":    2.54,
    "CPAY":  2.53,
    "BKNG":  2.46,
    "AZO":   2.45,
    "CFR":   2.34,
    "PEP":   2.32,
    "HSY":   2.30,
    "V":     2.27,
    "ROP":   2.23,
    "MDT":   2.17,
    "GWW":   2.15,
    "EW":    2.09,
    "CDW":   2.06,
    "META":  1.97,
    "SITE":  1.90,
    "EXP":   1.80,
    "UNH":   1.76,
    "MMC":   1.61,
    "EME":   1.57,
    "SNPS":  1.51,
    "VEEV":  1.38,
    "CASH":  1.34,
    "VRSN":  1.27,
    "CSL":   1.24,
    "OTCM":  0.95,
    "FND":   0.69,
}

# Tickers som faktisk handles (ekskl. CASH)
STOCK_TICKERS = [t for t in PORTFOLIO if t != "CASH"]

# Benchmark
BENCHMARK_TICKER = "SPY"
