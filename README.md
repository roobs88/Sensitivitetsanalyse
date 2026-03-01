# Portefølje Scenarioanalyse

Kvantitativ scenarioanalyse av en aksjeportefølje ved hjelp av en multifaktor regresjonsmodell.

## Installasjon

```bash
pip install -r requirements.txt
```

## Oppsett

1. Skaff en gratis FRED API-nøkkel fra https://fred.stlouisfed.org/
2. Start appen:

```bash
streamlit run app.py
```

3. Skriv inn FRED API-nøkkelen i sidebar

## Metode

Appen bruker OLS-regresjon til å estimere hver aksjes sensitivitet mot 9 makroøkonomiske faktorer:

| Faktor | Kilde | Beskrivelse |
|--------|-------|-------------|
| US_10Y | FRED | 10-års statsrente |
| US_2Y | FRED | 2-års statsrente |
| YIELD_CURVE | FRED | 10Y-2Y rentekurve |
| OIL_WTI | FRED | WTI oljepris |
| HY_SPREAD | FRED | High Yield kredittspread |
| USD_BROAD | FRED | USD handelsveiet indeks |
| VIX | Yahoo | Volatilitetsindeks |
| TECH_REL | Yahoo | Tech relativ styrke (QQQ/SPY) |
| CYCL_REL | Yahoo | Syklisk vs defensiv (XLI/XLP) |

Scenarioene definerer kumulative faktorsjokk, og porteføljeeffekten beregnes som summen av vektede aksjeeffekter.

## Scenarioer

- Dyp resesjon (2008-type)
- Mild resesjon
- Rentesjokk opp (+1.50%)
- Aggressivt rentekutt (-2.00%)
- Oljesjokk opp (+50%)
- Oljekollaps (-40%)
- Tech mega-rally
- Syklisk rotasjon / value rally
- Stagflasjon
- Goldilocks / soft landing
- Multippelkompresjon / valuation reset
- USD-kollaps (NOK-risiko)
- Handelskrig / tariff-eskalering
- AI-desillusjonering / capex-bakrus
- Regulatorisk / antitrust-sjokk
- Higher for longer (vedvarende 3-4% inflasjon)
