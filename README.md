# Magasinvann som realopsjon

Verdsetting av vannkraftproduksjon i prisområde **NO2** ved hjelp av en
stokastisk prismodell og dynamisk programmering. Prosjektet behandler magasinet
som en amerikansk realopsjon: retten — men ikke plikten — til å produsere kraft
når spotprisen er høy nok til å forsvare alternativkostnaden av vann.

> Status: under utvikling. Tallresultater og figurer legges inn etterhvert som
> de tre notebookene fylles ut.

## Hva prosjektet viser

- Hvordan en mean-reverting prismodell (Schwartz 1-faktor) kalibreres på
  historiske NO2-spotpriser, med en deterministisk sesongkomponent som
  skiller signal fra støy.
- Hvordan **vannverdien V(S, t)** — verdien av én ekstra MWh i magasinet på
  tidspunkt *t* og fyllingsgrad *S* — løses ut med bakoverinduksjon over et
  grid.
- Sesongmønsteret i vannverdien, og hvordan den responderer på endringer i
  volatilitet og styrken på mean-reversion.

Modellen er bevisst forenklet (ett magasin, fast effekt, deterministisk tilsig,
ingen ramping- eller balansemarkeder). Hensikten er å demonstrere metodisk
forståelse, ikke å bygge et produksjonssystem. Hva som er tatt ut og hvorfor
diskuteres i rapporten.

## Metode i ett avsnitt

Log-spot dekomponeres i en deterministisk sesongkomponent (sinus/cosinus med
ettårsperiode) og en stokastisk Ornstein–Uhlenbeck-residual. Parameterne κ
(reversjonshastighet) og σ (volatilitet) estimeres som AR(1) på residualen.
Optimal kjørestrategi over en horisont på 12 måneder løses med bakoverinduksjon
på et grid over magasinnivå, med daglige beslutninger om å produsere full effekt
eller holde vannet. Vannverdien leses ut som den marginale endringen i
forventet nåverdi ved én ekstra MWh i magasinet — ∂V/∂S.

## Struktur

```
magasinopsjon/
├── data/
│   ├── raw/                  # Rådata fra Nord Pool / NVE (ikke i git)
│   └── processed/            # Renset data klar til modellering (parquet)
├── src/
│   ├── data_loader.py        # Innlasting og rensing av prisdata
│   ├── price_model.py        # Schwartz 1-faktor + kalibrering
│   ├── option_value.py       # Bakoverinduksjon for V(S, t)
│   └── plotting.py           # Felles plottestil
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_price_model_calibration.ipynb
│   └── 03_option_valuation.ipynb
└── report/
    └── summary.md            # 2–3 siders skriftlig oppsummering
```

## Komme i gang

Forutsetter [uv](https://docs.astral.sh/uv/) og Python 3.11+.

```bash
uv sync                # oppretter .venv og installerer avhengigheter
uv run jupyter lab     # starter Jupyter
```

## Data

| Kilde | Hva | Frekvens | Bruk |
|-------|-----|----------|------|
| [Nord Pool](https://www.nordpoolgroup.com/) | Spotpris NO2 (EUR/MWh) | Time → daglig snitt | Modellinput |
| [NVE](https://www.nve.no/) | Magasinstatistikk Vestlandet | Ukentlig (% kapasitet) | Kontekst |

Last ned CSV manuelt og legg filene i `data/raw/`. Forventet kolonneformat
dokumenteres i `src/data_loader.py`.

## Begrensninger og naturlige utvidelser

V1 ignorerer eksplisitt:

- Stokastisk tilsig (modellert som konstant)
- Kaskademagasiner og hydrauliske avhengigheter
- Ramping- og produksjonsbegrensninger, min-stopp
- Deltakelse i FCR-N / mFRR og andre reservemarkeder
- Mer sofistikerte prismodeller (2-faktor, jumps, regimeskift)
- Kalibrering mot terminkurver i stedet for kun historisk spot

Hvert punkt drøftes i `report/summary.md` som mulige v2-utvidelser.

## Forfatter

[Håvard Rørtveit](https://github.com/haavardroertveit) — masterstudent i
finans. Prosjektet er bygget som faglig forberedelse til arbeid med
kraftforvaltning og produksjonsplanlegging.

## Referanser

- Schwartz, E. (1997). *The Stochastic Behavior of Commodity Prices:
  Implications for Valuation and Hedging.* Journal of Finance, 52(3).
- Longstaff, F. & Schwartz, E. (2001). *Valuing American Options by Simulation:
  A Simple Least-Squares Approach.* Review of Financial Studies, 14(1).
- Tseng, C.-L. & Barz, G. (2002). *Short-Term Generation Asset Valuation: A
  Real Options Approach.* Operations Research, 50(2).
