# Magasinvann som realopsjon — verdsetting av vannkraftproduksjon i NO2

**Håvard Rørtveit** · masterstudent i finans · mai 2026
GitHub: [haavardroertveit/magasinopsjon](https://github.com/haavardroertveit/magasinopsjon)

## Sammendrag

Et vannmagasin verdsettes som en serie amerikanske realopsjoner på fremtidig
produksjon. En Schwartz 1-faktor-prismodell kalibreres på NO2 dagspot
(2024–2026), og opsjonsverdsettingen løses ved bakoverinduksjon på et
$(S, X)$-grid med daglige beslutninger over 12 måneder. For et illustrativt
anlegg på 100 MW med 120 GWh magasin gir modellen en 12-måneders
opsjonsverdi på **32,8 MEUR** ved halvfull magasinfylling. **Vannverdien
$\partial V/\partial S$** varierer over et bredt område — typisk 40–95 EUR/MWh
— avtakende i fyllingsgrad og høyest i vintermånedene. En simulert
optimal kjørestrategi produserer på 146 av 365 dager med gjennomsnittlig
spotpris **+46,5 EUR/MWh** høyere på produksjonsdager enn på hold-dager.
Sensitivitetsanalysen viser at modellen er rimelig robust for både σ og κ
innenfor ±50 % rundt baseline, noe som er gunstig: parametrene som er
vanskeligst å estimere presist er også de minst kritiske for beslutningene.

## 1. Problemstilling

Et magasin er funksjonelt en serie amerikanske opsjoner: hver dag kan
eieren velge mellom å produsere (motta $Q \cdot \mathrm{spot}_t$) eller
holde (utsette beslutningen til prisen eventuelt blir bedre). Verdien
ligger ikke bare i forventet produksjon, men i **fleksibiliteten** —
retten til å vente på topp-priser uten å være forpliktet. Klassisk
neddiskontert kontantstrøm med forventet pris undervurderer denne
fleksibiliteten systematisk. Realopsjons-rammeverket
(Schwartz 1997, Tseng & Barz 2002) gir et veldefinert språk for å verdsette den.

For en kraftforvalter med produksjonsansvar er imidlertid den interessante
størrelsen ikke verdien av magasinet i seg selv, men den marginale
**vannverdien**

$$V_\partial(S, X, t) = \frac{\partial V}{\partial S},$$

— verdien av én ekstra MWh i magasinet i tilstand $(S, X, t)$. Den brukes
direkte som beslutningskriterium: *produser dag $t$ dersom
$\mathrm{spot}_t > V_\partial(S_t, X_t, t)$; hold ellers*. Dette prosjektet
beregner $V_\partial(S, t)$ for et illustrativt anlegg i NO2 og er eksplisitt
om hva som er forenklet bort. Resultatene er ikke ment som et pris-anslag for
et reelt anlegg, men som en demonstrasjon av at metoden er konsistent og kan
videreutvikles.

## 2. Data

Prosjektet bygger på to datakilder:

- **ENTSO-E Transparency Platform**: daglig snittpris (dagspot) for NO2,
  2020–2026, aggregert fra time- og 15-minutters MTU. 2 339 daglige
  observasjoner uten hull. Norsk dagsmarked gikk over fra time- til
  kvarter-MTU midt i 2025; daglig-aggregering håndterer overgangen
  transparent.
- **NVE Magasinstatistikk** for NO2 (ukentlig fyllingsgrad). Brukes som
  kontekst, ikke som modellinput i v1. Korrelasjonen mellom log-pris og
  fyllingsgrad er **−0,50** for overlappende perioder, som bekrefter den
  hydrologiske historien om priser i et hydro-dominert område.

Krise-perioden 2021–2023 dominerer datasettet: årssnittpris går fra
9 EUR/MWh (2020) til 211 EUR/MWh (2022). Det får konsekvenser for valget av
kalibreringsperiode (avsnitt 3).

Detaljer og figurer i [`notebooks/01_data_exploration.ipynb`](../notebooks/01_data_exploration.ipynb).

## 3. Prismodell: Schwartz 1-faktor

Log-pris dekomponeres i deterministisk sesong + mean-reverting støy:

$$\log P_t = f(t) + X_t, \qquad f(t) = \beta_0 + \beta_1 \sin(\omega t) + \beta_2 \cos(\omega t), \qquad \omega = \tfrac{2\pi}{365{,}25}$$

$$dX_t = -\kappa X_t\, dt + \sigma\, dW_t.$$

Kalibreres i tre OLS-steg: (i) $\beta$ fra log-pris på sesong-regressorer,
(ii) $\varphi$ fra AR(1) på residual uten intercept, (iii) $\kappa = -\ln\varphi/\Delta t$
og $\sigma$ fra eksakt OU-varians.

**To kalibreringer ble sammenlignet.** Full sample 2020–2026 gir
$R^2 = 2{,}2\,\%$ på sesongen og halveringstid 15,4 dager — modellen
"drukner" i krise-perioden, og residual-ACF avviker tydelig fra teoretisk
$\varphi^k$-decay. Post-krise-utvalget 2024–2026 gir $R^2 = 11{,}3\,\%$,
halveringstid 1,8 dager, og en ACF som matcher AR(1)-strukturen langt bedre.

Vi valgte post-krise-modellen som operasjonell, med eksplisitt begrunnelse:
*en kraftforvalter som tar beslutninger i 2026 baserer ikke sin
normalpris-forventning på et regime der gassen var notert til 300 EUR/MWh*.
Den motsatte posisjonen — at krise-perioden lærte oss om hva som er mulig
og bør vektes inn — har sine poenger og bør diskuteres med oppdragsgiver i
en faktisk beslutningssetting.

| Parameter | Post-krise 2024–2026 | Full 2020–2026 |
|---|---:|---:|
| $\beta_0$ (intercept) | 4,037 | 3,924 |
| $\varphi$ (AR(1)) | 0,685 | 0,956 |
| $\kappa$ (per dag) | 0,378 | 0,045 |
| $\sigma$ (per √dag) | 0,411 | 0,354 |
| Halveringstid | 1,8 d | 15,4 d |
| Sesong-$R^2$ | 11,3 % | 2,2 % |
| $E[P]$ stasjonært | 63,4 EUR/MWh | 101,3 EUR/MWh |

Q-Q-plot bekrefter at AR(1)-innovasjonen er rimelig — men ikke perfekt —
normalfordelt; eksess-kurtose > 0 og tunge haler er kjente svakheter ved
1-faktor-modellen.

Detaljer i [`notebooks/02_price_model_calibration.ipynb`](../notebooks/02_price_model_calibration.ipynb).

## 4. Opsjonsmodell og hovedresultat

Bellman-ligningen for hver tilstand $(S, X)$ på dag $t$:

$$V(t, S, X) = \max \left\{ Q \cdot P_t(X) + e^{-r\Delta t} \mathbb{E}[V(t{+}1, S{-}Q{+}I, X')], \;\; e^{-r\Delta t} \mathbb{E}[V(t{+}1, S{+}I, X')] \right\}$$

med terminal-betingelse $V(T, \cdot, \cdot) = 0$. Forventningen tas over
$X' | X \sim N(\varphi X, \sigma_\varepsilon^2)$, diskretisert med
midtpunkt-bins på et 41-noders grid for $X$ og 121-noders grid for $S$.
Lineær interpolasjon langs $S$ håndterer at handlinger kan lande mellom
grid-punkter. Full løsning på ~0,1 sekund.

**Referanse-anlegg:**
- Effekt: 100 MW (daglig produksjon $Q = 2\,400$ MWh)
- Magasinkapasitet $K$: 120 GWh
- Tilsig $I$: 800 MWh/dag (~292 GWh/år)
- Diskontering $r$: 0 (forenkling, v1)
- Horisont: 365 dager fra 1. januar

**12-måneders opsjonsverdi.** $V(t=0, S=K/2, X=0) = \mathbf{32{,}8\ \text{MEUR}}$.
Differansen mellom et fullt og et tomt magasin på dag 1 (henholdsvis 36,8 og
27,7 MEUR) er ca. **9 MEUR** — det er produksjons-optionaliteten av å ha
120 GWh tilgjengelig fremfor å vente på tilsiget.

**Vannverdi $\partial V/\partial S$ som funksjon av fyllingsgrad (X = 0):**

| Måned | $S = 10\,\%$ | $S = 50\,\%$ | $S = 90\,\%$ |
|---|---:|---:|---:|
| Januar | 95 | 73 | 63 |
| April | 82 | 61 | 54 |
| Juli | 66 | 55 | 47 |
| Oktober | 71 | 53 | 41 |

Tallene har riktige fortegn og rimelige nivåer: avtakende i fyllingsgrad
(knapphet), høyere om vinteren (sesong + lite tid igjen før horisontens
slutt), lavere om sommeren. Hovedplottet og en heatmap-visning over hele
$(S, t)$-flaten ligger i notebook 03.

**Optimal kjørestrategi** (én simulert bane med fast seed, halvfullt start):

- 146 produksjonsdager av 365 (≈ 40 %)
- Total inntekt: 31,7 MEUR
- Gjennomsnittlig spotpris på produksjonsdager: **90,5 EUR/MWh**
- Gjennomsnittlig spotpris på hold-dager: **44,0 EUR/MWh**
- Differanse: **+46,5 EUR/MWh** — modellen produserer på dramatisk
  høyere-pris dager enn den holder

Detaljer i [`notebooks/03_option_valuation.ipynb`](../notebooks/03_option_valuation.ipynb).

## 5. Sensitivitet

Vi varierer $\sigma$ og $\kappa$ med ±50 % rundt baseline, holder alt annet
konstant, og rapporterer årsgjennomsnittlig vannverdi ved $S = 50\,\% K$:

| Variant | Snitt-$V_\partial$ (EUR/MWh) | Endring |
|---|---:|---:|
| Baseline | 51,9 | — |
| +50 % σ | 53,8 | +3,7 % |
| −50 % σ | 51,0 | −1,8 % |
| +50 % κ | 50,9 | −1,9 % |
| −50 % κ | 56,0 | +7,9 % |

Volatiliteten har den ventede positive effekten (mer volatilitet → mer å
vente på), men beskjeden i størrelse. Mean-reversion-hastigheten har en
asymmetrisk effekt der **lavere κ** (mer persistente prisavvik) gir merkbart
høyere vannverdi — også konsistent med intuisjonen om at vedvarende
høy-pris-perioder gir mer å vente på.

Begge parametrene flytter vannverdien med < 10 % innenfor ±50 % variasjoner.
Det er positivt for praktisk anvendelse: vanskelig-estimerbare parametre
($\kappa$ særlig) er ikke beslutningskritiske i denne størrelsesordenen.

## 6. Begrensninger og naturlige utvidelser

Modellen er bevisst forenklet for å fokusere på kjernemekanikken. De
viktigste utvidelsene, prioritert etter forventet impact:

1. **Stokastisk tilsig.** Tilsiget i v1 er deterministisk og konstant. I
   virkeligheten varierer det med 1–2 størrelsesordener over året
   (snøsmelting, høstflom). Korrelasjonen vi observerte mellom magasinfylling
   og log-pris (−0,50) indikerer at koblingen mellom hydrologi og pris er
   sterk og ikke kan ignoreres for nøyaktig verdsetting. En v2-modell bør
   modellere $I_t$ som en stokastisk prosess, eventuelt korrelert med
   pris-residualen.

2. **Kalibrering på terminkurver.** Vi bruker historisk spot, som gir
   parametre under det subjektive målet $\mathbb{P}$. Risk-nøytrale parametre
   fra forward-kurver vil gi en finansielt mer korrekt verdsetting, og
   samtidig automatisk reflektere markedets pris på volatilitet og risiko.
   Strukturelt enkelt å koble på.

3. **Salvage-verdi ved horisont.** $V(T, S, X) = 0$ gjør at den optimale
   strategien presser ut vann mot slutten (sim. ender med $S \approx 1\,600$
   MWh fra start på 60 000 MWh). En rullerende uendelig-horisont-formulering,
   eller en eksplisitt salvage-verdi basert på forventet vannverdi i et nytt
   syklus, eliminerer denne biasen.

4. **Reservemarkeder (FCR-N, mFRR).** Disse markedene betaler for
   *tilgjengelighet*, ikke produksjon — og premier dermed nettopp den
   fleksibiliteten et magasin gir. De gir en ekstra inntektskilde, særlig i
   timer med lav spot, og bør med i en kommersiell verdsetting.

5. **2-faktor- eller jump-prosess.** Q-Q-plot avslører tunge haler som
   1-faktor-modellen underestimerer. Geman & Roncoroni (2006) er
   standardreferansen for jump-utvidelse. En 2-faktor-modell skiller
   kortsiktig avvik fra langsiktig nivå og er mer naturlig for lange
   horisonter.

6. **Ramping- og produksjonsbegrensninger.** Min-stopp, min-kjør,
   ramping-rater. Ikke-bindende for daglige beslutninger på et stort anlegg,
   men reelle for kortere tidshorisonter (intra-dag, regulering).

7. **Kaskademagasiner.** Anlegg som Sauda har flere magasiner i kaskade —
   vann fra et øvre magasin produserer i flere stasjoner nedover. Optimal
   løsning blir multi-dimensjonal og kompleks; LSM
   (Longstaff–Schwartz 2001) med flere tilstandsvariable er den vanlige
   tilnærmingen.

## Konklusjon

Det er konsistent å verdsette et magasin som en amerikansk realopsjon, og en
1-faktor Schwartz-modell er tilstrekkelig til å gi vannverdier i riktig
størrelsesorden og med riktig kvalitativ oppførsel. Den minst observerbare
parameteren ($\kappa$) er også blant de minst kritiske for beslutningene,
mens valget av kalibreringsperiode (post-krise vs. full historikk) har
*større* effekt enn ±50 % i de stokastiske parametrene — et viktig poeng for
en faktisk operasjonell modell. Modellen i v1 fanger fleksibilitetspremien
på flow-nivå, men hverken hydrologisk risiko, terminmarkedets risikopreferanser,
eller bidragene fra balansemarkeder. Hver av disse er en konkret v2-vei.

Kode, notebooks og rådata på [github.com/haavardroertveit/magasinopsjon](https://github.com/haavardroertveit/magasinopsjon).

---

**Referanser**

- Schwartz, E. (1997). *The Stochastic Behavior of Commodity Prices:
  Implications for Valuation and Hedging.* Journal of Finance 52(3).
- Longstaff, F. & Schwartz, E. (2001). *Valuing American Options by
  Simulation: A Simple Least-Squares Approach.* Review of Financial Studies 14(1).
- Tseng, C.-L. & Barz, G. (2002). *Short-Term Generation Asset Valuation:
  A Real Options Approach.* Operations Research 50(2).
- Geman, H. & Roncoroni, A. (2006). *Understanding the Fine Structure of
  Electricity Prices.* Journal of Business 79(3).
