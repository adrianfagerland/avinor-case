# Samidighetsmodell for AFIS – Avinor studentcase

Dette prosjektet er mitt forsøk på å løse Avinors studentkonkurranse: å forutsi sannsynligheten for at flere fly er aktive samtidig ved norske lufthavner i oktober 2025. Målet har vært å bygge en løsning som er både presis og robust, og som forstår de komplekse mønstrene som styrer flytrafikken.

Løsningen består i praksis av to hovedsteg. Først produserer jeg 120 Monte Carlo-simuleringer som genererer realistiske scenarioer for hvordan flytrafikken i oktober kan utfolde seg, minutt for minutt. Denne simuleringen tar høyde for historiske forsinkelser og kanselleringer for å skape et troverdig bilde av mulige utfall. Deretter blir innsikten fra disse simuleringene, sammen med en rekke andre datakilder, matet inn i en LightGBM-modell. Denne modellen veier all informasjonen og kommer med den endelige prediksjonen for hver time i perioden.

I løpet av prosjektet utforsket jeg også mer komplekse dyp lærings-modeller, spesifikt et Temporal Convolutional Network (TCN), men etter grundig validering viste den mer tradisjonelle LightGBM-tilnærmingen seg å være både mer treffsikker og mer stabil. Jeg testet også logistic regression, Catboost, XGBoost, men LightGBM/XGBoost ga best resultater.

## Eksterne datakilder

For å lage gode prediksjoner, må modellen forstå mer enn bare flyplanen. Derfor har jeg integrert flere eksterne datakilder for å gi den nødvendig kontekst. All innhenting av disse dataene er automatisert i skript som ligger i `scripts/`-mappen.

Været blir trukket fram som en viktig faktor i oppgavebeskrivelsen, og det er utvilsomt i praksis en kritisk faktor. Modellen henter derfor både historiske værdata og ferske varsler for oktober fra Meteorologisk institutts åpne API-er. Dette lar den lære hvordan spesifikke værforhold, som sterk vind eller mye nedbør, historisk har påvirket trafikken. For å knytte været til riktig sted, bruker jeg metadata fra det åpne OurAirports-datasettet til å finne presise koordinater for hver lufthavn, og koblet det mot værstasjonene som ligger nærmest.

I tillegg til været, påvirkes flytrafikken av menneskelige reisemønstre. Modellen tar hensyn til dette ved å inkludere en kalender over norske helligdager. Den er også designet for å kunne bruke data om skoleferier for å fange opp regionale reiseperioder, dersom slik data legges manuelt inn i prosjektet.

## Slik kjører du prosjektet

Det er en rett frem prosess å sette opp og kjøre hele pipelinen.

Først må du sette opp et isolert og reproduserbart Python-miljø ved hjelp av `uv`, som er verktøyet jeg har brukt for pakkehåndtering. Dette gjøres med en enkelt kommando:

```bash
uv sync
```

(Dette har allerede blitt gjort, dataen ligger her i repoet). Når miljøet er klart, må dataene på plass. Case-filene fra Avinor må kopieres inn i `data/`-mappen. Deretter kan du kjøre skriptene som henter de eksterne datakildene. For å hente historisk vær fra METs Frost-API, trenger du en personlig klient-ID.

```bash
# Hent metadata om lufthavner
uv run scripts/fetch_airport_metadata.py --mapping-path data/airportgroups.csv

# Hent værvarsel for oktober
uv run scripts/fetch_met_weather.py

# Hent historisk vær (krever Frost-ID)
FROST_CLIENT_ID=... uv run scripts/fetch_met_weather_history.py

# Generer kalender med helligdager
uv run scripts/generate_calendar_events.py
```

Med dataene på plass, kan selve modell-pipelinen kjøres. Dette er en to-stegs prosess hvor simuleringen kjøres først, etterfulgt av treningen av den endelige modellen.

```bash
# Steg 1: Kjør Monte Carlo-simuleringen for å generere fremtidsscenarioer
uv run scripts/run_simulation_and_ensemble.py --only-simulations

# Steg 2: Tren hovedmodellen og generer den endelige prediksjonen
uv run scripts/run_pipeline.py --log-level INFO
```

Resultatet skrives til `outputs/final_predictions.csv`, som er filen som skal leveres inn sammen med dokumentasjonen.

## Kreativ feature engineering

Caset ber om innovasjon og kreativ feature engineering i tillegg til eksterne datakilder. Derfor er det lagt vekt på operativt meningsfulle, forklarbare og effektive features.

Samtidighet beregnes minutt for minutt med en differanse‑array (“sweep line”) innenfor casets operasjonsvinduer (avgang −15/+8, landing −16/+5). Fra denne tidslinjen utledes p90, minutter med ≥2 og ≥3, antall samtidige par per time, near‑conflict‑minutter og minste mellomrom i minutter, som til sammen favner nivå, varighet og indikatorer på situasjoner tett på kapasitetsbrudd. I tillegg inngår vektoriserte tellinger for avganger/ankomster i ±30/60/90 minutter rundt timen, med forholdstall og 30‑minutters intensitet, for å fange bursts rett før og etter timegrensen.

Historiske basisrater per gruppe×time, måned×time og ukedag×time fungerer som datadrevne priors for forventet belastning når eksterne signaler er svake. Nær‑tidsdynamikk i målet modelleres med forrige time, rullerende middel og standardavvik (24/72/168/336 timer) og “hours since positive”, uten lekkasje av fremtidsinformasjon.

Monte Carlo‑simuleringen gir fordelingsmål (mean/std) for samtidighet, sannsynlighet for minst én samtidighet, areal under samtidighetskurven og variasjon, slik at både forventning og hale‑risiko vektlegges.

Kalender og syklikalitet inngår via norske helligdager (inkludert bevegelige), en enkel skoleferie‑heuristikk, uke‑ og dagposisjon og sykliske komponenter for time, ukedag og måned, samt sesongmapping. Værdata aggregeres og oversettes til binære indikatorer for høy vind, kraftig nedbør og lav skyhøyde, i tillegg til et vektet kompleksitetsmål og en enkel IFR‑proxy; symbolske værkoder klassifiseres videre til torden, snø og regn for tydeligere tolkning.

Lokasjon×tid‑profiler (“airport×hour/weekday”) og tidsdelt target‑encoding gir stabile, ikke‑lekkende profiler per kontekst.

## Viktigste features

Basert på LightGBM‑analysen (rapport i `outputs/analysis/lgbm_analysis.json`), er de viktigste signalene for modellen:

- Topp enkelt‑features (gain‑andel):
  - `feat_sched_concurrence` ≈ 59.4%
  - `sim_prob_any_overlap_mean` ≈ 7.2%
  - `sim_overlap_minutes_ge3_mean` ≈ 4.8%
  - `overlap_min_gap_minutes` ≈ 3.7%
  - `sim_overlap_mean_mean` ≈ 2.5%
  - `hist_rate_group_hour` ≈ 2.1%
  - `overlap_minutes_ge2` ≈ 1.2%
  - `sim_prob_any_overlap` ≈ 1.1%
  - `sim_concurrency_area_mean` ≈ 0.9%
  - `overlap_pairs_per_hour` ≈ 0.8%

- Viktigste feature‑familier (gain‑andel):
  - `schedule_counts` ≈ 61.4%
  - `simulation` ≈ 23.2%
  - `overlap` ≈ 6.4%
  - `history_encoding` ≈ 3.0%
  - øvrige familier (recent_target, weather, time_cyclical, m.fl.) står for resten.

Som vi kan se, og ikke overraskende, er den viktigste faktoren om det i følge planen er lagt opp til samtidig trafikk. Deretter kommer signaler fra simuleringene som fanger opp usikkerheten i planlagte tider, og hvorvidt fly faktisk overlapper i tid. Historiske mønstre og vær har også en rolle, men den er mindre fremtredende. Basert på denne analysen, kan det virke som vær ikke har en så stor rolle som tidligere planlagt. Det kan også være at MC-simuleringene ikke tar høyde for vær, og at vær har en mer indirekte effekt gjennom kanselleringer og forsinkelser.

## Antakelser og samsvar med case

- Operasjonsvinduer er hardkodet i koden i tråd med caset, med
  avgang `(-15, +8)` og landing `(-16, +5)`, se `src/avinor/feature_engineering.py:11`.
- Prosjektet er kjørt mot ruteplanen publisert 26. september 2025.
  En ny fil kan byttes inn og pipelinen kjøres på nytt.

## Struktur og reproduserbarhet

Hele prosjektet er strukturert med tanke på at resultatene skal være pålitelige og enkle å reprodusere. All kjernefunksjonalitet er samlet i `src/avinor/`, mens `scripts/` inneholder de kjørbare delene av flyten. Hver kjøring produserer logger og mellomlagrer resultater i `outputs/`, noe som gjør prosessen transparent. En grundig valideringsstrategi, basert på å splitte dataene i tid, sikrer at modellen testes på en måte som etterligner en ekte prediksjonssituasjon, uten at informasjon fra fremtiden lekker inn i treningen.

---

Takk for muligheten til å delta i konkurransen! Jeg stiller gjerne opp for en prat for å utdype noen av valgene som er tatt eller for å demonstrere løsningen.
