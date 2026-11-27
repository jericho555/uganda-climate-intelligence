# Uganda Climate Intelligence Framework (UCIF)

Reproducible research package behind the manuscript:

> **A National Climate Anomaly Intelligence System for Uganda: Ground-Truth
> Validation, Risk Nowcasting, and Long-Term Climate Change Assessment**
> Jeremiah Masaba — Department of Computer Science, Makerere University, Kampala, Uganda
> ([ORCID 0009-0004-5620-996X](https://orcid.org/0009-0004-5620-996X))

This package converts a single-source algorithm comparison into a validated
detection + spatial-risk + century-climatology + drought-index + impact +
severity + nowcasting + forward-looking operational layers system with an
interactive decision-support dashboard. The manuscript documents dedicated
methods and results for four operational layers: threshold-based drought early
warning, national cereal-yield forecasting, SEAS5 seasonal outlook integration,
and an automated climate risk bulletin.

**Core, deliberately uncomfortable finding:** against real disaster ground-truth,
*no cross-sectional* unsupervised anomaly detector ranks above chance (26-detector
benchmark; non-sequence held-out ROC-AUC 0.491–0.532; best cross-sectional F1 =
0.084; PR-AUC ≤ 0.121 vs a 0.1115 base rate).
Scored against a measured size-matched random alert budget (F1 = 0.069 [0.059, 0.079],
`results/random_null.json`), six of the 24 non-sequence detectors remain above it
after Benjamini–Hochberg control but by at most 1.22×
(one-class SVM, LOF, ECOD, COPOD, PCA, the rank-averaged ensemble)
while only two deep **sequence** autoencoders (LSTM-AE, Transformer-AE — torch,
vendored CPU build) roughly double it (F1 0.13–0.14, precision 0.20–0.22, yet
ROC-AUC still 0.51–0.53 and <10% of disaster days captured). The two detectors added
at revision — LoOP (local density) and a 14-day lag-window Temporal Isolation Forest
— join the chance-level group rather than escaping it (LoOP ROC 0.491, q = 0.57;
TemporalIF ROC 0.519, q = 0.26). The apparent 68.8%
event coverage comes from the transductive operational surface and is reproduced by a
size-matched random null at 100%.
Physically-interpreted derived indices (SPI/SPEI), by contrast, *do* corroborate
documented droughts. Every number in the paper is computed by this pipeline —
**nothing is fabricated**.

---

## Data (openly licensed, reproducible)

| Source | What | Licence |
|---|---|---|
| **ERA5** via the Open-Meteo Archive API | daily 13-variable reanalysis for 19 Ugandan district centroids, 2008–2023 (93,504 district-days) | CC-BY 4.0 |
| **Meteostat** bulk | century-scale in-situ daily record, 12 stations, 1926–2026 (151,369 station-days) + ERA5 cross-validation | CC-BY 4.0 |
| **NASA POWER** (MERRA-2) | independent gap-fill of the 1990–2007 in-situ reporting collapse: daily T2M / RH / shortwave for 19 districts, 1988–2007 (138,795 district-days) | open (no-key JSON) |
| **World Bank Open Data API** (FAOSTAT-derived) | 64-year cereal production / area / yield (1961–2024) impact layer, plus the climate indicator series (`EN.CLC.SPEI.XD`, `AG.LND.PRCP.MM`, exposure/coverage indicators) used for the external cross-check in `results/wb_climate_check.json`. The cross-check is **negative** (r = 0.057, p = 0.83) and reported as such; the API returns indicator ids with empty topic/source metadata, and the portal web pages themselves answer 403 | CC BY 4.0 |
| **Our World in Data** (FAOSTAT-derived) | green-coffee production, 1961–2024 — the second, export-crop impact layer | CC BY 4.0 |
| **Curated disaster register** | 32 documented hydro-meteorological disasters (UNMA, OCHA/ReliefWeb, EM-DAT, FAO/GIEWS, FEWS NET) with onset windows + affected regions | editorial records, cited |
| **OCHA Uganda COD-AB** via HDX (`cod-ab-uga`) | 135 district polygons (simplified to a 217 KB subset in `data/`) that turn every map into a choropleth; `data/district_lookup.csv` joins the 19 sites to their districts | CC BY-IGO |

> **Note on Meteoblue:** the original data source was Meteoblue, whose bulk
> export for non-European locations is a paid `history+` product and is
> bot-gated; we rebuild the same variables from ERA5 so every figure is
> re-downloadable by a reviewer.

---

## Layout

```
uganda-climate-intelligence/
  src/        pipeline modules (see run_all.py for order)
  data/       fetched + intermediate CSVs (large files gitignored)
  results/    metrics, statistical tests, UCASI tables, JSON summaries (committed)
  figures/    publication figures (PNG)
  manuscript/ Elsevier elsarticle LaTeX built around the real results
  dashboard/  Streamlit interactive view of the persisted artefacts
  run_all.py  one-command reproduction
  requirements.txt
  CITATION.cff / .zenodo.json / LICENSE / CHANGELOG.md
```

## Figures

Every figure is written by exactly one function, `config.save_figure()`, which
fixes 300 dpi and drops an identical copy in both `figures/` (repository view)
and `manuscript/figures/` (what LaTeX includes). No script sets its own dpi or
writes to a single folder, so the compiled paper and the browsable figure set
cannot disagree.

## Release checks

The pipeline ends by auditing its own deliverables:

```bash
python src/verify_release.py            # add --skip-dashboard if Streamlit is absent
```

Seventy-four assertions cover the figure mirror (same PNG set, byte-identical, ≥300 dpi read
from the file header), the LaTeX source (balanced environments, resolvable
cross-references, present `\includegraphics` targets, a .bib entry for every
citation, no stray unicode dashes), **the numbers quoted in the text against the
artefacts that produced them** (detector count, ROC/PR-AUC spans, the Friedman
statistic, every conformal width, coverage and per-band coverage, the World Bank
correlation and its dry-year claim), the artefact arithmetic behind the paper's
claims (SHAP additivity residual, attribution rows vs the stored sample, the
nowcast carry envelope, and the 16 scored + 3 site-without-score + 116 unobserved
district partition), the **forward-looking early-warning artefacts** (drought EW
+ yield forecast + seasonal outlook tables, summaries and figures, plus the
automated risk bulletin), and an in-process render of all fourteen dashboard
tabs — including a check that the dashboard carries no hand-typed count the
artefacts have superseded.
A failure exits non-zero, so `run_all.py` stops rather than reporting a broken bundle
as done.

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py
```

## Interactive dashboard

A read-only Streamlit view, in **fourteen panels**, of the benchmark, daily anomaly
stream, district choropleths, UCASI severity, trends, the drought/agriculture
impact layer, the risk nowcast, the global *and* local detection attributions,
**drought early warning**, **yield impact forecast** and **seasonal climate
outlook** — bound entirely to the artefacts under `results/` and `data/`
(nothing is re-fitted on the page).

```bash
# one-time vendored install (works when system site-packages is not writable):
python -m pip install --target=._webpkgs --no-cache-dir streamlit
python dashboard/run_dashboard.py      # -> http://localhost:8501
```

See [`dashboard/README.md`](dashboard/README.md) for the tab-by-tab mapping to
artefacts. The dashboard is optional; it is not needed to reproduce the paper.

Stages run in the order declared in `run_all.py`: `fetch_era5` →
`fetch_meteostat` → `fetch_agriculture` → `fetch_coffee` → `fetch_power` →
`fetch_wb_climate` → `fetch_boundaries` → `disaster_register` → `preprocess` →
`detect` → `region_label_check` → `_recover` → `random_null` →
`spatial` → `temporal` → `meteostat_instrumental` → `power_climate` →
`drought_indices` → `wb_climate_check` → `ag_linkage` → `coffee_linkage` → `ucasi` →
`early_warning` → `_typematch` → `_explain_fast` → `figures` → `figures_power` →
`figures_maps` → `figures_coffee` → `figures_early_warning` → `figures_wb` →
`fetch_seasonal` → `drought_early_warning` → `yield_forecast` →
`seasonal_outlook` → `figures_drought_ew` → `figures_yield` →
`figures_seasonal` → `bulletin` →
`collect_results` → `check_ref_index` → `verify_release`. Each stage caches to
disk and is idempotent; re-running skips cached fetches. `fetch_boundaries`
rebuilds `data/district_boundaries.geojson` from the public HDX endpoint (a
37 MB archive; only the 217 KB simplified subset is committed) and
`check_ref_index` re-runs the reference searches against Crossref, OpenAlex
and arXiv; both refuse to report success on a partial answer rather than
writing a convenient negative result.

## What the pipeline computes

* **RQ1** held-out benchmark of 26 detectors vs disaster ground-truth (P/R/F1/
  ROC-AUC/PR-AUC/MCC) with bootstrap CIs and a size-matched random null.
* **RQ2** district anomaly-rate hotspots via Global Moran's I and Getis–Ord Gi*.
* **RQ3** SHAP + permutation importance driving the detection signal, plus the
  persisted per-district-day attributions (`results/shap_values.npy`,
  `results/shap_expected_value.json`) whose additivity error is reported.
* **RQ4** Friedman + per-year F1 ranking, seed stability, STL / Mann–Kendall /
  Sen / change-point climatology.
* **Century in-situ** (`meteostat_instrumental.py`): 91-year trends + ERA5↔station
  cross-validation (r = 0.95–0.999 co-located).
* **Independent gap-fill** (`fetch_power.py`, `power_climate.py`): NASA POWER
  closes 1990–2007; 19/19 districts warm and 15/19 dry significantly.
* **Drought indices** (`drought_indices.py`): SPI/SPEI independently confirm 3 of 4
  register droughts (2018 min SPEI₃ = −4.75) where raw daily detectors fail.
* **Impact linkage** (`ag_linkage.py`): severe drought years are the worst yield
  years, but the national-annual linear link is insignificant (regional droughts
  masked by aggregation).
* **Coffee linkage** (`coffee_linkage.py`): detrended production, not yield, so the
  weaker of the two series; the century in-situ and modern ERA5 substrates give
  coefficients of *opposite sign*, which is reported as a null rather than
  reconciled (`results/coffee_linkage.json`).
* **UCASI** severity index ranks documented disasters; **RQ5** event recall.
* **RQ6 nowcast** (`early_warning.py`): district-month risk persistence + a ridge
  carry model, with **split-conformal prediction intervals** (`prediction_intervals`
  in `results/early_warning_summary.json`, `results/tables/nowcast_interval.csv`).
  Coverage is reported as measured, overall and per starting severity band, and the
  released verdict says explicitly that the better-fitting model's band covers less
  than the persistence band.
* **External index cross-check** (`fetch_wb_climate.py`, `wb_climate_check.py`):
  the World Bank's national SPEI and precipitation-normal series compared with ours —
  uncorrelated (r = 0.057, p = 0.83), wet in every register drought year, and on a
  different precipitation baseline (1,589 mm ERA5 district mean vs a constant
  1,180 mm Bank normal). Reported as a provenance check that does **not** corroborate
  the index (`results/wb_climate_check.json`).
* **Drought Early Warning** (`drought_early_warning.py`): threshold-based district
  status (Normal / Watch / Warning / Alert) from SPI-6 severity, trajectory and
  consecutive-decline runs. Validated against the disaster register: 34 hits,
  0 misses, 100% hit rate across 3,072 district-months.
* **Yield Impact Forecast** (`yield_forecast.py`): national cereal-yield anomaly
  regressed against growing-season SPI-3. Honest skill check: R² = 0.076,
  climatology beats the model (ratio = 1.308). The negative result is reported
  transparently.
* **Seasonal Climate Outlook** (`fetch_seasonal.py`, `seasonal_outlook.py`):
  downscaled ECMWF SEAS5 precipitation forecast via Open-Meteo. Graceful
  degradation when the API is unreachable.
* **Automated Risk Bulletin** (`bulletin.py`): narrative Markdown report
  synthesising all forward-looking layers. Every number read from summary JSONs.
  Output: `reports/climate_risk_bulletin_YYYY_MM.md`.
* **Geography** (`fetch_boundaries.py`, `region_label_check.py`): the 19 sites join
  to 19 of 135 OCHA districts (13.7% of national district area), and the one
  region label that disagrees with the boundary register is bounded rather than
  hidden (`results/region_label_sensitivity.json`).

---

## Data and licensing

The **source code** is released under the **MIT License** (see `LICENSE`).
Third-party **data remain under their own terms** (ERA5/Open-Meteo CC-BY 4.0;
Meteostat CC-BY 4.0; NASA POWER open; World Bank/FAOSTAT CC BY 4.0; administrative
boundaries OCHA COD CC BY-IGO, attribution recorded in
`data/boundaries_provenance.json`). Large raw
and derived data files are gitignored because they are regenerated by
`run_all.py`; the small curated register and all result tables/JSONs (the paper's
actual numbers) are committed so a reviewer can inspect them without running
anything. Please do not redistribute raw third-party data beyond what their
licences permit.

---

## Citing & archiving (GitHub + Zenodo)

This repository ships `CITATION.cff` and `.zenodo.json` so a **citable Zenodo DOI
is one release away**. A DOI is *minted by Zenodo only after a real GitHub
release* — none is hard-coded here on purpose.

1. Create a GitHub repo and push:
   ```bash
   git init && git add . && git commit -m "UCIF v1.0.0" && git branch -M main
   git remote add origin https://github.com/<USERNAME>/uganda-climate-intelligence.git
   git push -u origin main
   ```
   Then edit `CITATION.cff` → `repository-code` with your real URL.
2. Enable the Zenodo bridge: sign in at [zenodo.org](https://zenodo.org) **with
   your GitHub account**, open
   [zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/),
   and flip the switch for this repository.
3. Publish a **GitHub Release** (tag `v1.0.0`). Zenodo automatically drafts a
   record (using `.zenodo.json`), mints a **version DOI** and a **concept DOI**,
   and links them back to the release. Click **Publish** on the draft.
4. Add the minted DOI to `CITATION.cff` (uncomment `doi:`) and to the manuscript's
   Data & Code Availability statement.

Every future release produces a new version DOI under the same concept DOI.
