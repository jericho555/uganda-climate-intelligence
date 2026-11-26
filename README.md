# Uganda Climate Intelligence Framework (UCIF)

Reproducible research package behind the manuscript:

> **A National Climate Anomaly Intelligence System for Uganda: Detection,
> Spatial Risk Nowcasting, Impact Analysis and Explainability**
> Jeremiah Masaba — Department of Computer Science, Makerere University, Kampala, Uganda
> ([ORCID 0009-0004-5620-996X](https://orcid.org/0009-0004-5620-996X))

This converts the baseline study *"Evaluation of Anomaly Detection Techniques
Using Uganda Weather Dataset"* from a single-source algorithm comparison into a
validated detection + spatial-risk + century-climatology + drought-index +
impact + severity + nowcasting system with an interactive decision-support
dashboard.

**Core, deliberately uncomfortable finding:** against real disaster ground-truth,
*no cross-sectional* unsupervised anomaly detector ranks above chance (held-out ROC-AUC
0.49–0.53; best cross-sectional F1 = 0.084; PR-AUC ≤ 0.121 vs a 0.1115 base rate).
Scored against a measured size-matched random alert budget (F1 = 0.069 [0.059, 0.079],
`results/random_null.json`), five cross-sectional detectors stay within 1.22× of it
while only two deep **sequence** autoencoders (LSTM-AE, Transformer-AE — torch,
vendored CPU build) roughly double it (F1 0.13–0.14, precision 0.20–0.22, yet
ROC-AUC still 0.51–0.53 and <10% of disaster days captured). The apparent 68.8%
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
| **World Bank / FAOSTAT** | 64-year cereal production / area / yield (1961–2024) impact layer | CC BY 4.0 |
| **Curated disaster register** | 32 documented hydro-meteorological disasters (UNMA, OCHA/ReliefWeb, EM-DAT, FAO/GIEWS, FEWS NET) with onset windows + affected regions | editorial records, cited |

> **Note on Meteoblue:** the original study used a Meteoblue archive whose bulk
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

## Reproduce

```bash
pip install -r requirements.txt
python run_all.py
```

## Interactive dashboard

A read-only Streamlit view of the benchmark, daily anomaly stream, spatial
hotspots, UCASI severity, trends and the risk nowcast — bound entirely to the
artefacts under `results/` and `data/` (nothing is re-fitted on the page).

```bash
# one-time vendored install (works when system site-packages is not writable):
python -m pip install --target=._webpkgs --no-cache-dir streamlit
python dashboard/run_dashboard.py      # -> http://localhost:8501
```

See [`dashboard/README.md`](dashboard/README.md) for the tab-by-tab mapping to
artefacts. The dashboard is optional; it is not needed to reproduce the paper.

Stages run in the order declared in `run_all.py`: `fetch_era5` →
`fetch_meteostat` → `fetch_agriculture` → `fetch_coffee` → `fetch_power` →
`disaster_register` → `preprocess` → `detect` → `_recover` → `random_null` →
`spatial` → `temporal` → `meteostat_instrumental` → `power_climate` →
`drought_indices` → `ag_linkage` → `coffee_linkage` → `ucasi` → `early_warning` →
`_typematch` → `_explain_fast` → `figures` → `figures_power` → `figures_maps` →
`figures_coffee` → `figures_early_warning` → `collect_results`. Each stage caches to
disk and is idempotent; re-running skips cached fetches.

## What the pipeline computes

* **RQ1** held-out benchmark of detectors vs disaster ground-truth (P/R/F1/
  ROC-AUC/PR-AUC/MCC) with bootstrap CIs and a size-matched random null.
* **RQ2** district anomaly-rate hotspots via Global Moran's I and Getis–Ord Gi*.
* **RQ3** SHAP + permutation importance driving the detection signal.
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
* **UCASI** severity index ranks documented disasters; **RQ5** event recall.

---

## Data and licensing

The **source code** is released under the **MIT License** (see `LICENSE`).
Third-party **data remain under their own terms** (ERA5/Open-Meteo CC-BY 4.0;
Meteostat CC-BY 4.0; NASA POWER open; World Bank/FAOSTAT CC BY 4.0). Large raw
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
