# Implementation Plan: Forward-Looking Early Warning for Uganda Climate Intelligence

## Objective
Extend the existing UCIF pipeline with four forward-looking layers: threshold drought early warning, crop-yield impact forecasting, seasonal-outlook integration, and an automated risk bulletin — while preserving the project's two non-negotiable conventions: every number computed from data, and honest reporting of skill/uncertainty/limitations.

## Critical Findings That Reshape the Plan

Before implementation, three evidence-based corrections to the task assumptions are mandatory. Executing the request literally would violate the "no fabricated skill" and "nothing hard-coded" principles.

**Finding A — District-level crop yield data does NOT exist in this repo.**
`data/uganda_crop_yield.csv` is a **national annual** series (columns: `year, cereal_prod_t, cereal_area_ha, cereal_yield_kgha`) pulled from World Bank FAOSTAT-derived indicators by `src/fetch_agriculture.py:51-62`. Phase 2 asks to "fit a regression per district with yield data." Zero districts have yield data. FAOSTAT's district (sub-national) series for Uganda are not cleanly scriptable, which is exactly why `fetch_agriculture.py:5-7` chose the national backbone.
→ Correct honest design: build Phase 2 as a **national-scale** yield-impact forecast (extending the existing `ag_linkage.py` national regression), and report that district-scale yield projection is **not identifiable** from available data. Do not fabricate district yield.

**Finding B — SPI-12 is not computable at district scale on ERA5.**
`src/drought_indices.py:140-164` computes district SPI/SPEI only at **SPI3/SPI6/SPEI3/SPEI6**. The comment at `drought_indices.py:11-12` and the `_spi_from_accum` guard (`:47` needs ≥24 samples, `:55` needs ≥12 per calendar month) exist precisely because ERA5 is a 16-year window (`data/era5_daily.csv` starts 2008). SPI-12 needs 12-month accumulations → only ~4 usable full years, and per-calendar-month gamma fits become unstable. The **century SPI-12 exists only at 4 Meteostat stations** (`spi_meteostat.csv`, `station` column), not per district.
→ Phase 1 must compute rolling SPI3/SPI6 at district scale (feasible), cite the century SPI-12 only at station scale for trajectory context, and state the SPI-12 district limitation explicitly.

**Finding C — The verification gate and dashboard are coupled; they break together.**
`src/verify_release.py:344-345` asserts the dashboard exposes **exactly 10 tabs** (`len(labels) == 10`), and `check_figures` (`:77-92`) requires `figures/` and `manuscript/figures/` to hold the **identical PNG set, byte-for-byte, ≥300 dpi**. Adding three dashboard tabs and three figures without updating these assertions will make `run_all.py`'s final stage raise `SystemExit`. This is the highest-risk integration point.

## Target File Structure (new files marked +)

```
uganda-climate-intelligence/
├── src/
│   ├── config.py                     # EDIT: add REPORTS path, SPI thresholds, EW categories, lead constants
│   ├── drought_indices.py            # reuse (SPI/SPEI substrate) — no change needed
│   ├── + drought_early_warning.py    # Phase 1
│   ├── + yield_forecast.py           # Phase 2 (national)
│   ├── + fetch_seasonal.py           # Phase 3 ingest (Open-Meteo seasonal)
│   ├── + seasonal_outlook.py         # Phase 3 downscale + integrate with drought EW
│   ├── + bulletin.py                 # Phase 4 Markdown/HTML/PDF generator
│   ├── + figures_drought_ew.py       # Phase 1 figure
│   ├── + figures_yield.py            # Phase 2 figure
│   ├── + figures_seasonal.py         # Phase 3 figure
│   └── verify_release.py             # EDIT: tab count 10->13, new artefact/number checks
├── run_all.py                        # EDIT: add STAGES for Phases 1-4 + fetch_seasonal + figures
├── dashboard/app.py                  # EDIT: add 3 tabs + tab tuple unpacking
├── data/
│   └── + seasonal_forecast_raw.csv   # Phase 3 cached ingest output
├── results/
│   ├── tables/
│   │   ├── + drought_early_warning.csv
│   │   ├── + yield_forecast.csv
│   │   └── + seasonal_outlook.csv
│   ├── + drought_ew_summary.json
│   ├── + yield_forecast_summary.json
│   └── + seasonal_outlook_summary.json
├── figures/ + manuscript/figures/
│   ├── + fig_drought_ew.png          # must be mirrored by C.save_figure
│   ├── + fig_yield_forecast.png
│   └── + fig_seasonal_outlook.png
├── + reports/
│   └── climate_risk_bulletin_YYYY_MM.md
├── README.md / CHANGELOG.md          # EDIT: document new capabilities
└── manuscript/main.tex               # EDIT: new methods/results section (numbers must match artefacts)
```

## Phase Overview

### Phase 1 — Threshold-Based Drought Early Warning
Build on existing SPI/SPEI infrastructure:
- Define drought severity categories (normal, mild, moderate, severe, extreme) using SPI thresholds (WMO/McKee 1993)
- Compute rolling 3/6-month SPI for each district (SPI-12 infeasible at district scale)
- Track SPI trajectories (improving, stable, deteriorating)
- Generate district-level drought status: "Normal", "Watch", "Warning", "Alert"
- Flag districts where SPI has declined for 3+ consecutive months
- Validate against disaster register (honest hit/false-alarm counts)
- Output: `results/tables/drought_early_warning.csv`, `results/drought_ew_summary.json`
- Visualization: `figures/fig_drought_ew.png` (district map with drought status)

### Phase 2 — Crop Yield Impact Forecasting (national-scale, honest)
Leverage existing cereal yield data:
- National-scale regression: yield_anomaly ~ growing-season SPI (district yield unavailable)
- Compute current growing-season SPI trajectory (national average)
- Project expected yield based on current conditions
- Estimate probability of below-normal yield (< 10th percentile)
- Honest skill check: compare vs climatology null
- Output: `results/tables/yield_forecast.csv`, `results/yield_forecast_summary.json`
- Visualization: `figures/fig_yield_forecast.png`

### Phase 3 — Seasonal Climate Outlook Integration
Incorporate external seasonal forecasts:
- Data source: Open-Meteo Seasonal Forecast API (ECMWF SEAS5, no API key, CC-BY)
- Ingest seasonal precipitation/temperature forecasts (3-month lead time)
- Downscale to district level using ERA5 climatology as baseline
- Compare forecast vs climatology: "above normal", "near normal", "below normal"
- Integrate with drought early warning: escalation rule if outlook is below normal AND current SPI is declining
- Output: `results/tables/seasonal_outlook.csv`, `results/seasonal_outlook_summary.json`
- Visualization: `figures/fig_seasonal_outlook.png`

### Phase 4 — Automated Climate Risk Bulletin
Synthesize all layers into a narrative report:
- Monthly/quarterly report generator
- Sections: executive summary, drought status, seasonal outlook, impact projections, recommendations
- Format: Markdown (PDF optional if pypandoc/weasyprint available)
- Output: `reports/climate_risk_bulletin_YYYY_MM.md`
- Include: maps, tables, key metrics, uncertainty statements
- Every number read from summary JSONs (no hard-coding)

## Technical Requirements
- All code in `src/` as separate modules
- Integrate into `run_all.py` pipeline
- Follow existing patterns: config.py for constants, save_figure() for outputs
- Every number must be computed from data, nothing hard-coded
- Add verification checks to `verify_release.py`
- Update dashboard to show new tabs: "Drought Early Warning", "Yield Forecast", "Seasonal Outlook"
- Update manuscript/cover letter to mention new capabilities
- Maintain the "honest evaluation" principle: report skill, uncertainty, limitations

## Execution Order & Dependencies
1. config.py edits → (unblocks all)
2. Phase 1 `drought_early_warning.py` + `figures_drought_ew.py` (depends only on existing `spei_era5.csv`)
3. Phase 2 `yield_forecast.py` + `figures_yield.py` (depends on `uganda_crop_yield.csv`, `spei_era5.csv`)
4. Phase 3 `fetch_seasonal.py` → `seasonal_outlook.py` (depends on Phase 1 table for the escalation rule)
5. Phase 4 `bulletin.py` (depends on Phases 1-3 outputs)
6. `run_all.py` + `collect_results.py` integration
7. `dashboard/app.py` tabs
8. `verify_release.py` updates (tab count, new checks) — last, after artefacts exist
9. README / CHANGELOG / manuscript / cover letter

## Risks / Caveats
- **Network in Phase 3**: `fetch_seasonal` is the only new network step. Mitigate with graceful-degradation flag so `run_all`/verify never hard-fail if Open-Meteo seasonal is unreachable.
- **Weak yield skill is plausible**: `ag_linkage` correlations over 2008-2023 are the basis; if the yield regression does not beat climatology, the deliverable must say so.
- **Coordinated edits are brittle**: adding tabs/figures without the matching verify edits (Finding C) is the most likely failure mode. Land the verify + dashboard edits in the same change as the artifacts.
- **Do not add district yield or district SPI-12 by invention** (Findings A/B) — the honest, data-supported scope is national yield and district SPI3/SPI6.
