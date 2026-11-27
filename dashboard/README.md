# Uganda Climate Intelligence — interactive dashboard

A read-only **Streamlit** view of the artefacts produced by the pipeline
(`run_all.py`). Nothing is fitted or re-computed here; every number is loaded
from `data/` and `results/`.

## Launch

```bash
python dashboard/run_dashboard.py
```

This opens the launcher, which automatically prepends the workspace-local
vendored packages at `._webpkgs/` to `sys.path` before starting Streamlit, so
the dashboard works even when the environment's system site-packages is not
writable.

Then open **http://localhost:8501** in a browser.

Options pass straight through to Streamlit, e.g.

```bash
python dashboard/run_dashboard.py --server.port 8600 --server.headless true
```

## Installing the backend (one time)

If `._webpkgs/` does not exist yet (it is gitignored), vendor Streamlit into
the workspace:

```bash
python -m pip install --target=._webpkgs --no-cache-dir streamlit
```

`altair` and `pydeck` both ship as Streamlit dependencies, so no extra charting
install is required. The view needs `streamlit`, `pandas` and `numpy` only; the
maps additionally use `pydeck`, and it does **not** import torch/sklearn (the
heavy detector work happens in the pipeline, not in the view).

## Tabs

| Tab | Artefacts rendered |
|-----|--------------------|
| **Overview** | `benchmark.csv`, `friedman.json`, `district_coverage.json`, register, `district_anomaly.csv` |
| **Detection Benchmark** | `benchmark.csv` (all benchmarked detectors + bootstrap CIs, merged with the per-detector null ratio and BH q from `random_null.json`), `per_year_f1.npy`, `friedman.json` |
| **Time-Series Explorer** | `full_ensemble.csv` (daily stream, ~93k district-days) + `disaster_register.csv` |
| **Spatial Hotspots** | `district_anomaly.csv`, `spatial_stats.json` (Moran's I, Getis–Ord G*), `district_boundaries.geojson` |
| **Events & Severity (UCASI)** | `ucasi_events.csv`, `ucasi_summary.json`, `event_recall.csv` |
| **Trends & Climatology** | `temporal_trends.csv`, `meteostat_trends.csv` (Mann–Kendall, Sen's slope) |
| **Impacts & Agriculture** | `drought_summary.json`, `ag_linkage.json`, `coffee_linkage.json`, `spi_decade_drought.csv`, `spei_era5.csv`, `ag_yield_climate.csv`, `coffee_prod_climate.csv`, and the World Bank cross-check `wb_climate_check.json` + `wb_spei_compare.csv` + `wb_precip_compare.csv` |
| **Risk Nowcast (EWS)** | `risk_district_month.csv`, `early_warning_summary.json` (incl. `carry_uncertainty` and the split-conformal `prediction_intervals`), `nowcast_interval.csv`, `district_boundaries.geojson` |
| **Detection Drivers** | `shap_importance.csv`, `permutation_importance.csv`, and the persisted local attributions `shap_values.npy`, `shap_sample.csv`, `shap_expected_value.json`, `shap_row_outputs.npy` |
| **About / Reproducibility** | source and licence notes, artefact inventory |

## Maps

The two choropleths (Spatial Hotspots, Risk Nowcast) are drawn with `pydeck`
`GeoJsonLayer` over the COD-AB district polygons in
`data/district_boundaries.geojson`. Three deliberate choices:

- `map_provider=None` — no tile server is contacted, so the only geometry on
  screen is the district layer and nothing is loaded from a third party.
- Each colour class is a separate layer built from a Python-side classification,
  so the legend printed beside the map is provably the set of classes painted;
  the legend cannot drift from the data as a binning-built scale can.
- **No-data is never drawn as zero.** Of the 135 polygons, 16 carry an ERA5
  detection-panel score, 3 have a network site but no panel score (Hoima,
  Kabale, Kasese — shaded grey with a heavy outline, not merged into the
  no-data class), and 116 were never observed. If `pydeck` is unavailable, or
  the geometry is missing, the panel falls back to the paper's static 300-dpi
  map figure and says so.

## Integrity notes

- The headline counts are **read, not typed**. The sidebar and both benchmark
  panels take the detector count from `benchmark.csv`, the Friedman degrees of
  freedom from `friedman.json`, and the number of non-sequence detectors that
  survive Benjamini-Hochberg control from `random_null.json`. An earlier
  revision hard-coded "24 detectors" / "chi2(23)" and went stale the moment two
  detectors were added; `src/verify_release.py` now refuses those literals.
- The benchmark tab marks the **size-matched random-flag null (F1 = 0.069,
  Monte-Carlo band 0.059-0.079)** and the **base rate (0.1115)** on every
  relevant chart. Both are read from `results/random_null.json`, written by
  `src/random_null.py`; nothing else is typed in by hand. Against that reference the
  two torch sequence autoencoders (LSTM-AE, Transformer-AE) roughly double the
  alert-level F1, six of the twenty-four non-sequence detectors stay above it at
  BH q<0.05 by at most 1.22x, and every detector ranks at chance on ROC-AUC —
  including the two added at revision (LoOP, Temporal IF), which the tab labels
  as `temporal (lag-window)` and `cross-sectional` respectively.
- The Risk Nowcast tab reports the **measured** forward skill against the
  register, not a promotional framing (see `early_warning_summary.json`). The
  month-to-month carry is shown two ways: the *empirical quantiles* of the score
  change (`carry_uncertainty`), which describe the past, and the **split-conformal
  prediction intervals** (`prediction_intervals`, `nowcast_interval.csv`) fitted by
  `src/early_warning.py`, whose measured held-out coverage is printed next to the
  nominal level — including the shortfall, because a chronological split breaks the
  exchangeability the conformal guarantee assumes.
- The Impacts tab renders the **World Bank cross-check as the negative result it
  is** (r = 0.06, p = 0.83, and the Bank's national index is wet in every register
  drought year) rather than as corroboration.
- The Impacts & Agriculture tab renders the linkage tables field-by-field from
  `ag_linkage.json` / `coffee_linkage.json`, including the coefficients that are
  not significant and the coffee sign reversal between the in-situ and ERA5
  substrates. Yield anomalies are residuals of log-yield on a fitted linear
  trend; the coffee series is detrended log **production**, which is the weaker
  quantity and is labelled as such in the panel.
