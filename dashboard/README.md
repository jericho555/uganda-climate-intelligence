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

`altair` ships as a Streamlit dependency, so no extra charting install is
required. The dashboard only needs `streamlit`, `pandas` and `numpy`; it does
**not** import torch/sklearn (the heavy detector work happens in the pipeline,
not in the view).

## Tabs

| Tab | Artefacts rendered |
|-----|--------------------|
| **Overview** | `benchmark.csv`, `friedman.json`, register, districts |
| **Detection Benchmark** | `benchmark.csv` (24 detectors + bootstrap CIs), `per_year_f1.npy`, `friedman.json` |
| **Time-Series Explorer** | `full_ensemble.csv` (daily stream, ~93k district-days) + `disaster_register.csv` |
| **Spatial Hotspots** | `district_anomaly.csv`, `spatial_stats.json` (Moran's I, Getis–Ord G*) |
| **Events & Severity (UCASI)** | `ucasi_events.csv`, `ucasi_summary.json`, `event_recall.csv` |
| **Trends & Climatology** | `temporal_trends.csv`, `meteostat_trends.csv` (Mann–Kendall, Sen's slope) |
| **Risk Nowcast (EWS)** | `risk_district_month.csv`, `early_warning_summary.json` |
| **Detection Drivers** | `shap_importance.csv`, `permutation_importance.csv` |
| **About / Reproducibility** | source and licence notes |

## Integrity notes

- The benchmark tab marks the **size-matched random-flag null (F1 = 0.069,
  Monte-Carlo band 0.059-0.079)** and the **base rate (0.1115)** on every
  relevant chart. Both are read from `results/random_null.json`, written by
  `src/random_null.py`; nothing is typed in by hand. Against that reference the
  two torch sequence autoencoders (LSTM-AE, Transformer-AE) roughly double the
  alert-level F1, five cross-sectional detectors stay within 1.22x of it, and
  every detector ranks at chance on ROC-AUC.
- The Risk Nowcast tab reports the **measured** forward skill against the
  register, not a promotional framing (see `early_warning_summary.json`).
