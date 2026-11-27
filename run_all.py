"""One-command reproduction of the Uganda Climate Intelligence pipeline.

Usage:  python run_all.py
Order:  see STAGES below -- fetch -> preprocess -> detect -> random-null
        reference -> spatial/temporal -> climate layers -> impact layers ->
        UCASI -> early warning -> explainability -> figures -> digest ->
        release gate.

The final stage (``verify_release``) re-checks the deliverables themselves --
figure mirrors and resolution, cross-references, artefact arithmetic, dashboard
rendering -- and raises SystemExit if any check fails, so a broken bundle stops
the run instead of being reported as ``[done ...]``.
"""
import sys
import time
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

STAGES = [
    ("fetch_era5", "ERA5 (Open-Meteo) daily data"),
    ("fetch_meteostat", "Meteostat century-scale in-situ daily record (range extension)"),
    ("fetch_agriculture", "World Bank cereal production/area/yield (impact layer)"),
    ("fetch_coffee", "OWID/FAOSTAT Ugandan green-coffee production (signature export impact layer)"),
    ("fetch_power", "NASA POWER daily T2M/RH/solar 1988-2007 (independent gap-fill)"),
    ("fetch_boundaries", "OCHA Uganda COD district polygons + site->district join"),
    ("disaster_register", "Disaster ground-truth register"),
    ("preprocess", "Seasonal anomalies + labels"),
    ("detect", "Detector benchmark + evaluation"),
    ("region_label_check", "Sensitivity of event labels to the boundary register's regions"),
    ("_recover", "Friedman + full-period ensemble map (idempotent fallback)"),
    ("random_null", "Size-matched random-flag null + BH-controlled Monte-Carlo tests"),
    ("spatial", "Moran's I / Getis-Ord hotspots"),
    ("temporal", "STL / Mann-Kendall / change points"),
    ("meteostat_instrumental", "Century in-situ trends + ERA5 cross-validation"),
    ("power_climate", "POWER reliability + continuous temp/RH trends across dark period"),
    ("drought_indices", "SPI / SPEI derived drought indices + register validation"),
    ("ag_linkage", "Growing-season drought vs detrended cereal yield (impact novel finding)"),
    ("coffee_linkage", "Coffee production anomaly vs moisture across substrates (confounding demo)"),
    ("ucasi", "Uganda Climate Anomaly Severity Index"),
    ("early_warning", "UCASI forward risk nowcast + honest skill (RQ6)"),
    ("_typematch", "Type-matched early-warning + size-matched random null"),
    ("_explain_fast", "SHAP / feature importance (lightweight, deterministic)"),
    ("figures", "Publication figures"),
    ("figures_power", "Gap-fill + drought/yield figures"),
    ("figures_maps", "Study-area, risk, gap-fill and null-comparison maps"),
    ("figures_coffee", "Coffee production + sign-flip correlation figure"),
    ("figures_early_warning", "EWS architecture + risk-nowcast figures"),
    ("collect_results", "Results digest + event-recall/lead-time (RQ5)"),
    ("check_ref_index", "Crossref/OpenAlex/arXiv check on the refuted baseline draft"),
    ("verify_release", "Release gate: figure, manuscript, artefact and dashboard integrity"),
]


def main():
    for mod, desc in STAGES:
        print("\n" + "=" * 78)
        print(f"STAGE {mod}: {desc}")
        print("=" * 78)
        t0 = time.time()
        try:
            m = __import__(mod)
            m.main()
            print(f"[done {mod} in {time.time()-t0:.1f}s]")
        except Exception:
            print(f"[FAILED {mod}]")
            traceback.print_exc()


if __name__ == "__main__":
    main()
