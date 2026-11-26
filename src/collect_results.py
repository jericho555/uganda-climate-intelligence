"""Harvest every pipeline artifact into one machine-readable digest plus an RQ5
event-recall / lead-time analysis, so the manuscript only ever cites values that
were actually computed.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS


def event_recall_leadtime():
    df = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"])
    reg = pd.read_csv(DATA / "disaster_register.csv", parse_dates=["start", "end"])
    rows = []
    for _, e in reg.iterrows():
        regs = set(e.regions.split(";"))
        win = df[df.region.isin(regs) & (df.date >= e.start) & (df.date <= e.end)]
        look = df[df.region.isin(regs) &
                  (df.date >= e.start - pd.Timedelta(days=21)) & (df.date <= e.end)]
        flagged = int(win.is_anomaly.sum())
        detected = flagged > 0
        # lead time = days between earliest pre/onset anomaly and event start
        anom_days = look[look.is_anomaly == 1].date
        lead = None
        if len(anom_days):
            first = anom_days.min()
            lead = int((e.start - first).days)
        rows.append(dict(event_id=e.event_id, type=e.type, start=str(e.start.date()),
                         regions=e.regions, anomaly_days_in_window=flagged,
                         detected=detected, lead_time_days=lead,
                         n_loc_days=len(win)))
    ev = pd.DataFrame(rows)
    overall = float(ev.detected.mean())
    by_type = ev.groupby("type").detected.mean().round(3).to_dict()
    leads = ev.lead_time_days.dropna()
    return ev, {
        "event_recall": overall,
        "event_recall_by_type": by_type,
        "n_events": int(len(ev)),
        "median_lead_time_days": float(np.median(leads)) if len(leads) else None,
        "pct_early_warning_ge3d": float((leads >= 3).mean()) if len(leads) else None,
    }


def main():
    digest = {}
    for name in ["spatial_stats", "ucasi_summary", "explain_summary", "friedman", "change_points"]:
        p = RESULTS / f"{name}.json"
        digest[name] = json.loads(p.read_text()) if p.exists() else None

    if (RESULTS / "benchmark.csv").exists():
        digest["benchmark_top"] = pd.read_csv(RESULTS / "benchmark.csv").head(8).to_dict("records")

    ev, rl = event_recall_leadtime()
    ev.to_csv(RESULTS / "tables/event_recall.csv", index=False)
    digest["event_validation"] = rl

    # data provenance snapshot
    if (DATA / "era5_daily.csv").exists():
        raw = pd.read_csv(DATA / "era5_daily.csv", parse_dates=["date"])
        digest["data"] = {
            "rows": int(len(raw)), "locations": int(raw.location.nunique()),
            "location_list": sorted(raw.location.unique().tolist()),
            "date_min": str(raw.date.min().date()), "date_max": str(raw.date.max().date()),
            "variables": [c for c in raw.columns if c not in
                          ("location", "region", "date", "lat", "lon", "elevation_m")],
        }

    (RESULTS / "results_digest.json").write_text(json.dumps(digest, indent=2, default=str))
    print(json.dumps({k: (v if k != "benchmark_top" else "...") for k, v in digest.items()},
                     indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()
