"""Type-matched early-warning analysis (RQ5, RQ6).

The generic multivariate detectors are ~chance against disaster windows. This
module tests a physically-targeted alternative: for each documented event TYPE
we score days with a single antecedent index aligned to the hazard mechanism
(wetness spell for flood/landslide, dryness spell for drought) and evaluate
event detection and lead time against the same ground-truth register.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS

# hazard -> (index column, direction). direction +1 = high values are extreme.
HAZARD_INDEX = {
    "Flood":     ("precip_z_10d", +1),
    "Landslide": ("precip_z_5d",  +1),
    "Drought":   ("dry_z_60d",    +1),
}
Q = 1 - C.CONTAMINATION   # per-location top-5% threshold, matches ensemble flagging


def flag_by_index(df, col):
    """Per-location percentile flag on a chosen antecedent index."""
    out = np.zeros(len(df), int)
    for loc, g in df.groupby("location"):
        v = g[col].fillna(0).values
        thr = np.quantile(v, Q)
        m = df.location.values == loc
        out[m] = (v >= thr).astype(int)
    return out


def evaluate(df, reg, col):
    d = df.copy()
    d["flag"] = flag_by_index(d, col)
    rows = []
    for _, e in reg.iterrows():
        regs = set(e.regions.split(";"))
        win = d[d.region.isin(regs) & (d.date >= e.start) & (d.date <= e.end)]
        look = d[d.region.isin(regs) &
                 (d.date >= e.start - pd.Timedelta(days=30)) & (d.date <= e.end)]
        if len(win) == 0:
            continue
        flagged = int(win.flag.sum())
        detected = flagged > 0
        pre = look[(look.flag == 1) & (look.date < e.start)].date
        lead = int((e.start - pre.min()).days) if len(pre) else None
        rows.append(dict(event_id=e.event_id, type=e.type, start=str(e.start.date()),
                         n_days=len(win), flagged_days=flagged,
                         hit_rate=flagged / len(win), detected=detected, lead=lead))
    ev = pd.DataFrame(rows)
    leads = ev.lead.dropna()
    return ev, dict(
        index=col, n_events=int(len(ev)),
        detection_rate=round(float(ev.detected.mean()), 3),
        mean_hit_rate=round(float(ev.hit_rate.mean()), 3),
        median_lead_days=(float(np.median(leads)) if len(leads) else None),
        pct_lead_ge3d=(round(float((leads >= 3).mean()), 3) if len(leads) else None),
    )


def main():
    df = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"])
    reg = pd.read_csv(DATA / "disaster_register.csv", parse_dates=["start", "end"])

    summary = {}
    all_rows = []
    # generic ensemble baseline
    gen_ev, gen_stat = evaluate(df, reg, None) if False else (None, None)
    d = df.copy()
    gen_rows = []
    for _, e in reg.iterrows():
        regs = set(e.regions.split(";"))
        win = d[d.region.isin(regs) & (d.date >= e.start) & (d.date <= e.end)]
        if len(win) == 0:
            continue
        pre = d[d.region.isin(regs) & (d.date < e.start) &
                (d.date >= e.start - pd.Timedelta(days=30)) & (d.is_anomaly == 1)].date
        lead = int((e.start - pre.min()).days) if len(pre) else None
        gen_rows.append(dict(event_id=e.event_id, type=e.type, start=str(e.start.date()),
                             n_days=len(win), flagged_days=int(win.is_anomaly.sum()),
                             hit_rate=int(win.is_anomaly.sum()) / len(win),
                             detected=bool(win.is_anomaly.sum() > 0), lead=lead))
    gev = pd.DataFrame(gen_rows)
    gleads = gev.lead.dropna()
    # size-matched random null: P(>=1 flag) for a per-day rate Q over n_days rows
    null_touch = float(np.mean([1 - (1 - Q) ** n for n in gev.n_days]))
    summary["GenericEnsemble"] = dict(
        index="is_anomaly(ensemble)", n_events=int(len(gev)),
        detection_rate=round(float(gev.detected.mean()), 3),
        random_null_detection_rate=round(null_touch, 3),
        mean_hit_rate=round(float(gev.hit_rate.mean()), 3),
        median_lead_days=(float(np.median(gleads)) if len(gleads) else None),
        pct_lead_ge3d=(round(float((gleads >= 3).mean()), 3) if len(gleads) else None),
    )
    all_rows.append(gev.assign(index="GenericEnsemble"))

    # type-matched: overall + per-type detection
    matched = d.copy()
    matched["flag_tm"] = 0
    for _, e in reg.iterrows():
        pass
    # build a per-event matched flag using the hazard index for that event type
    tm_rows = []
    for _, e in reg.iterrows():
        col = HAZARD_INDEX.get(e.type)
        if col is None:
            continue
        col = col[0]
        regs = set(e.regions.split(";"))
        dd = d[d.region.isin(regs)].copy()
        dd["f"] = flag_by_index(dd, col)
        win = dd[(dd.date >= e.start) & (dd.date <= e.end)]
        if len(win) == 0:
            continue
        pre = dd[(dd.f == 1) & (dd.date < e.start) &
                 (dd.date >= e.start - pd.Timedelta(days=30))].date
        lead = int((e.start - pre.min()).days) if len(pre) else None
        tm_rows.append(dict(event_id=e.event_id, type=e.type, start=str(e.start.date()),
                            index=col, n_days=len(win), flagged_days=int(win.f.sum()),
                            hit_rate=int(win.f.sum()) / len(win),
                            detected=bool(win.f.sum() > 0), lead=lead))
    tev = pd.DataFrame(tm_rows)
    tleads = tev.lead.dropna()
    summary["TypeMatched_all"] = dict(
        n_events=int(len(tev)),
        detection_rate=round(float(tev.detected.mean()), 3),
        mean_hit_rate=round(float(tev.hit_rate.mean()), 3),
        median_lead_days=(float(np.median(tleads)) if len(tleads) else None),
        pct_lead_ge3d=(round(float((tleads >= 3).mean()), 3) if len(tleads) else None),
        by_type_detection=tev.groupby("type").detected.mean().round(3).to_dict(),
    )
    all_rows.append(tev.assign(index="TypeMatched"))

    (RESULTS / "typematch.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.concat(all_rows, ignore_index=True).to_csv(RESULTS / "tables/type_match_events.csv", index=False)
    for k, v in summary.items():
        print(k, "->", v)


if __name__ == "__main__":
    main()
