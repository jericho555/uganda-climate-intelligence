"""Uganda Climate Anomaly Severity Index (UCASI) - the core novel contribution.

UCASI fuses five dimensions computed from the validated ensemble anomaly map:
  intensity, duration, frequency, spatial spread, multi-variable impact
into a single 0-100 score per documented event and per district-year,
banded into Low / Moderate / High / Extreme.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS

ZCOLS = [f"{f}_z" for f in C.FEATURES]


def pct(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0, 1))


def event_ucasi(df, reg):
    # number of features concurrently anomalous on each day
    df = df.copy()
    df["n_feat_exceed"] = (df[ZCOLS].abs() > 2.0).sum(1)
    rows = []
    for _, e in reg.iterrows():
        regs = set(e.regions.split(";"))
        m = (df.date >= e.start) & (df.date <= e.end) & df.region.isin(regs)
        sub = df[m]
        anom = sub[sub.is_anomaly == 1]
        if len(sub) == 0:
            continue
        intensity = anom.ens_score.mean() if len(anom) else sub.ens_score.mean()
        duration = len(anom)                                   # anomaly-days
        spread = anom.location.nunique()                        # districts affected
        multivar = anom.n_feat_exceed.max() if len(anom) else sub.n_feat_exceed.max()
        rows.append(dict(event_id=e.event_id, start=e.start.date(), end=e.end.date(),
                         type=e.type, regions=e.regions,
                         n_days=len(sub), anomaly_days=duration,
                         districts_affected=spread,
                         intensity=float(intensity),
                         max_concurrent_vars=int(multivar),
                         matched_anomaly_frac=float(len(anom) / len(sub))))
    ev = pd.DataFrame(rows)
    # normalise dimensions across events
    ev["I"] = ev.intensity.apply(lambda v: pct(v, ev.intensity.min(), ev.intensity.max()))
    ev["D"] = ev.anomaly_days.apply(lambda v: pct(v, 0, ev.anomaly_days.max()))
    ev["S"] = ev.districts_affected.apply(lambda v: pct(v, 0, ev.districts_affected.max()))
    ev["M"] = ev.max_concurrent_vars.apply(lambda v: pct(v, 0, len(C.FEATURES)))
    # frequency: count of same-type events per year within dataset window
    yr = pd.to_datetime(ev.start).dt.year
    ev["year"] = yr
    freq = ev.groupby(["type", "year"]).size().rename("freq").reset_index()
    ev = ev.merge(freq, on=["type", "year"])
    ev["F"] = ev.freq.apply(lambda v: pct(v, 0, ev.freq.max()))
    W = dict(I=0.30, D=0.20, S=0.20, M=0.15, F=0.15)
    ev["UCASI"] = (100 * (W["I"]*ev.I + W["D"]*ev.D + W["S"]*ev.S +
                          W["M"]*ev.M + W["F"]*ev.F)).round(1)
    q = ev.UCASI.quantile
    ev["severity"] = pd.cut(ev.UCASI,
                            bins=[-1, q(0.4), q(0.7), q(0.9), 101],
                            labels=["Low", "Moderate", "High", "Extreme"])
    return ev, W


def district_year_ucasi(df, W):
    df = df.copy()
    df["n_feat_exceed"] = (df[ZCOLS].abs() > 2.0).sum(1)
    g = df.groupby(["location", "region", "year"])
    dy = g.apply(lambda x: pd.Series({
        "anomaly_days": int(x.is_anomaly.sum()),
        "intensity": float(x.loc[x.is_anomaly == 1, "ens_score"].mean()
                           if x.is_anomaly.any() else 0),
        "max_concurrent_vars": float(x.n_feat_exceed.max()),
    })).reset_index()
    dy["I"] = dy.intensity.apply(lambda v: pct(v, dy.intensity.min(), dy.intensity.max()))
    dy["D"] = dy.anomaly_days.apply(lambda v: pct(v, 0, dy.anomaly_days.max()))
    dy["M"] = dy.max_concurrent_vars.apply(lambda v: pct(v, 0, len(C.FEATURES)))
    dy["F"] = dy.groupby("location").anomaly_days.transform(
        lambda s: pct(s.max(), 0, dy.anomaly_days.max()))
    dy["S"] = 0.5  # placeholder spread at single-district resolution
    dy["UCASI"] = (100 * (W["I"]*dy.I + W["D"]*dy.D + W["S"]*dy.S +
                          W["M"]*dy.M + W["F"]*dy.F)).round(1)
    q = dy.UCASI.quantile
    dy["severity"] = pd.cut(dy.UCASI, bins=[-1, q(0.4), q(0.7), q(0.9), 101],
                            labels=["Low", "Moderate", "High", "Extreme"])
    return dy


def main():
    df = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"])
    reg = pd.read_csv(DATA / "disaster_register.csv", parse_dates=["start", "end"])
    ev, W = event_ucasi(df, reg)
    dy = district_year_ucasi(df, W)
    ev.to_csv(RESULTS / "tables/ucasi_events.csv", index=False)
    dy.to_csv(RESULTS / "tables/ucasi_district_year.csv", index=False)
    summary = {
        "weights": W,
        "n_events": int(len(ev)),
        "ucasi_range": [float(ev.UCASI.min()), float(ev.UCASI.max())],
        "severity_counts": ev.severity.value_counts().to_dict(),
        "top_events": ev.sort_values("UCASI", ascending=False)
            [["event_id", "type", "start", "regions", "UCASI",
              "severity", "districts_affected"]].head(8).to_dict("records"),
        "mean_ucasi_by_type": ev.groupby("type").UCASI.mean().round(1).to_dict(),
    }
    (RESULTS / "ucasi_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("UCASI weights:", W)
    print("severity counts:", summary["severity_counts"])
    print(ev.sort_values("UCASI", ascending=False)
          [["event_id", "type", "start", "regions", "UCASI", "severity"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
