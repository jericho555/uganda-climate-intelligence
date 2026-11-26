"""Century-scale in-situ climate intelligence + ERA5 cross-validation.

Adds the long observational record (Meteostat, 1926-2026) as an independent
complement to the ERA5 detection substrate:
  * Mann-Kendall / Sen's slope trends per station on tavg/tmax/tmin/precip,
    over up to ~91 years of monthly data (far more statistical power than the
    16-year ERA5 window).
  * Cross-validation of ERA5 against in-situ stations over their overlap,
    directly addressing the "no station cross-check" limitation.

Outputs: results/tables/meteostat_trends.csv, results/meteostat_summary.json,
         results/crossval.json
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS

import pymannkendall as mk


def haversine(a, b):
    R = 6371.0
    la1, lo1 = np.radians(a[0]), np.radians(a[1])
    la2, lo2 = np.radians(b[0]), np.radians(b[1])
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(h))


def sens_slope(y):
    y = np.asarray(y, float)
    n = len(y)
    slopes = [(y[j]-y[i])/(j-i) for i in range(n) for j in range(i+1, n)]
    return float(np.median(slopes)) if slopes else np.nan


def monthly(s, agg):
    s = s.dropna()
    if not len(s):
        return s
    return s.resample("MS").mean() if agg == "mean" else s.resample("MS").sum()


def trends(met):
    rows = []
    for sid, g in met.groupby("station"):
        g = g.sort_values("date")
        name = g["name"].iloc[0]
        for var, agg in [("tavg", "mean"), ("tmax", "mean"), ("tmin", "mean"), ("prcp", "sum")]:
            s = pd.to_numeric(g[var], errors="coerce")
            s.index = pd.to_datetime(g.date)
            m = monthly(s, agg)
            m = m[np.isfinite(m)]
            if len(m) < 120:
                continue
            tau_h = mk.original_test(m.values)
            rows.append(dict(station=str(sid), name=name, variable=var,
                             n_years=len(m)/12, tau=float(tau_h.Tau),
                             p_value=float(tau_h.p), trend=tau_h.trend,
                             sens_slope=round(sens_slope(m.values), 5),
                             first=str(m.index.min().date()), last=str(m.index.max().date())))
    return pd.DataFrame(rows)


def crossval(met):
    era5 = pd.read_csv(DATA / "era5_daily.csv", parse_dates=["date"])
    if not len(era5):
        return {"error": "no era5 data"}
    locs = era5.groupby("location")[["lat", "lon"]].first()
    era5["ym"] = era5.date.dt.to_period("M").dt.to_timestamp()
    em = era5.groupby(["location", "ym"]).agg(
        e_tmean=("temperature_2m_mean", "mean"), e_tmax=("temperature_2m_max", "mean"),
        e_tmin=("temperature_2m_min", "mean"), e_prcp=("precipitation_sum", "sum")).reset_index()
    out = {}
    for sid, g in met.groupby("station"):
        name = str(g["name"].iloc[0])
        lat, lon = float(g["lat"].iloc[0]), float(g["lon"].iloc[0])
        if not np.isfinite(lat) or not np.isfinite(lon):
            continue
        nearest = min(locs.index, key=lambda L: haversine((lat, lon), (locs.loc[L, "lat"], locs.loc[L, "lon"])))
        ed = em[em.location == nearest].set_index("ym")[["e_tmean", "e_tmax", "e_tmin", "e_prcp"]]
        g2 = g.copy()
        g2["ym"] = g2.date.dt.to_period("M").dt.to_timestamp()
        mm = g2.groupby("ym").agg(s_tavg=("tavg", "mean"), s_tmax=("tmax", "mean"),
                                  s_tmin=("tmin", "mean"), s_prcp=("prcp", "sum"))
        joined = mm.join(ed, how="inner")
        pairs = {"tmean~tavg": ("s_tavg", "e_tmean"), "tmax": ("s_tmax", "e_tmax"),
                 "tmin": ("s_tmin", "e_tmin"), "prcp": ("s_prcp", "e_prcp")}
        corrs = {}
        for label, (a, b) in pairs.items():
            v = joined[[a, b]].dropna()
            if len(v) > 24 and v[a].std() > 0 and v[b].std() > 0:
                corrs[label] = round(float(np.corrcoef(v[a], v[b])[0, 1]), 3)
        if corrs:
            out[f"{name}->{nearest}"] = dict(n_months=int(len(joined)), overlap=str(
                f"{joined.index.min().date()}..{joined.index.max().date()}"), **corrs)
    return out


def main():
    met = pd.read_csv(DATA / "meteostat_daily.csv", parse_dates=["date"])
    tr = trends(met)
    tr.to_csv(RESULTS / "tables/meteostat_trends.csv", index=False)
    cv = crossval(met)
    (RESULTS / "crossval.json").write_text(json.dumps(cv, indent=2, default=str))

    sig = tr.dropna(subset=["p_value"])
    warm = sig[(sig.variable.isin(["tavg", "tmax", "tmin"])) & (sig.tau > 0) & (sig.p_value < 0.05)]
    prec = sig[sig.variable == "prcp"]
    summary = {
        "n_stations_daily": int(met.station.nunique()),
        "record_start": str(met.date.min().date()), "record_end": str(met.date.max().date()),
        "years_covered": int(met.date.dt.year.nunique()),
        "n_daily_rows": int(len(met)),
        "n_trend_tests": int(len(sig)),
        "n_significant": int((sig.p_value < 0.05).sum()),
        "temp_warming_sig": int(len(warm)),
        "temp_tests": int(len(sig[sig.variable.isin(['tavg','tmax','tmin'])])),
        "precip_sig": int((prec.p_value < 0.05).sum()),
        "precip_tests": int(len(prec)),
        "strongest_warming": warm.sort_values("tau", ascending=False)
            [["name", "variable", "tau", "p_value", "sens_slope", "first", "last"]].head(6).to_dict("records"),
        "precip_trends": prec.sort_values("tau")
            [["name", "tau", "p_value", "sens_slope", "first", "last"]].head(10).to_dict("records"),
        "crossval": cv,
    }
    (RESULTS / "meteostat_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"stations={summary['n_stations_daily']}  rows={summary['n_daily_rows']}  "
          f"years={summary['years_covered']} ({summary['record_start']}..{summary['record_end']})")
    print(f"trend tests={summary['n_trend_tests']}  significant={summary['n_significant']}  "
          f"temp warming sig={summary['temp_warming_sig']}/{summary['temp_tests']}  "
          f"precip sig={summary['precip_sig']}/{summary['precip_tests']}")
    print("\nstrongest warming:\n", warm.sort_values('tau', ascending=False)
          [['name','variable','tau','p_value','sens_slope']].head(8).to_string(index=False))
    print("\ncross-val sample:", json.dumps({k: v for k, v in list(cv.items())[:4]}, indent=1))


if __name__ == "__main__":
    main()
