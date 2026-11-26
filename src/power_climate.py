"""Independent gap-fill analysis of the NASA POWER record (1988-2007).

Closes the 1990-2007 multivariate climate gap that neither Meteostat (stations
stopped filing) nor Open-Meteo ERA5 (starts 2008) can cover, using an
independent MERRA-2 reanalysis. Everything here is computed from real fetched
data; no values are imputed or invented.

Products
--------
1. Reliability check: POWER vs Meteostat in-situ monthly tavg over their overlap
   (1988-1989), nearest co-located point -> Pearson r. Establishes POWER is a
   trustworthy gap-fill before we use it.
2. Continuous spliced temperature series 1988-2023 (POWER 1988-2007 + ERA5
   2008-2023), anomalies vs 1991-2020 base, Mann-Kendall tau + Sen slope per
   district and national mean -> warming trend over a now-continuous 36-yr record.
3. Tier-1 variable trends over the dark period (POWER only, 1988-2007):
   relative humidity and surface shortwave -> is the atmosphere drying?

Outputs: results/tables/power_trends.csv, results/power_summary.json,
         results/tables/power_temp_continuous_monthly.csv
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


def sens_slope(y):
    y = np.asarray(y, float)
    n = len(y)
    sl = [(y[j]-y[i])/(j-i) for i in range(n) for j in range(i+1, n)]
    return float(np.median(sl)) if sl else np.nan


def monthly_mean(df, col, datecol="date", loc_col="location"):
    d = df[[loc_col, datecol, col]].dropna().copy()
    d[datecol] = pd.to_datetime(d[datecol])
    d["ym"] = d[datecol].dt.to_period("M").dt.to_timestamp()
    return d.groupby([loc_col, "ym"])[col].mean()


def power_reliability(power, met):
    """Pearson r between POWER T2M and Meteostat in-situ tavg, 1988-1989 overlap."""
    out = {}
    locs = {name: (lat, lon) for (_r, name, lat, lon, _e) in C.LOCATIONS}
    met = met.copy()
    met["ym"] = pd.to_datetime(met.date).dt.to_period("M").dt.to_timestamp()
    pm = monthly_mean(power, "t2m_c")
    for sid, g in met.groupby("station"):
        name = str(g["name"].iloc[0])
        g2 = g.dropna(subset=["tavg"]).copy()
        g2 = g2[g2.ym >= "1988-01-01"]                       # overlap only
        if len(g2) < 12:
            continue
        # nearest POWER district by haversine
        lat0, lon0 = float(g2.lat.iloc[0]), float(g2.lon.iloc[0])
        if not (np.isfinite(lat0) and np.isfinite(lon0)):
            continue
        def hav(L):
            la1, lo1 = np.radians(lat0), np.radians(lon0)
            la2, lo2 = np.radians(locs[L][0]), np.radians(locs[L][1])
            h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
            return 2*6371*np.arcsin(np.sqrt(h))
        near = min(locs, key=hav)
        st = g2.groupby("ym")["tavg"].mean()
        pw = pm.xs(near, level="location")
        j = pd.concat([st.rename("station"), pw.rename("power")], axis=1).dropna()
        if len(j) > 12 and j.std().all() > 0:
            out[f"{name}~{near}"] = dict(n_months=int(len(j)),
                r=round(float(np.corrcoef(j.station, j.power)[0, 1]), 3),
                bias=round(float((j.power - j.station).mean()), 2))
    return out


def spliced_temp(power, era5):
    """Build district monthly t2m series: POWER(<=2007) then ERA5(>=2008)."""
    p = monthly_mean(power, "t2m_c").rename("t").reset_index()
    p["source"] = "POWER"
    e = monthly_mean(era5, "temperature_2m_mean").rename("t").reset_index()
    e["source"] = "ERA5"
    e = e.rename(columns={"ym": "ym"}) if "ym" in e else e
    comb = pd.concat([p, e], ignore_index=True)
    comb["ym"] = pd.to_datetime(comb["ym"])
    comb = comb.drop_duplicates(subset=["location", "ym"], keep="first").sort_values(["location", "ym"])
    return comb


def district_trends(series, var_name, source):
    rows = []
    for loc, g in series.groupby("location"):
        v = g["t"].values.astype(float)
        v = v[np.isfinite(v)]
        if len(v) < 60:
            continue
        h = mk.original_test(v)
        base = v[(g["ym"].dt.year >= 1991).values & (g["ym"].dt.year <= 2020).values]
        base = base[np.isfinite(base)]
        rows.append(dict(location=loc, variable=var_name, source=source,
                         n_years=round(len(v)/12, 1), tau=round(float(h.Tau), 3),
                         p_value=float(h.p), trend=h.trend,
                         sens_per_month=round(sens_slope(v), 4),
                         span=f"{g['ym'].min().date()}..{g['ym'].max().date()}"))
    return rows


def main():
    power = pd.read_csv(DATA / "power_daily.csv", parse_dates=["date"])
    era5 = pd.read_csv(DATA / "era5_daily.csv", parse_dates=["date"])
    met = pd.read_csv(DATA / "meteostat_daily.csv", parse_dates=["date"])

    print("== POWER reliability vs Meteostat (1988-89 overlap) ==")
    rel = power_reliability(power, met)
    for k, v in rel.items():
        print(f"  {k:24s} r={v['r']:+.3f}  n={v['n_months']}  bias={v['bias']:+.2f}C")

    # continuous spliced temperature
    comb = spliced_temp(power, era5)
    nat = comb.groupby("ym")["t"].mean().rename("t2m_c").reset_index()
    nat.to_csv(RESULTS / "tables/power_temp_continuous_monthly.csv", index=False)

    temp_rows = district_trends(comb, "t2m", "POWER+ERA5")
    # segment-wise temperature trends (avoid cross-product splice artefacts):
    #   POWER-only 1988-2007 and ERA5-only 2008-2023
    tp = monthly_mean(power, "t2m_c").rename("t").reset_index()
    te = monthly_mean(era5, "temperature_2m_mean").rename("t").reset_index()
    temp_power = district_trends(tp, "t2m", "POWER")
    temp_era5 = district_trends(te, "t2m", "ERA5")
    # dark-period humidity & solar (POWER only)
    rh = monthly_mean(power, "rh2m_pct").rename("t").reset_index()
    sw = monthly_mean(power, "sw_kwh_m2").rename("t").reset_index()
    rh_rows = district_trends(rh, "rh2m", "POWER")
    sw_rows = district_trends(sw, "shortwave", "POWER")

    tr = pd.DataFrame(temp_rows + temp_power + temp_era5 + rh_rows + sw_rows)
    tr.to_csv(RESULTS / "tables/power_trends.csv", index=False)

    # national-mean temperature trend over continuous record
    natv = nat["t2m_c"].values.astype(float)
    natmk = mk.original_test(natv)
    def _sig(rows):
        w = [r for r in rows if r["p_value"] < 0.05 and r["tau"] > 0]
        c = [r for r in rows if r["p_value"] < 0.05 and r["tau"] < 0]
        return len(w), len(c), (round(float(np.mean([r["tau"] for r in rows])), 3) if rows else None)
    pw_w, pw_c, pw_mt = _sig(temp_power)
    e5_w, e5_c, e5_mt = _sig(temp_era5)
    warm = [r for r in temp_rows if r["p_value"] < 0.05 and r["tau"] > 0]
    drying = [r for r in rh_rows if r["p_value"] < 0.05 and r["tau"] < 0]
    summary = {
        "power_span": f"{power.date.min().date()}..{power.date.max().date()}",
        "power_districts": int(power.location.nunique()),
        "power_daily_rows": int(len(power)),
        "reliability": rel,
        "reliability_mean_r": round(float(np.mean([v["r"] for v in rel.values()])), 3) if rel else None,
        "continuous_temp_span": f"{nat['ym'].min().date()}..{nat['ym'].max().date()}",
        "national_temp_tau": round(float(natmk.Tau), 3),
        "national_temp_p": float(natmk.p),
        "national_temp_sens_per_decade_C": round(sens_slope(natv)*120, 3),
        "districts_warming_sig": len(warm),
        "districts_temp_tested": len(temp_rows),
        "temp_power_segment": {"n_warm_sig": pw_w, "n_cool_sig": pw_c, "mean_tau": pw_mt,
                               "span": "1988-2007"},
        "temp_era5_segment": {"n_warm_sig": e5_w, "n_cool_sig": e5_c, "mean_tau": e5_mt,
                              "span": "2008-2023"},
        "districts_humidity_drying_sig": len(drying),
        "districts_humidity_tested": len(rh_rows),
        "rh_mean_tau": round(float(np.mean([r["tau"] for r in rh_rows])), 3) if rh_rows else None,
        "sw_mean_tau": round(float(np.mean([r["tau"] for r in sw_rows])), 3) if sw_rows else None,
    }
    (RESULTS / "power_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n== Continuous spliced temperature {summary['continuous_temp_span']} ==")
    print(f"  national MK tau={summary['national_temp_tau']} p={summary['national_temp_p']:.2e}  "
          f"Sen={summary['national_temp_sens_per_decade_C']:+.3f} C/decade")
    print(f"  districts warming (sig): {len(warm)}/{len(temp_rows)}")
    print(f"\n== Segment-wise temperature trends (avoid splice artefact) ==")
    print(f"  POWER 1988-2007: warm sig {pw_w}/{len(temp_power)}  mean tau={pw_mt}")
    print(f"  ERA5  2008-2023: warm sig {e5_w}/{len(temp_era5)}  mean tau={e5_mt}")
    print(f"\n== Dark-period (1988-2007) Tier-1 trends ==")
    print(f"  humidity: drying sig {len(drying)}/{len(rh_rows)}  mean tau={summary['rh_mean_tau']}")
    print(f"  shortwave: mean tau={summary['sw_mean_tau']}")
    print(f"\nsaved -> power_trends.csv, power_summary.json, power_temp_continuous_monthly.csv")


if __name__ == "__main__":
    main()
