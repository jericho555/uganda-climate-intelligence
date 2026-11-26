"""Novel impact analysis: does growing-season drought explain cereal yield?

Closes the loop the reviewer asked for -- from "we detected anomalies" to
"the anomalies explain agricultural outcomes". Everything is genuinely
computed from the released pipeline + World Bank production data.

Method
------
1. Yield anomaly: log(cereal yield) detrended against a full-series linear
   time trend (removes 60 yr of intensification), then expressed as % deviation.
2. Growing-season moisture index per calendar year, from two independent
   substrates (kept SEPARATE because they are different observation systems
   and must not be pooled onto one scale):
     * METEOSTAT century SPI-3 (meteorological), station-averaged, 1961-1989
       (the well-covered pre-collapse part of the in-situ record).
     * ERA5 SPI-3 / SPEI-3 (agricultural), district-averaged, 2008-2023.
   Uganda is bimodal: main cereals grow in the 'long rains' (Mar-May); we use
   the long-rains seasonal mean and also the combined (Mar-May + Sep-Nov).
3. Correlate yield anomaly vs seasonal index (Pearson + Spearman + OLS slope),
   and contrast against the documented drought years in the register.

Outputs: results/ag_linkage.json, results/tables/ag_yield_climate.csv
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS
from scipy import stats

LONG_RAINS = [3, 4, 5]          # main cereal season
SECOND_RAINS = [9, 10, 11]


def yield_anomaly(path):
    y = pd.read_csv(path)
    y["logy"] = np.log(y["cereal_yield_kgha"].astype(float))
    # linear de-trend over the FULL series -> residual anomaly (log units)
    slope, intercept, *_ = stats.linregress(y["year"], y["logy"])
    y["trend"] = intercept + slope * y["year"]
    y["yield_anom_pct"] = 100.0 * (y["logy"] - y["trend"])   # ~ % deviation
    y["yield_trend_pct_yr"] = 100.0 * (np.exp(slope) - 1.0)
    return y


def national_seasonal(spi, cols, months_by_season):
    """spi: station x month SPI frame -> {season: year -> mean index}.
    Requires every season month present for a year after station-averaging."""
    d = spi.copy()
    d["month"] = pd.to_datetime(d["month"])
    d["year"] = d.month.dt.year
    d["m"] = d.month.dt.month
    out = {}
    for season, mlist in months_by_season.items():
        for col in cols:
            if col not in d.columns:
                continue
            sub = d[d.m.isin(mlist)].dropna(subset=[col])
            # station-average per year-month, then require all season months
            ym = sub.groupby(["year", "m"])[col].mean().reset_index()
            g = ym.groupby("year").agg(n=("m", "count"), val=(col, "mean"))
            ser = g[g.n == len(mlist)]["val"]
            out[f"{season}:{col}"] = {int(y): round(float(v), 4)
                                      for y, v in ser.items()}
    return out


def link(y, index_by_year):
    """correlate yield anomaly with a year->index mapping over overlap."""
    yrs = sorted(set(y.year) & set(index_by_year))
    if len(yrs) < 6:
        return {"n": len(yrs), "note": "insufficient overlap"}
    ya = y.set_index("year").loc[yrs, "yield_anom_pct"].values
    xi = np.array([index_by_year[yy] for yy in yrs])
    pr, pp = stats.pearsonr(xi, ya)
    sr, sp = stats.spearmanr(xi, ya)
    slope, intercept, *_ = stats.linregress(xi, ya)
    return {"n": len(yrs), "years": f"{yrs[0]}-{yrs[-1]}",
            "pearson_r": round(float(pr), 3), "pearson_p": float(f"{pp:.3g}"),
            "spearman_r": round(float(sr), 3), "spearman_p": float(f"{sp:.3g}"),
            "slope_pct_per_sd": round(float(slope), 2),
            "wet_minus_dry_yieldgap_pct": round(float(slope) * 2.0, 2)}


def main():
    y = yield_anomaly(DATA / "uganda_crop_yield.csv")
    spi = pd.read_csv(RESULTS / "tables/spi_meteostat.csv")
    es = pd.read_csv(RESULTS / "tables/spei_era5.csv")

    seasons = {"long_rains": LONG_RAINS,
               "bimodal": LONG_RAINS + SECOND_RAINS}

    # METEOSTAT century SPI-3 -> restrict to well-covered pre-collapse years
    m_nat = national_seasonal(spi, ["SPI3"], seasons)
    pre = {k: {yy: v for yy, v in d.items() if 1961 <= yy <= 1989}
           for k, d in m_nat.items()}

    # ERA5 modern SPI-3 / SPEI-3 (2008-2023)
    e_nat = national_seasonal(es, ["SPI3", "SPEI3"], seasons)
    mod = {k: {yy: v for yy, v in d.items() if 2008 <= yy <= 2023}
           for k, d in e_nat.items()}

    results = {"yield_trend_pct_per_year": round(float(y["yield_trend_pct_yr"].iloc[0]), 2),
               "yield_years": [int(y.year.min()), int(y.year.max())],
               "climate_gap_note": "no continuous in-situ or reanalysis daily "
                                   "moisture 1990-2007 (station reporting collapse)",
               "meteostat_century_linkage": {},
               "era5_modern_linkage": {},
               "worst_drought_years_vs_yield": {}}

    for k, d in pre.items():
        if d:
            results["meteostat_century_linkage"][k] = link(y, d)
    for k, d in mod.items():
        if d:
            results["era5_modern_linkage"][k] = link(y, d)

    # documented drought events vs the yield anomaly that year
    reg = pd.read_csv(DATA / "disaster_register.csv")
    dr = reg[reg["type"].str.lower().str.contains("drought", na=False)]
    ya = y.set_index("year")["yield_anom_pct"]
    checks = []
    for _, e in dr.iterrows():
        yy = pd.to_datetime(str(e["start"])).year
        if yy in ya.index and np.isfinite(ya[yy]):
            checks.append(dict(event=e["event_id"], year=yy,
                               yield_anom_pct=round(float(ya[yy]), 1),
                               regions=e["regions"]))
    results["drought_events_yield_anomaly"] = checks

    # save tidy merge table
    rows = []
    for _, r in y.iterrows():
        yy = int(r.year)
        rows.append(dict(year=yy, yield_kgha=r.cereal_yield_kgha,
                         yield_anom_pct=round(float(r.yield_anom_pct), 2),
                         met_SPI3_longrains=pre.get("long_rains:SPI3", {}).get(yy),
                         era_SPI3_longrains=mod.get("long_rains:SPI3", {}).get(yy),
                         era_SPEI3_longrains=mod.get("long_rains:SPEI3", {}).get(yy)))
    pd.DataFrame(rows).to_csv(RESULTS / "tables/ag_yield_climate.csv", index=False)

    (RESULTS / "ag_linkage.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"yield trend: +{results['yield_trend_pct_per_year']}%/yr "
          f"({results['yield_years'][0]}-{results['yield_years'][1]})")
    print("\nMETEOSTAT century linkage (1961-1989):")
    for k, v in results["meteostat_century_linkage"].items():
        print("  ", k, v)
    print("\nERA5 modern linkage (2008-2023):")
    for k, v in results["era5_modern_linkage"].items():
        print("  ", k, v)
    print("\ndocumented droughts vs yield anomaly:")
    for c in checks:
        print(f"  {c['event']} {c['year']}  yield_anom={c['yield_anom_pct']}%  ({c['regions']})")


if __name__ == "__main__":
    main()
