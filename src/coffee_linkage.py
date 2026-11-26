"""Coffee impact analysis: does growing-season moisture explain Ugandan coffee?

Extends the cereal-yield impact layer (ag_linkage.py) to Uganda's signature
export crop -- robusta- and arabica-dominant coffee, grown by ~1.7M smallholders
in the central/south-western "coffee belt" and Mt Elgon. Everything is genuinely
computed: coffee production is the real FAOSTAT/OWID series fetched by
fetch_coffee.py; the climate predictors are the SAME validated SPI/SPEI tables
used for cereals. We reuse ag_linkage.national_seasonal / link unchanged so the
two crops are compared on a common methodological footing.

Coffee vs cereal differences handled honestly:
  * Outcome is national ANNUAL PRODUCTION (tonnes), not per-ha yield -- harvested
    area is not published on a reachable OWID slug, so we detrend log-production.
    The residual anomaly therefore mixes weather response with area/price cycles;
    we disclose this (it makes the linkage even more confounded than yield).
  * Coffee is PERENNIAL: fruit set/fill tracks rainfall over a long antecedent
    window, so we add an ANNUAL (Jan-Dec) seasonal aggregate alongside the
    long-rains and bimodal windows used for cereals.

Outputs: results/coffee_linkage.json, results/tables/coffee_prod_climate.csv
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
from ag_linkage import national_seasonal, link   # reuse validated helpers

LONG_RAINS = [3, 4, 5]
SECOND_RAINS = [9, 10, 11]
ANNUAL = list(range(1, 13))


def coffee_anomaly(path):
    d = pd.read_csv(path)
    d["logy"] = np.log(d["coffee_prod_t"].astype(float))
    slope, intercept, *_ = stats.linregress(d["year"], d["logy"])
    d["trend"] = intercept + slope * d["year"]
    # column named yield_anom_pct so ag_linkage.link() works unchanged;
    # it is a PRODUCTION anomaly here (see module docstring).
    d["yield_anom_pct"] = 100.0 * (d["logy"] - d["trend"])
    d["prod_trend_pct_yr"] = 100.0 * (np.exp(slope) - 1.0)
    return d


def main():
    cf = coffee_anomaly(DATA / "uganda_coffee.csv")
    spi = pd.read_csv(RESULTS / "tables/spi_meteostat.csv")
    es = pd.read_csv(RESULTS / "tables/spei_era5.csv")

    seasons = {"long_rains": LONG_RAINS,
               "bimodal": LONG_RAINS + SECOND_RAINS,
               "annual": ANNUAL}

    m_nat = national_seasonal(spi, ["SPI3"], seasons)
    pre = {k: {yy: v for yy, v in d.items() if 1961 <= yy <= 1989}
           for k, d in m_nat.items()}
    e_nat = national_seasonal(es, ["SPI3", "SPEI3"], seasons)
    mod = {k: {yy: v for yy, v in d.items() if 2008 <= yy <= 2023}
           for k, d in e_nat.items()}

    results = {
        "outcome": "national coffee production (tonnes), FAOSTAT/OWID",
        "yield_note": "per-ha yield unavailable (area not on a reachable OWID "
                      "slug); anomaly is of LOG-PRODUCTION, so it mixes weather "
                      "with area/price cycles and is more confounded than yield",
        "prod_trend_pct_per_year": round(float(cf["prod_trend_pct_yr"].iloc[0]), 2),
        "prod_years": [int(cf.year.min()), int(cf.year.max())],
        "prod_range_t": [round(float(cf.coffee_prod_t.min()), 0),
                         round(float(cf.coffee_prod_t.max()), 0)],
        "meteostat_century_linkage": {},
        "era5_modern_linkage": {},
        "drought_events_coffee_anomaly": [],
        "worst_coffee_years": [],
    }

    for k, d in pre.items():
        if d:
            results["meteostat_century_linkage"][k] = link(cf, d)
    for k, d in mod.items():
        if d:
            results["era5_modern_linkage"][k] = link(cf, d)

    # documented droughts vs coffee production anomaly that year
    reg = pd.read_csv(DATA / "disaster_register.csv")
    dr = reg[reg["type"].str.lower().str.contains("drought", na=False)]
    ya = cf.set_index("year")["yield_anom_pct"]
    for _, e in dr.iterrows():
        yy = pd.to_datetime(str(e["start"])).year
        if yy in ya.index and np.isfinite(ya[yy]):
            results["drought_events_coffee_anomaly"].append(dict(
                event=e["event_id"], year=yy,
                coffee_anom_pct=round(float(ya[yy]), 1), regions=e["regions"]))

    # worst coffee production-anomaly years + their season index (ERA5 modern)
    for _, r in cf.nsmallest(4, "yield_anom_pct").iterrows():
        yy = int(r.year)
        results["worst_coffee_years"].append(dict(
            year=yy, coffee_anom_pct=round(float(r.yield_anom_pct), 1),
            prod_t=float(r.coffee_prod_t),
            era_bimodal_SPI3=mod.get("bimodal:SPI3", {}).get(yy),
            era_bimodal_SPEI3=mod.get("bimodal:SPEI3", {}).get(yy)))

    # tidy merge table
    rows = []
    for _, r in cf.iterrows():
        yy = int(r.year)
        rows.append(dict(year=yy, coffee_prod_t=r.coffee_prod_t,
                         coffee_anom_pct=round(float(r.yield_anom_pct), 2),
                         met_SPI3_longrains=pre.get("long_rains:SPI3", {}).get(yy),
                         met_SPI3_annual=pre.get("annual:SPI3", {}).get(yy),
                         era_SPI3_bimodal=mod.get("bimodal:SPI3", {}).get(yy),
                         era_SPEI3_bimodal=mod.get("bimodal:SPEI3", {}).get(yy)))
    pd.DataFrame(rows).to_csv(RESULTS / "tables/coffee_prod_climate.csv", index=False)
    (RESULTS / "coffee_linkage.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"coffee production trend: +{results['prod_trend_pct_per_year']}%/yr "
          f"({results['prod_years'][0]}-{results['prod_years'][1]}), "
          f"range {results['prod_range_t']} t")
    print("\nMETEOSTAT century (1961-89):")
    for k, v in results["meteostat_century_linkage"].items():
        print("  ", k, v)
    print("\nERA5 modern (2008-23):")
    for k, v in results["era5_modern_linkage"].items():
        print("  ", k, v)
    print("\ndocumented droughts vs coffee anomaly:")
    for c in results["drought_events_coffee_anomaly"]:
        print(f"  {c['event']} {c['year']}  coffee_anom={c['coffee_anom_pct']}%")
    print("\nworst coffee years:")
    for w in results["worst_coffee_years"]:
        print(f"  {w['year']} {w['coffee_anom_pct']}%  SPI3bim={w['era_bimodal_SPI3']}")


if __name__ == "__main__":
    main()
