"""Cross-check this paper's drought signal against the World Bank portal series.

The framework asks for validation against the World Bank Climate Portal. The
portal website refuses non-browser clients, but its indicator programme is
served by api.worldbank.org, which fetch_wb_climate.py pulls: a national annual
SPEI (EN.CLC.SPEI.XD, 1990-2023), a long-term precipitation normal
(AG.LND.PRCP.MM), an EM-DAT-derived population-affected share (EN.CLC.MDAT.ZS)
and CCDR/ThinkHazard! exposure values.

What is compared, and what is NOT claimed
----------------------------------------
1. Year-to-year agreement between the Bank's SPEI and the SPEI we derive from
   ERA5. The two are built from different inputs, different PET methods and
   different aggregation, so only their RELATIVE movement is comparable: Pearson
   and Spearman correlation on shared years, and agreement on wet/dry sign
   against each series' own mean. Level equality is not asserted.
2. Whether the Bank's driest national years coincide with the drought entries in
   the disaster register. This is the aggregation test: a national annual index
   averages away the sub-national, seasonal dryness that actually damages crops,
   so a register drought can be invisible in it. We report exactly which years
   each source calls dry rather than forcing them to agree.
3. The reanalysis level check: our district-mean annual precipitation against
   the Bank's published long-term normal.

Output: results/wb_climate_check.json, results/tables/wb_precip_compare.csv,
results/tables/wb_spei_compare.csv (the two small tables the figure reads)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

WB_CSV = C.DATA / "wb_climate.csv"
WB_META = C.DATA / "wb_climate_meta.json"
SPEI_CSV = C.RESULTS / "tables" / "spei_era5.csv"
OUT = C.RESULTS / "wb_climate_check.json"
OUT_TABLE = C.RESULTS / "tables" / "wb_precip_compare.csv"
OUT_SPEI_TABLE = C.RESULTS / "tables" / "wb_spei_compare.csv"


def ours_annual():
    """National annual aggregates of the ERA5-derived SPEI we compute ourselves."""
    sp = pd.read_csv(SPEI_CSV, parse_dates=["month"])
    sp["year"] = sp.month.dt.year
    g = sp.groupby("year")
    return dict(mean=g.SPEI6.mean(), min_=g.SPEI6.min(),
                spi_mean=g.SPI6.mean(), spi_min=g.SPI3.min())


def agreement(wb, ours):
    yrs = sorted(set(wb.index) & set(ours.index))
    a = wb.loc[yrs].values
    b = ours.loc[yrs].values
    pear = stats.pearsonr(a, b)
    spear = stats.spearmanr(a, b)
    sa = np.sign(a - a.mean())
    sb = np.sign(b - b.mean())
    return dict(n_years=len(yrs), years=[int(y) for y in yrs],
                pearson_r=round(float(pear.statistic), 3),
                pearson_p=float(pear.pvalue),
                spearman_rho=round(float(spear.statistic), 3),
                spearman_p=float(spear.pvalue),
                sign_agreement=round(float((sa == sb).mean()), 3),
                mean_abs_diff=round(float(np.mean(np.abs(a - b))), 3),
                wb_range=[round(float(a.min()), 2), round(float(a.max()), 2)],
                ours_range=[round(float(b.min()), 2), round(float(b.max()), 2)],
                reading=("correlation of year-to-year movement; the two series are "
                         "not on the same scale or inputs, so the level difference "
                         "is not an error term"))


def dry_year_check(wb, reg, ours_min):
    """Which years each source calls dry, and where the register agrees."""
    d = wb.sort_values().head(5)
    dry_rows = []
    for y, v in d.items():
        ev = reg[(reg.type.str.lower() == "drought") &
                 (reg.start.dt.year.between(y - 1, y + 1))]
        dry_rows.append(dict(year=int(y), wb_spei=round(float(v), 2),
                             ours_spei6_min=(round(float(ours_min[y]), 2)
                                             if y in ours_min.index else None),
                             register_drought_same_or_adjacent_year=
                             [f"{r.event_id} {r.start.date()} {r.type}"
                              for _, r in ev.iterrows()],
                             in_evaluation_window=bool(
                                 C.EVAL_TRAIN_YEARS[0] <= y <= C.EVAL_TEST_YEARS[1])))
    severe_any = wb[wb <= -1.0]
    in_win = wb[(wb.index >= C.EVAL_TRAIN_YEARS[0]) &
                (wb.index <= C.EVAL_TEST_YEARS[1])]
    return dict(wb_driest5=dry_rows,
                wb_years_le_minus_1=[int(y) for y in severe_any.index],
                wb_min_in_evaluation_window=round(float(in_win.min()), 2),
                wb_argmin_in_evaluation_window=int(in_win.idxmin()),
                register_drought_years=[int(y) for y in sorted(
                    set(reg[reg.type.str.lower() == "drought"].start.dt.year))],
                reading=("the Bank's index is a national annual mean: it can sit "
                         "above the -1 severe-drought threshold in a year the "
                         "register records droughts, because those droughts are "
                         "seasonal and district-local"))


def district_year_precip():
    """ERA5 district-year precipitation totals, complete district-years only.

    A district-year counts only when it has at least 360 daily records, which
    removes the truncated first and last years of the panel rather than letting a
    partial year deflate the mean.
    """
    e = pd.read_csv(C.DATA / "era5_daily.csv", parse_dates=["date"])
    e["year"] = e.date.dt.year
    counts = e.groupby(["location", "year"]).size().rename("n").reset_index()
    ann = (e.groupby(["location", "year"]).precipitation_sum.sum()
           .rename("mm").reset_index()).merge(counts, on=["location", "year"])
    return ann[ann.n >= 360]


def precip_year_table():
    """National (district-mean) annual precipitation used by the level check.

    Shared with the figure so the two can never disagree about what "the ERA5
    annual mean" means.
    """
    return (district_year_precip().groupby("year").mm.mean()
            .rename("era5_mean_mm"))


def precip_check(meta):
    """ERA5 district-mean annual precipitation vs the Bank's long-term normal.

    Only years with a complete daily record for a district are used, so a
    truncated final year cannot deflate the mean. The district spread is
    reported too: the 16 centroids are not a random sample of Ugandan land, they
    are the network's own sites, and several sit in the wet lake-region highlands.
    """
    full = district_year_precip()
    by_year = full.groupby("year").mm.mean()
    by_loc = full.groupby("location").mm.mean().sort_values()
    code = "AG.LND.PRCP.MM"
    entry = meta.get(code, {})
    vals = [v for v in (entry.get("values") or {}).values()]
    normal = float(np.mean(vals)) if vals else None
    return dict(wb_normal_mm=round(normal, 1) if normal else None,
                wb_normal_years=len(vals),
                wb_constant_across_years=bool(
                    len(vals) and max(vals) - min(vals) < 1e-9),
                wb_source=(entry.get("metadata") or {}).get("source", {}).get("value"),
                era5_mean_annual_mm=round(float(by_year.mean()), 1),
                era5_complete_year_loc_rows=int(len(full)),
                era5_years=[int(by_year.index.min()), int(by_year.index.max())],
                era5_year_min=round(float(by_year.min()), 1),
                era5_year_max=round(float(by_year.max()), 1),
                era5_driest_site=[by_loc.index[0], round(float(by_loc.iloc[0]), 1)],
                era5_wettest_site=[by_loc.index[-1], round(float(by_loc.iloc[-1]), 1)],
                pct_of_normal=round(float(100 * by_year.mean() / normal), 1)
                if normal else None,
                reading=("different products, different baselines, different "
                         "sampling of the country: this is a scale sanity check on "
                         "the reanalysis, not a validation of either number"))


def context(meta):
    out = {}
    for code, entry in meta.items():
        vals = entry.get("values") or {}
        md = entry.get("metadata") or {}
        out[code] = dict(name=entry.get("name"), label=md.get("name"),
                         source=(md.get("source") or {}).get("value"),
                         years={k: v for k, v in vals.items()},
                         n_values=len(vals))
    return out


def main():
    if not WB_CSV.exists():
        raise SystemExit("data/wb_climate.csv missing -- run fetch_wb_climate.py")
    wb = pd.read_csv(WB_CSV).set_index("year").wb_spei
    meta = json.loads(WB_META.read_text()) if WB_META.exists() else {}
    reg = pd.read_csv(C.DATA / "disaster_register.csv", parse_dates=["start", "end"])
    ours = ours_annual()

    res = dict(
        source_note=("World Bank indicator series retrieved through "
                     "api.worldbank.org by fetch_wb_climate.py; the Climate "
                     "Change Knowledge Portal pages themselves return HTTP 403 "
                     "to this environment, so no value here was read off a "
                     "portal chart."),
        agreement_vs_ours_spei6_mean=agreement(wb, ours["mean"]),
        agreement_vs_ours_spei6_min=agreement(wb, ours["min_"]),
        dry_year_check=dry_year_check(wb, reg, ours["min_"]),
        precip_normal_check=precip_check(meta),
        portal_indicators=context(meta),
    )
    OUT.write_text(json.dumps(res, indent=2, default=str))

    # Small joinable tables so the figure and the prose read the same numbers.
    drought_years = set(res["dry_year_check"]["register_drought_years"])
    sp = pd.DataFrame({"year": sorted(set(wb.index) | set(ours["mean"].index))})
    sp["wb_spei"] = sp.year.map(wb).round(3)
    sp["ours_spei6_mean"] = sp.year.map(ours["mean"]).round(3)
    sp["ours_spei6_min"] = sp.year.map(ours["min_"]).round(3)
    sp["register_drought_year"] = sp.year.isin(drought_years).astype(int)
    sp.to_csv(OUT_SPEI_TABLE, index=False)
    pt = precip_year_table().reset_index()
    pt["wb_normal_mm"] = res["precip_normal_check"]["wb_normal_mm"]
    pt.to_csv(OUT_TABLE, index=False)
    print(f"saved comparison tables -> {OUT_SPEI_TABLE.name}, {OUT_TABLE.name}")

    a = res["agreement_vs_ours_spei6_mean"]
    print(f"WB SPEI vs our annual SPEI6 mean: n={a['n_years']} "
          f"r={a['pearson_r']} (p={a['pearson_p']:.3g}) "
          f"rho={a['spearman_rho']} sign={a['sign_agreement']}")
    b = res["agreement_vs_ours_spei6_min"]
    print(f"WB SPEI vs our annual SPEI6 MIN : n={b['n_years']} "
          f"r={b['pearson_r']} (p={b['pearson_p']:.3g}) rho={b['spearman_rho']}")
    d = res["dry_year_check"]
    print("WB driest 5:", [(r["year"], r["wb_spei"]) for r in d["wb_driest5"]])
    print("register drought years:", d["register_drought_years"])
    print("WB years <= -1.0:", d["wb_years_le_minus_1"],
          "| WB min in eval window:", d["wb_min_in_evaluation_window"],
          f"({d['wb_argmin_in_evaluation_window']})")
    p = res["precip_normal_check"]
    print(f"precip: WB normal {p['wb_normal_mm']} mm vs ERA5 mean "
          f"{p['era5_mean_annual_mm']} mm = {p['pct_of_normal']}% of normal")


if __name__ == "__main__":
    main()
