"""Figures for the NASA POWER gap-fill and the drought/yield linkage.

Drawn only from files produced by fetch_power/power_climate/drought_indices/
ag_linkage; no values are hard-coded. Output -> manuscript/figures/ (where main.tex
reads them):
  fig_gapfill.png       : continuous 1988-2023 temperature anomaly (dark period now
                          filled by POWER) + 1988-2007 humidity-drying trend bars.
  fig_drought_yield.png : century drought chronology (Meteostat+ERA5 SPI/SPEI) and
                          detrended cereal yield vs growing-season drought index.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})
MF = C.MANUSCRIPT / "figures"
MF.mkdir(parents=True, exist_ok=True)


def sens_slope(y):
    y = np.asarray(y, float); n = len(y)
    sl = [(y[j]-y[i])/(j-i) for i in range(n) for j in range(i+1, n)]
    return float(np.median(sl)) if sl else np.nan


def fig_gapfill():
    tc = pd.read_csv(MF.parent.parent / "results/tables/power_temp_continuous_monthly.csv",
                     parse_dates=["ym"])
    base = tc[(tc.ym.dt.year >= 1991) & (tc.ym.dt.year <= 2020)].t2m_c.mean()
    tc["anom"] = tc.t2m_c - base
    # 12-month running mean to make the multi-decadal signal legible
    tc["sm12"] = tc.anom.rolling(12, center=True, min_periods=6).mean()
    pt = pd.read_csv(MF.parent.parent / "results/tables/power_trends.csv")
    rh = pt[(pt.variable == "rh2m")].sort_values("tau")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.axvspan(pd.Timestamp("1988-01-01"), pd.Timestamp("2007-12-01"),
               color="#f4c2c2", alpha=0.35, label="NASA POWER (gap filled)")
    ax.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2023-12-01"),
               color="#cfe0f3", alpha=0.45, label="ERA5 (Open-Meteo)")
    ax.plot(tc.ym, tc.anom, color="#bbbbbb", lw=0.6, alpha=0.7, label="monthly anomaly")
    ax.plot(tc.ym, tc.sm12, color="#2c7fb8", lw=1.8, label="12-mo running mean")
    x = np.arange(len(tc)); mk = stats.linregress(x, tc.anom.values)
    ax.plot(tc.ym, mk.intercept + mk.slope*x, "k--", lw=1.2,
            label=f"linear trend ({mk.slope*12:+.3f} \u00b0C/yr)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("temperature anomaly (\u00b0C, 1991-2020 base)")
    ax.set_title("Continuous national temperature, 1988-2023\n(dark period 1990-2007 filled by independent reanalysis)")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    sig = (rh.p_value < 0.05).values
    col = ["#d7301f" if (s and t < 0) else ("#4575b4" if s else "#bbbbbb") for s, t in zip(sig, rh.tau)]
    ax.barh(rh.location, rh.tau, color=col)
    ax.axvline(0, color="k", lw=0.6)
    nd = int(((rh.p_value < 0.05) & (rh.tau < 0)).sum())
    ax.set_xlabel("Mann-Kendall \u03c4 (relative humidity)")
    ax.set_title(f"Near-surface humidity, 1988-2007\n{nd}/{len(rh)} districts significantly drying")
    fig.tight_layout()
    fig.savefig(MF / "fig_gapfill.png"); plt.close(fig)
    return f"gap-fill: {nd}/{len(rh)} drying"


def fig_drought_yield():
    spi = pd.read_csv(MF.parent.parent / "results/tables/spi_meteostat.csv", parse_dates=["month"])
    spei = pd.read_csv(MF.parent.parent / "results/tables/spei_era5.csv", parse_dates=["month"])
    cy = pd.read_csv(C.DATA / "uganda_crop_yield.csv")

    # ---- panel A: annual minimum drought index, century chronology ----
    def annual_min(df, col, lo, hi):
        d = df[(df["month"].dt.year >= lo) & (df["month"].dt.year <= hi)]
        return d.groupby(d["month"].dt.year)[col].min()
    met = annual_min(spi, "SPI3", 1961, 1989)      # in-situ century block
    era = annual_min(spei, "SPEI3", 2008, 2023)    # reanalysis modern block
    reg = pd.read_csv(C.DATA / "disaster_register.csv", parse_dates=["start"])
    drought_years = sorted(set(reg[reg.type.str.contains("drought", case=False, na=False)].start.dt.year))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    ax.bar(met.index, met.values, color="#7b3294", label="Meteostat SPI\u2083 (1961-89, in-situ)")
    ax.bar(era.index, era.values, color="#2c7fb8", label="ERA5 SPEI\u2083 (2008-23, reanalysis)")
    ax.axhline(-1.5, color="#d7301f", ls="--", lw=1, label="exceptional drought (\u2264-1.5)")
    for y in drought_years:
        if 1961 <= y <= 2023:
            ax.axvline(y, color="#fdae61", alpha=0.4, lw=0.8)
    ax.annotate("register droughts", (2018, 0.4), fontsize=7, color="#b15a00")
    ax.set_ylabel("annual minimum index")
    ax.set_title("Century drought chronology (most-negative growing-season index/yr)")
    ax.legend(fontsize=6.5, loc="lower left")

    # ---- panel B: detrended yield vs growing-season drought ----
    cy = cy[(cy.year >= 1961) & (cy.year <= 2024)].copy()
    ly = np.log(cy.cereal_yield_kgha.values)
    fit = stats.linregress(cy.year.values, ly)
    cy["yield_anom_pct"] = (ly - (fit.intercept + fit.slope*cy.year.values))*100

    def season_index(df, col, months=(3, 4, 5)):
        d = df[df["month"].dt.month.isin(months)]
        return d.groupby(d["month"].dt.year)[col].mean()
    s_met = season_index(spi, "SPI3")
    s_era = season_index(spei, "SPEI3")

    ax = axes[1]
    pts = []
    for yr, v in s_met.items():
        if yr in set(cy.year):
            pts.append((yr, v, cy.loc[cy.year == yr, "yield_anom_pct"].iloc[0], "century"))
    for yr, v in s_era.items():
        if yr in set(cy.year):
            pts.append((yr, v, cy.loc[cy.year == yr, "yield_anom_pct"].iloc[0], "modern"))
    for blk, colr, lab in [("century", "#7b3294", "Meteostat SPI\u2083 (1961-86)"),
                           ("modern", "#2c7fb8", "ERA5 SPEI\u2083 (2008-23)")]:
        s = [p for p in pts if p[3] == blk]
        if s:
            ax.scatter([p[1] for p in s], [p[2] for p in s], c=colr, s=28, alpha=0.8, label=lab)
    # fit + annotate the worst year
    xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
    if len(xs) > 4:
        lr = stats.linregress(xs, ys)
        xx = np.linspace(xs.min(), xs.max(), 20)
        ax.plot(xx, lr.intercept + lr.slope*xx, "k--", lw=1,
                label=f"pooled r={lr.rvalue:.2f}, p={lr.pvalue:.2f}")
        ax.set_xlabel("growing-season drought index (long-rains SPI\u2083/SPEI\u2083)")
    worst = cy.loc[cy.yield_anom_pct.idxmin()]
    ax.annotate(f"{int(worst.year)}: worst\nyield {worst.yield_anom_pct:.0f}%",
                (0.02, 0.02), xycoords="axes fraction", fontsize=7.5,
                color="#d7301f", bbox=dict(boxstyle="round", fc="#fff2f2", ec="#d7301f", lw=0.6))
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("detrended cereal yield anomaly (%)")
    ax.set_title("Yield anomaly vs drought: significant years,\nweak national-linear link (aggregation masks regional drought)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(MF / "fig_drought_yield.png"); plt.close(fig)
    return f"drought/yield: {len(pts)} year-points, {len(drought_years)} register droughts"


def main():
    print(fig_gapfill())
    print(fig_drought_yield())
    print("wrote:", sorted(p.name for p in MF.glob("fig_gapfill.png")) +
          sorted(p.name for p in MF.glob("fig_drought_yield.png")))


if __name__ == "__main__":
    main()
