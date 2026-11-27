"""Coffee figure: national coffee production vs climate -- the confounding demo.

Drawn only from coffee_linkage.py outputs + the fetched coffee series; no values
are hard-coded. Output -> manuscript/figures/fig_coffee.png:

  panel A : coffee production (t) 1961-2024 with the fitted log-linear trend,
            worst negative-anomaly years shaded, documented droughts + the two
            record wet harvests (2020, 2022) annotated.
  panel B : Pearson r of detrended coffee anomaly vs seasonal moisture index,
            for two independent substrates (Meteostat century, ERA5 modern).
            The SIGN FLIPS between them -- the visual core of the finding: a
            coarse national crop aggregate is not a clean climate signal.
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

plt.rcParams.update({"font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3})            # dpi/bbox owned by config.save_figure
MF = C.MANUSCRIPT / "figures"
MF.mkdir(parents=True, exist_ok=True)


def fig_coffee():
    cf = pd.read_csv(C.DATA / "uganda_coffee.csv")
    res = json.loads((C.RESULTS / "coffee_linkage.json").read_text())
    slope_pct = res["prod_trend_pct_per_year"]

    # recompute the log-linear trend exactly as coffee_linkage did
    ly = np.log(cf.coffee_prod_t.values)
    slope, intercept, *_ = stats.linregress(cf.year.values, ly)
    trend = intercept + slope * cf.year.values
    anom = 100.0 * (ly - trend)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- panel A ----
    ax0.plot(cf.year, cf.coffee_prod_t, "-o", ms=3, lw=1.3, color="#6b3f1d",
             label="coffee production")
    ax0.plot(cf.year, np.exp(trend), "--", lw=1.2, color="#b08968",
             label=f"log-linear trend (+{slope_pct:.2f}%/yr)")
    # worst negative-anomaly years (excluding pre-1961) shaded
    worst = sorted(np.argsort(anom)[:4])
    for i in worst:
        ax0.axvspan(cf.year.iloc[i] - 0.5, cf.year.iloc[i] + 0.5,
                    color="#d62728", alpha=0.10)
        ax0.annotate(f"{int(cf.year.iloc[i])}\n{anom[i]:.0f}%",
                     (cf.year.iloc[i], cf.coffee_prod_t.iloc[i]),
                     textcoords="offset points", xytext=(0, -26),
                     ha="center", fontsize=7, color="#a11")
    # drought events + record wet harvests
    for ev in res["drought_events_coffee_anomaly"]:
        yy = ev["year"]
        if yy in set(cf.year.astype(int)):
            row = cf.index[cf.year == yy][0]
            ax0.annotate(f"drought {yy}\n({ev['coffee_anom_pct']:+.0f}%)",
                         (yy, cf.coffee_prod_t.iloc[row]),
                         textcoords="offset points", xytext=(6, 8),
                         fontsize=7, color="#333")
    ax0.set_ylabel("green coffee production (tonnes)")
    ax0.set_title("Uganda coffee: production +$\\pm$ detrended anomaly\n"
                  "(red bands = worst anomaly years)")
    ax0.legend(loc="upper left", fontsize=7)

    # ---- panel B: correlation, two substrates, sign flip ----
    order = ["long_rains", "bimodal", "annual"]
    cent = [res["meteostat_century_linkage"].get(f"{k}:SPI3", {}).get("pearson_r", np.nan)
            for k in order]
    mod = [res["era5_modern_linkage"].get(f"{k}:SPI3", {}).get("pearson_r", np.nan)
           for k in order]
    mod_p = [res["era5_modern_linkage"].get(f"{k}:SPI3", {}).get("pearson_p", 1.0)
             for k in order]
    y = np.arange(len(order))
    h = 0.36
    ax1.barh(y + h/2, cent, height=h, color="#4c72b0", label="Meteostat century (1961-86)")
    ax1.barh(y - h/2, mod, height=h, color="#c44e52", label="ERA5 modern (2008-23)")
    for i, (c, m, p) in enumerate(zip(cent, mod, mod_p)):
        if np.isfinite(m):
            ax1.annotate(f"{m:+.2f}" + ("*" if p < 0.05 else ""), (m, i - h/2),
                         xytext=(4 if m >= 0 else -4, 0), textcoords="offset points",
                         va="center", ha="left" if m >= 0 else "right", fontsize=7)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(order)
    ax1.set_xlabel("Pearson r: detrended coffee anomaly vs seasonal SPI$_3$")
    ax1.set_xlim(-0.8, 0.6)
    ax1.set_title("Sign flips between substrates\n(coarse national aggregate $\\neq$ climate signal)")
    ax1.legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    written = C.save_figure(fig, "fig_coffee")
    print("wrote", ", ".join(str(p) for p in written))


if __name__ == "__main__":
    fig_coffee()
