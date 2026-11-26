"""District reference maps and spatial risk maps for the manuscript.

Every quantity plotted here is read from an artefact already written by the
pipeline; nothing is re-derived, re-thresholded or re-fitted in this file:

  results/tables/district_anomaly.csv    16 analysed districts: rate, Gi*, cluster
  results/tables/temporal_trends.csv     Mann-Kendall tau per district x variable
  results/tables/risk_district_month.csv UCASI district-month nowcast surface
  results/tables/power_trends.csv        POWER 1988-2007 trends, 19 districts
  results/benchmark.csv                  held-out detector scores
  data/meteostat_stations.csv            in-situ stations (lon/lat)
  src/config.py                          district centroids (single source)

Districts are drawn at their WGS84 administrative centroids: markers are points,
not areal units, and no boundary geometry is implied.
"""
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS, FIGURES, MANUSCRIPT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as mp
from matplotlib.colors import LinearSegmentedColormap, Normalize

plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})

REGION_COLOR = {"Central": "#7b3294", "Eastern": "#005f73",
                "Northern": "#8a5a00", "Western": "#1b7837"}
BAND_COLOR = {"Low": "#4575b4", "Moderate": "#fee090",
              "High": "#fdae61", "Extreme": "#d7301f"}
DIV = LinearSegmentedColormap.from_list("tau", ["#2166ac", "#f7f7f7", "#b2182b"])

# Uganda spans roughly 29.6-33.9E and -1.5-4.4N for these centroids; the extra
# margin on the right and left is label space, not data range.
LON_LIM, LAT_LIM = (28.6, 36.4), (-2.2, 3.8)
COORDS = {name: (lon, lat) for _, name, lat, lon, _e in C.LOCATIONS}

# Manual label offsets (points) so the dense Central and Eastern clusters stay
# readable. The geometry is fixed by the data; only text placement is chosen.
LABEL_OFF = {
    "Arua": (0, 8, "center"), "Gulu": (8, 2, "left"), "Lira": (8, 1, "left"),
    "Masindi": (-8, 4, "right"), "FortPortal": (-4, 8, "center"),
    "Mbarara": (0, -12, "center"), "Kasese": (-8, -9, "right"),
    "Hoima": (-8, 3, "right"), "Kabale": (0, -12, "center"),
    "Kampala": (0, -14, "center"), "Wakiso": (-8, 6, "right"),
    "Mukono": (9, -6, "left"), "Mityana": (-8, -9, "right"),
    "Luweero": (0, 9, "center"), "Jinja": (-9, 6, "right"),
    "Soroti": (0, 9, "center"), "Kumi": (9, 3, "left"),
    "Mbale": (9, -3, "left"), "Tororo": (9, -8, "left"),
    "Lira": (-8, 4, "right"),
}


def _map_axes(ax):
    ax.set_xlim(*LON_LIM)
    ax.set_ylim(*LAT_LIM)
    ax.set_xlabel("longitude (\u00b0E)")
    ax.set_ylabel("latitude (\u00b0N)")
    ax.set_xticks(np.arange(30, 35, 2))
    ax.set_yticks(np.arange(-2, 4, 2))
    ax.set_aspect("equal")


def _label(ax, lon, lat, text):
    dx, dy, ha = LABEL_OFF.get(text.split()[0], (6, 4, "left"))
    ax.annotate(text, (lon, lat), fontsize=6.2, xytext=(dx, dy), ha=ha,
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none",
                          alpha=0.78))


def _save(fig, name):
    for folder in (FIGURES, MANUSCRIPT / "figures"):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------
def fig_study_area():
    """Where the observations actually are: districts, stations, topography."""
    da = pd.read_csv(RESULTS / "tables/district_anomaly.csv")
    st = pd.read_csv(DATA / "meteostat_stations.csv")
    power = pd.read_csv(RESULTS / "tables/power_trends.csv")
    n_power = power.location.nunique()
    power_only = sorted(set(power.location) - set(da.location))
    norm = lambda s: s.replace(" ", "").lower()
    district_keys = {norm(n) for n in da.location}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))

    ax = axes[0]
    _map_axes(ax)
    for region, grp in da.groupby("region"):
        ax.scatter(grp.lon, grp.lat, s=34, marker="o",
                   color=REGION_COLOR[region], edgecolor="k", linewidth=0.4,
                   label=f"ERA5 district ({region})", zorder=3)
        for _, r in grp.iterrows():
            _label(ax, r.lon, r.lat, r.location)
    ax.scatter(st.lon, st.lat, s=34, marker="^", facecolor="none",
               edgecolor="#c0392b", linewidth=1.0,
               label=f"Meteostat station (n={len(st)})", zorder=4)
    for _, r in st.iterrows():
        if norm(r["name"]) not in district_keys:
            _label(ax, r.lon, r.lat, r["name"])
    for name in power_only:
        lon, lat = COORDS[name]
        ax.scatter([lon], [lat], s=34, marker="s", facecolor="none",
                   edgecolor="#555555", linewidth=1.0, zorder=4)
        _label(ax, lon, lat, name)
    ax.legend(fontsize=6.2, loc="lower right", framealpha=0.9)
    ax.set_title(f"(a) Observing geometry: {len(da)} ERA5 districts, "
                 f"{len(st)} stations,\n{n_power} POWER gap-fill centroids",
                 fontsize=8.5)

    ax = axes[1]
    _map_axes(ax)
    elev = pd.Series({name: e for _, name, _lat, _lon, e in C.LOCATIONS})
    pts = np.array([COORDS[n] for n in elev.index])
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=90, c=elev.values, cmap="terrain",
                    vmin=elev.min(), vmax=elev.max(), edgecolor="k", linewidth=0.4)
    for name, (lon, lat) in COORDS.items():
        _label(ax, lon, lat, f"{name} {int(elev[name])} m")
    cb = fig.colorbar(sc, ax=ax, pad=0.03)
    cb.set_label("centroid elevation (m a.s.l.)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    erange = f"{int(elev.min())}\u2013{int(elev.max())}"
    ax.set_title(f"(b) Centroid elevation {erange} m: the orographic contrast\n"
                 "between the north and the southwest highlands", fontsize=8.5)

    fig.tight_layout()
    _save(fig, "fig_study_area")


# --------------------------------------------------------------------------
def fig_risk_maps():
    """Three district maps: detection rate, precipitation trend, forward nowcast."""
    da = pd.read_csv(RESULTS / "tables/district_anomaly.csv")
    tr = pd.read_csv(RESULTS / "tables/temporal_trends.csv")
    tr = tr[tr.variable == "precipitation_sum"].copy()
    rm = pd.read_csv(RESULTS / "tables/risk_district_month.csv",
                     parse_dates=False)
    rm["ym"] = rm.year.astype(int) * 100 + rm.month.astype(int)
    latest = rm[rm.ym == rm.ym.max()].copy()
    stamp = f"{int(latest.year.iloc[0])}-{int(latest.month.iloc[0]):02d}"

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4))

    # (a) detection rate with the Getis-Ord classification
    ax = axes[0]
    _map_axes(ax)
    norm = Normalize(vmin=0, vmax=da.anomaly_rate.max())
    ax.scatter(da.lon, da.lat, s=25 + 190 * norm(da.anomaly_rate),
               c=da.anomaly_rate, cmap="YlOrRd", vmin=0,
               vmax=da.anomaly_rate.max(), edgecolor="k", linewidth=0.4, zorder=3)
    for _, r in da.iterrows():
        _label(ax, r.lon, r.lat, f"{r.location} {100*r.anomaly_rate:.1f}%")
    hot = da[da.cluster == "hotspot"]
    for _, r in hot.iterrows():
        ax.add_patch(mp.Circle((r.lon, r.lat), 0.22, fill=False,
                               edgecolor="#b2182b", linewidth=1.6, zorder=4))
    txt = "; ".join(f"{r.location} $z$={r.Gi_star_z:.2f}" for _, r in hot.iterrows())
    ax.text(0.02, 0.03, f"significant $G_i^*$ hotspot (\u03b1=0.05): {txt}",
            transform=ax.transAxes, fontsize=6.6, color="#b2182b")
    cb = fig.colorbar(ax.collections[0], ax=ax, pad=0.03)
    cb.set_label("flagged district-days (fraction)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title(f"(a) Detected anomaly rate, 2008-2023\n(target 5%; "
                 f"circle = $G_i^*$ hotspot)", fontsize=8.5)

    # (b) precipitation Mann-Kendall tau
    ax = axes[1]
    _map_axes(ax)
    lim = max(abs(tr.tau.min()), abs(tr.tau.max()))
    sc = ax.scatter([COORDS[n][0] for n in tr.location],
                    [COORDS[n][1] for n in tr.location], s=45, c=tr.tau,
                    cmap=DIV, vmin=-lim, vmax=lim,
                    edgecolor=["k" if p < 0.05 else "#bbbbbb" for p in tr.p_value],
                    linewidth=[1.1 if p < 0.05 else 0.4 for p in tr.p_value],
                    zorder=3)
    for _, r in tr.iterrows():
        lon, lat = COORDS[r.location]
        sig = "*" if r.p_value < 0.05 else ""
        _label(ax, lon, lat, f"{r.location} {r.tau:+.2f}{sig}")
    cb = fig.colorbar(sc, ax=ax, pad=0.03)
    cb.set_label("Mann-Kendall $\\tau$ (precipitation, monthly)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title(f"(b) Precipitation trend, 2008-2023\n"
                 f"{int((tr.p_value < 0.05).sum())}/{len(tr)} significant "
                 f"($p<0.05$, bold edge)", fontsize=8.5)

    # (c) forward risk nowcast, most recent month
    ax = axes[2]
    _map_axes(ax)
    latest["c"] = latest.band.map(BAND_COLOR)
    ax.scatter([COORDS[n][0] for n in latest.location],
               [COORDS[n][1] for n in latest.location],
               s=25 + 3.2 * latest.risk, c=latest["c"], edgecolor="k",
               linewidth=0.5, zorder=3)
    for _, r in latest.iterrows():
        lon, lat = COORDS[r.location]
        _label(ax, lon, lat, f"{r.location} {r.risk:.0f}")
    handles = [mp.Patch(color=v, label=k) for k, v in BAND_COLOR.items()]
    ax.legend(handles=handles, fontsize=6.2, loc="lower left", ncol=2,
              framealpha=0.9, title="UCASI band", title_fontsize=6.5)
    top = latest.sort_values("risk", ascending=False).head(3)
    ax.set_title("(c) UCASI district-month nowcast, "
                 f"{stamp}\ntop: " + ", ".join(f"{r.location} {r.risk:.0f}"
                                               for _, r in top.iterrows()),
                 fontsize=8.5)

    fig.text(0.5, 0.005, "Districts plotted at WGS84 administrative centroids; "
             "markers are points, not areal units.", ha="center", fontsize=6.5,
             style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "fig_risk_maps")


# --------------------------------------------------------------------------
def fig_gapfill_maps():
    """POWER 1988-2007: warming and drying across the in-situ reporting collapse."""
    pw = pd.read_csv(RESULTS / "tables/power_trends.csv")
    warm = pw[(pw.variable == "t2m") & (pw.source == "POWER")]
    dry = pw[(pw.variable == "rh2m") & (pw.source == "POWER")]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    for i, (ax, tab, lab) in enumerate(
            ((axes[0], warm, "$T_{2m}$"),
             (axes[1], dry, "near-surface RH"))):
        _map_axes(ax)
        lim = max(abs(tab.tau.min()), abs(tab.tau.max()))
        sc = ax.scatter([COORDS[n][0] for n in tab.location],
                        [COORDS[n][1] for n in tab.location], s=55, c=tab.tau,
                        cmap=DIV, vmin=-lim, vmax=lim,
                        edgecolor=["k" if p < 0.05 else "#bbbbbb"
                                   for p in tab.p_value],
                        linewidth=[1.1 if p < 0.05 else 0.4 for p in tab.p_value],
                        zorder=3)
        for _, r in tab.iterrows():
            lon, lat = COORDS[r.location]
            sig = "*" if r.p_value < 0.05 else ""
            _label(ax, lon, lat, f"{r.location} {r.tau:+.2f}{sig}")
        cb = fig.colorbar(sc, ax=ax, pad=0.03)
        cb.set_label("Mann-Kendall $\\tau$, 1988-2007", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        nsig = int((tab.p_value < 0.05).sum())
        ax.set_title(f"({chr(ord('a') + i)}) POWER {lab}, 1988\u20132007\n"
                     f"{nsig}/{len(tab)} districts significant "
                     "($p<0.05$)", fontsize=8.5)
    fig.text(0.5, 0.005, "NASA POWER (MERRA-2 based) monthly series, "
             "1988-01 to 2007-12; centroids as in Table 1.", ha="center",
             fontsize=6.5, style="italic")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _save(fig, "fig_gapfill_maps")


# --------------------------------------------------------------------------
def fig_null_comparison():
    """Make the chance-level result legible: zoomed axes against the null."""
    import json
    nul = json.loads((RESULTS / "random_null.json").read_text())
    bm = pd.read_csv(RESULTS / "benchmark.csv").sort_values("pr_auc",
                                                            ascending=False)
    base = nul["base_rate"]
    null_f1 = nul["pooled"]["mean"]
    null_lo, null_hi = nul["pooled"]["lo"], nul["pooled"]["hi"]
    null_pr = nul["pooled"]["pr_auc_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.9))
    y = np.arange(len(bm))[::-1]
    seq = {n for n in bm.detector if n.upper() in ("LSTMAE", "TRANSFORMERAE")}
    col = ["#d95f02" if n in seq else "#2c7fb8" for n in bm.detector]

    ax = axes[0]
    ax.barh(y, bm.f1, color=col, height=0.72)
    ax.errorbar(bm.f1, y, xerr=[bm.f1 - bm.f1_lo, bm.f1_hi - bm.f1],
                fmt="none", ecolor="k", elinewidth=0.6, capsize=2)
    ax.axvline(null_f1, color="k", ls="--", lw=1.0)
    ax.axvspan(null_lo, null_hi, color="k", alpha=0.10, lw=0)
    ax.annotate(f"random 5% flag budget\n{null_f1:.3f} "
                f"[{null_lo:.3f}, {null_hi:.3f}]", (null_f1, y[-4]),
                xytext=(14, -30), textcoords="offset points", fontsize=6.6,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_yticks(y)
    ax.set_yticklabels(bm.detector, fontsize=6.4)
    ax.set_xlabel("held-out F1 (95% bootstrap CI)")
    ax.set_title("(a) F1 vs the random alert budget", fontsize=8.5)

    ax = axes[1]
    ax.scatter(bm.roc_auc, y, s=34, c=col, edgecolor="k", linewidth=0.4, zorder=3)
    ax.axvline(0.5, color="k", ls="--", lw=1.0)
    ax.set_xlim(0.44, 0.58)
    ax.set_yticks([])
    ax.set_xlabel("ROC-AUC (chance = 0.50)")
    ax.set_title("(b) Ranking skill: every detector\ninside a few points of "
                 "chance", fontsize=8.5)
    ax.annotate("random", (0.5, y[0]), xytext=(-34, 4), textcoords="offset points",
                fontsize=6.6)

    ax = axes[2]
    ax.scatter(bm.pr_auc, y, s=34, c=col, edgecolor="k", linewidth=0.4, zorder=3)
    ax.axvspan(nul["pooled"]["pr_auc_lo"], nul["pooled"]["pr_auc_hi"],
               color="k", alpha=0.10, lw=0)
    ax.axvline(null_pr, color="k", ls="--", lw=1.0)
    ax.set_xlim(0.10, 0.145)
    ax.set_yticks([])
    ax.set_xlabel(f"PR-AUC (base rate = {base:.4f})")
    ax.set_title("(c) Threshold-free precision-recall:\nmost detectors inside "
                 "the random-score band", fontsize=8.5)
    ax.annotate("random-score\nnull", (null_pr, y[4]), xytext=(-46, 14),
                textcoords="offset points", fontsize=6.6,
                arrowprops=dict(arrowstyle="->", lw=0.7))

    handles = [mp.Patch(color="#d95f02", label="sequence autoencoders"),
               mp.Patch(color="#2c7fb8", label="all other detectors")]
    fig.legend(handles=handles, fontsize=7, loc="lower center", ncol=2,
               frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig_null_comparison")


def main():
    fig_study_area()
    fig_risk_maps()
    fig_gapfill_maps()
    fig_null_comparison()


if __name__ == "__main__":
    main()
