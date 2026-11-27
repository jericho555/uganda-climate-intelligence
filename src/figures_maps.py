"""District maps for the manuscript: choropleths over Uganda's 135 districts.

Every quantity plotted here is read from an artefact already written by the
pipeline; nothing is re-derived, re-thresholded or re-fitted in this file:

  data/district_boundaries.geojson   135 districts (OCHA COD v01, CC BY-IGO)
  data/region_boundaries.geojson     4 regional outlines, drawn as context
  data/district_lookup.csv           observation point -> district + PCODE
  results/district_coverage.json     areal coverage of the analysed panel
  results/tables/district_anomaly.csv    16 districts: rate, Gi*, cluster
  results/tables/temporal_trends.csv     Mann-Kendall tau per district x variable
  results/tables/risk_district_month.csv UCASI district-month nowcast surface
  results/tables/power_trends.csv        POWER 1988-2007 trends, 19 districts
  data/meteostat_stations.csv            in-situ stations (lon/lat)
  src/config.py                          observation points (single source)

Each value is measured at a point (an ERA5 grid extraction or a station) and is
drawn over the district that contains that point, per data/district_lookup.csv.
Districts the pipeline does not observe are rendered as no-data, never as zero.
"""
import json
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
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})

REGION_COLOR = {"Central": "#b6a3cf", "Eastern": "#8fc4cd",
                "Northern": "#d9bd85", "Western": "#9dc9a5"}
BAND_COLOR = {"Low": "#4575b4", "Moderate": "#fee090",
              "High": "#fdae61", "Extreme": "#d7301f"}
DIV = LinearSegmentedColormap.from_list("tau", ["#2166ac", "#f7f7f7", "#b2182b"])
NODATA = "#e9e9e6"
CLS_COLOR = {"hotspot": "#d7301f", "coldspot": "#4575b4", "ns": "#c8c8c1"}
UNOBSERVED = "#f5f5f2"

# Uganda's true extent (29.5-35.0E, 1.5S-4.0N) plus a margin for labels.
LON_LIM, LAT_LIM = (29.1, 35.5), (-2.1, 4.4)
COORDS = {name: (lon, lat) for _, name, lat, lon, _e in C.LOCATIONS}

# Manual label offsets (points) so the dense Central and Eastern clusters stay
# readable. The geometry is fixed by the data; only text placement is chosen.
LABEL_OFF = {
    "Arua": (0, 9, "center"), "Gulu": (9, 2, "left"), "Lira": (9, 1, "left"),
    "Masindi": (-9, 5, "right"), "FortPortal": (-6, 9, "center"),
    "Mbarara": (0, -12, "center"), "Kasese": (0, -13, "center"),
    "Hoima": (-9, 3, "right"), "Kabale": (0, -12, "center"),
    "Kampala": (0, -15, "center"), "Wakiso": (-9, 7, "right"),
    "Mukono": (10, -8, "left"), "Mityana": (-9, -11, "right"),
    "Luweero": (0, 11, "center"), "Jinja": (10, 4, "left"),
    "Soroti": (0, 10, "center"), "Kumi": (10, 3, "left"),
    "Mbale": (10, -3, "left"), "Tororo": (10, -9, "left"),
}

_DISTRICTS = _REGIONS = _LOOKUP = None


def geometry():
    """Load the district/region polygons and the point-to-district lookup."""
    global _DISTRICTS, _REGIONS, _LOOKUP
    if _DISTRICTS is None:
        p = DATA / "district_boundaries.geojson"
        if not p.exists():
            raise SystemExit("missing %s: run python src/fetch_boundaries.py"
                             % p.relative_to(C.ROOT))
        _DISTRICTS = json.loads(p.read_text(encoding="utf-8"))["features"]
        rp = DATA / "region_boundaries.geojson"
        _REGIONS = (json.loads(rp.read_text(encoding="utf-8"))["features"]
                    if rp.exists() else [])
        _LOOKUP = pd.read_csv(DATA / "district_lookup.csv")
    return _DISTRICTS, _REGIONS, _LOOKUP


def district_of(loc):
    """Config name -> COD district name (Luweero->Luwero, FortPortal->Kabarole)."""
    _, _, lk = geometry()
    return dict(zip(lk.location, lk.adm2_name)).get(loc)


def values_by_district(table, column, key="location"):
    """Re-key a per-location table onto COD district names."""
    feats, _, lk = geometry()
    m = dict(zip(lk.location, lk.adm2_name))
    return {m[r[key]]: r[column] for _, r in table.iterrows() if r[key] in m}


def coverage_note():
    """One-line areal coverage statement taken from the committed artefact."""
    p = RESULTS / "district_coverage.json"
    if not p.exists():
        return ""
    cv = json.loads(p.read_text())
    return ("%d of %d districts, %.1f%% of the national district area"
            % (cv["districts_analysed"], cv["districts_total"],
               100 * cv["analysed_area_share"]))


def _rings(geom):
    """Exterior rings of a Polygon/MultiPolygon as (N, 2) arrays."""
    groups = (geom["coordinates"] if geom["type"] == "MultiPolygon"
              else [geom["coordinates"]])
    for poly in groups:
        if len(poly) and len(poly[0]) > 3:
            yield np.asarray(poly[0], dtype=float)[:, :2]


def _outline(ax, feats, color="#8c8c8c", lw=0.35, zorder=4, alpha=1.0):
    for f in feats:
        for ring in _rings(f["geometry"]):
            ax.add_patch(mp.Polygon(ring, closed=True, fill=False,
                                    edgecolor=color, linewidth=lw, zorder=zorder,
                                    joinstyle="round", alpha=alpha))


def fill(ax, values=None, cmap=None, vmin=None, vmax=None, colors=None,
         default=NODATA, edge="#4a4a4a", lw=0.4, zorder=1):
    """Draw every district. Numeric mode masks absent values to the bad colour."""
    feats, _, _ = geometry()
    patches, keys = [], []
    for f in feats:
        name = f["properties"]["adm2_name"]
        for ring in _rings(f["geometry"]):
            patches.append(mp.Polygon(ring, closed=True))
            keys.append(name)
    if colors is not None:
        fc = [colors.get(k, default) for k in keys]
        coll = PatchCollection(patches, facecolor=fc, edgecolor=edge,
                               linewidth=lw, zorder=zorder, match_original=False)
        ax.add_collection(coll)
        return coll
    cm = (matplotlib.colormaps[cmap].copy() if isinstance(cmap, str)
          else cmap.copy())
    cm.set_bad(NODATA)
    arr = np.array([values.get(k, np.nan) for k in keys], dtype=float)
    coll = PatchCollection(patches, edgecolor=edge, linewidth=lw, zorder=zorder,
                           match_original=False)
    coll.set_array(arr)
    coll.set_cmap(cm)
    coll.set_norm(Normalize(vmin=vmin, vmax=vmax))
    ax.add_collection(coll)
    return coll


def _frame(ax):
    ax.set_xlim(*LON_LIM)
    ax.set_ylim(*LAT_LIM)
    ax.set_xlabel("longitude (\u00b0E)")
    ax.set_ylabel("latitude (\u00b0N)")
    ax.set_xticks(np.arange(30, 35, 2))
    ax.set_yticks(np.arange(-2, 4, 2))
    ax.set_aspect("equal")


def _map_axes(ax, regions=True):
    """Grey district mosaic plus regional outlines: the national frame."""
    feats, regs, _ = geometry()
    _outline(ax, feats, color="#b9b9b4", lw=0.3, zorder=2)
    if regions and regs:
        _outline(ax, regs, color="#5a5a5a", lw=0.9, zorder=3, alpha=0.9)
    _frame(ax)


def _label(ax, lon, lat, text):
    dx, dy, ha = LABEL_OFF.get(text.split()[0], (6, 4, "left"))
    ax.annotate(text, (lon, lat), fontsize=6.2, xytext=(dx, dy), ha=ha,
                textcoords="offset points", zorder=9,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none",
                          alpha=0.78))


def _save(fig, name):
    for folder in (FIGURES, MANUSCRIPT / "figures"):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


def _select(names):
    """District features whose COD name is in `names`."""
    feats, _, _ = geometry()
    return [f for f in feats if f["properties"]["adm2_name"] in set(names)]


def _hatch(ax, names, hatch="////", color="#5f5f5f", zorder=5):
    """Mark districts whose trend is not significant, instead of shrinking them."""
    patches = [mp.Polygon(r, closed=True)
               for f in _select(names) for r in _rings(f["geometry"])]
    if not patches:
        return
    ax.add_collection(PatchCollection(patches, facecolor="none",
                                      edgecolor=color, linewidth=0.0,
                                      hatch=hatch, zorder=zorder, alpha=0.8))


# --------------------------------------------------------------------------
def fig_study_area():
    """Where the observations actually are: the district mosaic, and topography."""
    da = pd.read_csv(RESULTS / "tables/district_anomaly.csv")
    st = pd.read_csv(DATA / "meteostat_stations.csv")
    power = pd.read_csv(RESULTS / "tables/power_trends.csv")
    power_only = sorted(set(power.location) - set(da.location))
    feats, regs, lk = geometry()
    colors = {adm: REGION_COLOR.get(str(reg), "#cfcfca")
              for adm, reg in zip(lk.adm2_name, lk.adm1_name)}
    n_unobserved = len(feats) - len(colors)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.0))

    ax = axes[0]
    _frame(ax)
    fill(ax, colors=colors, default=NODATA, edge="#4a4a4a", lw=0.35, zorder=1)
    _outline(ax, regs, color="#3f3f3f", lw=0.9, zorder=3)
    for _, r in da.iterrows():
        ax.scatter([r.lon], [r.lat], s=13, marker="o", color="#1a1a1a", zorder=6)
    for name in power_only:
        lon, lat = COORDS[name]
        ax.scatter([lon], [lat], s=34, marker="s", facecolor="none",
                   edgecolor="#555555", linewidth=1.0, zorder=6)
    ax.scatter(st.lon, st.lat, s=44, marker="^", facecolor="none",
               edgecolor="#c0392b", linewidth=1.0, zorder=7)
    handles = [mp.Patch(facecolor=REGION_COLOR[r], edgecolor="#4a4a4a",
                        label="%s (observed)" % r) for r in
               ("Central", "Eastern", "Northern", "Western")]
    handles += [mp.Patch(facecolor=NODATA, edgecolor="#9a9a9a",
                         label="district, no data (n=%d)" % n_unobserved),
                plt.Line2D([], [], marker="^", ls="none", mec="#c0392b",
                           mfc="none", label="Meteostat station (n=%d)" % len(st)),
                plt.Line2D([], [], marker="s", ls="none", mec="#555555",
                           mfc="none", label="gap-fill only (n=%d)" % len(power_only))]
    ax.legend(handles=handles, fontsize=5.9, loc="lower left", ncol=2,
              framealpha=0.92)
    ax.set_title(f"(a) Uganda's {len(feats)} districts, {len(colors)} observed\n"
                 f"{coverage_note()}", fontsize=8.5)

    ax = axes[1]
    _map_axes(ax)
    elev = pd.Series({name: e for _, name, _lat, _lon, e in C.LOCATIONS})
    pts = np.array([COORDS[n] for n in elev.index])
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=95, c=elev.values, cmap="terrain",
                    vmin=elev.min(), vmax=elev.max(), edgecolor="k", linewidth=0.4,
                    zorder=6)
    for name, (lon, lat) in COORDS.items():
        _label(ax, lon, lat, f"{name} {int(elev[name])} m")
    cb = fig.colorbar(sc, ax=ax, pad=0.03)
    cb.set_label("observation-point elevation (m a.s.l.)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    erange = f"{int(elev.min())}\u2013{int(elev.max())}"
    ax.set_title(f"(b) Point elevation over the district mosaic, {erange} m:\n"
                 "the orographic contrast between north and southwest",
                 fontsize=8.5)

    fig.text(0.5, 0.004, "Every value is measured at a point and drawn over the "
             "district that contains it. Geometry: OCHA Uganda COD v01 admin-2 "
             "(CC BY-IGO).", ha="center", fontsize=6.4, style="italic")
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    _save(fig, "fig_study_area")


# --------------------------------------------------------------------------
def fig_hotspot_map():
    """Getis--Ord classification drawn as district categories, not bubble radii."""
    da = pd.read_csv(RESULTS / "tables/district_anomaly.csv")
    feats, regs, _ = geometry()
    hot = [district_of(n) for n in da.loc[da.cluster == "hotspot", "location"]]
    cold = [district_of(n) for n in da.loc[da.cluster == "coldspot", "location"]]
    colors = {k: CLS_COLOR.get(str(v), UNOBSERVED)
              for k, v in values_by_district(da, "cluster").items()}

    fig, ax = plt.subplots(figsize=(7.0, 6.4))
    _frame(ax)
    fill(ax, colors=colors, default=UNOBSERVED, edge="#9a9a9a", lw=0.3)
    _outline(ax, regs, color="#3f3f3f", lw=0.9, zorder=3)
    _outline(ax, _select(hot + cold), color="#1a1a1a", lw=2.2, zorder=5)
    for _, r in da.iterrows():
        ax.scatter([r.lon], [r.lat], s=8, marker="o", color="#1a1a1a", zorder=7)
        _label(ax, r.lon, r.lat, f"{r.location} {r.Gi_star_z:+.1f}")
    handles = [mp.Patch(facecolor=CLS_COLOR["hotspot"], edgecolor="#4a4a4a",
                        label="hot spot ($p<0.05$, n=%d)" % len(hot)),
               mp.Patch(facecolor=CLS_COLOR["coldspot"], edgecolor="#4a4a4a",
                        label="cold spot ($p<0.05$, n=%d)" % len(cold)),
               mp.Patch(facecolor=CLS_COLOR["ns"], edgecolor="#4a4a4a",
                        label="tested, not significant (n=%d)"
                              % int((da.cluster == "ns").sum())),
               mp.Patch(facecolor=UNOBSERVED, edgecolor="#9a9a9a",
                        label="district, no ERA5 panel (n=%d)"
                              % (len(feats) - len(colors)))]
    ax.legend(handles=handles, fontsize=6.2, loc="lower left", ncol=1,
              framealpha=0.92)
    ax.set_title("Getis--Ord $G_i^*$ classification by district, "
                 "2008\u20132023\n"
                 "z-values below each site; the national pattern is one hotspot",
                 fontsize=8.5)
    fig.text(0.5, 0.004, "One district, one class. Geometry: OCHA Uganda COD "
             "v01 admin-2 (CC BY-IGO).", ha="center", fontsize=6.4,
             style="italic")
    fig.tight_layout(rect=(0, 0.028, 1, 1))
    _save(fig, "fig_hotspot_map")


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
    hot = [district_of(n) for n in da.loc[da.cluster == "hotspot", "location"]]
    nonsig = [district_of(r.location) for _, r in tr.iterrows()
              if r.p_value >= 0.05]
    stamp = f"{int(latest.year.iloc[0])}-{int(latest.month.iloc[0]):02d}"

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.9))

    # (a) detection rate as a district choropleth, hotspots outlined
    ax = axes[0]
    _frame(ax)
    rates = values_by_district(da, "anomaly_rate")
    coll = fill(ax, values=rates, cmap="YlOrRd", vmin=0,
                vmax=da.anomaly_rate.max())
    _outline(ax, _select(hot), color="#1a1a1a", lw=2.2, zorder=5)
    for _, r in da.iterrows():
        ax.scatter([r.lon], [r.lat], s=8, marker="o", color="#1a1a1a", zorder=7)
        _label(ax, r.lon, r.lat, f"{r.location} {100*r.anomaly_rate:.1f}%")
    txt = "; ".join(f"{r.location} $z$={r.Gi_star_z:.2f}"
                    for _, r in da[da.cluster == "hotspot"].iterrows())
    ax.text(0.02, 0.03, f"significant $G_i^*$ hotspot (\u03b1=0.05, thick edge): "
            f"{txt}", transform=ax.transAxes, fontsize=6.6, color="#b2182b")
    cb = fig.colorbar(coll, ax=ax, pad=0.03)
    cb.set_label("flagged district-days (fraction)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title(f"(a) Detected anomaly rate by district, 2008-2023\n"
                 f"(target 5%; thick edge = $G_i^*$ hotspot)", fontsize=8.5)

    # (b) precipitation Mann-Kendall tau; non-significant districts hatched
    ax = axes[1]
    _frame(ax)
    lim = max(abs(tr.tau.min()), abs(tr.tau.max()))
    taus = values_by_district(tr, "tau")
    coll = fill(ax, values=taus, cmap=DIV, vmin=-lim, vmax=lim)
    _hatch(ax, nonsig)
    for _, r in tr.iterrows():
        lon, lat = COORDS[r.location]
        ax.scatter([lon], [lat], s=8, marker="o", color="#1a1a1a", zorder=7)
        sig = "*" if r.p_value < 0.05 else ""
        _label(ax, lon, lat, f"{r.location} {r.tau:+.2f}{sig}")
    cb = fig.colorbar(coll, ax=ax, pad=0.03)
    cb.set_label("Mann-Kendall $\\tau$ (precipitation, monthly)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title(f"(b) Precipitation trend by district, 2008-2023\n"
                 f"{int((tr.p_value < 0.05).sum())}/{len(tr)} significant "
                 f"($p<0.05$; hatch = not significant)", fontsize=8.5)

    # (c) forward risk nowcast, most recent month, as a banded choropleth
    ax = axes[2]
    _frame(ax)
    bandc = {k: BAND_COLOR.get(str(v), NODATA)
             for k, v in values_by_district(latest, "band").items()}
    fill(ax, colors=bandc, default=NODATA)
    for _, r in latest.iterrows():
        lon, lat = COORDS[r.location]
        ax.scatter([lon], [lat], s=8, marker="o", color="#1a1a1a", zorder=7)
        _label(ax, lon, lat, f"{r.location} {r.risk:.0f}")
    handles = [mp.Patch(color=v, edgecolor="#4a4a4a", label=k)
               for k, v in BAND_COLOR.items()]
    handles.append(mp.Patch(facecolor=NODATA, edgecolor="#9a9a9a",
                            label="district, no nowcast"))
    ax.legend(handles=handles, fontsize=6.2, loc="lower left", ncol=2,
              framealpha=0.92, title="UCASI band", title_fontsize=6.5)
    top = latest.sort_values("risk", ascending=False).head(3)
    ax.set_title("(c) UCASI district-month nowcast, "
                 f"{stamp}\ntop: " + ", ".join(f"{r.location} {r.risk:.0f}"
                                               for _, r in top.iterrows()),
                 fontsize=8.5)

    fig.text(0.5, 0.004, "Each value is measured at a point (ERA5 grid cell or "
             "station) and shaded over the district that contains it; districts "
             "the pipeline does not observe are grey, never zero. "
             "Geometry: OCHA Uganda COD v01 admin-2 (CC BY-IGO).",
             ha="center", fontsize=6.4, style="italic")
    fig.tight_layout(rect=(0, 0.028, 1, 1))
    _save(fig, "fig_risk_maps")


# --------------------------------------------------------------------------
def fig_gapfill_maps():
    """POWER 1988-2007: warming and drying across the in-situ reporting collapse."""
    pw = pd.read_csv(RESULTS / "tables/power_trends.csv")
    warm = pw[(pw.variable == "t2m") & (pw.source == "POWER")]
    dry = pw[(pw.variable == "rh2m") & (pw.source == "POWER")]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.0))
    for i, (ax, tab, lab) in enumerate(
            ((axes[0], warm, "$T_{2m}$"),
             (axes[1], dry, "near-surface RH"))):
        _frame(ax)
        lim = max(abs(tab.tau.min()), abs(tab.tau.max()))
        coll = fill(ax, values=values_by_district(tab, "tau"), cmap=DIV,
                    vmin=-lim, vmax=lim)
        _hatch(ax, [district_of(r.location) for _, r in tab.iterrows()
                    if r.p_value >= 0.05])
        for _, r in tab.iterrows():
            lon, lat = COORDS[r.location]
            ax.scatter([lon], [lat], s=8, marker="o", color="#1a1a1a", zorder=7)
            sig = "*" if r.p_value < 0.05 else ""
            _label(ax, lon, lat, f"{r.location} {r.tau:+.2f}{sig}")
        cb = fig.colorbar(coll, ax=ax, pad=0.03)
        cb.set_label("Mann-Kendall $\\tau$, 1988-2007", fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        nsig = int((tab.p_value < 0.05).sum())
        ax.set_title(f"({chr(ord('a') + i)}) POWER {lab} by district, "
                     "1988\u20132007\n"
                     f"{nsig}/{len(tab)} districts significant "
                     "($p<0.05$; hatch = not significant)", fontsize=8.5)
    fig.text(0.5, 0.004, "NASA POWER (MERRA-2 based) monthly series, "
             "1988-01 to 2007-12, measured at the 19 observation points and "
             "shaded over the containing district. "
             "Geometry: OCHA Uganda COD v01 admin-2 (CC BY-IGO).",
             ha="center", fontsize=6.4, style="italic")
    fig.tight_layout(rect=(0, 0.028, 1, 1))
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
    fig_hotspot_map()
    fig_risk_maps()
    fig_gapfill_maps()
    fig_null_comparison()


if __name__ == "__main__":
    main()
