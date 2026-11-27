"""Drought early warning figure: choropleth map + time-series panel.

Panel (a): District map showing current drought status (Normal/Watch/Warning/Alert)
           using the latest month from drought_early_warning.csv.
Panel (b): SPI-6 time-series for the 4 worst districts (most months in Warning/Alert)
           with documented drought events shaded.

Outputs: figures/fig_drought_ew.png (mirrored to manuscript/figures/)
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS, FIGURES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

# Import choropleth machinery from figures_maps
from figures_maps import geometry, district_of, fill, _frame, _map_axes, _label, NODATA, COORDS

plt.rcParams.update({"font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

# Status color palette (matches BAND_COLOR pattern)
STATUS_COLOR = {
    "Normal": "#4575b4",    # blue (low concern)
    "Watch": "#fee090",     # yellow
    "Warning": "#fdae61",   # orange
    "Alert": "#d7301f",     # red (high concern)
}


def fig_drought_ew():
    """Two-panel figure: (a) status choropleth, (b) SPI-6 time-series."""
    ew = pd.read_csv(RESULTS / "tables/drought_early_warning.csv")
    ew["month"] = pd.to_datetime(ew["month"])
    register = pd.read_csv(DATA / "disaster_register.csv")
    droughts = register[register["type"].str.lower().str.contains("drought", na=False)]
    
    # Latest month snapshot
    latest_month = ew["month"].max()
    latest = ew[ew["month"] == latest_month].copy()
    
    # Map config names to COD district names
    latest["district"] = latest["location"].apply(district_of)
    status_by_district = dict(zip(latest["district"], latest["status"]))
    # Convert status to color
    color_by_district = {d: STATUS_COLOR.get(s, NODATA) for d, s in status_by_district.items()}
    
    # Find the 4 worst districts (most months in Warning/Alert)
    flagged = ew[ew["status"].isin(["Warning", "Alert"])]
    worst = flagged.groupby("location").size().nlargest(4).index.tolist()
    worst_names = [district_of(loc) for loc in worst]
    
    # ---- Figure layout ----
    fig, (ax_map, ax_ts) = plt.subplots(1, 2, figsize=(12, 5),
                                         gridspec_kw={"width_ratios": [1, 1.2]})
    
    # ---- Panel (a): Status choropleth ----
    feats, regs, lk = geometry()
    _map_axes(ax_map, regions=True)
    fill(ax_map, colors=color_by_district, default=NODATA)
    
    # Legend
    legend_patches = [mpatches.Patch(facecolor=STATUS_COLOR[s], edgecolor="#4a4a4a",
                                     linewidth=0.5, label=s)
                      for s in ["Normal", "Watch", "Warning", "Alert"]]
    ax_map.legend(handles=legend_patches, loc="lower left", fontsize=7,
                  framealpha=0.9, edgecolor="#cccccc")
    ax_map.set_title(f"(a) Drought status — {latest_month.strftime('%Y-%m')}",
                     fontsize=9, fontweight="bold", loc="left")
    
    # District labels for worst 4
    for loc in worst:
        name = district_of(loc)
        if name and name in status_by_district:
            lon, lat = COORDS.get(loc, (None, None))
            if lon is not None:
                _label(ax_map, lon, lat, name)
    
    # ---- Panel (b): SPI-6 time-series for worst 4 districts ----
    for loc, name in zip(worst, worst_names):
        ts = ew[ew["location"] == loc][["month", "SPI6"]].sort_values("month")
        ax_ts.plot(ts["month"], ts["SPI6"], marker="o", markersize=3,
                   linewidth=1.2, label=name if name else loc)
    
    # Threshold lines
    ax_ts.axhline(-1.0, color="#fdae61", linestyle="--", linewidth=0.8,
                  label="Moderate drought (-1.0)")
    ax_ts.axhline(-1.5, color="#d7301f", linestyle="--", linewidth=0.8,
                  label="Severe drought (-1.5)")
    ax_ts.axhline(0, color="#999999", linestyle=":", linewidth=0.6)
    
    # Shade documented drought events
    for _, d in droughts.iterrows():
        s = pd.to_datetime(str(d["start"]), errors="coerce")
        e = pd.to_datetime(str(d["end"]), errors="coerce")
        if pd.isna(s):
            continue
        e = e if pd.notna(e) else s
        ax_ts.axvspan(s, e, alpha=0.15, color="#d7301f", zorder=0)
    
    ax_ts.set_xlabel("Month")
    ax_ts.set_ylabel("SPI-6")
    ax_ts.set_title("(b) SPI-6 trajectory — worstet districts",
                    fontsize=9, fontweight="bold", loc="left")
    ax_ts.legend(fontsize=6, loc="lower left", framealpha=0.9, edgecolor="#cccccc")
    ax_ts.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def main():
    fig = fig_drought_ew()
    paths = C.save_figure(fig, "fig_drought_ew")
    print(f"Drought early warning figure saved to: {[str(p) for p in paths]}")


if __name__ == "__main__":
    main()
