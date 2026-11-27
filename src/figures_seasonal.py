"""Seasonal outlook figure: choropleth map + forecast vs climatology bar chart.

Panel (a): District map showing seasonal precipitation outlook class
           (below_normal / near_normal / above_normal) for the latest forecast month.
Panel (b): Bar chart comparing forecast precipitation vs climatology for 4 representative
           districts.

Outputs: figures/fig_seasonal_outlook.png (mirrored to manuscript/figures/)
"""
import sys
import json
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

# Import choropleth machinery from figures_maps
from figures_maps import geometry, district_of, fill, _map_axes, _label, NODATA, COORDS

plt.rcParams.update({"font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

# Outlook class color palette
OUTLOOK_COLOR = {
    "below_normal": "#d7301f",   # red
    "near_normal": "#fee090",    # yellow
    "above_normal": "#4575b4",   # blue
    "unknown": NODATA,
}


def fig_seasonal_outlook():
    """Two-panel figure: (a) outlook choropleth, (b) forecast vs climatology bars."""
    # Load data
    outlook_df = pd.read_csv(RESULTS / "tables/seasonal_outlook.csv")
    summary = json.loads((RESULTS / "seasonal_outlook_summary.json").read_text())
    
    if not summary.get("available", False):
        # Create placeholder figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Seasonal outlook unavailable\n(no forecast data)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        return fig
    
    # Get latest forecast month
    latest_month = outlook_df["month"].max()
    latest = outlook_df[outlook_df["month"] == latest_month].copy()
    
    # Map config names to COD district names
    latest["district"] = latest["location"].apply(district_of)
    class_by_district = dict(zip(latest["district"], latest["forecast_class"]))
    color_by_district = {d: OUTLOOK_COLOR.get(c, NODATA) for d, c in class_by_district.items()}
    
    # Select 4 representative districts (one per region if possible)
    regions = latest["region"].unique()
    sample_districts = []
    for reg in ["Central", "Eastern", "Northern", "Western"]:
        reg_rows = latest[latest["region"] == reg]
        if len(reg_rows):
            sample_districts.append(reg_rows.iloc[0]["location"])
    sample_districts = sample_districts[:4]
    
    # ---- Figure layout ----
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(12, 5),
                                          gridspec_kw={"width_ratios": [1, 1.2]})
    
    # ---- Panel (a): Outlook choropleth ----
    feats, regs_geom, lk = geometry()
    _map_axes(ax_map, regions=True)
    fill(ax_map, colors=color_by_district, default=NODATA)
    
    # Legend
    legend_patches = [mpatches.Patch(facecolor=OUTLOOK_COLOR[c], edgecolor="#4a4a4a",
                                     linewidth=0.5, label=c.replace("_", " ").title())
                      for c in ["below_normal", "near_normal", "above_normal"]]
    ax_map.legend(handles=legend_patches, loc="lower left", fontsize=7,
                  framealpha=0.9, edgecolor="#cccccc")
    ax_map.set_title(f"(a) Seasonal precipitation outlook — {latest_month}",
                     fontsize=9, fontweight="bold", loc="left")
    
    # District labels for sample
    for loc in sample_districts:
        name = district_of(loc)
        if name:
            lon, lat = COORDS.get(loc, (None, None))
            if lon is not None:
                _label(ax_map, lon, lat, name)
    
    # ---- Panel (b): Forecast vs climatology bars ----
    sample_data = latest[latest["location"].isin(sample_districts)].copy()
    sample_data["district_name"] = sample_data["location"].apply(district_of)
    
    x = np.arange(len(sample_data))
    width = 0.35
    
    bars1 = ax_bar.bar(x - width/2, sample_data["precip_normal_mm"], width,
                       label="Climatology", color="#999999", edgecolor="white")
    bars2 = ax_bar.bar(x + width/2, sample_data["forecast_precip_mm"], width,
                       label="Forecast", color="#2171b5", edgecolor="white")
    
    ax_bar.set_xlabel("District")
    ax_bar.set_ylabel("Precipitation (mm)")
    ax_bar.set_title(f"(b) Forecast vs climatology — {latest_month}",
                     fontsize=9, fontweight="bold", loc="left")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(sample_data["district_name"], rotation=15, ha="right", fontsize=7)
    ax_bar.legend(fontsize=7, loc="upper right", framealpha=0.9, edgecolor="#cccccc")
    
    # Anomaly % annotation
    for i, (_, row) in enumerate(sample_data.iterrows()):
        anom = row["anomaly_pct"]
        if pd.notna(anom):
            color = "#d7301f" if anom < -10 else ("#4575b4" if anom > 10 else "#666666")
            ax_bar.text(i, max(row["precip_normal_mm"], row["forecast_precip_mm"]) + 5,
                        f"{anom:+.0f}%", ha="center", fontsize=6, color=color, fontweight="bold")
    
    fig.tight_layout()
    return fig


def main():
    fig = fig_seasonal_outlook()
    paths = C.save_figure(fig, "fig_seasonal_outlook")
    print(f"Seasonal outlook figure saved to: {[str(p) for p in paths]}")


if __name__ == "__main__":
    main()
