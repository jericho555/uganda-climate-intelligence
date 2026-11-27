"""Yield forecast figure: scatter plot + projection bar.

Panel (a): Scatter of observed yield anomaly vs growing-season SPI with OLS fit line
           and 95% confidence band.
Panel (b): Bar chart showing current-year projection with 95% prediction interval.

Outputs: figures/fig_yield_forecast.png (mirrored to manuscript/figures/)
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

plt.rcParams.update({"font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_yield_forecast():
    """Two-panel figure: (a) scatter + fit, (b) projection bar."""
    # Load data
    forecast_df = pd.read_csv(RESULTS / "tables/yield_forecast.csv")
    summary = json.loads((RESULTS / "yield_forecast_summary.json").read_text())
    model = summary["model"]
    projection = summary["projection"]
    
    fig, (ax_scatter, ax_bar) = plt.subplots(1, 2, figsize=(12, 5),
                                              gridspec_kw={"width_ratios": [1.3, 1]})
    
    # ---- Panel (a): Scatter + fit ----
    spi = np.array(model["spi"])
    ya = np.array(model["yield_anomaly"])
    fitted = np.array(model["fitted"])
    
    ax_scatter.scatter(spi, ya, s=50, c="#2171b5", alpha=0.7, edgecolor="white",
                       linewidth=0.5, zorder=5, label="Observed")
    
    # Fit line
    spi_range = np.linspace(spi.min(), spi.max(), 100)
    slope = model["slope"]
    intercept = model["intercept"]
    fit_line = intercept + slope * spi_range
    ax_scatter.plot(spi_range, fit_line, color="#d7301f", linewidth=2,
                    label=f"OLS fit (R²={model['r_squared']:.2f})")
    
    # 95% confidence band (compute residuals from observed - fitted)
    resid = ya - fitted
    resid_std = float(np.std(resid))
    ax_scatter.fill_between(spi_range, fit_line - 1.96 * resid_std,
                            fit_line + 1.96 * resid_std, alpha=0.2, color="#d7301f",
                            label="95% CI")
    
    # Zero line
    ax_scatter.axhline(0, color="#999999", linestyle=":", linewidth=0.8)
    
    ax_scatter.set_xlabel("Growing-season SPI-3 (national mean)")
    ax_scatter.set_ylabel("Yield anomaly (%)")
    ax_scatter.set_title("(a) Yield anomaly vs growing-season SPI",
                         fontsize=9, fontweight="bold", loc="left")
    ax_scatter.legend(fontsize=7, loc="upper left", framealpha=0.9, edgecolor="#cccccc")
    
    # Skill note
    skill = summary["skill_check"]
    skill_text = f"Skill: {skill.get('verdict', 'N/A').replace('_', ' ')}\n"
    skill_text += f"(ratio={skill.get('skill_ratio', 'N/A')})"
    ax_scatter.text(0.98, 0.02, skill_text, transform=ax_scatter.transAxes,
                    fontsize=6, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8))
    
    # ---- Panel (b): Projection bar ----
    year = summary["projection_year"]
    proj = projection["projected_anomaly_pct"]
    pi_lo, pi_hi = projection["prediction_interval_95"]
    
    # Clip to reasonable range for visualization
    pi_lo_clip = max(pi_lo, -50)
    pi_hi_clip = min(pi_hi, 50)
    
    ax_bar.bar([year], [proj], width=0.6, color="#2171b5", edgecolor="white",
               linewidth=1.5, zorder=5)
    
    # Prediction interval
    ax_bar.errorbar([year], [proj], yerr=[[proj - pi_lo_clip], [pi_hi_clip - proj]],
                    fmt="none", color="#333333", capsize=8, capthick=2, linewidth=2, zorder=6)
    
    # Zero line
    ax_bar.axhline(0, color="#999999", linestyle="-", linewidth=0.8)
    
    # Threshold line (10th percentile)
    threshold = projection["threshold_pct"]
    ax_bar.axhline(threshold, color="#d7301f", linestyle="--", linewidth=1,
                   label=f"Below-normal threshold ({threshold:.1f}%)")
    
    ax_bar.set_ylabel("Yield anomaly (%)")
    ax_bar.set_title(f"(b) {year} projection",
                     fontsize=9, fontweight="bold", loc="left")
    ax_bar.set_xticks([year])
    ax_bar.legend(fontsize=7, loc="upper left", framealpha=0.9, edgecolor="#cccccc")
    
    # P(below-normal) annotation
    p_below = projection["p_below_normal"]
    ax_bar.text(0.98, 0.98, f"P(below-normal)\n= {p_below:.1%}",
                transform=ax_bar.transAxes, fontsize=7, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fee090", ec="#fdae61", alpha=0.9))
    
    fig.tight_layout()
    return fig


def main():
    fig = fig_yield_forecast()
    paths = C.save_figure(fig, "fig_yield_forecast")
    print(f"Yield forecast figure saved to: {[str(p) for p in paths]}")


if __name__ == "__main__":
    main()
