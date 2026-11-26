"""Figures for the RQ6 early-warning deliverable.

Two products, both drawn from real pipeline outputs (no hard-coded values):
  fig_ews_architecture.png  -- the operational deployment architecture (schematic)
  fig_risk_nowcast.png      -- the district-month risk nowcast surface + honest skill
Run after early_warning.py. Writes directly into manuscript/figures/ (the path
main.tex includes from).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS, MANUSCRIPT
from spatial import centroid_table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import gridspec

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "savefig.bbox": "tight"})
FIGDIR = MANUSCRIPT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

BAND = {"Low": "#4575b4", "Moderate": "#fee090", "High": "#fdae61", "Extreme": "#d7301f"}


def _box(ax, x, y, w, h, title, body, fc, ec="#333333"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x + w/2, y + h - 0.045, title, ha="center", va="top",
            fontsize=8.6, fontweight="bold")
    ax.text(x + w/2, y + h/2 - 0.03, body, ha="center", va="center", fontsize=7.3)


def _arrow(ax, x1, y1, x2, y2, color="#444", style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, linewidth=1.4, color=color,
                                 shrinkA=1, shrinkB=1))


def fig_architecture():
    """Six-layer operational workflow of the Uganda Climate Early-Warning Framework."""
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    layers = [
        ("1. Data ingest",
         "ERA5 / Open-Meteo (daily)  +  Meteostat in-situ (century)\n"
         "NASA POWER gap-fill  +  disaster register  +  World Bank crops",
         "#e8f0fb"),
        ("2. Pre-processing",
         "running-window seasonal anomalies  +  antecedent indices\n"
         "(5/10/30/60-day)  +  leakage-safe climatology",
         "#eaf4ea"),
        ("3. Detection ensemble",
         "17 detectors (z-score, LOF, iForest, ECOD, COPOD,\n"
         "AutoEncoder, ...) -> ensemble score, validated vs register",
         "#fdf2df"),
        ("4. Risk scoring  (UCASI nowcast)",
         "per district-month: intensity, persistence, spatial\n"
         "coherence, multi-variable, frequency -> 0-100, Low..Extreme",
         "#fde4e0"),
        ("5. Forward nowcast + alert tiers",
         "persistence carry to next week / month / season;\n"
         "ranked district watch-list; measured (not assumed) skill",
         "#efe0f5"),
        ("6. Decision support",
         "UNMA / district prioritisation, allocation of\n"
         "preparedness resources; public risk surface",
         "#e0f0f2"),
    ]
    x, w = 0.07, 0.62
    ys, hh = [], 0.128
    top = 0.965
    for i, (t, b, fc) in enumerate(layers):
        y = top - (i + 1) * (hh + 0.028)
        ys.append(y)
        _box(ax, x, y, w, hh, t, b, fc)
        if i > 0:
            _arrow(ax, x + w/2, ys[i-1], x + w/2, y + hh + 0.002)

    # feedback loop: register -> (re)calibration
    ax.add_patch(FancyArrowPatch((x + w, ys[3] + hh/2), (x + w + 0.06, ys[1] + hh/2),
                                 connectionstyle="arc3,rad=-0.5", arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.1, color="#b24",
                                 linestyle="--"))
    ax.text(x + w + 0.10, (ys[1] + ys[3]) / 2 + 0.02,
            "ground-truth\nre-calibration", fontsize=6.8, color="#b24", rotation=0)

    # the honest gap box (what a weather-only space omits)
    gx = 0.755
    _box(ax, gx, ys[4], 0.235, hh*2 + 0.03,
         "Skill ceiling\n(known gap)",
         "Predictive event skill needs\ncoupling: soil moisture,\nexposure, geology/land-use,\nimpact data.",
         "#f2f2f2", ec="#c33")
    _arrow(ax, gx + 0.02, ys[4] + (hh*2+0.03)/2, x + w + 0.002, ys[4] + hh/2, color="#c33")
    _arrow(ax, gx + 0.02, ys[3], x + w + 0.002, ys[3] + hh/2, color="#c33")

    ax.set_title("Uganda Climate Early-Warning Framework: operational deployment architecture",
                 fontsize=10.5, pad=10)
    fig.savefig(FIGDIR / "fig_ews_architecture.png")
    plt.close(fig)


def fig_risk_nowcast():
    sm = json.loads((RESULTS / "early_warning_summary.json").read_text())
    dm = pd.read_csv(RESULTS / "tables/risk_district_month.csv")
    nat = pd.read_csv(RESULTS / "tables/risk_national_month.csv")
    ct = centroid_table()

    fig = plt.figure(figsize=(11, 5.4))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1], hspace=0.32, wspace=0.26)

    # ---- (a) district risk map (latest month) ----
    ax0 = fig.add_subplot(gs[:, 0])
    last = dm.sort_values(["year", "month"]).groupby("location").tail(1)
    last = last.merge(ct[["location", "lat", "lon"]], on="location")
    rmax = last.risk.max()
    for _, r in last.iterrows():
        c = BAND[str(r.band)]
        frac = r.risk / rmax
        ax0.add_patch(plt.Circle((r.lon, r.lat), radius=0.35 + 0.9*frac, color=c,
                                 alpha=0.25 + 0.45*frac, zorder=1))
        ax0.plot(r.lon, r.lat, "o", ms=5 + 16*frac, color=c, mec="k",
                 mew=0.5, zorder=2)
        ax0.annotate(f"{r.location} {r.risk:.0f}", (r.lon, r.lat), fontsize=6.0,
                     xytext=(3, 3), textcoords="offset points", zorder=3)
    ax0.set_xlim(29.2, 35.2); ax0.set_ylim(-1.8, 3.7)
    ax0.set_aspect("equal")
    ax0.set_xlabel("Longitude"); ax0.set_ylabel("Latitude")
    lm = last[["year", "month"]].iloc[0]
    ax0.set_title(f"(a) District risk nowcast, {int(lm.year)}-{int(lm.month):02d} "
                  f"({len(last)} districts)", fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=v, label=k)
               for k, v in BAND.items()]
    ax0.legend(handles=handles, fontsize=7, loc="lower left", title="risk band",
               title_fontsize=7)

    # ---- (b) national monthly risk with disaster windows ----
    ax1 = fig.add_subplot(gs[0, 1])
    idx = pd.to_datetime(dict(year=nat.year, month=nat.month, day=1))
    ax1.plot(idx, nat.risk, color="#2c7fb8", lw=1.1)
    ax1.fill_between(idx, 0, nat.risk, color="#2c7fb8", alpha=0.12)
    reg = pd.read_csv(DATA / "disaster_register.csv", parse_dates=["start", "end"])
    for _, e in reg.iterrows():
        ax1.axvspan(e.start, e.end, color="#d7301f", alpha=0.10, lw=0)
    ax1.set_ylabel("national mean risk")
    ax1.set_title("(b) Monthly risk nowcast vs disaster windows", fontsize=9)
    ax1.tick_params(axis="x", labelsize=7)

    # ---- (c) honest skill vs size-matched random null ----
    ax2 = fig.add_subplot(gs[1, 1])
    sk = sm["skill_vs_event_months"]; pe = sm["persistence"]
    labels = ["ROC-AUC", "recall@TopK"]
    model = [sk["auc"], sk["recall_at_k"]]
    null = [sk["auc_null_mean"], sk["recall_null_mean"]]
    xpos = np.arange(len(labels))
    ax2.bar(xpos - 0.18, model, 0.36, color="#2c7fb8", label="UCASI nowcast")
    ax2.bar(xpos + 0.18, null, 0.36, color="#bbbbbb", label="random null")
    for i, (m, n) in enumerate(zip(model, null)):
        ax2.text(i - 0.18, m + 0.01, f"{m:.2f}", ha="center", fontsize=7.5)
        ax2.text(i + 0.18, n + 0.01, f"{n:.2f}", ha="center", fontsize=7.5)
    ax2.set_xticks(xpos); ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 0.75); ax2.legend(fontsize=7)
    ax2.set_title(f"(c) Skill at chance (lag-1 persistence r={pe['pooled_lag1_r']:.2f})",
                  fontsize=9)

    fig.savefig(FIGDIR / "fig_risk_nowcast.png")
    plt.close(fig)


def main():
    made = []
    for f in (fig_architecture, fig_risk_nowcast):
        try:
            f(); made.append(f.__name__)
        except Exception as e:
            print(f"[fig skip] {f.__name__}: {e!r}")
    print("early-warning figures written:", made)
    print("dir:", FIGDIR)


if __name__ == "__main__":
    main()
