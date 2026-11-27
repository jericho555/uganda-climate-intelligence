"""Figures for the RQ6 early-warning deliverable.

Three products, all drawn from real pipeline outputs (no hard-coded values):
  fig_ews_architecture.png  -- the operational deployment architecture (schematic)
  fig_risk_nowcast.png      -- the district-month risk nowcast surface + honest skill
  fig_intervals.png         -- measured coverage of the split-conformal carry intervals
Run after early_warning.py. Every figure is written into both figures/ and
manuscript/figures/ by config.save_figure, at the resolution pinned there.
The map panel shares the district geometry layer used by every other map in the
paper (data/district_boundaries.geojson), so it is a choropleth, not a bubble plot.
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
from matplotlib import patches as mp
from figures_maps import fill, district_of, _frame, _label, _map_axes

plt.rcParams.update({"font.size": 9})               # dpi/bbox owned by config.save_figure
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
    C.save_figure(fig, "fig_ews_architecture")


def fig_risk_nowcast():
    sm = json.loads((RESULTS / "early_warning_summary.json").read_text())
    dm = pd.read_csv(RESULTS / "tables/risk_district_month.csv")
    nat = pd.read_csv(RESULTS / "tables/risk_national_month.csv")
    ct = centroid_table()

    fig = plt.figure(figsize=(11, 5.4))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.15, 1], hspace=0.32, wspace=0.26)

    # ---- (a) district risk map (latest month), as a district choropleth ----
    ax0 = fig.add_subplot(gs[:, 0])
    _map_axes(ax0)
    last = dm.sort_values(["year", "month"]).groupby("location").tail(1)
    last = last.merge(ct[["location", "lat", "lon"]], on="location")
    colors = {district_of(loc): BAND[str(b)] for loc, b in
              zip(last.location, last.band)}
    fill(ax0, colors=colors)
    _frame(ax0)
    for _, r in last.nlargest(6, "risk").iterrows():
        _label(ax0, r.lon, r.lat, f"{r.location} {r.risk:.0f}")
    lm = last[["year", "month"]].iloc[0]
    ax0.set_title(f"(a) District risk nowcast, {int(lm.year)}-{int(lm.month):02d}\n"
                  f"({len(last)} of 135 districts observed; unobserved = grey)",
                  fontsize=9)
    handles = [mp.Patch(color=v, label=k) for k, v in BAND.items()]
    handles.append(mp.Patch(color="#e9e9e6", label="not observed"))
    ax0.legend(handles=handles, fontsize=7, loc="lower left", title="risk band",
               title_fontsize=7, framealpha=0.9)

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

    C.save_figure(fig, "fig_risk_nowcast")


def fig_intervals():
    """What the conformal carry intervals are worth, measured on held-out years.

    Panel (a) shows the one-month carry itself: every 2019-2023 district-month
    transition plotted as (this month's score, next month's observed score), with
    the bands the conformal quantiles draw around the persistence point forecast.
    Panel (b) reports the coverage those bands actually achieved, overall and
    inside each starting band, for both point models. The widths come from the
    published summary, so the figure cannot drift from the paper's numbers; only
    the scatter points are recomputed, through the same carry_table() the
    calibration used.
    """
    import early_warning as EW
    sm = json.loads((RESULTS / "early_warning_summary.json").read_text())
    pi = sm.get("prediction_intervals") or {}
    if not pi.get("models"):
        raise RuntimeError("early_warning_summary.json has no prediction_intervals "
                           "-- run src/early_warning.py first")
    dm = pd.read_csv(RESULTS / "tables/risk_district_month.csv")
    X, y, tgt_year, _ = EW.carry_table(dm)
    test = tgt_year >= C.EVAL_TEST_YEARS[0]
    now = X[:, EW.CARRY_FEATS.index("risk")]
    pm = pi["models"]["persistence"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    xg = np.linspace(0, 100, 400)
    for lvl, col in (("level_90", "#cfe0f3"), ("level_80", "#74a9cf")):
        q = pm[lvl]["conformal_quantile"]
        ax.fill_between(xg, np.clip(xg - q, 0, 100), np.clip(xg + q, 0, 100),
                        color=col, alpha=0.6, lw=0,
                        label=f"{lvl[-2:]}% band ($\\pm${q:.1f})")
    ax.plot(xg, xg, "k--", lw=0.8, label="perfect carry")
    ax.scatter(now[test], y[test], s=9, alpha=0.35, color="#251c70",
               edgecolor="none", label="held-out transition")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("this month's risk score (persistence point forecast)")
    ax.set_ylabel("next month's observed risk score")
    ax.set_title(f"(a) The one-month carry, {C.EVAL_TEST_YEARS[0]}-"
                 f"{C.EVAL_TEST_YEARS[1]}\n"
                 f"90% band covers {pm['level_90']['coverage_test']:.1%} of "
                 f"{pi['n_test']} held-out transitions", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    bands = [b for b in ["Low", "Moderate", "High", "Extreme"]
             if any(b in pi["models"][m][lvl]["coverage_by_starting_band"]
                    for m in pi["models"] for lvl in ("level_80", "level_90"))]
    cats = ["all"] + bands
    xpos = np.arange(len(cats))
    series = [("persistence", "level_90", "#2c7fb8", "persistence, 90%"),
              ("ridge", "level_90", "#a63603", "fitted carry, 90%"),
              ("persistence", "level_80", "#92c5de", "persistence, 80%"),
              ("ridge", "level_80", "#f4a582", "fitted carry, 80%")]
    w = 0.2
    for i, (m, lvl, col, lab) in enumerate(series):
        rec = pi["models"][m][lvl]
        vals = [rec["coverage_test"]] + [rec["coverage_by_starting_band"]
                                         .get(b, {}).get("coverage", np.nan)
                                         for b in bands]
        ax.bar(xpos + (i - 1.5) * w, vals, w, color=col, label=lab)
    ax.axhline(0.90, color="#2c7fb8", lw=0.9, ls=":")
    ax.axhline(0.80, color="#a63603", lw=0.9, ls=":")
    ns = [pi["n_test"]] + [pm["level_90"]["coverage_by_starting_band"]
                           .get(b, {}).get("n", 0) for b in bands]
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{c}\n(n={m:,})" for c, m in zip(cats, ns)], fontsize=7.5)
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("measured coverage")
    ax.set_xlabel("starting risk band of the origin month")
    ax.set_title("(b) Coverage against nominal\n"
                 "dotted lines mark the 0.90 / 0.80 targets", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper center", ncol=2, framealpha=0.95)

    fig.tight_layout()
    C.save_figure(fig, "fig_intervals")


def main():
    made = []
    for f in (fig_architecture, fig_risk_nowcast, fig_intervals):
        try:
            f(); made.append(f.__name__)
        except Exception as e:
            print(f"[fig skip] {f.__name__}: {e!r}")
    print("early-warning figures written:", made)
    for folder in C.FIG_FOLDERS:
        print("dir:", folder)


if __name__ == "__main__":
    main()
