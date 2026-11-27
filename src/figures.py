"""Generate publication figures from the real pipeline outputs.

Every figure is drawn from files produced by the earlier stages; no values are
hard-coded. Run after detect/spatial/temporal/ucasi/explain.
"""
import sys, warnings, json
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
from sklearn.metrics import roc_curve, precision_recall_curve, auc

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})
ZCOLS = [f"{f}_z" for f in C.FEATURES]


def _top_det_order(res, k=6):
    return res.sort_values("pr_auc", ascending=False).detector.head(k).tolist()


def fig_benchmark():
    res = pd.read_csv(RESULTS / "benchmark.csv")
    res = res.sort_values("pr_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(res))
    ax.bar(x - 0.2, res.pr_auc, 0.4, label="PR-AUC", color="#2c7fb8")
    ax.bar(x + 0.2, res.roc_auc, 0.4, label="ROC-AUC", color="#7fcdbb")
    ax.errorbar(x, res.f1, yerr=[res.f1 - res.f1_lo, res.f1_hi - res.f1],
                fmt="ko-", ms=4, lw=1, capsize=3, label="F1 (95% CI)")
    ax.set_xticks(x); ax.set_xticklabels(res.detector, rotation=35, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title("Held-out anomaly-detection benchmark vs disaster ground-truth")
    ax.legend(fontsize=8)
    fig.savefig(FIGURES / "fig_benchmark.png"); plt.close(fig)


def fig_curves():
    z = np.load(RESULTS / "test_scores.npz", allow_pickle=False)
    y = z["y"]
    res = pd.read_csv(RESULTS / "benchmark.csv")
    res = res[res.detector.isin(list(z.files) + ["y", "year"])]
    order = _top_det_order(res, 5)
    order = [n for n in order if n in z.files][:5]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(order)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for name, col in zip(order, colors):
        s = z[name]
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, color=col, lw=1.4, label=f"{name} ({auc(fpr,tpr):.2f})")
        prec, rec, _ = precision_recall_curve(y, s)
        axes[1].plot(rec, prec, color=col, lw=1.4, label=f"{name} ({auc(rec,prec):.2f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (legend = AUC)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (legend = AP)")
    for a in axes:
        a.legend(fontsize=7)
    fig.savefig(FIGURES / "fig_roc_pr.png"); plt.close(fig)


def fig_map():
    """District Getis-Ord classification, drawn as a district choropleth.

    Rendering lives in figures_maps so that every map in the paper shares one
    geometry layer (data/district_boundaries.geojson) and one no-data rule.
    """
    from figures_maps import fig_hotspot_map
    fig_hotspot_map()


def fig_ucasi():
    ev = pd.read_csv(RESULTS / "tables/ucasi_events.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    ev = ev.sort_values("UCASI", ascending=False)
    cmap = {"Low": "#4575b4", "Moderate": "#fee090", "High": "#fdae61", "Extreme": "#d7301f"}
    axes[0].bar(range(len(ev)), ev.UCASI, color=[cmap[str(s)] for s in ev.severity])
    axes[0].set_xticks(range(len(ev)))
    axes[0].set_xticklabels(ev.event_id, rotation=90, fontsize=6)
    axes[0].set_ylabel("UCASI (0-100)"); axes[0].set_title("Per-event UCASI severity")
    from matplotlib import patches as mp
    handles = [mp.Patch(color=v, label=k) for k, v in cmap.items()]
    axes[0].legend(handles=handles, fontsize=7, loc="upper right")
    types = sorted(ev.type.unique())
    axes[1].boxplot([g.UCASI.values for _, g in ev.groupby("type")])
    axes[1].set_xticks(range(1, len(types) + 1)); axes[1].set_xticklabels(types)
    axes[1].set_ylabel("UCASI"); axes[1].set_title("UCASI by event type")
    fig.savefig(FIGURES / "fig_ucasi.png"); plt.close(fig)


def fig_temporal():
    df = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"])
    monthly = df.groupby(pd.Grouper(key="date", freq="MS")).is_anomaly.sum()
    cp = json.loads((RESULTS / "change_points.json").read_text())
    idxs = cp.get("monthly_anom_count_bkps_index", [])[:-1] if cp.get("monthly_anom_count_bkps_index") else []
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(monthly.index, monthly.values, color="#333", lw=1)
    for i in idxs:
        if i < len(monthly):
            ax.axvline(monthly.index[i], color="#d7301f", ls="--", lw=0.8)
    reg = pd.read_csv(DATA / "disaster_register.csv", parse_dates=["start"])
    for _, e in reg.iterrows():
        if e.start in monthly.index:
            ax.axvline(e.start, color="#2c7fb8", alpha=0.25, lw=0.7)
    ax.set_ylabel("district-days flagged")
    ax.set_title("National monthly anomaly burden; red = change points, blue = disaster events")
    fig.savefig(FIGURES / "fig_temporal.png"); plt.close(fig)

    tr = pd.read_csv(RESULTS / "tables/temporal_trends.csv").dropna(subset=["tau"])
    tr = tr[tr.variable == "precipitation_sum"].sort_values("tau")
    fig, ax = plt.subplots(figsize=(6, 4))
    c = ["#d7301f" if (abs(t) > 0 and p < 0.05 and t > 0) else ("#4575b4" if p < 0.05 else "#bbbbbb")
         for t, p in zip(tr.tau, tr.p_value)]
    ax.barh(tr.location, tr.tau, color=c)
    ax.set_xlabel("Mann-Kendall tau (precipitation, monthly)"); ax.axvline(0, color="k", lw=0.6)
    ax.set_title("Precipitation trends by district (red=wetting, blue=drying, p<.05)")
    fig.savefig(FIGURES / "fig_mannkendall.png"); plt.close(fig)


def fig_shap():
    sv = np.load(RESULTS / "shap_values.npy")
    si = pd.read_csv(RESULTS / "tables/shap_importance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].barh(si.variable[::-1], si.shap_mean_abs[::-1], color="#2c7fb8")
    axes[0].set_xlabel("mean |SHAP|"); axes[0].set_title("Global feature importance (SHAP)")
    try:
        shap.summary_plot(sv, feature_names=C.FEATURES, show=False, max_display=9)
        axes[1].get_subplots()  # no-op
        fig2 = plt.gcf()
        fig2.savefig(FIGURES / "fig_shap_beeswarm.png"); plt.close(fig2)
    except Exception as e:
        axes[1].text(0.5, 0.5, f"beeswarm skipped: {e!r}", ha="center")
    fig.savefig(FIGURES / "fig_shap.png"); plt.close(fig)


def fig_case():
    df = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"])
    # pick the district-year with the most extreme UCASI event window
    ev = pd.read_csv(RESULTS / "tables/ucasi_events.csv").sort_values("UCASI", ascending=False)
    top = ev.iloc[0]
    regs = top.regions.split(";")
    loc = df[df.region.isin(regs)].groupby("location").is_anomaly.sum().idxmax()
    g = df[df.location == loc].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(g.date, g.precipitation_sum, color="#4575b4", lw=0.8, label="precip (mm/day)")
    anom = g[g.is_anomaly == 1]
    ax.scatter(anom.date, anom.precipitation_sum, color="#d7301f", s=12, label="detected anomaly")
    ax.axvline(pd.Timestamp(top.start), color="k", ls=":", label=f"{top.event_id} onset")
    ax.set_title(f"Case study: {loc} (region {regs[0]}) around {top.type} event {top.event_id}, UCASI={top.UCASI}")
    ax.set_ylabel("precipitation (mm/day)"); ax.legend(fontsize=7, ncol=3)
    fig.savefig(FIGURES / "fig_case_study.png"); plt.close(fig)


def main():
    import shap
    figs = [fig_benchmark, fig_curves, fig_map, fig_ucasi, fig_temporal, fig_shap, fig_case]
    made = []
    for f in figs:
        try:
            f(); made.append(f.__name__)
        except Exception as e:
            print(f"[fig skip] {f.__name__}: {e!r}")
    print("figures written:", made)
    print("dir:", FIGURES)


if __name__ == "__main__":
    main()
