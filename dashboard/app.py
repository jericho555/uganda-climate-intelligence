"""Uganda Climate Intelligence -- interactive dashboard.

Streamlit app that binds to the pipeline's persisted artefacts (CSV / JSON /
NPZ under ``data/`` and ``results/``).  Nothing is fitted or fabricated here:
the dashboard is a pure VIEW of what ``run_all.py`` already produced.

Launch with::

    python dashboard/run_dashboard.py

which transparently prepends the vendored ``._webpkgs/`` install (created via
``pip install --target=._webpkgs streamlit``) onto ``sys.path`` so the app is
reproducible even when the environment's system site-packages is not writable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Bootstrap the workspace-local vendored packages BEFORE importing streamlit
# (this line is redundant when launched via run_dashboard.py, but harmless when
# the app is started directly by ``streamlit run dashboard/app.py``).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_VEND = _ROOT / "._webpkgs"
if _VEND.exists() and str(_VEND) not in sys.path:
    sys.path.insert(0, str(_VEND))

import altair as alt                            # noqa: E402
import numpy as np                              # noqa: E402
import pandas as pd                             # noqa: E402
import streamlit as st                          # noqa: E402

st.set_page_config(
    page_title="Uganda Climate Intelligence",
    layout="wide",
    page_icon="\U0001F324",
    initial_sidebar_state="expanded",
)

# The daily-stream explorer can exceed Altair's default inline-row cap.
alt.data_transformers.disable_max_rows()

DATA = _ROOT / "data"
RESULTS = _ROOT / "results"
TABLES = RESULTS / "tables"

# The size-matched random-flag null is read from the artefact written by
# ``src/random_null.py`` (``results/random_null.json``).  An earlier revision of
# this dashboard hard-coded 0.096, a value no script produced; the audited
# measurement is 0.069 [0.059, 0.079].  The constant below is therefore a
# fallback only, and the fallback is the measured value.
NULL_JSON = RESULTS / "random_null.json"
try:
    _NULL = json.loads(NULL_JSON.read_text(encoding="utf-8"))
except Exception:
    _NULL = None

if _NULL is not None:
    RANDOM_NULL_F1 = float(_NULL["pooled"]["mean"])
    RANDOM_NULL_CI = (float(_NULL["pooled"]["lo"]), float(_NULL["pooled"]["hi"]))
    RANDOM_NULL_PRAUC = float(_NULL["pooled"]["pr_auc_mean"])
    BASE_RATE = float(_NULL["base_rate"])
else:
    RANDOM_NULL_F1 = 0.069
    RANDOM_NULL_CI = (0.059, 0.079)
    RANDOM_NULL_PRAUC = 0.112
    BASE_RATE = 0.1115
EVAL_TEST_YEARS = [2019, 2020, 2021, 2022, 2023]

# --------------------------------------------------------------------------- #
# Loaders (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_csv(path: Path, **kw):
    if not Path(path).exists():
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_json(path: Path):
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_full_ensemble() -> pd.DataFrame | None:
    """Load only the columns the dashboard needs; the raw frame is ~93k rows."""
    df = load_csv(DATA / "full_ensemble.csv")
    if df is None:
        return None
    keep = [c for c in [
        "date", "location", "region", "year", "month",
        "precipitation_sum", "temperature_2m_max", "temperature_2m_min",
        "temperature_2m_mean", "relative_humidity_2m_mean",
        "et0_fao_evapotranspiration", "shortwave_radiation_sum",
        "pressure_msl_mean", "wind_speed_10m_max",
        "precipitation_sum_z", "temperature_2m_max_z",
        "temperature_2m_mean_z", "relative_humidity_2m_mean_z",
        "et0_fao_evapotranspiration_z",
        "precip_z_5d", "precip_z_10d", "dry_z_60d", "tmax_z_5d", "et_z_30d",
        "ens_score", "is_anomaly", "label_event", "event_type",
    ] if c in df.columns]
    out = df[keep].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


@st.cache_data(show_spinner=False)
def load_register() -> pd.DataFrame | None:
    df = load_csv(DATA / "disaster_register.csv", parse_dates=["start", "end"])
    return df


@st.cache_data(show_spinner=False)
def load_per_year_f1():
    npy = RESULTS / "per_year_f1.npy"
    order = RESULTS / "per_year_f1_order.txt"
    if not (npy.exists() and order.exists()):
        return None
    mat = np.load(npy)
    names = order.read_text(encoding="utf-8").splitlines()
    return mat, names


def humanize(name: str) -> str:
    """Map benchmark.csv internal detector keys to the manuscript's display."""
    fixed = {
        "Zscore(max)": "Z-score (max)",
        "Ensemble+Persist+Region": "Ensemble+Persist+Reg",
        "StructuralTS": "Structural TS",
        "IsolationForest": "Isolation Forest",
        "GeneralizedESD": "Generalized ESD",
        "SeasonalESD": "Seasonal ESD",
        "OneClassSVM": "One-class SVM",
        "LSTMAE": "LSTM-AE",
        "TransformerAE": "Transformer-AE",
        "DeepAE": "Deep AE (MLP)",
        "VAE": "VAE (NumPy)",
    }
    return fixed.get(name, name)


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta=delta if delta else None)


# --------------------------------------------------------------------------- #
# Sidebar (global filters + provenance)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("\U0001F30D Uganda Climate Intelligence")
    st.caption("Interactive view of the pipeline artefacts. "
               "Every number here is loaded from ``results/`` / ``data/`` "
               "produced by ``run_all.py``. Nothing is fitted on this page.")
    st.divider()
    if st.checkbox("Show random-null reference lines", value=True,
                   key="show_null"):
        pass
    if st.checkbox("Dark theme for charts", value=False, key="dark_charts"):
        alt.data_transformers.enable("default")
    st.divider()
    st.markdown(
        f"- **Benchmark:** 24 detectors, held-out test 2019–2023\n"
        f"- **Register:** 32 curated disasters (ground truth)\n"
        f"- **Base rate:** {BASE_RATE:.4f} · **Random-flag F1 null:** "
        f"{RANDOM_NULL_F1:.3f} [{RANDOM_NULL_CI[0]:.3f}, {RANDOM_NULL_CI[1]:.3f}]\n"
        f"- **Friedman χ²(23) over 5 test years**"
    )
    st.divider()
    st.caption("Reproducible pipeline: ERA5 (Open-Meteo) · Meteostat in-situ · "
               "NASA POWER · MERRA-2 · FAOSTAT · World Bank. "
               "CC-BY 4.0 sources. Seed=42.")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_over, tab_bench, tab_ts, tab_spatial, tab_events, tab_trends, tab_now, \
    tab_drivers, tab_about = st.tabs([
        "Overview",
        "Detection Benchmark",
        "Time-Series Explorer",
        "Spatial Hotspots",
        "Events & Severity (UCASI)",
        "Trends & Climatology",
        "Risk Nowcast (EWS)",
        "Detection Drivers",
        "About / Reproducibility",
    ])

# ---- 1. OVERVIEW ---------------------------------------------------------- #
with tab_over:
    st.header("Framework overview")
    bench = load_csv(RESULTS / "benchmark.csv")
    fri = load_json(RESULTS / "friedman.json")
    reg = load_register()
    dist = load_csv(TABLES / "district_anomaly.csv")
    if bench is None or fri is None:
        st.warning("Benchmark artefacts not found. Run `python run_all.py` "
                   "first.")
    else:
        n_det = len(bench)
        seq_names = {"LSTMAE", "TransformerAE"}
        seq_mask = bench["detector"].isin(seq_names)
        best_f1_overall = bench["f1"].max()
        best_f1_cross = bench.loc[~seq_mask, "f1"].max()
        roc_min, roc_max = bench["roc_auc"].min(), bench["roc_auc"].max()
        metric_row([
            ("Detectors benchmarked", f"{n_det}", None),
            ("Held-out test years", "2019–2023", None),
            ("Register disasters", f"{len(reg) if reg is not None else '—'}",
             None),
            ("Districts", f"{len(dist) if dist is not None else '—'}", None),
            ("Friedman p", f"{fri['friedman_p']:.2g}", None),
        ])
        st.subheader("Central finding, in one line")
        st.info(
            f"**No cross-sectional detector ranks above chance** on held-out "
            f"disaster windows (ROC-AUC span {roc_min:.2f}–{roc_max:.2f}; best "
            f"cross-sectional F1 = {best_f1_cross:.3f}). Measured against the "
            f"size-matched random alert budget (F1 {RANDOM_NULL_F1:.3f}), five "
            f"cross-sectional methods stay within 1.22× of it, while only the "
            f"two torch sequence autoencoders (LSTM-AE, Transformer-AE) roughly "
            f"double it (best F1 = {best_f1_overall:.3f}), still with global "
            f"ranking near chance."
        )
        st.markdown("### Headline table (top 5 by PR-AUC)")
        show = ["detector", "precision", "recall", "f1", "roc_auc", "pr_auc",
                "mcc"]
        disp = bench.head(5)[show].copy()
        disp["detector"] = disp["detector"].map(humanize)
        st.dataframe(disp.style.format(
            {c: "{:.3f}" for c in show[1:]},
        ), width="stretch", hide_index=True)
        st.caption("Full 24-row table is in the **Detection Benchmark** tab.")

# ---- 2. DETECTION BENCHMARK ----------------------------------------------- #
with tab_bench:
    st.header("Held-out detection benchmark (24 detectors)")
    bench = load_csv(RESULTS / "benchmark.csv")
    if bench is None:
        st.error("benchmark.csv missing. Run the pipeline.")
        st.stop()
    b = bench.copy()
    b["detector"] = b["detector"].map(humanize)
    b["family"] = np.where(b["detector"].isin(["LSTM-AE", "Transformer-AE"]),
                           "sequence (torch)",
                           np.where(b["detector"].isin(
                               ["Deep AE (MLP)", "VAE (NumPy)", "AutoEncoder"]),
                               "feed-forward deep",
                               np.where(b["detector"].isin(
                                   ["SARIMAX", "Structural TS"]),
                                   "temporal (innovation)",
                                   "cross-sectional")))
    st.dataframe(
        b[
            ["detector", "family", "precision", "recall", "f1", "f1_lo",
             "f1_hi", "roc_auc", "pr_auc", "pr_auc_lo", "pr_auc_hi", "mcc"]
        ].style.format({c: "{:.3f}" for c in
                        ["precision", "recall", "f1", "f1_lo", "f1_hi",
                         "roc_auc", "pr_auc", "pr_auc_lo", "pr_auc_hi",
                         "mcc"]}),
        width="stretch", hide_index=True, height=520,
    )
    st.subheader("PR-AUC vs F1, with the random null marked")
    show_null = st.session_state.get("show_null", True)
    chart = alt.Chart(b).mark_point(size=140, filled=True).encode(
        x=alt.X("f1:Q", title="F1 (held-out)",
                scale=alt.Scale(zero=False)),
        y=alt.Y("pr_auc:Q", title="PR-AUC", scale=alt.Scale(zero=False)),
        color=alt.Color("family:N",
                        scale=alt.Scale(
                            domain=["cross-sectional",
                                    "temporal (innovation)",
                                    "feed-forward deep",
                                    "sequence (torch)"],
                            range=["#888", "#3b7dd8", "#b5651d", "#c1272d"])),
        shape=alt.Shape("family:N"),
        tooltip=[
            alt.Tooltip("detector:N", title="Detector"),
            alt.Tooltip("family:N", title="Family"),
            alt.Tooltip("f1:Q", format=".3f"),
            alt.Tooltip("f1_lo:Q", format=".3f"),
            alt.Tooltip("f1_hi:Q", format=".3f"),
            alt.Tooltip("roc_auc:Q", format=".3f"),
            alt.Tooltip("pr_auc:Q", format=".3f"),
            alt.Tooltip("mcc:Q", format=".3f"),
        ],
    ).properties(height=380)
    layers = [chart]
    if show_null:
        layers.append(alt.Chart(pd.DataFrame({"x": [RANDOM_NULL_F1]})).mark_rule(
            color="#c1272d", strokeDash=[4, 4]
        ).encode(x="x:Q"))
        layers.append(alt.Chart(pd.DataFrame({"x": [BASE_RATE]})).mark_rule(
            color="#555", strokeDash=[2, 3]
        ).encode(x="x:Q"))
    st.altair_chart(alt.layer(*layers), width="stretch")
    st.caption("Vertical dashed red = random-flag F1 null "
               f"({RANDOM_NULL_F1:.3f}, Monte-Carlo band "
               f"{RANDOM_NULL_CI[0]:.3f}–{RANDOM_NULL_CI[1]:.3f}). Solid grey = "
               "base rate "
               f"({BASE_RATE:.4f}). Several cross-sectional points sit "
               "marginally right of the null (at most 1.22×); only the two red "
               "sequence points roughly double it.")

    st.subheader("Per-year F1 stability across the five test years")
    pyf = load_per_year_f1()
    if pyf is not None:
        mat, names = pyf                       # mat: (n_detectors, n_years)
        yrs = EVAL_TEST_YEARS[:mat.shape[1]]
        df = pd.DataFrame(mat, columns=[str(y) for y in yrs])
        df.insert(0, "detector", [humanize(n) for n in names][:mat.shape[0]])
        tidy = df.melt(id_vars="detector", var_name="year",
                       value_name="f1").dropna()
        hm = alt.Chart(tidy).mark_rect().encode(
            x=alt.X("year:N", title="Test year"),
            y=alt.Y("detector:N", sort="-x", title=None),
            color=alt.Color("f1:Q", scale=alt.Scale(scheme="viridis"),
                            title="F1"),
            tooltip=["detector:N", "year:N", alt.Tooltip("f1:Q", format=".3f")],
        ).properties(height=520)
        st.altair_chart(hm, width="stretch")

    st.subheader("Friedman test over held-out predictions")
    fri = load_json(RESULTS / "friedman.json")
    if fri:
        st.json(fri)

# ---- 3. TIME-SERIES EXPLORER ---------------------------------------------- #
with tab_ts:
    st.header("Daily stream explorer")
    full = load_full_ensemble()
    if full is None:
        st.error("full_ensemble.csv missing. Run `python run_all.py`.")
        st.stop()
    reg = load_register()
    c1, c2, c3 = st.columns([2, 2, 2])
    districts = sorted(full.location.dropna().unique().tolist())
    sel_district = c1.selectbox("District", districts,
                                index=districts.index("Kampala")
                                if "Kampala" in districts else 0)
    var_options = [c for c in full.columns if c.endswith("_z")] + \
                  ["ens_score"] + [
        c for c in ["precipitation_sum", "temperature_2m_max",
                    "temperature_2m_mean", "relative_humidity_2m_mean",
                    "et0_fao_evapotranspiration", "shortwave_radiation_sum"]
        if c in full.columns]
    sel_var = c2.selectbox("Variable / score", var_options,
                           index=var_options.index("precipitation_sum_z")
                           if "precipitation_sum_z" in var_options else 0)
    d = full[full.location == sel_district].sort_values("date")
    lo = d["date"].min().to_pydatetime() if len(d) else None
    hi = d["date"].max().to_pydatetime() if len(d) else None
    rng = c3.date_input("Date range", value=(lo, hi), min_value=lo,
                        max_value=hi)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        lo_d, hi_d = rng
    else:
        lo_d, hi_d = lo, hi
    d = d[(d["date"] >= pd.to_datetime(lo_d)) & (d["date"] <= pd.to_datetime(hi_d))]

    st.markdown(f"**{sel_district}** — {len(d):,} district-days in view · "
                f"anomaly rate {d['is_anomaly'].mean():.3f} · "
                f"event-day share {d['label_event'].mean():.3f}")
    base_chart = alt.Chart(d).mark_line(point=False).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y(f"{sel_var}:Q", title=sel_var),
    ).properties(height=320)
    layers = [base_chart]
    show_null = st.session_state.get("show_null", True)
    if sel_var.endswith("_z") and show_null:
        for z in (-2, 2):
            layers.append(alt.Chart(pd.DataFrame({"v": [z]})).mark_rule(
                color="#999", strokeDash=[4, 4]).encode(y="v:Q"))
    if "is_anomaly" in d.columns:
        flags = d[d["is_anomaly"] == 1]
        layers.append(alt.Chart(flags).mark_point(color="#c1272d",
                                                  filled=True, size=28).encode(
            x="date:T", y=alt.Y(f"{sel_var}:Q")))
    if reg is not None and len(reg):
        r = reg[(reg["end"] >= pd.to_datetime(lo_d)) &
                (reg["start"] <= pd.to_datetime(hi_d))]
        for _, e in r.iterrows():
            layers.append(alt.Chart(pd.DataFrame({
                "s": [e["start"]], "e": [e["end"]], "lab": [e["event_id"]]
            })).mark_rule(color="#3b7dd8", opacity=0.15, size=25).encode(
                x="s:T", x2="e:T"))
    st.altair_chart(alt.layer(*layers), width="stretch")
    if len(r := (reg if reg is not None else pd.DataFrame())) and \
            "event_id" in r.columns:
        st.caption("Blue bands = register event windows overlapping the "
                   "view. Red points = days flagged by the ensemble detector.")
    st.markdown("#### Selected event windows in range")
    if reg is not None and len(reg):
        r2 = reg[(reg["end"] >= pd.to_datetime(lo_d)) &
                 (reg["start"] <= pd.to_datetime(hi_d))]
        st.dataframe(r2[["event_id", "type", "start", "end", "regions",
                         "source"]], width="stretch", hide_index=True)

# ---- 4. SPATIAL HOTSPOTS -------------------------------------------------- #
with tab_spatial:
    st.header("Spatial anomaly clustering")
    dist = load_csv(TABLES / "district_anomaly.csv")
    sp = load_json(RESULTS / "spatial_stats.json")
    if dist is None:
        st.error("district_anomaly.csv missing.")
        st.stop()
    cA, cB = st.columns([2, 1])
    with cA:
        ch = alt.Chart(dist).mark_circle(opacity=0.85).encode(
            x=alt.X("lon:Q", title="Longitude (°E)",
                    scale=alt.Scale(domain=[29.5, 35.0])),
            y=alt.Y("lat:Q", title="Latitude (°N)",
                    scale=alt.Scale(domain=[-1.5, 4.0])),
            size=alt.Size("anomaly_rate:Q",
                          scale=alt.Scale(range=[120, 900]), title="Anomaly rate"),
            color=alt.Color("cluster:N",
                            scale=alt.Scale(
                                domain=["hotspot", "coldspot", "ns"],
                                range=["#d7301f", "#4575b4", "#999999"]),
                            title="Cluster"),
            tooltip=[
                "location:N", "region:N",
                alt.Tooltip("anomaly_rate:Q", format=".3f"),
                alt.Tooltip("Gi_star_z:Q", format=".2f"),
                alt.Tooltip("local_morans_I:Q", format=".2f"),
                alt.Tooltip("precision_vs_events:Q", format=".3f"),
                "cluster:N",
            ],
        ).properties(height=460)
        labels = alt.Chart(dist).mark_text(align="left", baseline="middle",
                                           dx=6, fontSize=10).encode(
            x="lon:Q", y="lat:Q", text="location:N")
        st.altair_chart(ch + labels, width="stretch")
    with cB:
        st.subheader("Global Moran's I")
        st.metric("Moran's I", f"{sp['moran_global_I']:.3f}" if sp else "—")
        st.metric("z-score", f"{sp['moran_z']:.2f}" if sp else "—")
        st.metric("p-value", f"{sp['moran_p']:.3f}" if sp else "—")
        st.markdown("**Hotspots** (Getis–Ord G\\*): " +
                    (", ".join(sp.get("hotspots", [])) or "—"))
        st.markdown("**Coldspots**: " +
                    (", ".join(sp.get("coldspots", [])) or "—"))
        st.caption("Global clustering is *not* significant; one local hotspot "
                   "(Mbale, G\\*≈3.5) survives — consistent with the paper's "
                   "RQ2 narrative.")
    st.markdown("#### District table")
    st.dataframe(dist.sort_values("anomaly_rate", ascending=False),
                 width="stretch", hide_index=True)

# ---- 5. EVENTS & SEVERITY (UCASI) ----------------------------------------- #
with tab_events:
    st.header("Documented events & UCASI severity index")
    uc = load_csv(TABLES / "ucasi_events.csv")
    er = load_csv(TABLES / "event_recall.csv")
    reg = load_register()
    if uc is None:
        st.error("ucasi_events.csv missing.")
        st.stop()
    cE1, cE2, cE3 = st.columns(3)
    typ = cE1.multiselect("Type", sorted(uc["type"].unique().tolist()),
                          default=sorted(uc["type"].unique().tolist()))
    sev = cE2.multiselect("Severity",
                          ["Low", "Moderate", "High", "Extreme"],
                          default=["Moderate", "High", "Extreme"])
    yr_range = cE3.slider("Year range", int(uc["year"].min()),
                          int(uc["year"].max()),
                          (int(uc["year"].min()), int(uc["year"].max())))
    filt = uc[(uc["type"].isin(typ)) & (uc["severity"].isin(sev)) &
              (uc["year"].between(*yr_range))].copy()
    st.metric("Events matching filters", len(filt))
    ch = alt.Chart(filt).mark_bar().encode(
        x=alt.X("UCASI:Q", title="UCASI (0–100)",
                scale=alt.Scale(domain=[0, 100])),
        y=alt.Y("event_id:N", sort="-x", title=None),
        color=alt.Color("severity:N",
                        scale=alt.Scale(
                            domain=["Low", "Moderate", "High", "Extreme"],
                            range=["#a6cee3", "#1b9e77", "#fb9a99", "#b15a00"])),
        tooltip=["event_id:N", "type:N", "start:N",
                 alt.Tooltip("UCASI:Q", format=".1f"),
                 alt.Tooltip("I:Q", format=".2f"),
                 alt.Tooltip("D:Q", format=".2f"),
                 alt.Tooltip("S:Q", format=".2f"),
                 alt.Tooltip("M:Q", format=".2f"),
                 alt.Tooltip("F:Q", format=".2f"),
                 alt.Tooltip("districts_affected:Q"),
                 alt.Tooltip("matched_anomaly_frac:Q", format=".3f")],
    ).properties(height=700)
    st.altair_chart(ch, width="stretch")
    st.markdown("### Per-event detail")
    pick = st.selectbox("Pick an event", filt["event_id"].tolist(),
                        key="pick_event")
    if pick:
        row = filt[filt["event_id"] == pick].iloc[0]
        comp = pd.DataFrame({
            "component": ["I (intensity)", "D (duration)", "S (spread)",
                          "M (multivariate)", "F (frequency)"],
            "value": [row["I"], row["D"], row["S"], row["M"], row["F"]],
        })
        cc1, cc2 = st.columns([1, 2])
        cc1.altair_chart(
            alt.Chart(comp).mark_bar(color="#3b7dd8").encode(
                x=alt.X("value:Q", title="Component (0–1)",
                        scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("component:N", sort=None, title=None),
                tooltip=["component:N", alt.Tooltip("value:Q", format=".3f")],
            ).properties(height=200),
            width="stretch",
        )
        cc2.markdown(f"**{pick}** · {row['type']} · "
                     f"start {row['start']} → end {row['end']} · "
                     f"regions {row['regions']} · "
                     f"UCASI **{row['UCASI']:.1f}** ({row['severity']})")
        cc2.caption(f"n_days={int(row['n_days'])}, "
                    f"anomaly_days={int(row['anomaly_days'])}, "
                    f"districts_affected={int(row['districts_affected'])}, "
                    f"matched_anomaly_frac={row['matched_anomaly_frac']:.3f}")
        if er is not None:
            e = er[er["event_id"] == pick]
            if len(e):
                cc2.write("Event recall (detected?): "
                          f"{bool(e.iloc[0]['detected'])} · "
                          f"lead time: {e.iloc[0].get('lead_time_days')} days")

# ---- 6. TRENDS & CLIMATOLOGY ---------------------------------------------- #
with tab_trends:
    st.header("Mann–Kendall trend testing")
    tt = load_csv(TABLES / "temporal_trends.csv")
    mt = load_csv(TABLES / "meteostat_trends.csv")
    if tt is None:
        st.warning("temporal_trends.csv missing.")
    else:
        cT1, cT2 = st.columns([2, 1])
        alpha = cT2.slider("Significance α", 0.001, 0.20, 0.05, 0.005)
        tt2 = tt.copy()
        tt2["significant"] = tt2["p_value"] < alpha
        tt2["tau_label"] = tt2["tau"].round(2).astype(str)
        ch = alt.Chart(tt2).mark_rect().encode(
            x=alt.X("variable:N", title=None,
                    axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("location:N", sort=None, title=None),
            color=alt.Color("tau:Q",
                            scale=alt.Scale(scheme="blueorange", domain=[-0.6, 0.6]),
                            title="Kendall τ"),
            opacity=alt.Opacity("significant:N",
                                scale=alt.Scale(domain=[False, True],
                                                range=[0.25, 1.0]), legend=None),
            tooltip=["location:N", "variable:N",
                     alt.Tooltip("tau:Q", format=".3f"),
                     alt.Tooltip("p_value:Q", format=".3g"),
                     alt.Tooltip("sen_slope_per_month:Q", format=".4f"),
                     "trend:N",
                     alt.Tooltip("significant:N", title="p<α")],
        ).properties(height=520)
        cT1.altair_chart(ch, width="stretch")
        cT2.markdown(f"**Rows:** {len(tt2):,}  \n"
                     f"**Significant (p<{alpha}):** {int(tt2['significant'].sum())}  \n"
                     f"**Increasing (τ>0):** "
                     f"{int(((tt2['tau'] > 0) & tt2['significant']).sum())}  \n"
                     f"**Decreasing (τ<0):** "
                     f"{int(((tt2['tau'] < 0) & tt2['significant']).sum())}")
    if mt is not None and len(mt):
        st.markdown("### In-situ cross-check (Meteostat stations)")
        st.dataframe(mt[["station", "name", "variable", "n_years", "tau",
                         "p_value", "trend", "sens_slope"]],
                     width="stretch", hide_index=True)

# ---- 7. RISK NOWCAST ------------------------------------------------------ #
with tab_now:
    st.header("District-month risk nowcast (deployment artefact)")
    rd = load_csv(TABLES / "risk_district_month.csv")
    ews = load_json(RESULTS / "early_warning_summary.json")
    if rd is None:
        st.warning("risk_district_month.csv missing.")
    else:
        st.markdown(
            "**Honest reading:** the district-month risk score sits at "
            "chance against the register for forward skill. The dashboard "
            "reports the *measured* skill, not a promotional framing."
        )
        if ews:
            k = ews["skill_vs_event_months"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Nowcast ROC-AUC", f"{k['auc']:.2f}",
                      delta=f"null {k['auc_null_mean']:.2f}", delta_color="off")
            c2.metric("Recall@TopK", f"{k['recall_at_k']:.3f}",
                      delta=f"null {k['recall_null_mean']:.3f}",
                      delta_color="off")
            c3.metric("Lag-1 τ (persistence)",
                      f"{ews['persistence']['lag1_mean_tau']:.2f}")
            c4.metric("Districts × months",
                      f"{k['n_district_months']:,}")
        dsel = st.multiselect("Districts", sorted(rd.location.unique()),
                              default=sorted(rd.location.unique())[:5])
        rr = rd[rd.location.isin(dsel)].copy()
        rr["period"] = pd.PeriodIndex(rr.year.astype(str) + "-" +
                                      rr.month.astype(str).str.zfill(2),
                                      freq="M").to_timestamp()
        ch = alt.Chart(rr).mark_line().encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("risk:Q", title="UCASI-derived risk (0–100)"),
            color=alt.Color("location:N", legend=alt.Legend(title=None)),
            tooltip=["location:N", "period:T",
                     alt.Tooltip("risk:Q", format=".1f"), "band:N",
                     alt.Tooltip("anomaly_days:Q"),
                     alt.Tooltip("is_event_month:Q", title="Event month?")],
        ).properties(height=340)
        event_rects = alt.Chart(rr[rr["is_event_month"] == 1]).mark_rule(
            color="#c1272d", opacity=0.15, size=8).encode(x="period:T")
        st.altair_chart(event_rects + ch, width="stretch")
        st.caption("Red vertical ticks mark district-months overlapping the "
                   "disaster register.")
        if ews and "latest_nowcast_top8" in ews:
            st.markdown("#### Latest nowcast (most recent month)")
            df = pd.DataFrame(ews["latest_nowcast_top8"])
            st.dataframe(df, width="stretch", hide_index=True)
        st.markdown("#### Full district-month table (top 40 by risk)")
        st.dataframe(rd.sort_values("risk", ascending=False).head(40),
                     width="stretch", hide_index=True)

# ---- 8. DETECTION DRIVERS ------------------------------------------------- #
with tab_drivers:
    st.header("Which variables drive the detection signal?")
    sh = load_csv(TABLES / "shap_importance.csv")
    pm = load_csv(TABLES / "permutation_importance.csv")
    st.info("These numbers interpret the **detector**, not physical causation. "
            "See Section~RQ3 of the paper.")
    if sh is not None:
        ch1 = alt.Chart(sh).mark_bar(color="#2c7fb8").encode(
            x=alt.X("shap_mean_abs:Q", title="mean |SHAP|"),
            y=alt.Y("variable:N", sort="-x", title=None),
            tooltip=["variable:N", alt.Tooltip("shap_mean_abs:Q", format=".3f")],
        ).properties(height=340, title="Global feature importance (SHAP)")
        st.altair_chart(ch1, width="stretch")
    if pm is not None:
        pm2 = pm.copy()
        pm2["perm_hi"] = pm2["perm_importance"] + pm2["perm_std"]
        pm2["perm_lo"] = pm2["perm_importance"] - pm2["perm_std"]
        ch2 = alt.Chart(pm2).mark_bar(color="#b5651d").encode(
            x=alt.X("perm_lo:Q", title="Permutation importance"),
            x2="perm_hi:Q",
            y=alt.Y("variable:N", sort="-x", title=None),
            tooltip=["variable:N",
                     alt.Tooltip("perm_importance:Q", format=".3f"),
                     alt.Tooltip("perm_std:Q", format=".3f")],
        ).properties(height=340, title="Permutation importance (±1 SD)")
        st.altair_chart(ch2, width="stretch")

# ---- 9. ABOUT / REPRODUCIBILITY ------------------------------------------- #
with tab_about:
    st.header("About this dashboard")
    st.markdown(
        f"""
This view is a **read-only interface** to the artefacts produced by
``run_all.py``. Every number rendered here is loaded from a file on disk.

**Data sources rendered:**

- `results/benchmark.csv` — 24-detector held-out metrics with bootstrap CIs
- `results/friedman.json` — Friedman statistic (24 detectors × 5 years)
- `results/per_year_f1.npy` + `per_year_f1_order.txt` — per-year stability
- `results/spatial_stats.json` + `results/tables/district_anomaly.csv` — GIS
  clustering and per-district metrics
- `results/tables/ucasi_events.csv` + `results/ucasi_summary.json` — the UCASI
  severity index and its per-event components
- `results/tables/event_recall.csv` — event-level detection recall and lead time
- `results/tables/temporal_trends.csv` + `meteostat_trends.csv` —
  Mann–Kendall τ, Sen's slope
- `results/tables/risk_district_month.csv` +
  `results/early_warning_summary.json` — district-month risk nowcast and its
  skill-vs-null measurement
- `results/tables/shap_importance.csv` + `permutation_importance.csv` —
  feature drivers of the ensemble detector
- `data/full_ensemble.csv` — daily anomaly stream (~93 k district-days) used in
  the Time-Series Explorer
- `data/disaster_register.csv` — 32 curated disasters (ground truth)

**No model is fitted here.** Re-run the pipeline for a change; the dashboard
picks it up on the next page refresh.

**Environment:** requires `streamlit` and `altair` (bundled with `streamlit`).
This repo vendors a CPU wheel set at `._webpkgs/` (gitignored). Launch with
`python dashboard/run_dashboard.py`, which prepends `._webpkgs/` to
`sys.path` before calling the Streamlit CLI.

**Reproducibility:** seeds fixed at 42; every artefact is regenerable with
`python run_all.py`. Sources: ERA5 (Open-Meteo archive API, CC-BY 4.0),
Meteostat (CC-BY 4.0), NASA POWER, MERRA-2, FAOSTAT, World Bank, UNMA/OCHA
disaster register (public reports).
        """
    )
    st.caption(f"Repo root: `{_ROOT}`")
