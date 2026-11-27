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
FIGURES = _ROOT / "figures"

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


# --------------------------------------------------------------------------- #
# District polygons: the OCHA COD-AB layer behind every map in the paper
# --------------------------------------------------------------------------- #
try:
    import pydeck as pdk                  # ships alongside streamlit
except Exception:                         # a minimal env still gets the PNG
    pdk = None

GEOJSON = DATA / "district_boundaries.geojson"
NODATA_RGB = (233, 233, 230)              # the no-data grey used throughout


@st.cache_data(show_spinner=False)
def load_district_features():
    """The 135 COD-AB district polygons and the site->district join that
    ``src/fetch_boundaries.py`` published.  The join is loaded, never recomputed,
    so this map and the paper's maps agree by construction."""
    if not GEOJSON.exists():
        return None, {}
    feats = json.loads(GEOJSON.read_text(encoding="utf-8"))["features"]
    lk = load_csv(DATA / "district_lookup.csv")
    lookup = {} if lk is None else dict(zip(lk["location"], lk["adm2_name"]))
    return feats, lookup


def hex_rgb(value, alpha=215):
    """``"#d7301f"`` -> ``[215, 48, 31, 215]``, the colour deck.gl is given."""
    h = value.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def district_features(features, lookup, values_by_site, extra_by_district=None):
    """Copy the polygons with this panel's numbers folded into ``properties``.

    Districts that no site falls in keep ``observed = "no"`` and carry no
    numbers at all, so an unobserved district can never be read as a zero.
    ``extra_by_district`` supplies districts that hold a network site but no
    score for the variable being shaded; they get their own class for the same
    reason.
    """
    by_name = {lookup[site]: vals for site, vals in values_by_site.items()
               if site in lookup}
    for name, vals in (extra_by_district or {}).items():
        by_name.setdefault(name, vals)
    out = []
    for f in features:
        name = f["properties"]["adm2_name"]
        extra = dict(by_name.get(name, {}))
        extra.setdefault("observed", "yes" if extra else "no")
        extra.setdefault("klass", "not observed")
        out.append({**f, "properties": {**f["properties"], **extra}})
    return out


def site_gap_districts(lookup_df, covered_names):
    """COD districts with a network site but no score in the drawn panel."""
    if lookup_df is None:
        return {}
    return {r.adm2_name: {"klass": "site, no panel score",
                          "observed": "yes (no detection score)"}
            for r in lookup_df.itertuples()
            if r.adm2_name not in covered_names}


def choropleth(features, class_of, styles, height=460, tooltip_fields=()):
    """Draw one deck.gl layer per class, with the class chosen in Python.

    Colour is therefore a property of the layer rather than a JavaScript
    accessor, which keeps the legend below identical to what is drawn.
    ``map_provider=None`` means no third-party tile server is contacted: the
    districts are the map.  Returns False if pydeck or the geometry is
    unavailable, so the caller can fall back to the paper's static figure.
    """
    if pdk is None or not features:
        return False
    buckets = {}
    for f in features:
        buckets.setdefault(class_of(f["properties"]), []).append(f)
    layers = [
        pdk.Layer("GeoJsonLayer",
                  {"type": "FeatureCollection", "features": feats},
                  name=label, pickable=True,
                  get_fill_color=styles[label]["fill"],
                  get_line_color=styles[label]["line"],
                  line_width_min_pixels=styles[label]["lw"])
        for label, feats in buckets.items() if label in styles
    ]
    tip = {"html": "<b>{adm2_name}</b>" + "".join(
        f"<br/>{field}: {{{field}}}" for field in tooltip_fields),
        "static": True, "style": {"font-size": "12px"}}
    deck = pdk.Deck(
        layers=layers, map_provider=None, tooltip=tip,
        initial_view_state=pdk.ViewState(latitude=1.4, longitude=32.4,
                                         zoom=5.1, pitch=0, bearing=0,
                                         width=900, height=height))
    st.pydeck_chart(deck, width="stretch", height=height)
    return True


def legend_md(styles, order):
    """Swatches generated from the same dict that coloured the layers."""
    cells = "".join(
        "<span style='display:inline-block;margin-right:14px'>"
        "<span style='display:inline-block;width:11px;height:11px;"
        f"background:rgb{tuple(styles[k]['fill'][:3])};"
        "border:1px solid #666;margin-right:5px'></span>"
        f"{k}</span>"
        for k in order if k in styles)
    st.markdown(cells, unsafe_allow_html=True)


def class_from_props(props):
    """Class label folded into a district's properties by the calling panel."""
    return props.get("klass", "not observed")


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


def linkage_rows(payload, block):
    """Flatten one linkage block of a ``*_linkage.json`` into a display table.

    Nothing is recomputed here: the coefficients are exactly those written by
    ``ag_linkage.py`` / ``coffee_linkage.py``.
    """
    rows = []
    for key, v in (payload or {}).get(block, {}).items():
        if "pearson_r" not in v:
            continue
        season, index = key.split(":", 1)
        rows.append({"season window": season.replace("_", " "),
                     "index": index, "n": v["n"], "years": v["years"],
                     "pearson r": v["pearson_r"], "pearson p": v["pearson_p"],
                     "spearman r": v["spearman_r"],
                     "spearman p": v["spearman_p"],
                     "slope %/SD": v["slope_pct_per_sd"],
                     "wet-dry gap %": v["wet_minus_dry_yieldgap_pct"]})
    return pd.DataFrame(rows)


def p_values(payload, blocks):
    """Every reported p-value in the given linkage blocks."""
    out = []
    for b in blocks:
        for v in (payload or {}).get(b, {}).values():
            out += [v[k] for k in ("pearson_p", "spearman_p") if k in v]
    return out


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
               "NASA POWER · MERRA-2 · FAOSTAT · World Bank · OCHA COD-AB "
               "district polygons (CC BY-IGO). CC-BY 4.0 sources. Seed=42.")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_over, tab_bench, tab_ts, tab_spatial, tab_events, tab_trends, \
    tab_impact, tab_now, tab_drivers, tab_about = st.tabs([
        "Overview",
        "Detection Benchmark",
        "Time-Series Explorer",
        "Spatial Hotspots",
        "Events & Severity (UCASI)",
        "Trends & Climatology",
        "Impacts & Agriculture",
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
        cv = load_json(RESULTS / "district_coverage.json")
        district_label = ("Districts observed"
                          if cv else "Districts")
        district_value = (f"{cv['districts_analysed']} of "
                          f"{cv['districts_total']}"
                          if cv else
                          f"{len(dist) if dist is not None else '—'}")
        metric_row([
            ("Detectors benchmarked", f"{n_det}", None),
            ("Held-out test years", "2019–2023", None),
            ("Register disasters", f"{len(reg) if reg is not None else '—'}",
             None),
            (district_label, district_value,
             (f"{100 * cv['analysed_area_share']:.1f}% of district area"
              if cv else None)),
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
    feats, lookup = load_district_features()
    lk_all = load_csv(DATA / "district_lookup.csv")
    cv = load_json(RESULTS / "district_coverage.json")
    shade = cA.radio("Shade districts by",
                     ["Anomaly rate", "Getis–Ord cluster"], horizontal=True)
    vals = {r.location: {
        "region": str(r.region),
        "anomaly_rate": round(float(r.anomaly_rate), 4),
        "Gi_star_z": round(float(r.Gi_star_z), 2),
        "local_morans_I": round(float(r.local_morans_I), 2),
        "precision_vs_events": round(float(r.precision_vs_events), 3),
        "cluster": str(r.cluster)} for _, r in dist.iterrows()}
    UNOBSERVED = {"fill": [*NODATA_RGB, 255], "line": [185, 185, 180],
                  "lw": 1}
    SITE_STYLE = {"fill": [214, 214, 208, 255], "line": [120, 120, 120],
                  "lw": 2}
    cluster_styles = {
        "hotspot": {"fill": hex_rgb("#d7301f"), "line": [26, 26, 26], "lw": 2},
        "coldspot": {"fill": hex_rgb("#4575b4"), "line": [74, 74, 74], "lw": 1},
        "ns": {"fill": hex_rgb("#c8c8c1"), "line": [74, 74, 74], "lw": 1},
        "site, no panel score": SITE_STYLE,
        "not observed": UNOBSERVED,
    }
    rates = sorted(v["anomaly_rate"] for v in vals.values())
    breaks = [float(np.quantile(rates, q)) for q in (0.25, 0.5, 0.75)]
    quartiles = [f"Q1 ≤ {breaks[0]:.3f}",
                 f"Q2 {breaks[0]:.3f}–{breaks[1]:.3f}",
                 f"Q3 {breaks[1]:.3f}–{breaks[2]:.3f}",
                 f"Q4 > {breaks[2]:.3f}"]
    rate_styles = {lab: {"fill": hex_rgb(c), "line": [74, 74, 74], "lw": 1}
                   for lab, c in zip(quartiles,
                                     ["#fee5d9", "#fcae91", "#fb6a4a",
                                      "#cb181d"])}
    rate_styles["not observed"] = UNOBSERVED
    rate_styles["site, no panel score"] = SITE_STYLE
    by_quartile = {}
    for site, v in vals.items():
        r = v["anomaly_rate"]
        by_quartile[site] = dict(v, klass=(
            quartiles[3] if r > breaks[2] else quartiles[2] if r > breaks[1]
            else quartiles[1] if r > breaks[0] else quartiles[0]))
    by_cluster = {site: dict(v, klass=v["cluster"]) for site, v in vals.items()}
    gap = site_gap_districts(lk_all, {lookup[s] for s in vals if s in lookup})
    with cA:
        if feats is None:
            st.warning("`data/district_boundaries.geojson` is missing — run "
                       "`python src/fetch_boundaries.py`. Showing the paper's "
                       "static choropleth instead.")
            st.image(str(FIGURES / "fig_hotspot_map.png"))
        else:
            if shade == "Anomaly rate":
                drawn = choropleth(
                    district_features(feats, lookup, by_quartile, gap),
                    class_from_props, rate_styles, height=460,
                    tooltip_fields=("region", "anomaly_rate", "Gi_star_z",
                                    "local_morans_I", "precision_vs_events",
                                    "cluster", "observed"))
                legend_md(rate_styles, quartiles + ["site, no panel score",
                                                    "not observed"])
            else:
                drawn = choropleth(
                    district_features(feats, lookup, by_cluster, gap),
                    class_from_props, cluster_styles, height=460,
                    tooltip_fields=("region", "cluster", "Gi_star_z",
                                    "anomaly_rate", "observed"))
                legend_md(cluster_styles,
                          ["hotspot", "coldspot", "ns",
                           "site, no panel score", "not observed"])
            if not drawn:
                st.info("pydeck is unavailable in this environment, so the "
                        "interactive choropleth is replaced by the paper's "
                        "static map.")
                st.image(str(FIGURES / "fig_hotspot_map.png"))
            if cv:
                breaks_note = (
                    " Class breaks are quartiles of the "
                    f"{len(vals):d} observed districts." if shade ==
                    "Anomaly rate" else "")
                st.caption(
                    f"{cv['districts_analysed']} of {cv['districts_total']} "
                    "COD districts hold a network site "
                    f"({100 * cv['analysed_area_share']:.1f}% of the national "
                    f"district area), and only {len(vals)} of those sit inside "
                    "the ERA5 detection panel and therefore carry a shaded "
                    "value here. Districts with a site but no panel score are "
                    "marked as such; districts with neither are drawn as "
                    "no-data. Neither is ever shown as a zero." + breaks_note +
                    " Polygons: OCHA Uganda COD-AB v01 (CC BY-IGO).")
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

# ---- 7. IMPACTS & AGRICULTURE --------------------------------------------- #
with tab_impact:
    st.header("Drought indices and the agricultural impact layer")
    dr = load_json(RESULTS / "drought_summary.json")
    agl = load_json(RESULTS / "ag_linkage.json")
    cfl = load_json(RESULTS / "coffee_linkage.json")
    ay = load_csv(TABLES / "ag_yield_climate.csv")
    cp = load_csv(TABLES / "coffee_prod_climate.csv")
    dec = load_csv(TABLES / "spi_decade_drought.csv")
    era = load_csv(TABLES / "spei_era5.csv")
    if dr is None or ay is None:
        st.warning("Drought and impact artefacts are missing. Run "
                   "`python src/drought_indices.py`, `python src/ag_linkage.py` "
                   "and `python src/coffee_linkage.py` first.")
    else:
        n_chk = dr.get("register_droughts_checked") or 0
        n_cor = dr.get("register_droughts_with_spei_le_minus1") or 0
        st.info(
            "**This layer is a two-sided finding.** Physically interpreted "
            f"SPI/SPEI corroborate {n_cor} of {n_chk} documented droughts and "
            "single out 2018 as the worst detrended cereal-yield year of the "
            "series, yet the national-annual drought-yield coefficients are "
            "not significant, and the coffee correlation reverses sign between "
            "the in-situ and reanalysis substrates. Every number below is read "
            "from the artefacts, including the ones that undercut the "
            "headline."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("In-situ SPI record", f"{dr['spi_stations']} stations",
                  delta=f"{dr['spi_months']:,} station-months",
                  delta_color="off")
        m2.metric("ERA5 SPI/SPEI panel", f"{dr['spei_locations']} districts",
                  delta=f"{dr['spei_months']:,} district-months",
                  delta_color="off")
        m3.metric("Register droughts corroborated", f"{n_cor} of {n_chk}",
                  delta="index <= -1 inside window", delta_color="off")
        if agl:
            m4.metric("Cereal yield trend",
                      f"{agl['yield_trend_pct_per_year']:+.1f}%/yr",
                      delta=(f"{agl['yield_years'][0]}-{agl['yield_years'][1]}, "
                             "World Bank/FAOSTAT"), delta_color="off")

        i1, i2 = st.columns([3, 2])
        with i1:
            st.markdown("#### Century in-situ drought frequency by decade")
            if dec is None or not len(dec):
                st.info("`results/tables/spi_decade_drought.csv` is empty.")
            else:
                dt = dec.melt(id_vars=["decade", "n_months"], var_name="class",
                              value_name="share")
                dt["class"] = dt["class"].map({
                    "drought_freq": "SPI6 < -1",
                    "severe_freq": "SPI6 < -1.5",
                    "extreme_freq": "SPI6 < -2"})
                dt["decade"] = dt["decade"].astype(int).astype(str) + "s"
                st.altair_chart(
                    alt.Chart(dt).mark_bar().encode(
                        x=alt.X("decade:N", title=None),
                        y=alt.Y("share:Q", title="share of station-months"),
                        color=alt.Color("class:N",
                                        scale=alt.Scale(
                                            domain=["SPI6 < -1",
                                                    "SPI6 < -1.5",
                                                    "SPI6 < -2"],
                                            range=["#fee090", "#f46d43",
                                                   "#b10026"])),
                        tooltip=["decade:N", "class:N",
                                 alt.Tooltip("share:Q", format=".3f"),
                                 alt.Tooltip("n_months:Q",
                                             title="station-months")],
                    ).properties(height=300),
                    width="stretch")
                st.caption(
                    "Rows are exactly what ``decade_drought()`` wrote: a decade "
                    "is listed only if it carries >=24 station-months with a "
                    f"valid SPI6. The fetch runs {dr['spi_record_start'][:4]}-"
                    f"{dr['spi_record_end'][:4]} but holds no 1990s-2010s rows "
                    "at all, so those decades are absent because of the "
                    "reporting collapse, not because they were wet.")
        with i2:
            st.markdown("#### Index validation")
            val = dr.get("index_validation") or {}
            if val:
                vv = pd.DataFrame(
                    [{"series": k, "n": v["n"], "mean": v["mean"],
                      "sd": v["sd"], "min": v["min"], "max": v["max"]}
                     for k, v in val.items()])
                st.dataframe(vv.style.format(
                    {"mean": "{:.3f}", "sd": "{:.3f}", "min": "{:.2f}",
                     "max": "{:.2f}"}), width="stretch", hide_index=True,
                    height=290)
                st.caption("SPI/SPEI are unit-normal by construction; this is "
                           "the check that each fitted distribution behaves, "
                           "as written by ``drought_indices.py``.")
            wd = dr.get("worst_century_droughts") or []
            if wd:
                deep = min(wd, key=lambda e: float(e["SPI12"]))
                names = sorted({str(e["name"]) for e in wd})
                months = sorted(str(e["month"]) for e in wd)
                st.markdown(
                    f"**Deepest century months (in-situ SPI6/SPI12):** "
                    f"{float(deep['SPI12']):.2f} at {deep['name']}, "
                    f"{deep['month']}. All {len(wd)} entries in the artefact "
                    f"sit at {', '.join(names)} between {months[0]} and "
                    f"{months[-1]}.")

        st.markdown("#### Do the derived indices see the documented droughts?")
        rc = dr.get("register_drought_spei") or []
        if rc:
            rcd = pd.DataFrame(rc)
            rcd["corroborated"] = [
                "yes" if (float(r["min_SPEI3"]) <= -1
                          or float(r["min_SPI6"]) <= -1) else "no"
                for _, r in rcd.iterrows()]
            st.dataframe(rcd, width="stretch", hide_index=True)
            missed = ", ".join(rcd.loc[rcd["corroborated"] == "no",
                                       "event"].astype(str)) or "none"
            st.caption("A register drought counts as corroborated when the "
                       "affected-region district-months reach moderate drought "
                       f"(index <= -1) anywhere in the window. Missed: "
                       f"{missed}. Windows come from the disaster register, "
                       "which is never used for fitting.")

        if era is not None and len(era):
            st.markdown("#### ERA5 district moisture indices, month by month")
            e1, e2 = st.columns([1, 5])
            locs = sorted(era["location"].dropna().unique().tolist())
            pick = e1.selectbox("District", locs,
                                index=locs.index("Mbale")
                                if "Mbale" in locs else 0)
            de = era[era["location"] == pick].copy()
            de["month"] = pd.to_datetime(de["month"], errors="coerce")
            tidy = de.melt(id_vars=["month"], value_name="index",
                           var_name="series").dropna(
                               subset=["index"])
            tidy = tidy[tidy["series"].isin(["SPI3", "SPI6", "SPEI3", "SPEI6"])]
            lines = [alt.Chart(tidy).mark_line().encode(
                x=alt.X("month:T", title=None),
                y=alt.Y("index:Q", title="standardised index"),
                color=alt.Color("series:N", legend=alt.Legend(title=None)),
                tooltip=["series:N", "month:T",
                         alt.Tooltip("index:Q", format=".2f")])]
            for thr in (-1.0, -1.5, -2.0):
                lines.append(alt.Chart(pd.DataFrame({"v": [thr]})).mark_rule(
                    color="#c1272d", opacity=0.45, strokeDash=[3, 3]
                ).encode(y="v:Q"))
            with e2:
                st.altair_chart(alt.layer(*lines), width="stretch")
                n_dry = int((tidy["series"].isin(["SPEI3", "SPI6"])
                             & (tidy["index"] <= -1)).sum())
                st.caption(f"{pick}: {de['month'].min():%Y-%m} to "
                           f"{de['month'].max():%Y-%m} "
                           f"({len(de):,} district-months), {n_dry} index "
                           "readings at or below -1. Red rules mark the "
                           "moderate / severe / extreme drought thresholds. "
                           "Source: ``results/tables/spei_era5.csv``.")

        st.markdown("#### Cereal: documented drought years on the yield series")
        ev = {int(e["year"]): e
              for e in (agl or {}).get("drought_events_yield_anomaly", [])}
        yv = ay.dropna(subset=["yield_anom_pct"]).copy()
        yv["register drought"] = yv["year"].map(
            lambda y: "documented drought" if int(y) in ev
            else "no register drought")
        yv["event"] = yv["year"].map(lambda y: ev.get(int(y), {}).get("event",
                                                                      ""))
        yv["regions"] = yv["year"].map(lambda y: ev.get(int(y), {}).get(
            "regions", ""))
        st.altair_chart(
            alt.Chart(yv).mark_bar().encode(
                x=alt.X("year:O", title=None),
                y=alt.Y("yield_anom_pct:Q",
                        title="detrended cereal-yield anomaly (%)"),
                color=alt.Color("register drought:N",
                                scale=alt.Scale(
                                    domain=["no register drought",
                                            "documented drought"],
                                    range=["#b0b6bd", "#d7301f"]),
                                legend=alt.Legend(title=None)),
                tooltip=[alt.Tooltip("year:O"),
                         alt.Tooltip("yield_kgha:Q", format=".0f",
                                     title="yield kg/ha"),
                         alt.Tooltip("yield_anom_pct:Q", format=".1f"),
                         "register drought:N", "event:N", "regions:N",
                         alt.Tooltip("met_SPI3_longrains:Q", format=".2f",
                                     title="in-situ SPI3"),
                         alt.Tooltip("era_SPEI3_longrains:Q", format=".2f",
                                     title="ERA5 SPEI3")],
            ).properties(height=320),
            width="stretch")
        worst = yv.loc[yv["yield_anom_pct"].idxmin()]
        st.caption(f"Series spans {int(yv['year'].min())}-"
                   f"{int(yv['year'].max())}; the weakest detrended year is "
                   f"{int(worst['year'])} at {float(worst['yield_anom_pct']):.1f}%"
                   " (``ag_yield_climate.csv``). Empty SPI/SPEI cells in the "
                   "tooltip are the 1990-2007 in-situ gap and the pre-2008 "
                   "absence of the ERA5 panel, not zero moisture anomaly.")
        if agl:
            c9, c10 = st.columns(2)
            c9.markdown("**Century in-situ substrate (Meteostat)**")
            c9.dataframe(linkage_rows(agl, "meteostat_century_linkage"),
                         width="stretch", hide_index=True)
            c10.markdown("**Modern reanalysis substrate (ERA5)**")
            c10.dataframe(linkage_rows(agl, "era5_modern_linkage"),
                          width="stretch", hide_index=True)
            allp = p_values(agl, ["meteostat_century_linkage",
                                  "era5_modern_linkage"])
            if allp:
                sig = sum(1 for p in allp if float(p) < 0.05)
                st.caption(
                    f"{len(allp)} coefficients across both blocks; "
                    + (f"{sig} reach p<0.05 (smallest "
                       f"p={min(float(p) for p in allp):.3f})"
                       if sig else f"none reach p<0.05 (smallest "
                                   f"p={min(float(p) for p in allp):.3f})")
                    + ". The national aggregate is the confounder: regional "
                    "droughts are averaged away. Source: "
                    "``results/ag_linkage.json``.")
        evd = pd.DataFrame((agl or {}).get("drought_events_yield_anomaly", []))
        if len(evd):
            st.markdown("**Yield anomaly in each documented drought year**")
            st.dataframe(evd, width="stretch", hide_index=True)
            above = ", ".join(
                f"{r['event']} ({r['yield_anom_pct']:+.1f}%)"
                for _, r in evd.iterrows() if r["yield_anom_pct"] > 0)
            st.caption("Droughts that fell in an above-trend national yield "
                       f"year: {above or 'none'}. That is the aggregation "
                       "masking, reported rather than hidden.")

        st.markdown("#### Coffee: the signature export, and its confounding")
        if cfl is None or cp is None:
            st.info("Coffee artefacts missing. Run "
                    "`python src/coffee_linkage.py`.")
        else:
            k1, k2, k3 = st.columns(3)
            k1.metric("Production trend",
                      f"{cfl['prod_trend_pct_per_year']:+.2f}%/yr",
                      delta=(f"{cfl['prod_years'][0]}-"
                             f"{cfl['prod_years'][1]}, FAOSTAT/OWID"),
                      delta_color="off")
            k2.metric("Production range",
                      f"{cfl['prod_range_t'][0]:,.0f}-"
                      f"{cfl['prod_range_t'][1]:,.0f} t",
                      delta="log-detrended, not per-ha yield",
                      delta_color="off")
            cfy = cp.dropna(subset=["coffee_anom_pct"]).copy()
            cev = {int(e["year"]) for e
                   in cfl.get("drought_events_coffee_anomaly", [])}
            cfy["register drought"] = cfy["year"].map(
                lambda y: "documented drought" if int(y) in cev else "other")
            worst_years = cfl.get("worst_coffee_years", [])
            k3.metric("Worst production years", f"{len(worst_years)}",
                      delta="all before the ERA5 panel", delta_color="off")
            q1, q2 = st.columns([3, 2])
            with q1:
                st.altair_chart(
                    alt.Chart(cfy).mark_bar().encode(
                        x=alt.X("year:O", title=None),
                        y=alt.Y("coffee_anom_pct:Q",
                                title="detrended production anomaly (%)"),
                        color=alt.Color("register drought:N",
                                        scale=alt.Scale(
                                            domain=["other",
                                                    "documented drought"],
                                            range=["#b0b6bd", "#d7301f"]),
                                        legend=alt.Legend(title=None)),
                        tooltip=[alt.Tooltip("year:O"),
                                 alt.Tooltip("coffee_prod_t:Q", format=".0f",
                                             title="production t"),
                                 alt.Tooltip("coffee_anom_pct:Q",
                                             format=".1f"),
                                 "register drought:N",
                                 alt.Tooltip("met_SPI3_longrains:Q",
                                             format=".2f",
                                             title="in-situ SPI3"),
                                 alt.Tooltip("era_SPEI3_bimodal:Q",
                                             format=".2f",
                                             title="ERA5 SPEI3")],
                    ).properties(height=300),
                    width="stretch")
            with q2:
                st.markdown("**Century in-situ substrate**")
                st.dataframe(linkage_rows(cfl, "meteostat_century_linkage"),
                             width="stretch", hide_index=True)
                st.markdown("**Modern ERA5 substrate**")
                st.dataframe(linkage_rows(cfl, "era5_modern_linkage"),
                             width="stretch", hide_index=True)
            cent = [v["pearson_r"] for v in
                    cfl.get("meteostat_century_linkage", {}).values()
                    if "pearson_r" in v]
            mod = [v["pearson_r"] for v in
                   cfl.get("era5_modern_linkage", {}).values()
                   if "pearson_r" in v]
            if cent and mod:
                reverses = min(cent) > 0 > max(mod)
                st.caption(
                    "Pearson r against the century in-situ substrate ranges "
                    f"{min(cent):+.2f} to {max(cent):+.2f}; against modern "
                    f"ERA5 it ranges {min(mod):+.2f} to {max(mod):+.2f}. The "
                    + ("signs reverse between two independent observation "
                       "substrates, so no stable climate signal survives the "
                       "national aggregate" if reverses else
                       "signs are consistent across the two substrates")
                    + ". Production, not yield, is detrended here: "
                    + str(cfl.get("yield_note", "")))
            wc = pd.DataFrame(cfl.get("worst_coffee_years", []))
            if len(wc):
                st.markdown("**Weakest production years and their moisture "
                            "index**")
                st.dataframe(wc, width="stretch", hide_index=True)
                no_idx = wc["era_bimodal_SPI3"].isna().sum()
                st.caption(f"{no_idx} of {len(wc)} have no ERA5 bimodal-season "
                           "index at all, because they precede the reanalysis "
                           "panel. The record's worst coffee shortfalls cannot "
                           "be attributed to measured drought with the data "
                           "this pipeline has.")

        with st.expander("The paper's own impact figures (static, 300 dpi)"):
            st.markdown(
                "Generated by ``src/figures_coffee.py`` from the same artefacts "
                "rendered above; identical copies live in ``figures/`` and "
                "``manuscript/figures/``."
            )
            f1, f2 = st.columns(2)
            for col, name in ((f1, "fig_drought_yield.png"),
                              (f2, "fig_coffee.png")):
                if (FIGURES / name).exists():
                    col.image(str(FIGURES / name))
                else:
                    col.info(f"{name} not built yet. Run "
                             "`python src/figures_coffee.py`.")

# ---- 8. RISK NOWCAST ------------------------------------------------------ #
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
            cu = ews.get("carry_uncertainty") or {}
            am, hm = (cu.get("all_district_months") or {}), \
                (cu.get("from_high_or_extreme") or {})
            if am:
                st.markdown("**Uncertainty of the forward carry** — the "
                            "empirical spread of month-to-month change in the "
                            "district-month score, measured over the whole "
                            "record. It is the honest interval around the "
                            "persistence nowcast, not a fitted prediction.")
                u1, u2, u3, u4 = st.columns(4)
                u1.metric("Median |Δ|, next month", f"{am['median_abs_change']:.1f}")
                u2.metric("80th pct |Δ|", f"{am['p80_abs_change']:.1f}",
                          delta=f"from High/Extreme: "
                                f"{hm.get('p80_abs_change', '—')}",
                          delta_color="off")
                u3.metric("95th pct |Δ|", f"{am['p95_abs_change']:.1f}",
                          delta=f"from High/Extreme: "
                                f"{hm.get('p95_abs_change', '—')}",
                          delta_color="off")
                u4.metric("Transitions measured",
                          f"{am['n_transitions']:,}",
                          delta=f"{hm.get('n_transitions', '—')} from "
                                "High/Extreme", delta_color="off")
                st.caption(f"Signed 10th–90th percentile of the monthly change: "
                           f"{am['q10_signed']:+.1f} to {am['q90_signed']:+.1f} "
                           "risk points, so the carry is close to symmetric "
                           "rather than drifting in one direction. Source: "
                           "``carry_uncertainty`` in "
                           "``results/early_warning_summary.json``.")
        feats, lookup = load_district_features()
        lk_all = load_csv(DATA / "district_lookup.csv")
        last = rd.sort_values(["year", "month"]).groupby("location").tail(1)
        lm = last[["year", "month"]].iloc[0]
        band_vals = {r.location: {"region": str(r.region),
                                  "risk": round(float(r.risk), 1),
                                  "band": str(r.band), "klass": str(r.band)}
                     for _, r in last.iterrows()}
        BAND_HEX = {"Low": "#4575b4", "Moderate": "#fee090",
                    "High": "#fdae61", "Extreme": "#d7301f"}
        band_styles = {lab: {"fill": hex_rgb(h, 225), "line": [74, 74, 74],
                             "lw": 1} for lab, h in BAND_HEX.items()}
        band_styles["not observed"] = {"fill": [*NODATA_RGB, 255],
                                       "line": [185, 185, 180], "lw": 1}
        band_styles["site, no panel score"] = {"fill": [214, 214, 208, 255],
                                               "line": [120, 120, 120], "lw": 2}
        st.markdown(f"#### Current risk surface, {int(lm.year)}-"
                    f"{int(lm.month):02d} (most recent complete month)")
        if feats is None:
            st.image(str(FIGURES / "fig_risk_nowcast.png"))
        else:
            gap_now = site_gap_districts(lk_all, {lookup[s] for s in band_vals
                                                  if s in lookup})
            drawn = choropleth(district_features(feats, lookup, band_vals,
                                                 gap_now),
                               class_from_props, band_styles, height=430,
                               tooltip_fields=("region", "risk", "band",
                                               "observed"))
            if not drawn:
                st.image(str(FIGURES / "fig_risk_nowcast.png"))
            legend_md(band_styles,
                      ["Low", "Moderate", "High", "Extreme",
                       "site, no panel score", "not observed"])
            st.caption("Same district polygons and the same no-data rule as the "
                       "paper's maps: a district is shaded only if the "
                       "district-month panel scores it, otherwise it is marked "
                       "as a site without a panel score or drawn as no-data.")
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

# ---- 9. DETECTION DRIVERS ------------------------------------------------- #
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

    st.markdown("#### Local attribution for a single district-day")
    smp = load_csv(TABLES / "shap_sample.csv")
    ev = load_json(RESULTS / "shap_expected_value.json")
    npy = RESULTS / "shap_values.npy"
    outnpy = RESULTS / "shap_row_outputs.npy"
    if smp is None or ev is None or not npy.exists():
        st.info("No persisted local-attribution matrix. Run "
                "`python src/_explain_fast.py`, which writes "
                "`results/shap_values.npy`, `results/tables/shap_sample.csv` "
                "and `results/shap_expected_value.json` together.")
    else:
        sv = np.load(npy)
        order = sh["variable"].tolist() if sh is not None else list(smp.columns)
        names = list(smp.columns)
        pos = {c: k for k, c in enumerate(names)}
        cols = [c for c in order if c in names] + \
               [c for c in names if c not in order]
        n = int(min(len(sv), len(smp)))
        i = st.slider("Which district-day? (row of the persisted sample)",
                      0, n - 1, 0)
        # sv columns follow the sample's own order, so re-pair by name rather
        # than by position whenever the display is sorted by global importance.
        row = pd.DataFrame({"variable": cols,
                            "shap": [float(sv[i][pos[c]]) for c in cols],
                            "value_z": [float(smp.iloc[i][c]) for c in cols]})
        ch3 = alt.Chart(row).mark_bar().encode(
            x=alt.X("shap:Q",
                    title="Attribution (surrogate probability points)"),
            y=alt.Y("variable:N", sort=order, title=None),
            color=alt.Color("shap:Q",
                            scale=alt.Scale(scheme="blueorange"),
                            legend=alt.Legend(title="pushes")),
            tooltip=["variable:N",
                     alt.Tooltip("shap:Q", format="+.4f"),
                     alt.Tooltip("value_z:Q", format="+.2f",
                                 title="standardised value")],
        ).properties(height=340)
        st.altair_chart(ch3, width="stretch")
        base = float(ev["expected_value"])
        total = float(sv[i].sum())
        note = (f"Base value (the surrogate's average output) {base:.3f}; these "
                f"attributions sum to {total:+.3f}. ")
        if outnpy.exists():
            note += (f"The surrogate's own output for this row is "
                     f"{float(np.load(outnpy)[i]):.3f}, and the largest "
                     f"additivity error over all {ev['n_rows']} persisted rows "
                     f"is {ev['additivity_max_abs_error']:.2e}. ")
        note += (f"Rows are {ev['rows']}; the standardised values beside each "
                 f"bar are the {ev['columns']}.")
        st.caption(note)

# ---- 10. ABOUT / REPRODUCIBILITY ------------------------------------------ #
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
- `results/drought_summary.json` + `results/tables/spi_decade_drought.csv` +
  `spi_meteostat.csv` + `spei_era5.csv` — the derived drought indices, their
  ~N(0,1) validation and the register-drought corroboration
- `results/ag_linkage.json` + `results/tables/ag_yield_climate.csv` and
  `results/coffee_linkage.json` + `coffee_prod_climate.csv` — the cereal and
  coffee impact linkage, including the non-significant and sign-reversing
  coefficients
- `results/tables/risk_district_month.csv` +
  `results/early_warning_summary.json` — district-month risk nowcast, its
  skill-vs-null measurement and the measured spread of the one-month carry
- `results/tables/shap_importance.csv` + `permutation_importance.csv` —
  feature drivers of the ensemble detector
- `results/shap_values.npy` + `results/tables/shap_sample.csv` +
  `results/shap_expected_value.json` + `results/shap_row_outputs.npy` — the
  per-district-day local attributions and the base value they explain
- `data/district_boundaries.geojson` + `data/district_lookup.csv` +
  `results/district_coverage.json` — the OCHA Uganda COD-AB admin-2 polygons
  (v01, CC BY-IGO) that every map here and in the paper is drawn on, the
  site-to-district join, and the areal coverage those two imply
- `data/full_ensemble.csv` — daily anomaly stream (~93 k district-days) used in
  the Time-Series Explorer
- `data/disaster_register.csv` — 32 curated disasters (ground truth)

**No model is fitted here.** Re-run the pipeline for a change; the dashboard
picks it up on the next page refresh.

**Environment:** requires `streamlit` and `altair` (bundled with `streamlit`).
The two district choropleths are drawn with `pydeck`, which ships with
Streamlit; if it is missing the panels fall back to the paper's own map images.
No basemap tiles are requested — `map_provider=None` — so the only geometry on
screen is the COD-AB district layer. This repo vendors a CPU wheel set at
`._webpkgs/` (gitignored). Launch with `python dashboard/run_dashboard.py`,
which prepends `._webpkgs/` to `sys.path` before calling the Streamlit CLI.

**Reproducibility:** seeds fixed at 42; every artefact is regenerable with
`python run_all.py`. Sources: ERA5 (Open-Meteo archive API, CC-BY 4.0),
Meteostat (CC-BY 4.0), NASA POWER, MERRA-2, FAOSTAT, World Bank, UNMA/OCHA
disaster register (public reports).
        """
    )
    st.caption(f"Repo root: `{_ROOT}`")
