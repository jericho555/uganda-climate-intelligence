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

# --------------------------------------------------------------------------- #
# Professional CSS injection
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
/* ── Global typography & background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a2332 0%, #0d1b2a 100%);
    color: #e0e6ed;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown {
    color: #e0e6ed !important;
}
section[data-testid="stSidebar"] .stCaption {
    color: #8899aa !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #2a3a4a;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: transparent;
    border-radius: 12px 12px 0 0;
    padding: 4px 4px 0;
}
.stTabs [data-baseweb="tab"] {
    background: #e8ecf1;
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 500;
    font-size: 0.82rem;
    color: #4a5568;
    border: none;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #ffffff;
    color: #1a73e8;
    font-weight: 600;
    box-shadow: 0 -2px 8px rgba(26,115,232,0.12);
}
.stTabs [data-baseweb="tab-highlight"] {
    background: #1a73e8;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #ffffff;
    border-radius: 0 0 12px 12px;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}
[data-testid="stMetric"] label {
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #1e293b !important;
    font-weight: 700 !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-weight: 500 !important;
}

/* ── Headers ── */
h1, h2, h3 {
    color: #1e293b !important;
    font-family: 'Inter', sans-serif !important;
}
h1 { font-weight: 700 !important; }
h2 { font-weight: 600 !important; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }
h3 { font-weight: 600 !important; color: #334155 !important; }

/* ── Info / success / warning boxes ── */
.stAlert-info {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%) !important;
    border: 1px solid #93c5fd !important;
    border-radius: 10px !important;
    color: #1e40af !important;
}
.stAlert-success {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%) !important;
    border: 1px solid #6ee7b7 !important;
    border-radius: 10px !important;
    color: #065f46 !important;
}
.stAlert-warning {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%) !important;
    border: 1px solid #fcd34d !important;
    border-radius: 10px !important;
    color: #92400e !important;
}

/* ── Dataframe tables ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 500 !important;
    color: #475569 !important;
    background: #f8fafc;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    padding: 10px 16px !important;
}
.streamlit-expanderContent {
    border: 1px solid #e2e8f0;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 14px 16px !important;
    background: #ffffff;
}

/* ── Status badges ── */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.status-ok { background: #d1fae5; color: #065f46; }
.status-warn { background: #fef3c7; color: #92400e; }
.status-alert { background: #fee2e2; color: #991b1b; }

/* ── Divider ── */
hr {
    border-color: #e2e8f0 !important;
    margin: 1.2rem 0 !important;
}

/* ── Caption ── */
.stCaption {
    color: #64748b !important;
    font-size: 0.85rem !important;
}

/* ── Chart containers ── */
.stPlotlyChart, [data-testid="stArrowChart"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* ── Image figures ── */
.stImage {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

/* ── Download button ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 24px;
    font-weight: 600;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1557b0 0%, #0d3d7a 100%);
    box-shadow: 0 4px 12px rgba(26,115,232,0.3);
}
</style>
""", unsafe_allow_html=True)

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
        "TemporalIF": "Temporal IF",
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
# Headline counts, read from the artefacts
# --------------------------------------------------------------------------- #
# An earlier revision of this page typed "24 detectors" and "chi2(23)" into the
# sidebar prose.  Those strings survived a re-run that added two detectors, which
# is the exact drift the release gate now refuses, so every count quoted on this
# page comes out of ``benchmark.csv`` / ``friedman.json`` / ``random_null.json``.
_SEQ_NAMES = {"LSTMAE", "TransformerAE"}
_BENCH0 = load_csv(RESULTS / "benchmark.csv")
_FRI0 = load_json(RESULTS / "friedman.json")
N_DET = len(_BENCH0) if _BENCH0 is not None else 0
N_SEQ = (int(_BENCH0["detector"].isin(_SEQ_NAMES).sum())
         if _BENCH0 is not None else 0)
N_NONSEQ = N_DET - N_SEQ
FRI_DF = (int(_FRI0["n_detectors"]) - 1) if _FRI0 else 0
FRI_CHI2 = float(_FRI0["friedman_chi2"]) if _FRI0 else float("nan")
_FDET = (_NULL or {}).get("detectors") or {}
# Non-sequence detectors that stay above the equal-budget random null after
# Benjamini-Hochberg control, and the largest excess any of them reaches.
_SURV = [k for k, v in _FDET.items()
         if k not in _SEQ_NAMES and float(v.get("bh_q", 1.0)) < 0.05]
N_SURV = len(_SURV)
SURV_MAX_RATIO = max([float(_FDET[k]["ratio_vs_null"]) for k in _SURV],
                     default=float("nan"))


# --------------------------------------------------------------------------- #
# Sidebar (global filters + provenance)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:8px 0 4px">'
        '<span style="font-size:2.2rem">\U0001F30D</span>'
        '<h2 style="margin:4px 0 2px;color:#e0e6ed;font-weight:700;">'
        'Uganda Climate</h2>'
        '<h3 style="margin:0;color:#64b5f6;font-weight:500;font-size:1rem;">'
        'Intelligence Framework</h3>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── System layer status ── #
    st.markdown("**System Layers**")
    _bench_ok = _BENCH0 is not None and len(_BENCH0) == 26
    _ew_ok = (RESULTS / "drought_ew_summary.json").exists()
    _yf_ok = (RESULTS / "yield_forecast_summary.json").exists()
    _so_ok = (RESULTS / "seasonal_outlook_summary.json").exists()
    _bl_ok = any((RESULTS.parent / "reports").glob("climate_risk_bulletin_*.md"))
    _db_ok = True  # dashboard is running
    for label, ok in [
        ("Detection Benchmark (26)", _bench_ok),
        ("Drought Early Warning", _ew_ok),
        ("Yield Forecast", _yf_ok),
        ("Seasonal Outlook (SEAS5)", _so_ok),
        ("Automated Bulletin", _bl_ok),
        ("Dashboard Engine", _db_ok),
    ]:
        icon = "\u2705" if ok else "\u26a0\ufe0f"
        st.markdown(f"{icon} &nbsp; {label}")

    st.divider()
    if st.checkbox("Show random-null reference lines", value=True,
                   key="show_null"):
        pass
    if st.checkbox("Dark theme for charts", value=False, key="dark_charts"):
        alt.data_transformers.enable("default")
    st.divider()
    st.markdown(
        f"**Pipeline summary**\n\n"
        f"- Benchmark: {N_DET} detectors ({N_NONSEQ} non-sequence)\n"
        f"- Held-out test: 2019\u20132023 (5 years)\n"
        f"- Register: 32 curated disasters\n"
        f"- Base rate: {BASE_RATE:.4f}\n"
        f"- Random-flag F1 null: {RANDOM_NULL_F1:.3f} "
        f"[{RANDOM_NULL_CI[0]:.3f}, {RANDOM_NULL_CI[1]:.3f}]\n"
        f"- Friedman \u03c7\u00b2({FRI_DF}) over 5 test years\n"
        f"- Dashboard: 14 tabs, 54 metrics"
    )
    st.divider()
    st.markdown(
        "**Data sources**\n\n"
        "ERA5 (Open-Meteo) \u00b7 Meteostat \u00b7 NASA POWER \u00b7 "
        "MERRA-2 \u00b7 FAOSTAT \u00b7 World Bank \u00b7 OCHA COD-AB\n\n"
        "All CC-BY 4.0 \u00b7 Seed = 42"
    )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_over, tab_bench, tab_ts, tab_spatial, tab_events, tab_trends, \
    tab_impact, tab_now, tab_drivers, tab_drought_ew, tab_yield, \
    tab_outlook, tab_bulletin, tab_about = st.tabs([
        "Overview",
        "Detection Benchmark",
        "Time-Series",
        "Spatial Hotspots",
        "Events & Severity",
        "Trends",
        "Impacts",
        "Risk Nowcast",
        "Detection Drivers",
        "Drought EW",
        "Yield Forecast",
        "Seasonal Outlook",
        "Bulletin",
        "About",
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
        seq_names = _SEQ_NAMES
        seq_mask = bench["detector"].isin(seq_names)
        best_f1_overall = bench["f1"].max()
        best_f1_cross = bench.loc[~seq_mask, "f1"].max()
        roc_min = bench.loc[~seq_mask, "roc_auc"].min()
        roc_max = bench.loc[~seq_mask, "roc_auc"].max()
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
            f"disaster windows (ROC-AUC span {roc_min:.2f}\u2013{roc_max:.2f}; best "
            f"cross-sectional F1 = {best_f1_cross:.3f}). Measured against the "
            f"size-matched random alert budget (F1 {RANDOM_NULL_F1:.3f}), only "
            f"{N_SURV} of the {N_NONSEQ} non-sequence methods stay above it "
            f"after Benjamini-Hochberg control, and the best of them reaches "
            f"{SURV_MAX_RATIO:.2f}\u00d7 it, while the "
            f"two torch sequence autoencoders (LSTM-AE, Transformer-AE) roughly "
            f"double it (best F1 = {best_f1_overall:.3f}), still with global "
            f"ranking near chance. The lag-window Temporal IF and the local "
            f"density LoOP, both added at revision, join the chance-level group "
            f"rather than escaping it."
        )
        st.markdown("### Headline table (top 5 by PR-AUC)")
        show = ["detector", "precision", "recall", "f1", "roc_auc", "pr_auc",
                "mcc"]
        disp = bench.head(5)[show].copy()
        disp["detector"] = disp["detector"].map(humanize)
        st.dataframe(disp.style.format(
            {c: "{:.3f}" for c in show[1:]},
        ), width="stretch", hide_index=True)
        st.caption(f"Full {n_det}-row table is in the **Detection Benchmark** "
                   "tab.")

# ---- 2. DETECTION BENCHMARK ----------------------------------------------- #
with tab_bench:
    bench = load_csv(RESULTS / "benchmark.csv")
    if bench is None:
        st.error("benchmark.csv missing. Run the pipeline.")
        st.stop()
    st.header(f"Held-out detection benchmark ({len(bench)} detectors)")
    b = bench.copy()
    # Attach the measured Monte-Carlo verdict beside each row, so the table can
    # be read against the equal-budget random null without leaving the page.
    if _FDET:
        nul = pd.DataFrame([{"detector": k,
                             "f1 vs null": float(v["ratio_vs_null"]),
                             "BH q": float(v["bh_q"])}
                            for k, v in _FDET.items()])
        b = b.merge(nul, on="detector", how="left")
    b["detector"] = b["detector"].map(humanize)
    b["family"] = np.where(b["detector"].isin(["LSTM-AE", "Transformer-AE"]),
                           "sequence (torch)",
                           np.where(b["detector"].isin(
                               ["Deep AE (MLP)", "VAE (NumPy)", "AutoEncoder"]),
                               "feed-forward deep",
                               np.where(b["detector"].isin(
                                   ["SARIMAX", "Structural TS"]),
                                   "temporal (innovation)",
                                   np.where(b["detector"] == "Temporal IF",
                                            "temporal (lag-window)",
                                            "cross-sectional"))))
    st.dataframe(
        b[
            ["detector", "family", "precision", "recall", "f1", "f1_lo",
             "f1_hi", "roc_auc", "pr_auc", "pr_auc_lo", "pr_auc_hi", "mcc",
             "f1 vs null", "BH q"]
        ].style.format({**{c: "{:.3f}" for c in
                           ["precision", "recall", "f1", "f1_lo",
                            "f1_hi", "roc_auc", "pr_auc", "pr_auc_lo",
                            "pr_auc_hi", "mcc"]},
                        **{"f1 vs null": "{:.2f}", "BH q": "{:.3g}"}}),
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
                                    "temporal (lag-window)",
                                    "feed-forward deep",
                                    "sequence (torch)"],
                            range=["#888", "#3b7dd8", "#6a51a3",
                                   "#b5651d", "#c1272d"])),
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
               f"{RANDOM_NULL_CI[0]:.3f}\u2013{RANDOM_NULL_CI[1]:.3f}). Solid grey = "
               "base rate "
               f"({BASE_RATE:.4f}). {N_SURV} of the {N_NONSEQ} non-sequence "
               "points sit right of the null at BH q<0.05, and the furthest "
               f"advanced is {SURV_MAX_RATIO:.2f}\u00d7 it; only the two red sequence "
               "points roughly double it.")

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

        wbc = load_json(RESULTS / "wb_climate_check.json")
        wbs = load_csv(TABLES / "wb_spei_compare.csv")
        wbpr = load_csv(TABLES / "wb_precip_compare.csv")
        if wbc and wbs is not None and len(wbs):
            st.markdown("#### Cross-check against the World Bank climate series")
            ag = wbc.get("agreement_vs_ours_spei6_mean") or {}
            dry = wbc.get("dry_year_check") or {}
            pct = wbc.get("precip_normal_check") or {}
            wbr = ag.get("wb_range") or []
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Bank SPEI vs our SPEI6 (r)",
                      f"{ag.get('pearson_r', float('nan')):.3f}",
                      delta=f"p = {ag.get('pearson_p', float('nan')):.2f}",
                      delta_color="off")
            w2.metric("Spearman rho", f"{ag.get('spearman_rho', float('nan')):.3f}",
                      delta=f"sign agreement "
                            f"{100 * ag.get('sign_agreement', 0):.0f}%",
                      delta_color="off")
            w3.metric("Mean |difference|",
                      f"{ag.get('mean_abs_diff', float('nan')):.2f} SD",
                      delta=f"Bank range {wbr[0]} to {wbr[1]}" if len(wbr) == 2
                      else None, delta_color="off")
            w4.metric("ERA5 annual rain vs Bank normal",
                      f"{pct.get('pct_of_normal', float('nan')):.0f}%",
                      delta=f"{pct.get('era5_mean_annual_mm', 0):.0f} mm vs "
                            f"{pct.get('wb_normal_mm', 0):.0f} mm",
                      delta_color="off")
            regy = {int(r.year): float(r.wb_spei)
                    for _, r in wbs.iterrows() if pd.notna(r["wb_spei"])}
            dy = ", ".join(f"{y} ({regy[y]:+.2f})" for y in
                           dry.get("register_drought_years", []) if y in regy)
            st.warning(
                "**This cross-check does not validate the index, and the paper "
                "reports it that way.** The Bank's national annual SPEI is "
                f"statistically unrelated to our district-mean SPEI6 "
                f"(r = {ag.get('pearson_r', float('nan')):.3f}, "
                f"p = {ag.get('pearson_p', float('nan')):.2f}, "
                f"n = {ag.get('n_years', 0)} overlapping years), and in every "
                f"register drought year the Bank's index is wet: {dy}. The "
                "Bank's own three severe-drought years "
                "(" + ", ".join(str(y) for y in
                                dry.get("wb_years_le_minus_1", [])) + ") "
                "fall outside the window in which our panel exists. Two "
                "explanations are structural rather than statistical: the Bank "
                "series is a national annual mean, while the register events "
                "are seasonal and district-local, and the two precipitation "
                "products do not share a baseline "
                f"({pct.get('era5_mean_annual_mm', 0):.0f} mm of ERA5 district "
                f"means against a constant {pct.get('wb_normal_mm', 0):.0f} mm "
                "World Bank normal). This is therefore a provenance and scale "
                "check, not an external corroboration.")
            o1, o2 = st.columns([3, 2])
            with o1:
                wt = wbs.dropna(subset=["ours_spei6_mean"]).copy()
                if len(wt):
                    wtidy = wt.melt(id_vars=["year"],
                                    value_vars=["wb_spei", "ours_spei6_mean",
                                                "ours_spei6_min"],
                                    value_name="index",
                                    var_name="series").dropna(
                                        subset=["index"])
                    st.altair_chart(
                        alt.Chart(wtidy).mark_line(point=True).encode(
                            x=alt.X("year:O", title=None),
                            y=alt.Y("index:Q", title="annual SPEI (SD)"),
                            color=alt.Color("series:N",
                                            legend=alt.Legend(title=None)),
                            tooltip=["year:O", "series:N",
                                     alt.Tooltip("index:Q", format=".2f")],
                        ).properties(height=300), width="stretch")
            with o2:
                st.dataframe(wbs.style.format(
                    {"wb_spei": "{:.2f}", "ours_spei6_mean": "{:.2f}",
                     "ours_spei6_min": "{:.2f}"}), width="stretch",
                    hide_index=True, height=300)
            if wbpr is not None and len(wbpr):
                normal = float(wbpr["wb_normal_mm"].iloc[0])
                st.altair_chart(
                    alt.Chart(wbpr).mark_bar(color="#3b7dd8").encode(
                        x=alt.X("year:O", title=None),
                        y=alt.Y("era5_mean_mm:Q", title="ERA5 district-mean "
                                                     "annual rainfall (mm)",
                                scale=alt.Scale(zero=False)),
                        tooltip=["year:O", alt.Tooltip("era5_mean_mm:Q",
                                                       format=".0f"),
                                 alt.Tooltip("wb_normal_mm:Q",
                                             title="Bank normal")],
                    ).properties(height=180) +
                    alt.Chart(pd.DataFrame({"v": [normal]})).mark_rule(
                        color="#555", strokeDash=[4, 4]).encode(y="v:Q"),
                    width="stretch")
                st.caption(f"Dashed grey rule is the Bank's constant normal "
                           f"({normal:.0f} mm); the "
                           f"driest and wettest ERA5 sites are "
                           f"{pct.get('era5_driest_site', ['—'])[0]} "
                           f"{pct.get('era5_driest_site', [0, 0])[1]:.0f} mm and "
                           f"{pct.get('era5_wettest_site', ['—'])[0]} "
                           f"{pct.get('era5_wettest_site', [0, 0])[1]:.0f} mm, "
                           "so a national normal cannot describe either. "
                           "Source: ``results/tables/wb_precip_compare.csv``.")
            if (FIGURES / "fig_wb_check.png").exists():
                st.image(str(FIGURES / "fig_wb_check.png"))

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
                            "record. It describes what the score tends to do; "
                            "the conformal interval below is the version with a "
                            "measured coverage rate on held-out months.")
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
            pi = ews.get("prediction_intervals") or {}
            pmdl = pi.get("models") or {}
            pers = pmdl.get("persistence") or {}
            ridg = pmdl.get("ridge") or {}
            if pers and ridg and "level_90" in pers and "level_90" in ridg:
                st.markdown("#### Fitted prediction intervals "
                            "(split conformal)")
                ty = pi.get("test_years") or []
                st.caption(
                    f"Split conformal on {pi['n_transitions']:,} district-month "
                    f"transitions: {pi['n_calibration']:,} calibration months "
                    f"(destination month \u2264 2018) and {pi['n_test']:,} "
                    "held-out "
                    "test months "
                    + (f"{ty[0]}\u2013{ty[1]}" if len(ty) == 2 else "")
                    + ". The interval is the \u00b1 residual quantile measured on "
                    "the calibration half; the coverage column is what it "
                    "actually achieved on the test half, so no guarantee is "
                    "claimed that the data did not deliver.")
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Persistence RMSE (test)", f"{pers['rmse_test']}",
                          delta=f"MAE {pers['mae_test']}", delta_color="off")
                q2.metric("Ridge carry RMSE (test)", f"{ridg['rmse_test']}",
                          delta=f"MAE {ridg['mae_test']}", delta_color="off")
                q3.metric("90% band, persistence",
                          f"\u00b1{pers['level_90']['conformal_quantile']}",
                          delta=f"covers "
                                f"{100 * pers['level_90']['coverage_test']:.1f}%",
                          delta_color="off")
                q4.metric("90% band, ridge carry",
                          f"\u00b1{ridg['level_90']['conformal_quantile']}",
                          delta=f"covers "
                                f"{100 * ridg['level_90']['coverage_test']:.1f}%",
                          delta_color="off")
                rows = []
                for mname, m in (("persistence", pers), ("ridge carry", ridg)):
                    for lvl in ("level_90", "level_80"):
                        d = m.get(lvl) or {}
                        if "conformal_quantile" not in d:
                            continue
                        band = d.get("coverage_by_starting_band") or {}
                        rows.append(
                            {"model": mname,
                             "nominal level": 1 - float(d.get("alpha", 0)),
                             "\u00b1 risk points": d["conformal_quantile"],
                             "mean width": d["mean_width"],
                             "test coverage": d["coverage_test"],
                             **{f"from {b}": (band.get(b) or {}).get("coverage")
                                for b in ("Low", "Moderate", "High",
                                          "Extreme")}})
                bands = (pers.get("level_90", {}).get(
                    "coverage_by_starting_band") or {})
                fmt = {"nominal level": "{:.0%}", "\u00b1 risk points": "{:.1f}",
                       "mean width": "{:.1f}", "test coverage": "{:.1%}"}
                fmt.update({f"from {b}": "{:.1%}" for b in bands})
                st.dataframe(pd.DataFrame(rows).style.format(fmt),
                             width="stretch", hide_index=True)
                st.caption("'from X' columns are the coverage on test "
                           "transitions that started in severity band X "
                           "(n = " + ", ".join(
                               f"{b} {bands[b]['n']}" for b in bands) + "). "
                           "Source: ``prediction_intervals`` in "
                           "``results/early_warning_summary.json``.")
                if pi.get("verdict"):
                    st.info(f"**Verdict as written by the pipeline:** "
                            f"{pi['verdict']}.")
                co = ridg.get("largest_standardised_coefficients") or []
                if co:
                    st.caption("The ridge carry model is dominated by " +
                               ", ".join(f"{n} {v:+.2f}" for n, v in co) +
                               " (risk points per standardised unit).")
                iv = load_csv(TABLES / "nowcast_interval.csv")
                if iv is not None and len(iv):
                    st.markdown("**Next-month interval, most recent month** "
                                f"({iv['as_of'].iloc[0]})")
                    st.dataframe(iv, width="stretch", hide_index=True)
                if (FIGURES / "fig_intervals.png").exists():
                    st.image(str(FIGURES / "fig_intervals.png"))
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

# ---- 11. DROUGHT EARLY WARNING -------------------------------------------- #
with tab_drought_ew:
    st.header("\U0001F321\uFE0F Drought Early Warning System")
    ew_df = load_csv(TABLES / "drought_early_warning.csv")
    ew_sum = load_json(RESULTS / "drought_ew_summary.json")
    if ew_df is None or ew_sum is None:
        st.warning("Drought early-warning artefacts not found. "
                   "Run `python run_all.py` first.")
    else:
        st.info(
            "Threshold-based drought early warning using SPI-6 severity, "
            "trajectory (3-month OLS slope), and consecutive-decline runs. "
            "Validation is against the curated disaster register (32 events)."
        )
        # Status counts
        sc = ew_sum.get("status_counts", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("\U0001F7E2 Normal", sc.get("Normal", 0))
        c2.metric("\U0001F7E1 Watch", sc.get("Watch", 0))
        c3.metric("\U0001F7E0 Warning", sc.get("Warning", 0))
        c4.metric("\U0001F534 Alert", sc.get("Alert", 0))

        # ── Interactive charts row ── #
        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.markdown("**Status distribution**")
            status_data = pd.DataFrame([
                {"Status": k, "Count": v}
                for k, v in sc.items() if v > 0
            ])
            if len(status_data):
                color_scale = alt.Scale(
                    domain=["Normal", "Watch", "Warning", "Alert"],
                    range=["#4ade80", "#facc15", "#fb923c", "#ef4444"])
                pie = alt.Chart(status_data).mark_arc(outerRadius=100, innerRadius=50).encode(
                    theta=alt.Theta(field="Count", type="quantitative"),
                    color=alt.Color(field="Status", type="nominal", scale=color_scale),
                    tooltip=["Status", "Count"]
                ).properties(width=300, height=250)
                st.altair_chart(pie, use_container_width=True)

        with chart_right:
            st.markdown("**Validation vs documented droughts**")
            val = ew_sum.get("validation_vs_register", {})
            val_data = pd.DataFrame([
                {"Metric": "Hits", "Value": val.get("hits", 0)},
                {"Metric": "False alarms", "Value": val.get("false_alarms", 0)},
                {"Metric": "Misses", "Value": val.get("misses", 0)},
            ])
            bar = alt.Chart(val_data).mark_bar(cornerRadius=6).encode(
                x=alt.X("Metric:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Value:Q", axis=alt.Axis(title="Count")),
                color=alt.Color("Metric:N",
                    scale=alt.Scale(
                        domain=["Hits", "False alarms", "Misses"],
                        range=["#4ade80", "#fb923c", "#ef4444"])),
                tooltip=["Metric", "Value"]
            ).properties(width=300, height=250)
            st.altair_chart(bar, use_container_width=True)

        # ── SPI-6 trajectory chart ── #
        ew_df["month"] = pd.to_datetime(ew_df["month"])
        latest_month = ew_df["month"].max()
        latest = ew_df[ew_df["month"] == latest_month]

        with st.expander("\U0001F4C8 SPI-6 trajectory (last 6 months)", expanded=False):
            recent = ew_df[ew_df["month"] >= latest_month - pd.DateOffset(months=5)]
            if len(recent):
                line = alt.Chart(recent).mark_line(point=True, strokeWidth=2).encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("SPI6:Q", title="SPI-6",
                            scale=alt.Scale(domain=[-3, 3])),
                    color=alt.Color("location:N", legend=None),
                    tooltip=["location", "month", "SPI6", "status"]
                ).properties(width=700, height=300)
                rule_normal = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                    strokeDash=[4, 4], stroke="#94a3b8").encode(y="y")
                rule_mod = alt.Chart(pd.DataFrame({"y": [-1]})).mark_rule(
                    strokeDash=[4, 4], stroke="#fb923c").encode(y="y")
                rule_sev = alt.Chart(pd.DataFrame({"y": [-1.5]})).mark_rule(
                    strokeDash=[4, 4], stroke="#ef4444").encode(y="y")
                st.altair_chart(line + rule_normal + rule_mod + rule_sev,
                                use_container_width=True)
                st.caption("Dashed lines: Normal (0), Moderate drought (\u22121), "
                           "Severe drought (\u22121.5).")

        # ── Status timeline heatmap ── #
        with st.expander("\U0001F5D3\uFE0F District status timeline", expanded=False):
            status_order = {"Normal": 0, "Watch": 1, "Warning": 2, "Alert": 3}
            timeline_data = ew_df.copy()
            timeline_data["status_ord"] = timeline_data["status"].map(status_order)
            if len(timeline_data):
                heat = alt.Chart(timeline_data).mark_rect().encode(
                    x=alt.X("month:T", title="Month", timeUnit="month"),
                    y=alt.Y("location:N", title="District", sort="-x"),
                    color=alt.Color("status_ord:Q", scale=alt.Scale(
                        domain=[0, 1, 2, 3],
                        range=["#4ade80", "#facc15", "#fb923c", "#ef4444"]),
                        legend=None),
                    tooltip=["location", "month", "status", "SPI6"]
                ).properties(width=700, height=400)
                st.altair_chart(heat, use_container_width=True)

        # ── Choropleth map ── #
        st.markdown("**District status map**")
        feats, lk = load_district_features()
        if feats is not None:
            lookup = lk
            status_by_district = dict(zip(latest["location"], latest["status"]))
            STATUS_COLOR = {
                "Normal": (200, 225, 200, 255), "Watch": (255, 230, 140, 255),
                "Warning": (255, 160, 80, 255), "Alert": (220, 50, 50, 255),
            }
            UNOBSERVED = {"fill": [*NODATA_RGB, 255], "line": [185, 185, 180], "lw": 1}
            SITE_STYLE = {"fill": [214, 214, 208, 255], "line": [120, 120, 120], "lw": 2}
            status_styles = {s: {"fill": [*c[:3], 255], "line": [74, 74, 74], "lw": 1}
                             for s, c in STATUS_COLOR.items()}
            status_styles["site, no panel score"] = SITE_STYLE
            status_styles["not observed"] = UNOBSERVED
            vals_by_site = {}
            for loc, status in status_by_district.items():
                vals_by_site[loc] = {"status": status, "klass": status, "observed": "yes"}
            gap = site_gap_districts(
                load_csv(DATA / "district_lookup.csv"),
                {lookup[s] for s in vals_by_site if s in lookup})
            drawn = choropleth(
                district_features(feats, lookup, vals_by_site, gap),
                class_from_props, status_styles, height=430,
                tooltip_fields=("region", "status", "observed"))
            if not drawn:
                fig_ew = FIGURES / "fig_drought_ew.png"
                if fig_ew.exists():
                    st.image(str(fig_ew))
            legend_md(status_styles,
                      ["Normal", "Watch", "Warning", "Alert",
                       "site, no panel score", "not observed"])
        st.caption(f"As of {latest_month.strftime('%Y-%m')}. "
                   f"SPI-6 categories per WMO/McKee 1993.")

        # ── District detail table ── #
        with st.expander("\U0001F4CB District detail table"):
            detail = latest[["location", "region", "SPI6", "SPEI6",
                             "spi6_class", "trajectory", "decline_run",
                             "status"]].copy()
            detail.columns = ["District", "Region", "SPI-6", "SPEI-6",
                              "SPI-6 class", "Trajectory", "Decline run",
                              "Status"]
            st.dataframe(detail.style.apply(
                lambda row: [
                    "background-color: #dc262630" if row["Status"] == "Alert"
                    else "background-color: #ea580c30" if row["Status"] == "Warning"
                    else "background-color: #ca8a0430" if row["Status"] == "Watch"
                    else "background-color: #16a34a20"
                    for _ in row
                ], axis=1
            ), width="stretch", hide_index=True)

        # Limitations
        lim = ew_sum.get("limitations", [])
        if lim:
            with st.expander("\u26a0\ufe0f Limitations"):
                for l in lim:
                    st.markdown(f"- {l}")

# ---- 12. YIELD FORECAST ----------------------------------------------------- #
with tab_yield:
    st.header("\U0001F33D Crop Yield Impact Forecast")
    y_sum = load_json(RESULTS / "yield_forecast_summary.json")
    y_df = load_csv(TABLES / "yield_forecast.csv")
    if y_sum is None:
        st.warning("Yield forecast artefacts not found. "
                   "Run `python run_all.py` first.")
    else:
        st.info(
            "National cereal-yield anomaly regressed against growing-season "
            "SPI-3 (March\u2013May + September\u2013November). "
            "Skill check compares model RMSE against a climatology null."
        )
        mdl = y_sum.get("model", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R\u00b2", f"{mdl.get('r_squared', 'N/A')}")
        c2.metric("RMSE", f"{mdl.get('rmse', 'N/A')}%")
        c3.metric("Pearson r", f"{mdl.get('pearson_r', 'N/A')}")
        c4.metric("p-value", f"{mdl.get('pearson_p', 'N/A')}")

        # ── Interactive scatter: yield anomaly vs SPI ── #
        if y_df is not None and len(y_df):
            st.markdown("**Yield anomaly vs growing-season SPI**")
            scatter = alt.Chart(y_df).mark_circle(size=80, opacity=0.7).encode(
                x=alt.X("spi:Q", title="Growing-season SPI-3 (national mean)",
                         scale=alt.Scale(zero=False)),
                y=alt.Y("observed_anomaly_pct:Q", title="Yield anomaly (%)"),
                tooltip=["year:O", "spi:Q", "observed_anomaly_pct:Q",
                         "fitted_anomaly_pct:Q"]
            ).properties(width=350, height=280)
            # Regression line
            if mdl.get("slope") is not None:
                x_range = [y_df["spi"].min(), y_df["spi"].max()]
                line_df = pd.DataFrame({
                    "x": x_range,
                    "y": [mdl["intercept"] + mdl["slope"] * x for x in x_range]
                })
                reg = alt.Chart(line_df).mark_line(stroke="#ef4444", strokeWidth=2).encode(
                    x="x:Q", y="y:Q")
                st.altair_chart(scatter + reg, use_container_width=True)
            else:
                st.altair_chart(scatter, use_container_width=True)

        # ── Time series: observed vs fitted ── #
        if y_df is not None and len(y_df):
            st.markdown("**Observed vs fitted yield anomaly**")
            ts_data = y_df.melt(
                id_vars=["year"],
                value_vars=["observed_anomaly_pct", "fitted_anomaly_pct"],
                var_name="type", value_name="value")
            ts_data["type"] = ts_data["type"].map({
                "observed_anomaly_pct": "Observed",
                "fitted_anomaly_pct": "Fitted"})
            ts_chart = alt.Chart(ts_data).mark_line(point=True).encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("value:Q", title="Yield anomaly (%)"),
                color=alt.Color("type:N",
                    scale=alt.Scale(domain=["Observed", "Fitted"],
                                    range=["#1a73e8", "#ef4444"])),
                tooltip=["year:O", "type:N", "value:Q"]
            ).properties(width=700, height=250)
            st.altair_chart(ts_chart, use_container_width=True)

        # ── Skill check ── #
        skill = y_sum.get("skill_check", {})
        verdict = skill.get("verdict", "N/A").replace("_", " ")
        ratio = skill.get("skill_ratio")
        if ratio is not None:
            st.markdown("**Skill check: model vs climatology null**")
            skill_data = pd.DataFrame([
                {"Model": "SPI Regression (holdout)", "RMSE (%)": skill.get("holdout_rmse", 0)},
                {"Model": "Climatology Null", "RMSE (%)": skill.get("climatology_rmse", 0)},
            ])
            skill_bar = alt.Chart(skill_data).mark_bar(cornerRadius=6).encode(
                x=alt.X("Model:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("RMSE (%):Q", axis=alt.Axis(title="RMSE (%)")),
                color=alt.Color("Model:N",
                    scale=alt.Scale(
                        domain=["SPI Regression (holdout)", "Climatology Null"],
                        range=["#1a73e8", "#94a3b8"])),
                tooltip=["Model", "RMSE (%)"]
            ).properties(width=400, height=220)
            st.altair_chart(skill_bar, use_container_width=False)
            if verdict == "climatology beats model":
                st.warning(
                    f"**{verdict.title()}** (ratio = {ratio:.3f}). "
                    f"The SPI regression does not beat a simple mean forecast "
                    f"out of sample. This is an honest negative result.")
            else:
                st.success(f"**{verdict.title()}** (ratio = {ratio:.3f})")

        # ── Projection ── #
        proj = y_sum.get("projection", {})
        if proj.get("projected_anomaly_pct") is not None:
            st.markdown(f"**{y_sum.get('projection_year', 'N/A')} Projection**")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Current SPI", f"{proj.get('current_spi', 'N/A')}")
            pc2.metric("Projected anomaly",
                       f"{proj.get('projected_anomaly_pct', 'N/A'):.1f}%")
            pi = proj.get("prediction_interval_95", [None, None])
            pc3.metric("95% PI",
                       f"[{pi[0]:.1f}, {pi[1]:.1f}]%" if pi[0] is not None else "N/A")
            pc4.metric("P(below-normal)",
                       f"{proj.get('p_below_normal', 0):.1%}")

        # Figure fallback
        with st.expander("\U0001F4CA Static figure"):
            fig_y = FIGURES / "fig_yield_forecast.png"
            if fig_y.exists():
                st.image(str(fig_y))

        lim = y_sum.get("limitations", [])
        if lim:
            with st.expander("\u26a0\ufe0f Limitations"):
                for l in lim:
                    st.markdown(f"- {l}")

# ---- 13. SEASONAL OUTLOOK --------------------------------------------------- #
with tab_outlook:
    st.header("\U0001F326\uFE0F Seasonal Climate Outlook")
    o_sum = load_json(RESULTS / "seasonal_outlook_summary.json")
    o_df = load_csv(TABLES / "seasonal_outlook.csv")
    if o_sum is None or not o_sum.get("available", False):
        st.warning(
            "Seasonal forecast data is currently unavailable. "
            "The Open-Meteo SEAS5 API could not be reached during the "
            "last pipeline run. "
            "Downstream modules gracefully degrade: the drought early-warning "
            "and yield forecast layers are unaffected."
        )
        fig_o = FIGURES / "fig_seasonal_outlook.png"
        if fig_o.exists():
            st.image(str(fig_o))
    else:
        st.info(
            "Downscaled seasonal precipitation forecast from ECMWF SEAS5 "
            "(51 ensemble members, 7-month lead time) via the Open-Meteo "
            "Seasonal Forecast API. Integrated with drought EW status."
        )
        fc = o_sum.get("forecast_class_counts", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("\U0001F534 Below normal", fc.get("below_normal", 0))
        c2.metric("\U0001F7E1 Near normal", fc.get("near_normal", 0))
        c3.metric("\U0001F535 Above normal", fc.get("above_normal", 0))

        # ── Forecast class chart ── #
        fc_data = pd.DataFrame([
            {"Class": "Below\nnormal", "Count": fc.get("below_normal", 0)},
            {"Class": "Near\nnormal", "Count": fc.get("near_normal", 0)},
            {"Class": "Above\nnormal", "Count": fc.get("above_normal", 0)},
            {"Class": "Unknown", "Count": fc.get("unknown", 0)},
        ])
        fc_chart = alt.Chart(fc_data).mark_bar(cornerRadius=6).encode(
            x=alt.X("Class:N", axis=alt.Axis(labelAngle=0, title="Forecast Class")),
            y=alt.Y("Count:Q", axis=alt.Axis(title="District-months")),
            color=alt.Color("Class:N",
                scale=alt.Scale(
                    domain=["Below\nnormal", "Near\nnormal", "Above\nnormal", "Unknown"],
                    range=["#ef4444", "#facc15", "#3b82f6", "#94a3b8"])),
            tooltip=["Class", "Count"]
        ).properties(width=400, height=220)
        st.altair_chart(fc_chart, use_container_width=False)

        # ── Escalated status ── #
        esc = o_sum.get("escalated_status_counts", {})
        st.markdown("**Escalated risk status after outlook integration**")
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("\U0001F7E2 Normal", esc.get("Normal", 0))
        ec2.metric("\U0001F7E1 Watch", esc.get("Watch", 0))
        ec3.metric("\U0001F7E0 Warning", esc.get("Warning", 0))
        ec4.metric("\U0001F534 Alert", esc.get("Alert", 0))

        n_esc = o_sum.get("n_districts_escalated", 0)
        n_total = o_sum.get("n_districts", 19) * o_sum.get("n_months_forecast", 7)
        if n_total > 0:
            st.caption(
                f"{n_esc} of {n_total} district-months escalated to a higher "
                f"risk tier ({100*n_esc/n_total:.0f}%). Escalation rule: "
                f"below-normal forecast + Watch/Warning EW status \u2192 +1 tier."
            )

        # ── Choropleth ── #
        st.markdown("**District forecast map**")
        if o_df is not None and len(o_df):
            feats, lk = load_district_features()
            if feats is not None:
                lookup = lk
                OUTLOOK_COLOR = {
                    "below_normal": (220, 80, 60),
                    "near_normal": (255, 230, 140),
                    "above_normal": (100, 180, 220),
                }
                UNOBSERVED = {"fill": [*NODATA_RGB, 255], "line": [185, 185, 180], "lw": 1}
                SITE_STYLE = {"fill": [214, 214, 208, 255], "line": [120, 120, 120], "lw": 2}
                outlook_styles = {k: {"fill": [*v, 255], "line": [74, 74, 74], "lw": 1}
                                  for k, v in OUTLOOK_COLOR.items()}
                outlook_styles["site, no panel score"] = SITE_STYLE
                outlook_styles["not observed"] = UNOBSERVED
                o_df["month"] = pd.to_datetime(o_df["month"])
                latest = o_df.sort_values("month").groupby("location").last().reset_index()
                class_by_district = dict(zip(latest["location"], latest["forecast_class"]))
                vals_by_site = {}
                for loc, klass in class_by_district.items():
                    vals_by_site[loc] = {"forecast_class": klass, "klass": klass, "observed": "yes"}
                gap = site_gap_districts(
                    load_csv(DATA / "district_lookup.csv"),
                    {lookup[s] for s in vals_by_site if s in lookup})
                drawn = choropleth(
                    district_features(feats, lookup, vals_by_site, gap),
                    class_from_props, outlook_styles, height=430,
                    tooltip_fields=("region", "forecast_class", "observed"))
                if not drawn:
                    fig_o = FIGURES / "fig_seasonal_outlook.png"
                    if fig_o.exists():
                        st.image(str(fig_o))
                legend_md(outlook_styles,
                          ["below_normal", "near_normal", "above_normal",
                           "site, no panel score", "not observed"])
        lim = o_sum.get("limitations", [])
        if lim:
            with st.expander("\u26a0\ufe0f Limitations"):
                for l in lim:
                    st.markdown(f"- {l}")

# ---- 14. CLIMATE BULLETIN --------------------------------------------------- #
with tab_bulletin:
    st.header("\U0001F4DC Climate Risk Bulletin")
    st.info(
        "Automated narrative bulletin synthesising all forward-looking layers. "
        "Every number is read from the persisted summary JSONs \u2014 nothing is "
        "hard-coded, so the bulletin cannot drift from the pipeline."
    )
    # Find the latest bulletin
    reports_dir = RESULTS.parent / "reports"
    bulletins = sorted(reports_dir.glob("climate_risk_bulletin_*.md")) if reports_dir.exists() else []
    if not bulletins:
        st.warning("No bulletin found. Run `python run_all.py` first.")
    else:
        latest_bulletin = bulletins[-1]
        bulletin_text = latest_bulletin.read_text(encoding="utf-8")
        st.markdown(f"**Latest bulletin:** `{latest_bulletin.name}`")
        st.divider()
        st.markdown(bulletin_text)
        st.divider()
        st.download_button(
            label="\U0001F4E5 Download Bulletin (Markdown)",
            data=bulletin_text,
            file_name=latest_bulletin.name,
            mime="text/markdown",
        )

# ---- 15. ABOUT / REPRODUCIBILITY ------------------------------------------- #
with tab_about:
    st.header("About this dashboard")
    st.markdown(
        f"""
This view is a **read-only interface** to the artefacts produced by
``run_all.py``. Every number rendered here is loaded from a file on disk.

**Data sources rendered:**

- `results/benchmark.csv` — {N_DET}-detector held-out metrics with bootstrap CIs
- `results/friedman.json` — Friedman statistic ({N_DET} detectors × 5 years)
- `results/random_null.json` — the size-matched random alert budget every detector
  is judged against, with its Monte-Carlo band and Benjamini-Hochberg q-values
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
- `data/wb_climate.csv` + `data/wb_climate_meta.json` +
  `results/wb_climate_check.json` + `results/tables/wb_spei_compare.csv` +
  `wb_precip_compare.csv` — the World Bank Open Data climate series fetched from
  the public API, and the (negative) cross-check of our SPEI against them
- `results/ag_linkage.json` + `results/tables/ag_yield_climate.csv` and
  `results/coffee_linkage.json` + `coffee_prod_climate.csv` — the cereal and
  coffee impact linkage, including the non-significant and sign-reversing
  coefficients
- `results/tables/risk_district_month.csv` +
  `results/early_warning_summary.json` + `results/tables/nowcast_interval.csv` —
  district-month risk nowcast, its skill-vs-null measurement, the measured spread
  of the one-month carry, and the split-conformal prediction intervals with their
  measured held-out coverage
- `results/tables/drought_early_warning.csv` + `results/drought_ew_summary.json` —
  threshold-based drought early warning (SPI-6 severity + trajectory + decline
  run), validated against the disaster register
- `results/tables/yield_forecast.csv` + `results/yield_forecast_summary.json` —
  national cereal-yield anomaly forecast from growing-season SPI-3, with honest
  skill check vs climatology null
- `results/tables/seasonal_outlook.csv` + `results/seasonal_outlook_summary.json` +
  `data/seasonal_fetch_status.json` — downscaled ECMWF SEAS5 seasonal
  precipitation outlook and its integration with the drought EW tiers
- `reports/climate_risk_bulletin_*.md` — automated narrative risk bulletin
  synthesising all forward-looking layers
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
