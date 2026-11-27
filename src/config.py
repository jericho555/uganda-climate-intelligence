"""Central configuration for the Uganda Climate Intelligence framework.

All paths are relative to this package root so the pipeline is reproducible
from any machine. Data are pulled from openly-licensed, citable sources:
  * ERA5 reanalysis via the Open-Meteo Archive API (CC-BY 4.0)
  * In-situ weather stations via Meteostat (CC-BY 4.0)
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # .../uganda-climate-intelligence
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MANUSCRIPT = ROOT / "manuscript"
for _d in (DATA, RESULTS, FIGURES, MANUSCRIPT, DATA / "raw", RESULTS / "tables"):
    _d.mkdir(parents=True, exist_ok=True)

# Every figure the manuscript includes is also kept in figures/ so the release
# carries the artwork its own generator produced. One resolution everywhere:
# Elsevier asks for >=300 dpi for halftone and combination artwork.
FIG_DPI = 300
FIG_FOLDERS = (FIGURES, MANUSCRIPT / "figures")


def save_figure(fig, name):
    """Write ``fig`` as ``<name>.png`` into every folder that must carry it.

    Both copies come from the same Figure object at the same dpi, so the working
    directory and the manuscript bundle cannot drift apart -- a stale copy in
    manuscript/figures/ would otherwise be submitted without anyone noticing.
    """
    import matplotlib.pyplot as plt
    written = []
    for folder in FIG_FOLDERS:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written

# ----------------------------------------------------------------------------
# Study period and spatial units
# ----------------------------------------------------------------------------
START_DATE = "1950-01-01"
END_DATE = "2025-12-31"
TIMEZONE = "Africa/Kampala"

# District/urban centroids spanning Uganda's four administrative regions,
# with approximate elevation (m a.s.l.). Coordinates are WGS84 (decimal deg).
# region, name, lat, lon, elevation_m
LOCATIONS = [
    # Central
    ("Central", "Kampala",   0.3476, 32.5825, 1190),
    ("Central", "Wakiso",    0.4031, 32.4714, 1180),
    ("Central", "Mukono",    0.3506, 32.7508, 1217),
    ("Central", "Luweero",   0.9500, 32.4830, 1160),
    ("Central", "Mityana",   0.4158, 32.0283, 1380),
    # Eastern
    ("Eastern", "Mbale",     1.0850, 34.1750, 1110),
    ("Eastern", "Jinja",     0.4429, 33.1974, 1180),
    ("Eastern", "Soroti",    1.7147, 33.6111, 1120),
    ("Eastern", "Tororo",    0.6910, 34.1810, 1120),
    ("Eastern", "Kumi",      1.4550, 33.9060, 1060),
    # Northern
    ("Northern", "Gulu",     2.7747, 32.2990, 1120),
    ("Northern", "Lira",     2.2472, 32.9058, 1110),
    ("Northern", "Arua",     3.0257, 30.9140, 1190),
    ("Northern", "Masindi",  1.7083, 31.7340, 1130),
    # Western
    ("Western", "Mbarara",  -0.6073, 30.6545, 1354),
    ("Western", "FortPortal", 0.6667, 30.2833, 1480),
    ("Western", "Kasese",   0.1830, 30.0930, 1000),
    ("Western", "Hoima",    1.4349, 31.3540, 1160),
    ("Western", "Kabale",   -1.2489, 29.9930, 1960),
]

# Meteostat IgRA/others: representative in-situ stations for ERA5 cross-validation.
# (meteostat station id, name, region)
STATIONS = [
    ("63630", "Entebbe Airport", "Central"),
    ("63619", "Jinja", "Eastern"),
    ("63641", "Gulu", "Northern"),
    ("63633", "Mbarara", "Western"),
]

# ----------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------
# Open-Meteo ERA5 Archive daily variables (all confirmed available on v1/era5).
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "precipitation_hours",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "pressure_msl_mean",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

# Anomaly-detection feature set (directional/derived handled in preprocess).
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "pressure_msl_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

# Event-oriented persistence features (physically motivated antecedent indices):
# name -> (base variable, window_days, sign). sign=-1 turns a wet accumulated
# anomaly into a dryness index (drought precursor); +1 captures persistent
# heavy-rain / heat spells (flood & landslide precursors).
DERIVED = {
    "precip_z_5d":  ("precipitation_sum", 5, 1),
    "precip_z_10d": ("precipitation_sum", 10, 1),
    "dry_z_60d":    ("precipitation_sum", 60, -1),
    "tmax_z_5d":    ("temperature_2m_max", 5, 1),
    "et_z_30d":     ("et0_fao_evapotranspiration", 30, -1),
}

# ----------------------------------------------------------------------------
# Method hyper-parameters
# ----------------------------------------------------------------------------
SEED = 42
CONTAMINATION = 0.05            # expected anomaly rate used by detectors
TRAIN_TEST_YEARS = (2008, 2018, 2024)  # legacy: superseded by the EVAL_* window
# Supervised disaster-evaluation window = curated register coverage only.
# Years outside it (1950-2007, 2024-2025) feed climatology / trends / spatial
# only and are excluded from the leak-free detection benchmark.
EVAL_TRAIN_YEARS = (2008, 2018)  # unsupervised detector fit + threshold
EVAL_TEST_YEARS = (2019, 2023)   # held-out, register-covered evaluation
N_BOOTSTRAP = 800
CLIMATOLOGY_WINDOW = 15         # running-window climatology for seasonal anomalies
PERSISTENCE_WINDOW = 5          # days over which the ensemble score is smoothed
DETECT_FEATURES = [f"{f}_z" for f in FEATURES] + list(DERIVED)
