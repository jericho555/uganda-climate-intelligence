"""Preprocessing: seasonal climatology anomalies + ground-truth labelling.

The baseline study fed RAW variables to detectors and reported precision=1.0 /
recall=1.0 with no labels. Here we (a) remove the strong annual cycle by
computing per-location day-of-year standardized anomalies (z of value minus a
smoothed climatology), and (b) attach supervised ground-truth labels from the
curated disaster register so detection quality can actually be measured.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

NDAYS = 366


def _circular_smooth(arr, width):
    if width <= 1:
        return arr
    k = np.ones(width) / width
    pad = width // 2
    ext = np.concatenate([arr[-pad:], arr, arr[:pad]])
    sm = np.convolve(ext, k, mode="valid")
    return sm[: len(arr)]


def _fill_invalid(vals):
    """Linear-fill NaNs in a doy climatology array (circular)."""
    s = pd.Series(vals)
    s = s.interpolate(limit_direction="both")
    out = s.bfill().ffill().values
    return out


def seasonal_z(series, doy, width):
    """Standardized anomaly vs a smoothed per-doy climatology."""
    mu = np.full(NDAYS, np.nan)
    sd = np.full(NDAYS, np.nan)
    for d in range(NDAYS):
        v = series.values[doy == d]
        if len(v) >= 3:
            mu[d] = np.nanmean(v)
            sd[d] = np.nanstd(v, ddof=1)
    mu = _fill_invalid(mu)
    sd = _fill_invalid(sd)
    mu = _circular_smooth(mu, width)
    sd = _circular_smooth(sd, width)
    sd = np.where(sd <= 1e-6, 1.0, sd)
    return (series.values - mu[doy]) / sd[doy]


def add_labels(df, reg):
    """label=1 if the (date, region) falls inside a documented event window."""
    lab = np.zeros(len(df), dtype=int)
    ev_type = np.array([""] * len(df), dtype=object)
    for _, e in reg.iterrows():
        regs = set(e.regions.split(";"))
        mask = (df.date >= e.start) & (df.date <= e.end) & df.region.isin(regs)
        lab[mask.values] = 1
        # keep a single (first-seen) event type per row
        types = ev_type[mask.values]
        new = np.where((types == "") , e.type, types)
        ev_type[mask.values] = new
    df["label_event"] = lab
    df["event_type"] = ev_type
    return df


def build_features(raw):
    """Compute seasonal-standardised anomalies for FEATURES plus the event-
    oriented persistence indices in config.DERIVED. Returns a long frame."""
    width = C.CLIMATOLOGY_WINDOW * 2 + 1
    zcols = [f"{f}_z" for f in C.FEATURES]
    parts = []
    for loc, g in raw.groupby("location", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()
        doy = g.date.dt.dayofyear.values - 1
        for f in C.FEATURES:
            g[f"{f}_z"] = seasonal_z(g[f].astype(float), doy, width)
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    df[zcols] = df[zcols].replace([np.inf, -np.inf], np.nan).clip(-8, 8)
    # derived persistence indices (rolling mean of the seasonal z, signed)
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    for name, (base, w, sign) in C.DERIVED.items():
        s = sign * df.groupby("location")[f"{base}_z"].transform(
            lambda x: x.rolling(w, min_periods=max(2, w // 3)).mean())
        # re-standardise so all inputs share a comparable scale
        s = (s - s.mean()) / (s.std(ddof=1) + 1e-9)
        df[name] = s.replace([np.inf, -np.inf], np.nan).clip(-8, 8)
    return df


def main():
    raw = pd.read_csv(C.DATA / "era5_daily.csv", parse_dates=["date"])
    reg = pd.read_csv(C.DATA / "disaster_register.csv", parse_dates=["start", "end"])
    print(f"raw rows={len(raw)}  locations={raw.location.nunique()}  events={len(reg)}")

    df = build_features(raw)
    allcols = C.DETECT_FEATURES
    df[allcols] = df[allcols].fillna(0.0)
    df = add_labels(df, reg)

    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    tr_lo, tr_hi = C.EVAL_TRAIN_YEARS
    te_lo, te_hi = C.EVAL_TEST_YEARS
    df["split"] = np.where(
        (df.year >= tr_lo) & (df.year <= tr_hi), "train",
        np.where((df.year >= te_lo) & (df.year <= te_hi), "test", "unlabeled"))

    df.to_csv(C.DATA / "processed.csv", index=False)
    print(f"processed rows={len(df)}  pos_label_rate={df.label_event.mean():.4f}")
    print(f"train rows={ (df.split=='train').sum() }  test rows={ (df.split=='test').sum() }  "
          f"unlabeled rows={ (df.split=='unlabeled').sum() }")
    print("test-set positive rate:", df.loc[df.split == 'test', 'label_event'].mean().round(4))
    print("detect features:", len(allcols), "->", allcols)


if __name__ == "__main__":
    main()
