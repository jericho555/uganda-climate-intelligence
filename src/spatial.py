"""Spatial analysis: district-level anomaly hotspots via Moran's I (global +
local) and Getis-Ord Gi*, using a distance-band spatial weights matrix built
from district centroids (RQ2). Implemented directly so it is reproducible
without the full PySAL dependency chain.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS


def centroid_table():
    rows = [dict(location=n, region=r, lat=la, lon=lo, elev=e)
            for r, n, la, lo, e in C.LOCATIONS]
    return pd.DataFrame(rows)


def haversine(a, b):
    R = 6371.0
    la1, lo1 = np.radians(a[0]), np.radians(a[1])
    la2, lo2 = np.radians(b[0]), np.radians(b[1])
    dla, dlo = la2 - la1, lo2 - lo1
    h = np.sin(dla/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlo/2)**2
    return 2*R*np.arcsin(np.sqrt(h))


def distance_weights(coords, threshold_km=220.0):
    n = len(coords)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine(coords[i], coords[j])
                if d <= threshold_km:
                    W[i, j] = 1.0 / d  # inverse-distance within band
    return W


def _moran_I_raw(x, W):
    x = np.asarray(x, float)
    n = len(x); xbar = x.mean(); z = x - xbar
    S0 = W.sum()
    den = S0 * (z**2).sum()
    return (n * (W * np.outer(z, z)).sum() / den) if den else np.nan


def moran_global(x, W, n_perm=9999, seed=None):
    """Global Moran's I with permutation-based inference (randomisation null).
    Returns (I, z, p) where z/p come from the permutation distribution."""
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    x = np.asarray(x, float)
    I = _moran_I_raw(x, W)
    obs = np.array([_moran_I_raw(rng.permutation(x), W) for _ in range(n_perm)])
    mu, sd = obs.mean(), obs.std(ddof=1)
    z = (I - mu) / sd if sd > 0 else np.nan
    p = (1.0 + np.sum(np.abs(obs - mu) >= np.abs(I - mu))) / (n_perm + 1)
    return I, z, float(p)


def _ncdf(x):
    from math import erf, sqrt
    return 0.5*(1+erf(x/sqrt(2)))


def local_morans_I(x, W):
    x = np.asarray(x, float); n = len(x)
    z = x - x.mean()
    s2 = (z**2).sum()/n
    Wsum = W.sum(1)
    Ii = np.zeros(n)
    for i in range(n):
        lag = (W[i] * z).sum()
        Ii[i] = (n/Wsum[i]) * z[i] * lag / s2 if Wsum[i] > 0 else 0.0
    return Ii


def getis_ord_gi(x, W):
    """Return Gi* z-scores (positive = hot spot)."""
    x = np.asarray(x, float); n = len(x)
    Wd = W.copy()
    for i in range(n):
        Wd[i, i] = 1.0                     # include self (Gi*)
    sumW = Wd.sum(1)
    sumW2 = (Wd**2).sum(1)
    xbar = x.mean()
    s = np.sqrt((x**2).sum()/n - xbar**2)
    E = sumW * xbar
    V = sumW2 * ((x**2).sum()/n - xbar**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Gi = np.where(V > 0, (x * sumW - E) / np.sqrt(V), 0.0)
    return Gi


def main():
    df = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"])
    ct = centroid_table()
    # district-level anomaly rate (% of days flagged) and mean intensity
    agg = df.groupby("location").agg(
        anomaly_rate=("is_anomaly", "mean"),
        mean_score=("ens_score", "mean"),
        n_days=("is_anomaly", "size"),
    ).reset_index()
    agg = agg.merge(ct, on="location")
    agg["region"] = agg.region_x if "region_x" in agg else agg.get("region")
    # event-linked rate: fraction of anomaly days that coincide with documented events
    ev = df.groupby("location").apply(
        lambda g: (g.is_anomaly * g.label_event).sum() / max(g.is_anomaly.sum(), 1)
    ).rename("precision_vs_events").reset_index()
    agg = agg.merge(ev, on="location")

    coords = list(zip(agg.lat.values, agg.lon.values))
    W = distance_weights(coords)
    # row-standardise for Moran
    Ws = W / np.where(W.sum(1, keepdims=True) > 0, W.sum(1, keepdims=True), 1)

    I, zI, p = moran_global(agg.anomaly_rate.values, Ws)
    Mi = local_morans_I(agg.anomaly_rate.values, Ws)
    Gi = getis_ord_gi(agg.anomaly_rate.values, W)
    agg["local_morans_I"] = Mi
    agg["Gi_star_z"] = Gi
    agg["cluster"] = np.where(Gi > 1.96, "hotspot", np.where(Gi < -1.96, "coldspot", "ns"))

    agg.to_csv(RESULTS / "tables/district_anomaly.csv", index=False)
    (RESULTS / "spatial_stats.json").write_text(json.dumps({
        "moran_global_I": float(I), "moran_z": float(zI), "moran_p": float(p),
        "n_districts": int(len(agg)),
        "hotspots": agg.loc[agg.cluster == "hotspot", "location"].tolist(),
        "coldspots": agg.loc[agg.cluster == "coldspot", "location"].tolist(),
        "top5_anomaly_rate": agg.sort_values("anomaly_rate", ascending=False)
            [["location", "region", "anomaly_rate"]].head(5).to_dict("records"),
    }, indent=2))

    print(f"Global Moran's I={I:.3f} (z={zI:.2f}, p={p:.4g})")
    print("Hotspots:", agg.loc[agg.cluster == 'hotspot', 'location'].tolist())
    print(agg.sort_values("anomaly_rate", ascending=False)
          [["location", "region", "anomaly_rate", "Gi_star_z", "cluster"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
