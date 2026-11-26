"""Recovery stage: recompute Friedman/per-year F1 from saved test scores and
build the full-period persistence ensemble map used by downstream modules.
Reuses results/test_scores.npz so the (slow) detector benchmark is NOT re-run.
"""
import sys
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import RESULTS, DATA
import preprocess as P
from detect import score, ranknorm

DATA_FULL = DATA / "full_ensemble.csv"


def persist(scores, loc, window=5):
    out = np.empty(len(scores))
    for L in np.unique(loc):
        m = loc == L
        out[np.where(m)[0]] = pd.Series(scores[m]).rolling(
            window, center=True, min_periods=1).mean().values
    return out


def flag_local(score, groups, cont):
    pred = np.zeros(len(score), int)
    for g in np.unique(groups):
        m = groups == g
        pred[m] = (score[m] >= np.quantile(score[m], 1 - cont)).astype(int)
    return pred


def friedman():
    z = np.load(RESULTS / "test_scores.npz")
    y = z["y"]
    year = z["year"]
    bench = pd.read_csv(RESULTS / "benchmark.csv")
    order = bench.detector.tolist()
    years = sorted(set(year.tolist()))
    mat, names = [], []
    for name in order:
        pk = "pred__" + name
        if pk in z.files:
            pred = np.asarray(z[pk])
        elif name in z.files:
            s = z[name]
            thr = np.quantile(s, 1 - C.CONTAMINATION)
            pred = (s >= thr).astype(int)
        else:
            continue
        row = []
        for yr in years:
            m = year == yr
            if y[m].sum() == 0 or y[m].sum() == m.sum():
                row.append(np.nan)
            else:
                row.append(f1_score(y[m], pred[m]))
        mat.append(row)
        names.append(name)
    mat = np.array(mat, dtype=float)
    cols = [mat[:, j] for j in range(mat.shape[1])]
    cols = [c[~np.isnan(c)] for c in cols]
    k = min(len(c) for c in cols)
    cols = [c[:k] for c in cols]
    stat_f, pval = stats.friedmanchisquare(*cols)
    (RESULTS / "friedman.json").write_text(json.dumps(
        {"friedman_chi2": float(stat_f), "friedman_p": float(pval),
         "years": [int(v) for v in years], "n_detectors": len(names)}, indent=2))
    np.save(RESULTS / "per_year_f1.npy", mat)
    (RESULTS / "per_year_f1_order.txt").write_text("\n".join(names))
    print(f"Friedman chi2={stat_f:.3f} p={pval:.4g} over {len(years)} years, {len(names)} detectors")


def fullmap():
    df = pd.read_csv(C.DATA / "processed.csv", parse_dates=["date"])
    feats = C.DETECT_FEATURES
    te_mask = df.split == "test"
    zf = df[feats].fillna(0).clip(-8, 8).values
    FAST = ["IsolationForest", "HBOS", "ECOD", "COPOD", "PCA", "LODA"]
    ens = np.zeros(len(df))
    for name in FAST:
        sc = score(name, zf, zf)
        ens += ranknorm(np.nan_to_num(sc, nan=0))
        print("  scored", name)
    ens /= len(FAST)
    pers = persist(ens, df.location.values, 5)
    df["ens_score"] = pers
    df["is_anomaly"] = flag_local(pers, df.region.values, C.CONTAMINATION)
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    df.to_csv(DATA_FULL, index=False)
    print("full-period ensemble saved:", len(df), "rows; anomaly rate:", round(df.is_anomaly.mean(), 4))


if __name__ == "__main__":
    friedman()
    fullmap()
