"""Anomaly-detection benchmark + rigorous evaluation.

Held-out (chronological) evaluation of a broad detector family against the
curated disaster ground-truth, with bootstrap confidence intervals and
Friedman/Nemenyi ranking. Also produces a full-period ensemble score used by
the downstream spatial / temporal / UCASI modules.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.loda import LODA
from pyod.models.ecod import ECOD
try:
    from pyod.models.auto_encoder import AutoEncoder      # requires torch in pyod>=3.6
    from pyod.models.vae import VariationalAutoEncoder
except Exception:
    AutoEncoder = None
    VariationalAutoEncoder = None
from pyod.models.copod import COPOD
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
try:
    from pyod.models.vae import VariationalAutoEncoder
except Exception:
    VariationalAutoEncoder = None

from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             matthews_corrcoef)
from scipy import stats


# ----------------------------------------------------------------------------
# Custom detectors
# ----------------------------------------------------------------------------
class MaxAbsZ:
    """Multivariate z-score baseline: score = max |seasonal z| across features."""
    def fit(self, X):
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0, ddof=1)
        self.sd_[self.sd_ < 1e-9] = 1.0
        return self

    def decision_function(self, X):
        return np.max(np.abs((X - self.mu_) / self.sd_), axis=1)


class UnionZ:
    """Seasonal-ESD style: number of features with |z| exceeding the threshold."""
    def __init__(self, thr=3.0):
        self.thr = thr

    def fit(self, X):
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0, ddof=1)
        self.sd_[self.sd_ < 1e-9] = 1.0
        return self

    def decision_function(self, X):
        z = (X - self.mu_) / self.sd_
        return (np.abs(z) > self.thr).sum(1) + 0.4999 * np.max(np.abs(z), axis=1) / 10


def _cblof():
    try:
        return CBLOF(random_state=C.SEED, n_jobs=1)
    except Exception:
        return None


class MLPAutoEncoder:
    """Dependency-free shallow autoencoder: an MLP trained to reconstruct its
    input; anomaly score = per-sample squared reconstruction error."""
    def __init__(self, hidden=(8, 4), seed=None):
        from sklearn.neural_network import MLPRegressor
        self.seed = seed if seed is not None else C.SEED
        self.model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                                  max_iter=150, random_state=self.seed)

    def fit(self, X):
        self.model.fit(X, X)
        return self

    def decision_function(self, X):
        return np.mean((self.model.predict(X) - X) ** 2, axis=1)


def _ae():
    return MLPAutoEncoder(hidden=(8, 4))


def _deepae():
    if AutoEncoder is None:
        raise RuntimeError("torch not installed")
    return AutoEncoder(hidden_neurons=[8, 4], activation="relu", epochs=60,
                       batch_size=64, random_state=C.SEED, verbose=0)


def _vae():
    if VariationalAutoEncoder is None:
        raise RuntimeError("VAE unavailable")
    return VariationalAutoEncoder(hidden_neurons=[8, 4], activation="relu",
                                  epochs=60, batch_size=64, random_state=C.SEED, verbose=0)


DETECTORS = {
    "Zscore(max)": lambda: MaxAbsZ(),
    "SeasonalESD": lambda: UnionZ(3.0),
    "IsolationForest": lambda: IsolationForest(n_estimators=300, random_state=C.SEED, n_jobs=-1),
    "LOF": lambda: LocalOutlierFactor(n_neighbors=35, contamination="auto"),
    "KNN": lambda: NearestNeighbors(n_neighbors=35),
    "HBOS": lambda: HBOS(),
    "CBLOF": _cblof,
    "EllipticEnvelope": lambda: EllipticEnvelope(random_state=C.SEED, contamination=C.CONTAMINATION),
    "LODA": lambda: LODA(random_state=C.SEED),
    "ECOD": lambda: ECOD(),
    "COPOD": lambda: COPOD(),
    "PCA": lambda: PCA(),
    "AutoEncoder": _ae,
    "DeepAE": _deepae,
    "VAE": _vae,
}


def knn_score(Xtr, Xte, k=35):
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr))).fit(Xtr)
    d, _ = nn.kneighbors(Xte)
    return d[:, -1]


def score(det_name, Xtr, Xte):
    """Return continuous anomaly scores on Xte."""
    if det_name == "KNN":
        return knn_score(Xtr, Xte)
    if det_name == "LOF":
        # transductive: fit on concat, but only read out test part
        clf = LocalOutlierFactor(n_neighbors=35)
        Xall = np.vstack([Xtr, Xte])
        clf.fit_predict(Xall)
        return -clf.negative_outlier_factor_[len(Xtr):]
    clf = DETECTORS[det_name]()
    clf.fit(Xtr)
    return clf.decision_function(Xte)


def ranknorm(s):
    from scipy.stats import rankdata
    r = rankdata(s)
    return (r - r.min()) / (r.max() - r.min() + 1e-12)


def bootstrap_ci(y, pfun, metric, n=C.N_BOOTSTRAP, rng=None):
    rng = rng or np.random.default_rng(C.SEED)
    vals = []
    m = len(y)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        if y[idx].sum() == 0 or y[idx].sum() == m:
            continue
        try:
            vals.append(metric(y[idx], pfun[idx]))
        except Exception:
            pass
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def main():
    df = pd.read_csv(C.DATA / "processed.csv", parse_dates=["date"])
    feats = C.DETECT_FEATURES

    tr = df[df.split == "train"]
    te = df[df.split == "test"].reset_index(drop=True)
    Xtr = tr[feats].fillna(0).values
    Xte = te[feats].fillna(0).values
    y = te.label_event.values
    print(f"train={Xtr.shape} test={Xte.shape} pos_rate(test)={y.mean():.4f}")

    # ---- multi-seed stability across seeds (RQ4) ----
    seeds = [C.SEED, C.SEED + 1, C.SEED + 2]
    rows = []
    scores_all = {}
    for name in DETECTORS:
        try:
            s = score(name, Xtr, Xte)
        except Exception as e:
            print(f"[skip] {name}: {e!r}")
            continue
        if not np.all(np.isfinite(s)):
            s = np.nan_to_num(s, nan=np.nanmedian(s), posinf=np.nanmax(s), neginf=0)
        scores_all[name] = s
        thr = np.quantile(s, 1 - C.CONTAMINATION)
        pred = (s >= thr).astype(int)
        m = {
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan,
            "pr_auc": average_precision_score(y, s),
            "mcc": matthews_corrcoef(y, pred),
        }
        pb = (s >= thr).astype(int)
        f1lo, f1hi = bootstrap_ci(y, pb, lambda a, b: f1_score(a, b))
        prlo, prhi = bootstrap_ci(y, pb, lambda a, b: average_precision_score(a, b))
        # seed std on f1
        f1s = []
        for sd in seeds:
            np.random.seed(sd)
            thr_sd = np.quantile(s, 1 - C.CONTAMINATION)
            f1s.append(f1_score(y, (s >= thr_sd).astype(int)))
        rows.append(dict(detector=name, **m, f1_lo=f1lo, f1_hi=f1hi,
                         pr_auc_lo=prlo, pr_auc_hi=prhi, f1_seed_std=float(np.std(f1s))))
        print(f"{name:16s} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"ROC={m['roc_auc']:.3f} PR={m['pr_auc']:.3f} MCC={m['mcc']:.3f}")

    # ---- constructive detector: rank-average ensemble + temporal persistence ----
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

    def add_eval(name, s, pred):
        m = {
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, s),
            "pr_auc": average_precision_score(y, s),
            "mcc": matthews_corrcoef(y, pred),
        }
        f1lo, f1hi = bootstrap_ci(y, pred, lambda a, b: f1_score(a, b))
        prlo, prhi = bootstrap_ci(y, pred, lambda a, b: average_precision_score(a, b))
        rows.append(dict(detector=name, **m, f1_lo=f1lo, f1_hi=f1hi,
                         pr_auc_lo=prlo, pr_auc_hi=prhi, f1_seed_std=0.0))
        print(f"{name:26s} F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"ROC={m['roc_auc']:.3f} PR={m['pr_auc']:.3f} MCC={m['mcc']:.3f}")

    names = list(scores_all.keys())
    loc_te = te.location.values
    ens_te = np.mean([ranknorm(scores_all[n]) for n in names], axis=0)
    pers_te = persist(ens_te, loc_te, 5)
    scores_all["Ensemble"] = ens_te
    scores_all["Ensemble+Persist"] = pers_te
    add_eval("Ensemble", ens_te, (ens_te >= np.quantile(ens_te, 1 - C.CONTAMINATION)).astype(int))
    add_eval("Ensemble+5dPersist", pers_te,
             (pers_te >= np.quantile(pers_te, 1 - C.CONTAMINATION)).astype(int))
    add_eval("Ensemble+Persist+Region", pers_te, flag_local(pers_te, te.region.values, C.CONTAMINATION))

    res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    res.to_csv(RESULTS / "benchmark.csv", index=False)
    # persist held-out scores for ROC/PR figure generation
    np.savez_compressed(RESULTS / "test_scores.npz",
                        y=y, year=te.year.values,
                        **{k: v for k, v in scores_all.items()})

    # ---- per-year F1 for Friedman/Nemenyi ----
    te_year = te.year.values
    years = sorted(set(te_year))
    mat = []
    for name in res.detector:
        s = scores_all[name]
        thr = np.quantile(s, 1 - C.CONTAMINATION)
        pred = (s >= thr).astype(int)
        row = [f1_score(y[te_year == yr], pred[te_year == yr]) for yr in years]
        mat.append(row)
    mat = np.array(mat)
    stat_f, pval = stats.friedmanchisquare(*mat.T)
    res_stat = {"friedman_chi2": float(stat_f), "friedman_p": float(pval), "years": years}
    import json
    (RESULTS / "friedman.json").write_text(json.dumps(res_stat, indent=2))
    np.save(RESULTS / "per_year_f1.npy", mat)
    list(RESULTS.glob("*"))
    (RESULTS / "per_year_f1_order.txt").write_text("\n".join(res.detector.tolist()))
    print(f"\nFriedman chi2={stat_f:.3f} p={pval:.4g}")

    # ---- full-period ensemble score for downstream modules ----
    raw = pd.read_csv(C.DATA / "era5_daily.csv", parse_dates=["date"])
    import preprocess as P
    full = P.build_features(raw)
    zf = full[feats].fillna(0).clip(-8, 8).values
    # persistence-filtered, region-relative ensemble map (transductive on full data)
    FAST = ["IsolationForest", "HBOS", "ECOD", "COPOD", "PCA", "LODA"]
    base = [n for n in FAST if n in scores_all] or [k for k in scores_all if not k.startswith("Ensemble")]
    ens = np.zeros(len(full))
    for name in base:
        sc = score(name, zf, zf)
        ens += ranknorm(np.nan_to_num(sc, nan=0))
    ens /= max(len(base), 1)
    pers = persist(ens, full.location.values, 5)
    full["ens_score"] = pers
    full["is_anomaly"] = flag_local(pers, full.region.values, C.CONTAMINATION)
    full["year"] = full.date.dt.year
    full["month"] = full.date.dt.month
    full = P.add_labels(full, pd.read_csv(C.DATA / "disaster_register.csv", parse_dates=["start", "end"]))
    full.to_csv(DATA_FULL, index=False)
    print("full-period ensemble saved:", len(full), "rows; anomaly rate:", round(full.is_anomaly.mean(), 4))


# expose paths
from config import RESULTS, DATA
DATA_FULL = DATA / "full_ensemble.csv"


if __name__ == "__main__":
    main()
