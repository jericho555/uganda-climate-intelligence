"""Lightweight, deterministic explainability (surrogate RF -> SHAP + permutation
importance) sized to complete reliably on a constrained machine. Same outputs as
explain.py. Uses a single held-out split instead of repeated CV refits.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import shap

ZCOLS = [f"{f}_z" for f in C.FEATURES]


def main():
    df = pd.read_csv(DATA / "full_ensemble.csv", usecols=ZCOLS + ["is_anomaly", "ens_score", "location"])
    X = df[ZCOLS].fillna(0).values
    y = df.is_anomaly.values
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(df), size=min(24000, len(df)), replace=False)
    X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=C.SEED, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                 random_state=C.SEED, n_jobs=2, class_weight="balanced")
    clf.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    print("surrogate held-out AUC =", round(auc, 3))

    pi = permutation_importance(clf, Xte, yte, n_repeats=5, random_state=C.SEED,
                                scoring="average_precision", n_jobs=2)
    perm = pd.DataFrame({"variable": C.FEATURES, "perm_importance": pi.importances_mean,
                         "perm_std": pi.importances_std}).sort_values("perm_importance", ascending=False)

    samp = Xte[:1200]
    sv = shap.TreeExplainer(clf).shap_values(samp)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                 # (samples, features, classes) -> positive class
        sv = sv[:, :, 1]
    shap_imp = pd.DataFrame({"variable": C.FEATURES, "shap_mean_abs": np.abs(sv).mean(0)}
                            ).sort_values("shap_mean_abs", ascending=False)

    shap_imp.to_csv(RESULTS / "tables/shap_importance.csv", index=False)
    perm.to_csv(RESULTS / "tables/permutation_importance.csv", index=False)
    np.save(RESULTS / "shap_values.npy", sv)

    raw = pd.read_csv(DATA / "full_ensemble.csv", usecols=ZCOLS + ["ens_score"])
    corr = {f: float(np.corrcoef(raw[f + "_z"].fillna(0), raw.ens_score)[0, 1]) for f in C.FEATURES}

    (RESULTS / "explain_summary.json").write_text(json.dumps({
        "surrogate_holdout_auc": auc,
        "n_train": int(len(Xtr)),
        "top_shap": shap_imp.head(5).to_dict("records"),
        "top_perm": perm.head(5).to_dict("records"),
        "corr_with_score": {k: round(v, 3) for k, v in corr.items()},
    }, indent=2))
    print("SHAP ranking:\n", shap_imp.to_string(index=False))
    print("corr with score:", {k: round(v, 3) for k, v in corr.items()})


if __name__ == "__main__":
    main()
