"""Explainability (XAI) + feature-contribution and uncertainty analysis (RQ3, RQ4).

A supervised surrogate (RandomForest) is trained on the validated ensemble
labels, then SHAP values explain which weather variables drive anomalies.
Permutation importance and a correlation cross-check are reported alongside.
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
from sklearn.model_selection import cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import shap

ZCOLS = [f"{f}_z" for f in C.FEATURES]


def main():
    df = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"])
    X = df[ZCOLS].fillna(0).values
    y = df.is_anomaly.values
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(df), size=min(60000, len(df)), replace=False)
    Xs, ys = X[idx], y[idx]

    clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                 random_state=C.SEED, n_jobs=-1, class_weight="balanced")
    clf.fit(Xs, ys)
    proba = cross_val_predict(clf, Xs, ys, cv=4, method="predict_proba")[:, 1]
    surrogate_auc = float(roc_auc_score(ys, proba))

    # permutation importance on the surrogate
    pi = permutation_importance(clf, Xs, ys, n_repeats=8, random_state=C.SEED,
                                scoring="average_precision", n_jobs=-1)
    perm = pd.DataFrame({"variable": C.FEATURES, "perm_importance": pi.importances_mean,
                         "perm_std": pi.importances_std}).sort_values("perm_importance", ascending=False)

    # SHAP (TreeExplainer) on a sample
    samp = Xs[rng.choice(len(Xs), size=min(3000, len(Xs)), replace=False)]
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(samp)
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv).mean(0)
    shap_imp = pd.DataFrame({"variable": C.FEATURES, "shap_mean_abs": mean_abs}
                            ).sort_values("shap_mean_abs", ascending=False)
    shap_imp.to_csv(RESULTS / "tables/shap_importance.csv", index=False)
    perm.to_csv(RESULTS / "tables/permutation_importance.csv", index=False)
    np.save(RESULTS / "shap_values.npy", sv)

    # correlation of each raw seasonal z with ensemble score (model-free cross-check)
    corr = {f: float(np.corrcoef(df[f + "_z"].fillna(0), df.ens_score)[0, 1]) for f in C.FEATURES}

    (RESULTS / "explain_summary.json").write_text(json.dumps({
        "surrogate_cv_auc": surrogate_auc,
        "top_shap": shap_imp.head(5).to_dict("records"),
        "top_perm": perm.head(5).to_dict("records"),
        "corr_with_score": {k: round(v, 3) for k, v in corr.items()},
    }, indent=2))
    print(f"surrogate CV AUC={surrogate_auc:.3f}")
    print("SHAP ranking:\n", shap_imp.to_string(index=False))
    print("corr with ensemble score:", {k: round(v, 3) for k, v in corr.items()})


if __name__ == "__main__":
    main()
