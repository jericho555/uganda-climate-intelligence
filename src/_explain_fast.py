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


def _positive_class(expl, sv):
    """Reduce a binary TreeExplainer output to (positive-class matrix, base value).

    SHAP has changed this return type between releases -- a list per class, or one
    array with a class axis -- so all three shapes are handled rather than
    pinned, and the base value is taken for the positive class in each case.
    """
    base = [float(v) for v in np.ravel(np.asarray(expl.expected_value, dtype=float))]
    if isinstance(sv, list):
        return np.asarray(sv[-1]), (base[-1] if len(base) > 1 else base[0])
    arr = np.asarray(sv)
    if arr.ndim == 3:                       # (samples, features, classes)
        return arr[:, :, -1], (base[-1] if len(base) > 1 else base[0])
    return arr, base[0]


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
    expl = shap.TreeExplainer(clf)
    sv, exp_val = _positive_class(expl, expl.shap_values(samp))

    shap_imp = pd.DataFrame({"variable": C.FEATURES, "shap_mean_abs": np.abs(sv).mean(0)}
                            ).sort_values("shap_mean_abs", ascending=False)

    shap_imp.to_csv(RESULTS / "tables/shap_importance.csv", index=False)
    perm.to_csv(RESULTS / "tables/permutation_importance.csv", index=False)
    np.save(RESULTS / "shap_values.npy", sv)
    # The local-attribution panel of fig_shap draws the matrix above, so it also
    # needs the rows the matrix was computed on and the base value it explains.
    # Both are persisted rather than re-derived, so the figure can never show a
    # sample that differs from the one that was attributed.
    pd.DataFrame(samp, columns=C.FEATURES).to_csv(
        RESULTS / "tables/shap_sample.csv", index=False)
    # The surrogate's own output for exactly those rows, so that "base value +
    # attributions = the prediction being explained" is a checked statement
    # rather than an assumption, in the figure caption and in the dashboard.
    proba = clf.predict_proba(samp)[:, 1]
    np.save(RESULTS / "shap_row_outputs.npy", proba)
    additivity = float(np.max(np.abs(exp_val + sv.sum(axis=1) - proba)))
    (RESULTS / "shap_expected_value.json").write_text(json.dumps({
        "expected_value": exp_val,
        "n_rows": int(len(samp)),
        "rows": "held-out (test) district-days, first 1200 in split order",
        "columns": "the 9 base z-scored features, in config.FEATURES order",
        "model": "RandomForest surrogate of the ensemble flag, seed " + str(C.SEED),
        "surrogate_holdout_auc": auc,
        "output_scale": "surrogate probability of the anomaly class",
        "additivity_max_abs_error": additivity,
    }, indent=2))

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
    print("additivity check: max |base + sum(shap) - predict_proba| =",
          f"{additivity:.2e}")
    print("corr with score:", {k: round(v, 3) for k, v in corr.items()})


if __name__ == "__main__":
    main()
