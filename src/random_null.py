"""Size-matched random-flag null for the held-out detection benchmark.

The manuscript quotes a random-flag baseline next to every detector score. This
script derives it from the persisted held-out labels instead of assuming it, so
the reference line in Table 2 / Figure (null comparison) is reproducible.

Inputs : results/test_scores.npz  (y = held-out labels, year = test year)
Outputs: results/random_null.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import RESULTS

GAMMA = C.CONTAMINATION          # 0.05, the detector alert budget
B = 20000                        # Monte-Carlo draws


def null_draws(y, q, rng, draws):
    """F1 achieved by `draws` random flaggers with alert budget q."""
    m = len(y)
    k = max(1, int(round(q * m)))
    out = np.empty(draws)
    for i in range(draws):
        flag = np.zeros(m, dtype=int)
        flag[rng.permutation(m)[:k]] = 1
        out[i] = f1_score(y, flag)
    return out


def null_for(y, q, rng, draws=B):
    """F1 of a random flagger that marks fraction q of rows, and its PR-AUC."""
    m = len(y)
    f1s = null_draws(y, q, rng, draws)
    prs = np.empty(200)
    for i in range(200):
        score = rng.random(m)
        prs[i] = average_precision_score(y, score)
    p = y.mean()
    analytic = 2 * p * q / (p + q)
    return dict(k=int(round(q * m)), mean=float(f1s.mean()), sd=float(f1s.std()),
                lo=float(np.percentile(f1s, 2.5)),
                hi=float(np.percentile(f1s, 97.5)),
                analytic=analytic, pr_auc_mean=float(prs.mean()),
                pr_auc_lo=float(np.percentile(prs, 2.5)),
                pr_auc_hi=float(np.percentile(prs, 97.5)), _draws=f1s)


def pooled_frames():
    import pandas as pd
    return pd.read_csv(RESULTS / "benchmark.csv")


def main():
    z = np.load(RESULTS / "test_scores.npz", allow_pickle=False)
    y = np.asarray(z["y"]).astype(int)
    year = np.asarray(z["year"]).astype(int) if "year" in z.files else None
    rng = np.random.default_rng(C.SEED)

    pooled = null_for(y, GAMMA, rng)
    draws = pooled.pop("_draws")
    out = {"n_test": int(len(y)), "base_rate": float(y.mean()),
           "gamma": GAMMA, "draws": B, "pooled": pooled}

    # Per-detector Monte-Carlo test against the same-budget null distribution,
    # with Benjamini-Hochberg control across the benchmark.
    per_det = {}
    bm = pooled_frames()
    names = bm.detector.tolist()
    pv = []
    for _, r in bm.iterrows():
        ge = int((draws >= r.f1).sum())
        pv.append((1 + ge) / (len(draws) + 1))
        per_det[r.detector] = {
            "f1": float(r.f1),
            "ratio_vs_null": float(r.f1 / pooled["mean"]),
            "mc_p_one_sided": float(pv[-1]),
        }
    m = len(pv)
    order = sorted(range(m), key=lambda i: pv[i])
    q = [0.0] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):   # largest p first
        k = m - rank + 1
        prev = min(prev, pv[i] * m / k)
        q[i] = prev
    for n, qi in zip(names, q):
        per_det[n]["bh_q"] = float(qi)
    out["detectors"] = per_det
    if year is not None:
        per = {}
        macro = []
        for yr in sorted(set(year.tolist())):
            yy = y[year == yr]
            r = null_for(yy, GAMMA, rng, draws=4000)
            r.pop("_draws")
            per[str(int(yr))] = {"base_rate": float(yy.mean()),
                                 "n": int(len(yy)), "f1_mean": r["mean"],
                                 "f1_lo": r["lo"], "f1_hi": r["hi"],
                                 "analytic": r["analytic"]}
            macro.append(r["mean"])
        out["per_year"] = per
        out["per_year_macro_f1_mean"] = float(np.mean(macro))

    (RESULTS / "random_null.json").write_text(json.dumps(out, indent=2))
    signif = sorted((n for n, d in per_det.items() if d["bh_q"] < 0.05),
                    key=lambda n: -per_det[n]["ratio_vs_null"])
    print("BH-significant above null:",
          [(n, round(per_det[n]["ratio_vs_null"], 3),
            round(per_det[n]["bh_q"], 5)) for n in signif])
    print(json.dumps({"base_rate": out["base_rate"],
                      "pooled_f1_mean": out["pooled"]["mean"],
                      "pooled_f1_ci": [out["pooled"]["lo"], out["pooled"]["hi"]],
                      "analytic": out["pooled"]["analytic"],
                      "macro_f1_mean": out.get("per_year_macro_f1_mean"),
                      "pr_auc": [out["pooled"]["pr_auc_mean"],
                                 out["pooled"]["pr_auc_lo"],
                                 out["pooled"]["pr_auc_hi"]]}, indent=2))


if __name__ == "__main__":
    main()
