"""Forward risk nowcasting: UCASI recast as an operational district-month risk
score (RQ6 deliverable).

Motivation
----------
The generic daily detector benchmark sits at chance against disaster ground truth,
so a *predictive* forecast cannot be honestly claimed. What CAN be delivered is a
**nowcast**: a per-district-month risk surface that (i) condenses the validated
anomaly stream into the same five UCASI dimensions, (ii) ranks where climate-
anomaly severity is concentrated right now, and (iii) is carried forward to the
coming week/month/season by a *persistence* rule whose skill we MEASURE rather than
assume. Every number written here is computed from data/full_ensemble.csv; nothing
is hard-coded. If the skill is modest we report it as modest.

Honest evaluation
-----------------
* lag-1 month-to-month autocorrelation of the score (justifies a persistence nowcast)
* ROC-AUC of the district-month score for flagging documented event-months
* recall@TopK (share of documented event-months captured by the K highest-risk cells)
* the same two metrics for a size-matched random null (so "skill" means skill-above-chance)

Outputs: results/tables/risk_district_month.csv, results/early_warning_summary.json
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS
from spatial import centroid_table, distance_weights
from sklearn.metrics import roc_auc_score

ZCOLS = [f"{f}_z" for f in C.FEATURES]
W = dict(I=0.30, D=0.20, S=0.20, M=0.15, F=0.15)   # identical to UCASI weights


def longest_run(bools):
    """Longest consecutive True run in a 1-D boolean array."""
    best = cur = 0
    for b in bools:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def build_district_month(df, Wmat, nbr_idx, loc_index):
    """Aggregate the validated daily anomaly stream into one row per district-month,
    expressing the five UCASI dimensions in observational (nowcastable) form."""
    df = df.copy()
    df["n_feat_exceed"] = (df[ZCOLS].abs() > 2.0).sum(1)
    rows = []
    for (loc, yr, mo), g in df.groupby(["location", "year", "month"]):
        g = g.sort_values("date")
        anom = g[g.is_anomaly == 1]
        # intensity: mean ensemble score on anomaly days (peak-condition magnitude)
        intensity = float(anom.ens_score.mean()) if len(anom) else 0.0
        peak = float(g.ens_score.max())
        # duration/persistence: anomaly-day count and longest consecutive run
        dur_days = int(len(anom))
        run = longest_run((g.is_anomaly == 1).values)
        # multivar: most variables concurrently beyond |z|>2 on any day
        multivar = int(g.n_feat_exceed.max())
        # frequency: anomaly-day rate this month
        freq = float(len(anom) / max(len(g), 1))
        # event co-occurrence (for external evaluation only, never for scoring)
        event_days = int(g.label_event.sum())
        rows.append(dict(location=loc, region=g.region.iloc[0], year=int(yr),
                         month=int(mo), intensity=intensity, peak_score=peak,
                         anomaly_days=dur_days, longest_run=run,
                         max_concurrent_vars=multivar, anomaly_rate=freq,
                         event_days=event_days,
                         is_event_month=int(event_days > 0)))
    dm = pd.DataFrame(rows).sort_values(["location", "year", "month"]).reset_index(drop=True)

    # spatial coherence: fraction of neighbouring districts also in their own top
    # tercile of anomaly_rate this month (regional, not single-point, signal)
    dm["loc_i"] = dm.location.map(loc_index)
    top_third = dm.groupby("location").anomaly_rate.transform(
        lambda s: s >= s.quantile(2/3))
    dm["_elev"] = top_third.astype(int)
    # build a (year-month x district) elevation matrix for neighbour lookup
    piv = dm.pivot_table(index=["year", "month"], columns="loc_i",
                         values="_elev", aggfunc="first")
    nloc = len(loc_index)
    coh = {}
    for (yr, mo), prow in piv.iterrows():
        for li in range(nloc):
            nb = nbr_idx.get(li, [])
            vals = [prow.get(n, 0) for n in nb if not pd.isna(prow.get(n, np.nan))]
            coh[(yr, mo, li)] = float(np.mean(vals)) if vals else 0.0
    dm["spatial_coherence"] = [coh.get((r.year, r.month, int(r.loc_i)), 0.0)
                               for _, r in dm.iterrows()]
    dm = dm.drop(columns=["_elev"])

    # normalise each dimension to [0,1] by its own cross-cell percentile rank,
    # then fuse with UCASI weights -> risk score 0-100 (same construction as UCASI)
    for src, dst in [("intensity", "I"), ("anomaly_days", "D"),
                     ("spatial_coherence", "S"), ("max_concurrent_vars", "M"),
                     ("anomaly_rate", "F")]:
        dm[dst] = dm[src].rank(pct=True)
    dm["risk"] = (100 * (W["I"]*dm.I + W["D"]*dm.D + W["S"]*dm.S +
                         W["M"]*dm.M + W["F"]*dm.F)).round(1)
    q = dm.risk.quantile
    dm["band"] = pd.cut(dm.risk, bins=[-1, q(0.4), q(0.7), q(0.9), 101],
                        labels=["Low", "Moderate", "High", "Extreme"])
    return dm


def evaluate_skill(dm, seed=0):
    """Honest skill of the nowcast score for flagging documented event-months,
    against a size-matched random null (same number of positive cells)."""
    rng = np.random.default_rng(C.SEED + seed)
    y = dm.is_event_month.values
    s = dm.risk.values
    auc = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    k = int(y.sum())
    topk = np.argsort(-s)[:k]
    recall_at_k = float(y[topk].sum() / max(k, 1))
    # random null: k cells picked at random, averaged over draws
    recalls, aucs = [], []
    for _ in range(500):
        pick = rng.choice(len(y), size=k, replace=False)
        recalls.append(y[pick].sum() / max(k, 1))
        perm = rng.permutation(len(y))
        aucs.append(roc_auc_score(y, s[perm]))
    return dict(
        n_district_months=int(len(dm)), n_event_months=int(y.sum()),
        k_selected=k,
        auc=round(auc, 3), auc_null_mean=round(float(np.mean(aucs)), 3),
        recall_at_k=round(recall_at_k, 3),
        recall_null_mean=round(float(np.mean(recalls)), 3),
        recall_null_p95=round(float(np.quantile(recalls, 0.95)), 3),
    )


def persistence(dm):
    """Month-to-month (lag-1) autocorrelation of the risk score. A high value is the
    empirical warrant for carrying the current nowcast forward to the next period."""
    per = []
    pooled_pairs = []
    for _, g in dm.sort_values(["year", "month"]).groupby("location"):
        x = g.risk.values
        if len(x) > 3:
            per.append(np.corrcoef(x[:-1], x[1:])[0, 1])
            pooled_pairs.extend(zip(x[:-1], x[1:]))
    a = np.array([p for p in per if not np.isnan(p)])
    px = np.array([p[0] for p in pooled_pairs]); py = np.array([p[1] for p in pooled_pairs])
    return dict(lag1_mean_tau=round(float(np.mean(a)), 3),
                lag1_min=round(float(np.min(a)), 3),
                lag1_max=round(float(np.max(a)), 3),
                pooled_lag1_r=round(float(np.corrcoef(px, py)[0, 1]), 3),
                n_districts=int(len(a)))


def latest_nowcast(dm):
    """The current risk surface: highest-risk districts in the most recent complete month."""
    last = dm.sort_values(["year", "month"]).groupby("location").tail(1)
    last = last.sort_values("risk", ascending=False)
    top = last.head(8)[["location", "region", "year", "month", "risk", "band",
                        "anomaly_days", "longest_run", "max_concurrent_vars"]]
    return top.to_dict("records")


def forward_tiers(dm):
    """Nowcast carried forward by the persistence rule at three lead times.
    Skill claim = persistence-lag1 r; we report it, not an assumed forecast."""
    last = dm.sort_values(["year", "month"]).groupby("location").tail(1)
    last = last.sort_values("risk", ascending=False)
    return last[["location", "region", "risk", "band"]].to_dict("records")


def main():
    df = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"])
    ct = centroid_table()
    coords = list(zip(ct.lat.values, ct.lon.values))
    Wmat = distance_weights(coords, threshold_km=220.0)
    locs = ct.location.tolist()
    loc_index = {l: i for i, l in enumerate(locs)}
    nbr_idx = {loc_index[locs[i]]: [loc_index[locs[j]] for j in np.where(Wmat[i] > 0)[0]]
               for i in range(len(locs))}

    dm = build_district_month(df, Wmat, nbr_idx, loc_index)
    dm.to_csv(RESULTS / "tables/risk_district_month.csv", index=False)

    skill = evaluate_skill(dm)
    pers = persistence(dm)
    nowcast = latest_nowcast(dm)
    band_counts = dm.band.value_counts().to_dict()
    # national monthly risk for the time-series figure
    nat = dm.groupby(["year", "month"]).risk.mean().reset_index()

    summary = dict(
        weights=W,
        construction=("district-month UCASI recombination of the validated daily "
                      "anomaly stream; forward carry = persistence rule"),
        band_counts={str(k): int(v) for k, v in band_counts.items()},
        skill_vs_event_months=skill,
        persistence= pers,
        latest_nowcast_top8=nowcast,
        note=("This is a monitoring/prioritisation NOWCAST. The forward-lead skill "
              "equals the measured lag-1 persistence; it is NOT a causal forecast. "
              "Predictive skill requires coupling exposure, soil moisture and impact "
              "data that a weather-only feature space omits."),
    )
    (RESULTS / "early_warning_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    nat.to_csv(RESULTS / "tables/risk_national_month.csv", index=False)

    print("district-months:", len(dm), "event-months:", int(dm.is_event_month.sum()))
    print("band counts:", band_counts)
    print("SKILL:", json.dumps(skill, indent=2))
    print("PERSISTENCE:", json.dumps(pers, indent=2))
    print("LATEST NOWCAST:")
    for r in nowcast:
        print(f"  {r['location']:<11} {r['region']:<9} risk={r['risk']:5.1f} {r['band']}")


if __name__ == "__main__":
    main()
