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


def carry_uncertainty(dm):
    """Observed spread of the month-to-month carry, i.e. the honest uncertainty
    envelope around the persistence nowcast.

    The forward tier is produced by carrying the current score into the next week,
    month and season. That rule needs an interval, not just a point: here it is the
    empirical distribution of how far the district-month score actually moved in the
    16-year record. No model is fitted and nothing is assumed -- if carries are
    volatile we report a wide envelope.
    """
    deltas, hot = [], []
    for _, g in dm.sort_values(["year", "month"]).groupby("location"):
        r = g.risk.values
        b = g.band.values
        if len(r) < 2:
            continue
        d = np.diff(r)                       # d[i] = month i -> month i+1
        deltas.append(d)
        hot.append(d[np.isin(b[:-1], ["High", "Extreme"])])

    def block(x):
        if x is None or not len(x):
            return None
        a = np.abs(x)
        return dict(n_transitions=int(len(x)),
                    median_abs_change=round(float(np.median(a)), 1),
                    p80_abs_change=round(float(np.quantile(a, 0.80)), 1),
                    p95_abs_change=round(float(np.quantile(a, 0.95)), 1),
                    q10_signed=round(float(np.quantile(x, 0.10)), 1),
                    q90_signed=round(float(np.quantile(x, 0.90)), 1))

    dd = np.concatenate(deltas) if deltas else np.array([])
    cc = np.concatenate(hot) if hot else np.array([])
    return dict(all_district_months=block(dd),
                from_high_or_extreme=block(cc),
                reading=("carrying the current score forward one month, 80% of "
                         "observed transitions moved it by at most the p80 value "
                         "below; that is the width of the nowcast, not a model fit"))


CARRY_FEATS = ["risk", "I", "D", "S", "M", "F", "anomaly_days", "longest_run",
               "max_concurrent_vars", "anomaly_rate", "spatial_coherence"]


def carry_table(dm):
    """One row per observed month -> month+1 transition for a district.

    A transition is only counted when the two rows are genuinely adjacent
    calendar months (year*12+month differs by exactly 1), so a gap in the record
    can never be read as a one-month step. Features are everything measurable at
    the ORIGIN month; the target is the destination month's risk score."""
    X, y, tgt_year, start_band = [], [], [], []
    for _, g in dm.sort_values(["year", "month"]).groupby("location"):
        if len(g) < 2:
            continue
        key = g.year.values * 12 + g.month.values
        keep = np.diff(key) == 1
        for i in np.where(keep)[0]:
            X.append(g[CARRY_FEATS].iloc[i].values.astype(float))
            y.append(float(g.risk.values[i + 1]))
            tgt_year.append(int(g.year.values[i + 1]))
            start_band.append(str(g.band.values[i]))
    return (np.array(X, float), np.array(y, float),
            np.array(tgt_year, int), np.array(start_band, object))


def conformal_intervals(dm, alphas=(0.10, 0.20), cal_end=None, ridge_alpha=1.0):
    """Fitted, calibrated PREDICTION INTERVALS for the one-month carry.

    carry_uncertainty() above describes what carries did (empirical quantiles).
    This is the modelled version the uncertainty module asks for: a point forecast
    plus a distribution-free interval with MEASURED coverage.

    Split conformal (Lei et al. 2018), run chronologically so no future data is
    ever used: transitions whose DESTINATION month falls in or before
    EVAL_TRAIN_YEARS[1] (2018) are calibration, the rest (2019-2023) are the test
    block on which coverage is scored. Two point models are carried side by side
    and neither is selected on test data:
      persistence -- next score = this score, i.e. the framework's actual rule;
      ridge       -- a linear carry model on the origin-month features (the five
                     UCASI dimensions in rank form, anomaly days, longest run,
                     concurrent variables, anomaly rate, spatial coherence),
                     standardised on calibration only.
    Nonconformity is the absolute residual; the interval is point +/- the
    ceil((n+1)(1-alpha)) order statistic of the calibration residuals, clipped to
    the score's legal [0, 100] range. Coverage on the untouched test block is
    reported, per overall and per starting band, because an interval that covers
    on average but not inside High/Extreme cells is useless to an operator.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    cal_end = C.EVAL_TEST_YEARS[0] - 1 if cal_end is None else cal_end
    X, y, tgt_year, start_band = carry_table(dm)
    if len(y) < 50:
        return {"note": f"only {len(y)} transitions -- too few to calibrate"}
    cal = tgt_year <= cal_end
    test = ~cal
    scaler = StandardScaler().fit(X[cal])
    ridge = Ridge(alpha=ridge_alpha).fit(scaler.transform(X[cal]), y[cal])
    models = {"persistence": X[:, CARRY_FEATS.index("risk")].copy(),
              "ridge": ridge.predict(scaler.transform(X))}

    out = dict(
        method=("split conformal on chronologically split district-month "
                f"transitions; calibration = destination year <= {cal_end}"),
        features=CARRY_FEATS, ridge_alpha=ridge_alpha,
        n_transitions=int(len(y)), n_calibration=int(cal.sum()),
        n_test=int(test.sum()),
        test_years=[int(tgt_year[test].min()), int(tgt_year[test].max())],
        models={},
    )
    coef = sorted(zip(CARRY_FEATS[1:], ridge.coef_[1:]), key=lambda kv: -abs(kv[1]))
    for mname, yhat in models.items():
        e_cal = np.abs(y[cal] - yhat[cal])
        n = len(e_cal)
        rec = dict(rmse_calibration=round(float(np.sqrt(np.mean(e_cal ** 2))), 2),
                   rmse_test=round(float(np.sqrt(np.mean((y[test] - yhat[test]) ** 2))), 2),
                   mae_test=round(float(np.mean(np.abs(y[test] - yhat[test]))), 2))
        if mname == "ridge":
            rec["largest_standardised_coefficients"] = [
                [k, round(float(v), 2)] for k, v in coef[:4]]
        for a in alphas:
            lvl = int(round(100 * (1 - a)))
            k = min(n, int(np.ceil((n + 1) * (1 - a))))
            q = float(np.sort(e_cal)[k - 1])
            lo = np.clip(yhat[test] - q, 0.0, 100.0)
            hi = np.clip(yhat[test] + q, 0.0, 100.0)
            hit = (y[test] >= lo) & (y[test] <= hi)
            by_band = {}
            for b in ["Low", "Moderate", "High", "Extreme"]:
                m = start_band[test] == b
                if m.any():
                    by_band[b] = dict(n=int(m.sum()),
                                      coverage=round(float(hit[m].mean()), 3))
            rec[f"level_{lvl}"] = dict(
                alpha=a, conformal_quantile=round(q, 1),
                mean_width=round(float(np.mean(hi - lo)), 1),
                coverage_test=round(float(hit.mean()), 3),
                coverage_by_starting_band=by_band)
        out["models"][mname] = rec

    rp = out["models"]["persistence"]["rmse_test"]
    rr = out["models"]["ridge"]["rmse_test"]
    cp = out["models"]["persistence"]["level_90"]["coverage_test"]
    cr = out["models"]["ridge"]["level_90"]["coverage_test"]
    point = (
        f"the fitted carry model reaches test RMSE {rr} against {rp} for plain "
        "persistence, so its point forecast is the better one" if rr < rp else
        f"the fitted carry model reaches test RMSE {rr} against {rp} for plain "
        "persistence, so it adds nothing over carrying the score forward")
    band = (
        f"yet its conformal band covers only {cr} of held-out transitions against "
        f"{cp} for the persistence band at the same nominal 90%: a chronological "
        "split breaks the exchangeability the coverage guarantee assumes, so the "
        "measured shortfall is reported rather than smoothed. The wider "
        "persistence band is the conservative interval to quote"
        if cr < cp else
        f"and its conformal band covers {cr} of held-out transitions against {cp} "
        "for persistence at the same nominal 90%, so the fitted band is the "
        "narrower one and is also no worse covered")
    out["verdict"] = point + "; " + band
    # latest observed month -> forward interval per district, for the dashboard
    last = dm.sort_values(["year", "month"]).groupby("location").tail(1)
    last = last.reset_index(drop=True)
    rows = []
    q80 = out["models"]["persistence"]["level_80"]["conformal_quantile"]
    q90 = out["models"]["persistence"]["level_90"]["conformal_quantile"]
    Xf = last[CARRY_FEATS].values.astype(float)
    ridge_next = ridge.predict(scaler.transform(Xf))
    for i, r in last.iterrows():
        p = float(Xf[i, 0])
        rows.append(dict(location=r.location, region=r.region,
                         as_of=f"{int(r.year)}-{int(r.month):02d}",
                         risk_now=round(p, 1), band=str(r.band),
                         next_point=round(p, 1),
                         next_ridge=round(float(np.clip(ridge_next[i], 0, 100)), 1),
                         pi80_lo=round(max(p - q80, 0.0), 1),
                         pi80_hi=round(min(p + q80, 100.0), 1),
                         pi90_lo=round(max(p - q90, 0.0), 1),
                         pi90_hi=round(min(p + q90, 100.0), 1)))
    out["latest_forward_intervals"] = sorted(
        rows, key=lambda d: -d["risk_now"])
    return out


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
    carry = carry_uncertainty(dm)
    pi = conformal_intervals(dm)
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
        carry_uncertainty=carry,
        prediction_intervals=pi,
        latest_nowcast_top8=nowcast,
        note=("This is a monitoring/prioritisation NOWCAST. The forward-lead skill "
              "equals the measured lag-1 persistence; it is NOT a causal forecast. "
              "Predictive skill requires coupling exposure, soil moisture and impact "
              "data that a weather-only feature space omits."),
    )
    (RESULTS / "early_warning_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    nat.to_csv(RESULTS / "tables/risk_national_month.csv", index=False)
    pd.DataFrame(pi.get("latest_forward_intervals", [])).to_csv(
        RESULTS / "tables" / "nowcast_interval.csv", index=False)

    print("district-months:", len(dm), "event-months:", int(dm.is_event_month.sum()))
    print("band counts:", band_counts)
    print("SKILL:", json.dumps(skill, indent=2))
    print("PERSISTENCE:", json.dumps(pers, indent=2))
    print("CARRY UNCERTAINTY:", json.dumps(carry, indent=2))
    print("PREDICTION INTERVALS:", json.dumps(
        {k: v for k, v in pi.items() if k != "latest_forward_intervals"}, indent=2,
        default=str))
    print("LATEST NOWCAST:")
    for r in nowcast:
        print(f"  {r['location']:<11} {r['region']:<9} risk={r['risk']:5.1f} {r['band']}")


if __name__ == "__main__":
    main()
