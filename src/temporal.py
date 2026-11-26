"""Temporal analysis: STL decomposition, Mann-Kendall trend, Sen's slope,
and change-point detection on the Uganda climate series (RQ4, RQ5).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS

from statsmodels.tsa.seasonal import STL
import pymannkendall as mk
import ruptures as rpt


def sens_slope(x, y):
    n = len(x)
    slopes = [(y[j] - y[i]) / (j - i) for i in range(n) for j in range(i + 1, min(n, i + 400))]
    return float(np.median(slopes)) if slopes else np.nan


def main():
    df = pd.read_csv(DATA / "full_ensemble.csv", parse_dates=["date"])
    out_trend, stl_info, changepoints = [], [], []

    for loc, g in df.groupby("location"):
        g = g.sort_values("date")
        for var in ["precipitation_sum", "temperature_2m_max", "et0_fao_evapotranspiration"]:
            s = g[var].astype(float).copy()
            s.index = pd.to_datetime(g.date)
            s = s.resample("MS").mean().dropna()          # monthly series
            if len(s) < 36:
                continue
            # Mann-Kendall + Sen's slope on monthly series
            try:
                tau_h = mk.original_test(s.values)
                ss = sens_slope(np.arange(len(s.values)), s.values)
                out_trend.append(dict(location=loc, variable=var,
                                      tau=float(tau_h.Tau), p_value=float(tau_h.p),
                                      z=float(tau_h.z), trend=tau_h.trend,
                                      sen_slope_per_month=ss))
            except Exception as e:
                out_trend.append(dict(location=loc, variable=var, error=repr(e)))
            # STL
            try:
                stl = STL(s, period=12, robust=True).fit()
                stl_info.append(dict(location=loc, variable=var,
                                     resid_std=float(np.std(stl.resid)),
                                     seasonal_strength=float(1 - np.var(stl.resid) /
                                                             np.var(s - stl.trend))))
            except Exception:
                pass

    trend = pd.DataFrame(out_trend)
    trend.to_csv(RESULTS / "tables/temporal_trends.csv", index=False)
    pd.DataFrame(stl_info).to_csv(RESULTS / "tables/stl_strength.csv", index=False)

    # change points on the national anomaly-count series
    daily_anom = df.groupby("date").is_anomaly.sum().astype(float)
    daily_anom.index = pd.to_datetime(daily_anom.index)
    monthly = daily_anom.resample("MS").sum().values
    bkps = []
    if len(monthly) > 24:
        algo = rpt.Pelt(model="rbf", min_size=6).fit(monthly)
        bkps = algo.predict(pen=12)
    monthly_series = daily_anom.resample("MS").sum()
    (RESULTS / "change_points.json").write_text(json.dumps({
        "monthly_anom_count_bkps_index": [int(b) for b in bkps],
        "bkp_months": [str(monthly_series.index[b].date()) for b in bkps if b < len(monthly_series)],
    }, indent=2))

    # significance summary
    sig = trend.dropna(subset=["p_value"]) if "p_value" in trend else pd.DataFrame()
    n_sig = int((sig.p_value < 0.05).sum()) if len(sig) else 0
    n_dec = int(((sig.p_value < 0.05) & (sig.tau < 0)).sum()) if len(sig) else 0
    n_inc = int(((sig.p_value < 0.05) & (sig.tau > 0)).sum()) if len(sig) else 0
    print(f"trend tests={len(sig)} significant={n_sig} increasing={n_inc} decreasing={n_dec}")
    if len(sig):
        print(sig.sort_values('p_value').head(12).to_string(index=False))
    print("change points (monthly idx):", bkps)


if __name__ == "__main__":
    main()
