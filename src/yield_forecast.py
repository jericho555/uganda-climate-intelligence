"""National cereal-yield impact forecasting from growing-season SPI.

Because district-level yield data is unavailable (FAOSTAT provides only national
annual totals), this module builds a NATIONAL-scale yield forecast. The honest
limitation is stated: district-level yield projection is not identifiable from
available data.

Method:
  1. Yield anomaly: log(cereal yield) detrended against a full-series linear
     time trend (removes 60+ years of intensification), expressed as % deviation.
  2. Growing-season moisture index per calendar year: district SPI-3 averaged
     nationally over GROWING_MONTHS (Mar-May + Sep-Nov, bimodal Uganda).
  3. OLS regression: yield_anomaly_pct ~ growing-season SPI.
  4. Honest skill check: chronological holdout RMSE vs climatology null.
  5. Current-year projection: given the latest complete growing-season SPI,
     project expected yield anomaly and P(below-normal).

Outputs:
  results/tables/yield_forecast.csv          year x {spi, observed, fitted, resid}
  results/yield_forecast_summary.json        model stats, projection, skill, limitations
"""
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS
from scipy import stats


def yield_anomaly(path):
    """Log yield detrended against full-series linear trend -> % anomaly."""
    y = pd.read_csv(path)
    y["logy"] = np.log(y["cereal_yield_kgha"].astype(float))
    slope, intercept, *_ = stats.linregress(y["year"], y["logy"])
    y["trend"] = intercept + slope * y["year"]
    y["yield_anom_pct"] = 100.0 * (y["logy"] - y["trend"])
    y["yield_trend_pct_yr"] = 100.0 * (np.exp(slope) - 1.0)
    return y


def national_growing_spi(spei_df, growing_months):
    """District SPI-3 averaged nationally over growing months each year.
    
    Returns: dict year -> mean SPI-3 over growing season (requires all months present).
    """
    df = spei_df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["mo"] = df["month"].dt.month
    
    # Filter to growing months
    sub = df[df["mo"].isin(growing_months)].dropna(subset=["SPI3"])
    
    # Station-average per year-month, then require all growing months
    ym = sub.groupby(["year", "mo"])["SPI3"].mean().reset_index()
    g = ym.groupby("year").agg(n=("mo", "count"), val=("SPI3", "mean"))
    ser = g[g["n"] == len(growing_months)]["val"]
    return {int(y): float(v) for y, v in ser.items()}


def fit_yield_model(yield_df, spi_by_year):
    """OLS: yield_anom_pct ~ growing-season SPI.
    
    Returns: dict with slope, intercept, R^2, Pearson/Spearman r & p, n_years,
    95% CI on slope, residuals.
    """
    yrs = sorted(set(yield_df["year"]) & set(spi_by_year))
    if len(yrs) < 6:
        return {"n": len(yrs), "note": "insufficient overlap", "verdict": "no_model"}
    
    ya = yield_df.set_index("year").loc[yrs, "yield_anom_pct"].values
    xi = np.array([spi_by_year[yy] for yy in yrs])
    
    pr, pp = stats.pearsonr(xi, ya)
    sr, sp = stats.spearmanr(xi, ya)
    slope, intercept, r_value, p_value, std_err = stats.linregress(xi, ya)
    
    # 95% CI on slope
    n = len(yrs)
    t_crit = stats.t.ppf(0.975, n - 2)
    ci_lo = slope - t_crit * std_err
    ci_hi = slope + t_crit * std_err
    
    fitted = intercept + slope * xi
    resid = ya - fitted
    rmse = float(np.sqrt(np.mean(resid**2)))
    
    return {
        "n": n,
        "years": f"{yrs[0]}-{yrs[-1]}",
        "slope": round(float(slope), 3),
        "intercept": round(float(intercept), 3),
        "r_squared": round(float(r_value**2), 3),
        "pearson_r": round(float(pr), 3),
        "pearson_p": float(f"{pp:.3g}"),
        "spearman_r": round(float(sr), 3),
        "spearman_p": float(f"{sp:.3g}"),
        "slope_95ci": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
        "rmse": round(rmse, 2),
        "residuals": resid.tolist(),
        "fitted": fitted.tolist(),
        "years_list": yrs,
        "yield_anomaly": ya.tolist(),
        "spi": xi.tolist(),
        "verdict": "model_fitted",
    }


def skill_check(model, yield_df, spi_by_year):
    """Honest skill check: chronological holdout RMSE vs climatology null.
    
    Returns: dict with holdout_rmse, climatology_rmse, skill_ratio.
    If skill_ratio > 1, model beats climatology; if < 1, climatology is better.
    """
    if model.get("verdict") != "model_fitted":
        return {"note": "no model to check"}
    
    yrs = model["years_list"]
    # Chronological split: train on first 80%, test on last 20%
    split = int(len(yrs) * 0.8)
    train_yrs = yrs[:split]
    test_yrs = yrs[split:]
    
    if len(test_yrs) < 2:
        return {"note": "insufficient test years for holdout"}
    
    # Refit on train
    train_ya = yield_df.set_index("year").loc[train_yrs, "yield_anom_pct"].values
    train_xi = np.array([spi_by_year[yy] for yy in train_yrs])
    slope, intercept, *_ = stats.linregress(train_xi, train_ya)
    
    # Predict on test
    test_xi = np.array([spi_by_year[yy] for yy in test_yrs])
    test_ya = yield_df.set_index("year").loc[test_yrs, "yield_anom_pct"].values
    pred = intercept + slope * test_xi
    holdout_rmse = float(np.sqrt(np.mean((test_ya - pred)**2)))
    
    # Climatology null: predict mean
    clim_pred = np.full_like(test_ya, train_ya.mean())
    clim_rmse = float(np.sqrt(np.mean((test_ya - clim_pred)**2)))
    
    skill_ratio = holdout_rmse / clim_rmse if clim_rmse > 0 else float("nan")
    
    return {
        "holdout_rmse": round(holdout_rmse, 2),
        "climatology_rmse": round(clim_rmse, 2),
        "skill_ratio": round(skill_ratio, 3),
        "verdict": "model_beats_climatology" if skill_ratio < 1 else "climatology_beats_model",
    }


def project_yield(model, current_spi, yield_df):
    """Given current growing-season SPI, project expected yield anomaly.
    
    Returns: dict with projected_anomaly_pct, prediction_interval, p_below_normal.
    """
    if model.get("verdict") != "model_fitted":
        return {"note": "no model for projection"}
    
    slope = model["slope"]
    intercept = model["intercept"]
    resid_std = float(np.std(model["residuals"]))
    
    projected = intercept + slope * current_spi
    pi_lo = projected - 1.96 * resid_std
    pi_hi = projected + 1.96 * resid_std
    
    # P(below-normal): share of predictive distribution below 10th pct of yield anomaly
    threshold = float(np.percentile(yield_df["yield_anom_pct"], C.YIELD_BELOW_NORMAL_PCTL))
    # Approximate using normal CDF
    p_below = float(stats.norm.cdf(threshold, loc=projected, scale=resid_std))
    
    return {
        "current_spi": round(float(current_spi), 3),
        "projected_anomaly_pct": round(float(projected), 2),
        "prediction_interval_95": [round(float(pi_lo), 2), round(float(pi_hi), 2)],
        "p_below_normal": round(float(p_below), 3),
        "threshold_pct": round(float(threshold), 2),
    }


def main():
    yield_df = yield_anomaly(DATA / "uganda_crop_yield.csv")
    spei_df = pd.read_csv(RESULTS / "tables/spei_era5.csv")
    
    # National growing-season SPI
    spi_by_year = national_growing_spi(spei_df, C.GROWING_MONTHS)
    
    # Fit model
    model = fit_yield_model(yield_df, spi_by_year)
    
    # Skill check
    skill = skill_check(model, yield_df, spi_by_year)
    
    # Current-year projection (latest complete growing season)
    latest_year = max(spi_by_year.keys())
    current_spi = spi_by_year[latest_year]
    projection = project_yield(model, current_spi, yield_df)
    
    # Save forecast table
    rows = []
    for yr in model.get("years_list", []):
        idx = model["years_list"].index(yr)
        rows.append({
            "year": yr,
            "spi": round(model["spi"][idx], 3),
            "observed_anomaly_pct": round(model["yield_anomaly"][idx], 2),
            "fitted_anomaly_pct": round(model["fitted"][idx], 2),
            "residual_pct": round(model["yield_anomaly"][idx] - model["fitted"][idx], 2),
        })
    pd.DataFrame(rows).to_csv(RESULTS / "tables/yield_forecast.csv", index=False)
    
    # Summary
    summary = {
        "yield_years": [int(yield_df["year"].min()), int(yield_df["year"].max())],
        "yield_trend_pct_per_year": round(float(yield_df["yield_trend_pct_yr"].iloc[0]), 2),
        "growing_months": list(C.GROWING_MONTHS),
        "model": {k: v for k, v in model.items() if k != "residuals"},
        "skill_check": skill,
        "projection": projection,
        "projection_year": latest_year,
        "note": "National-scale yield forecast from growing-season SPI. District-level "
                "yield projection is not identifiable from available data (FAOSTAT provides "
                "only national annual totals).",
        "limitations": [
            "National scale only; no district-level yield data",
            f"Few overlapping years ({model.get('n', 0)}) -> wide confidence intervals",
            "Correlation does not imply causation; technology trend removed by detrending",
            f"Skill check: {skill.get('verdict', 'unknown')}",
        ],
    }
    (RESULTS / "yield_forecast_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    
    print(f"Yield forecast model: n={model.get('n', 0)} years ({model.get('years', 'N/A')})")
    print(f"  R^2 = {model.get('r_squared', 'N/A')}, RMSE = {model.get('rmse', 'N/A')}%")
    print(f"  Pearson r = {model.get('pearson_r', 'N/A')} (p={model.get('pearson_p', 'N/A')})")
    print(f"Skill check: {skill.get('verdict', 'N/A')} (ratio={skill.get('skill_ratio', 'N/A')})")
    print(f"\nProjection for {latest_year}:")
    print(f"  Current SPI = {projection.get('current_spi', 'N/A')}")
    print(f"  Projected yield anomaly = {projection.get('projected_anomaly_pct', 'N/A')}%")
    print(f"  95% PI: [{projection.get('prediction_interval_95', ['N/A', 'N/A'])[0]}, "
          f"{projection.get('prediction_interval_95', ['N/A', 'N/A'])[1]}]%")
    print(f"  P(below-normal) = {projection.get('p_below_normal', 'N/A'):.1%}")


if __name__ == "__main__":
    main()
