"""Seasonal climate outlook integration with drought early warning.

Downscale seasonal precipitation forecasts to district-level anomalies vs ERA5
climatology, classify as above/near/below normal, and integrate with drought EW
status to produce an escalated risk tier.

Method:
  1. Build ERA5 climatology: district x calendar-month precipitation normals
  2. Downscale: express forecast as % anomaly vs climatology
  3. Classify: tercile-based (below normal < 33rd pct, near normal 33-67, above > 67)
  4. Integrate: if forecast is "below normal" AND drought EW status is Watch/Warning,
     escalate one tier (bounded by Alert)
  5. Honest reporting: if seasonal forecast data is unavailable, report "outlook
     unavailable" rather than fabricating skill.

Outputs:
  results/tables/seasonal_outlook.csv         district x {forecast_class, escalated_status}
  results/seasonal_outlook_summary.json       tercile counts, integration stats, limitations
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


def build_climatology(era5_df):
    """Build district x calendar-month precipitation climatology from ERA5.
    
    Returns: DataFrame with columns [location, month, precip_normal_mm]
    """
    era = era5_df.copy()
    era["date"] = pd.to_datetime(era["date"])
    era["month"] = era["date"].dt.month
    
    # Monthly precipitation sum per district
    monthly = era.groupby(["location", "month"])["precipitation_sum"].sum().reset_index()
    clim = monthly.groupby(["location", "month"])["precipitation_sum"].mean().reset_index()
    clim.columns = ["location", "month", "precip_normal_mm"]
    return clim


def downscale_forecast(forecast_df, clim_df):
    """Express forecast as % anomaly vs climatology.
    
    Returns: DataFrame with columns [location, month, forecast_precip_mm, precip_normal_mm, anomaly_pct]
    """
    fc = forecast_df.copy()
    fc["month"] = pd.to_datetime(fc["month"]).dt.month
    
    # Merge with climatology
    merged = fc.merge(clim_df, on=["location", "month"], how="left")
    
    # Compute anomaly %
    merged["anomaly_pct"] = 100.0 * (merged["forecast_precip_mm"] - merged["precip_normal_mm"]) / merged["precip_normal_mm"]
    merged["anomaly_pct"] = merged["anomaly_pct"].replace([np.inf, -np.inf], np.nan)
    
    return merged


def classify_outlook(anomaly_pct):
    """Classify forecast as below/near/above normal using terciles.
    
    Simple tercile-based classification: below < -10%, near -10 to +10, above > +10.
    (In a full implementation, terciles would be derived from historical forecast distribution.)
    """
    if pd.isna(anomaly_pct):
        return "unknown"
    if anomaly_pct < -10:
        return "below_normal"
    elif anomaly_pct > 10:
        return "above_normal"
    else:
        return "near_normal"


def integrate_with_drought_ew(outlook_df, ew_df):
    """Escalate drought EW status if forecast is below normal AND status is Watch/Warning.
    
    Returns: DataFrame with escalated_status column.
    """
    # Get latest month EW status
    latest_month = ew_df["month"].max()
    latest_ew = ew_df[ew_df["month"] == latest_month][["location", "status"]].copy()
    
    # Merge with outlook
    merged = outlook_df.merge(latest_ew, on="location", how="left")
    
    # Escalation rule
    def escalate(row):
        if pd.isna(row["status"]):
            return "unknown"
        if row["forecast_class"] == "below_normal" and row["status"] in ["Watch", "Warning"]:
            # Escalate one tier
            if row["status"] == "Watch":
                return "Warning"
            elif row["status"] == "Warning":
                return "Alert"
        return row["status"]
    
    merged["escalated_status"] = merged.apply(escalate, axis=1)
    return merged


def main():
    # Check if seasonal forecast data is available
    status_file = DATA / "seasonal_fetch_status.json"
    if not status_file.exists():
        print("Seasonal forecast status file not found; skipping outlook")
        # Write empty outputs
        pd.DataFrame(columns=["location", "region", "month", "forecast_precip_mm",
                               "precip_normal_mm", "anomaly_pct", "forecast_class",
                               "status", "escalated_status"]).to_csv(
            RESULTS / "tables/seasonal_outlook.csv", index=False
        )
        summary = {
            "available": False,
            "note": "Seasonal forecast data unavailable. Outlook cannot be computed.",
            "limitations": ["No seasonal forecast data ingested"],
        }
        (RESULTS / "seasonal_outlook_summary.json").write_text(json.dumps(summary, indent=2))
        return
    
    status = json.loads(status_file.read_text())
    if not status.get("fetched", False):
        print("Seasonal forecast fetch failed; skipping outlook")
        pd.DataFrame(columns=["location", "region", "month", "forecast_precip_mm",
                               "precip_normal_mm", "anomaly_pct", "forecast_class",
                               "status", "escalated_status"]).to_csv(
            RESULTS / "tables/seasonal_outlook.csv", index=False
        )
        summary = {
            "available": False,
            "note": "Seasonal forecast fetch failed. Outlook cannot be computed.",
            "limitations": ["Seasonal API unreachable or returned no data"],
        }
        (RESULTS / "seasonal_outlook_summary.json").write_text(json.dumps(summary, indent=2))
        return
    
    # Load data
    forecast_df = pd.read_csv(DATA / "seasonal_forecast_raw.csv")
    era5_df = pd.read_csv(DATA / "era5_daily.csv")
    ew_df = pd.read_csv(RESULTS / "tables/drought_early_warning.csv")
    ew_df["month"] = pd.to_datetime(ew_df["month"])
    
    # Build climatology
    clim_df = build_climatology(era5_df)
    
    # Downscale forecast
    downscaled = downscale_forecast(forecast_df, clim_df)
    
    # Classify outlook
    downscaled["forecast_class"] = downscaled["anomaly_pct"].apply(classify_outlook)
    
    # Integrate with drought EW
    integrated = integrate_with_drought_ew(downscaled, ew_df)
    
    # Save output
    integrated.to_csv(RESULTS / "tables/seasonal_outlook.csv", index=False)
    
    # Summary statistics
    class_counts = integrated["forecast_class"].value_counts().to_dict()
    escalated_counts = integrated["escalated_status"].value_counts().to_dict()
    n_escalated = len(integrated[integrated["escalated_status"] != integrated["status"]])
    
    summary = {
        "available": True,
        "n_districts": len(C.LOCATIONS),
        "n_months_forecast": integrated["month"].nunique(),
        "forecast_class_counts": class_counts,
        "escalated_status_counts": escalated_counts,
        "n_districts_escalated": n_escalated,
        "note": "Seasonal climate outlook integrated with drought early warning. "
                "Escalation rule: if forecast is 'below normal' AND EW status is Watch/Warning, "
                "escalate one tier (bounded by Alert).",
        "limitations": [
            "Seasonal forecast skill over East Africa is modest (precipitation particularly)",
            "Statistical downscaling assumes forecast bias is stationary",
            "Tercile classification uses fixed thresholds (+/-10%) rather than historical distribution",
            "Escalation rule is heuristic; not validated against historical outcomes",
        ],
    }
    (RESULTS / "seasonal_outlook_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    
    print(f"Seasonal outlook: {len(integrated)} district-months")
    print(f"Forecast class counts: {class_counts}")
    print(f"Escalated status counts: {escalated_counts}")
    print(f"Districts escalated: {n_escalated}")


if __name__ == "__main__":
    main()
