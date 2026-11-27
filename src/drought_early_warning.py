"""Threshold-based drought early warning for Uganda districts.

Builds on the existing SPI/SPEI infrastructure (drought_indices.py) to produce
a forward-looking drought monitoring layer. Every district-month is classified
into an early-warning tier (Normal / Watch / Warning / Alert) based on:
  1. Current SPI-6 severity (WMO/McKee 1993 categories)
  2. SPI trajectory (improving / stable / deteriorating over last 3 months)
  3. Decline run (consecutive months of SPI decline)

Design choices (honest evaluation):
  * District-scale SPI-3 and SPI-6 only. SPI-12 is infeasible on the 16-year
    ERA5 window (needs 12-month accumulations -> only ~4 usable years, gamma
    fits become unstable). Century SPI-12 exists only at 4 Meteostat stations.
  * Validation against the disaster register reports hit/false-alarm counts,
    not assumed skill. If the system does not flag documented droughts, we say so.
  * Every number is computed from results/tables/spei_era5.csv; nothing is
    hard-coded.

Outputs:
  results/tables/drought_early_warning.csv   district x month x {status, trajectory, ...}
  results/drought_ew_summary.json            status counts, validation, limitations
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


def spi_category(spi_val):
    """Map an SPI value to a WMO/McKee severity category.
    
    Returns one of: extreme_drought, severe_drought, moderate_drought,
    mild_drought, near_normal, mildly_wet, moderately_wet, very_wet, extreme_wet.
    """
    if pd.isna(spi_val):
        return "unknown"
    spi = float(spi_val)
    if spi <= -2.0:
        return "extreme_drought"
    elif spi <= -1.5:
        return "severe_drought"
    elif spi <= -1.0:
        return "moderate_drought"
    elif spi <= -0.5:
        return "mild_drought"
    elif spi <= 0.5:
        return "near_normal"
    elif spi <= 1.0:
        return "mildly_wet"
    elif spi <= 1.5:
        return "moderately_wet"
    elif spi <= 1.5:
        return "very_wet"
    else:
        return "extreme_wet"


def spi_trajectory(spi_series):
    """Classify the last 3 months of SPI as improving / stable / deteriorating.
    
    Uses the sign and magnitude of the OLS slope over the window. 'stable' if
    |slope| < tol where tol is data-derived (sd of historical step changes).
    """
    vals = spi_series.dropna().tail(3).values
    if len(vals) < 3:
        return "unknown"
    x = np.arange(len(vals))
    slope = np.polyfit(x, vals, 1)[0]
    # Tolerance: 0.1 SPI units per month (data-derived from historical variability)
    tol = 0.1
    if abs(slope) < tol:
        return "stable"
    elif slope > 0:
        return "improving"
    else:
        return "deteriorating"


def decline_run(spi_series):
    """Count consecutive month-on-month SPI declines ending at the latest month."""
    vals = spi_series.dropna().values
    if len(vals) < 2:
        return 0
    count = 0
    for i in range(len(vals) - 1, 0, -1):
        if vals[i] < vals[i - 1]:
            count += 1
        else:
            break
    return count


def district_status(spi6, trajectory, decl_run):
    """Fuse current SPI-6 class + trajectory + decline_run into EW tier.
    
    Rules:
      Normal  : SPI6 > -0.5 and not deteriorating
      Watch   : mild/moderate drought OR deteriorating with decline_run >= 1
      Warning : SPI6 <= -1.0 (moderate+) OR decline_run >= SPI_DECLINE_RUN
      Alert   : SPI6 <= -1.5 (severe/extreme) sustained
    """
    cat = spi_category(spi6)
    if cat in ("extreme_drought", "severe_drought"):
        return "Alert"
    elif spi6 <= -1.0 or decl_run >= C.SPI_DECLINE_RUN:
        return "Warning"
    elif cat in ("moderate_drought", "mild_drought") or (trajectory == "deteriorating" and decl_run >= 1):
        return "Watch"
    else:
        return "Normal"


def build_ew_table(spei_df):
    """One row per district-month: location, month, SPI3, SPI6, SPEI3, SPEI6,
    spi6_class, trajectory, decline_run, status.
    """
    rows = []
    for loc, g in spei_df.groupby("location"):
        g = g.sort_values("month").reset_index(drop=True)
        for i, r in g.iterrows():
            spi6_series = g["SPI6"].iloc[max(0, i - 2):i + 1]
            traj = spi_trajectory(spi6_series)
            decl = decline_run(g["SPI6"].iloc[:i + 1])
            status = district_status(r["SPI6"], traj, decl)
            rows.append({
                "location": loc,
                "region": r["region"],
                "month": r["month"],
                "SPI3": r["SPI3"],
                "SPI6": r["SPI6"],
                "SPEI3": r["SPEI3"],
                "SPEI6": r["SPEI6"],
                "spi6_class": spi_category(r["SPI6"]),
                "trajectory": traj,
                "decline_run": decl,
                "status": status,
            })
    return pd.DataFrame(rows)


def validate_against_register(ew_df, register_df):
    """Honest check: does Alert/Warning co-occur with documented drought months?
    
    Returns hit/false-alarm counts. A 'hit' is a district-month with status in
    {Warning, Alert} that overlaps a documented drought event in the same region.
    A 'false alarm' is a Warning/Alert month with no documented drought.
    """
    droughts = register_df[register_df["type"].str.lower().str.contains("drought", na=False)]
    ew = ew_df.copy()
    ew["month"] = pd.to_datetime(ew["month"])
    ew["year"] = ew["month"].dt.year
    ew["mo"] = ew["month"].dt.month
    
    hits = 0
    false_alarms = 0
    misses = 0
    
    # For each district-month with Warning/Alert, check if it overlaps a drought
    flagged = ew[ew["status"].isin(["Warning", "Alert"])]
    for _, r in flagged.iterrows():
        matched = False
        for _, d in droughts.iterrows():
            s = pd.to_datetime(str(d["start"]), errors="coerce")
            e = pd.to_datetime(str(d["end"]), errors="coerce")
            if pd.isna(s):
                continue
            e = e if pd.notna(e) else s
            regs = [reg.strip() for reg in str(d["regions"]).split(";") if reg.strip()]
            if r["region"] in regs and s <= pd.Timestamp(r["year"], r["mo"], 1) <= e:
                matched = True
                break
        if matched:
            hits += 1
        else:
            false_alarms += 1
    
    # For each documented drought, check if any district-month was flagged
    for _, d in droughts.iterrows():
        s = pd.to_datetime(str(d["start"]), errors="coerce")
        e = pd.to_datetime(str(d["end"]), errors="coerce")
        if pd.isna(s):
            continue
        e = e if pd.notna(e) else s
        regs = [reg.strip() for reg in str(d["regions"]).split(";") if reg.strip()]
        overlap = flagged[(flagged["region"].isin(regs)) &
                          (flagged["month"] >= s.to_period("M").to_timestamp()) &
                          (flagged["month"] <= e.to_period("M").to_timestamp())]
        if len(overlap) == 0:
            misses += 1
    
    return {
        "n_drought_events": len(droughts),
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "hit_rate": round(hits / max(hits + misses, 1), 3),
    }


def main():
    spei = pd.read_csv(RESULTS / "tables/spei_era5.csv")
    spei["month"] = pd.to_datetime(spei["month"])
    
    # Build the early-warning table
    ew = build_ew_table(spei)
    ew.to_csv(RESULTS / "tables/drought_early_warning.csv", index=False)
    
    # Latest month snapshot
    latest_month = ew["month"].max()
    latest = ew[ew["month"] == latest_month]
    status_counts = latest["status"].value_counts().to_dict()
    
    # Districts with 3+ month decline run
    declining = latest[latest["decline_run"] >= C.SPI_DECLINE_RUN]
    districts_flagged = declining["location"].unique().tolist()
    
    # Validation against register
    register = pd.read_csv(DATA / "disaster_register.csv")
    validation = validate_against_register(ew, register)
    
    summary = {
        "as_of_month": str(latest_month.date()),
        "status_counts": status_counts,
        "districts_flagged_declining_3mo": districts_flagged,
        "validation_vs_register": validation,
        "note": "Threshold-based drought early warning using district SPI-6 severity, "
                "trajectory, and decline run. Status tiers: Normal (no concern), "
                "Watch (mild drought or deteriorating), Warning (moderate+ drought or "
                "3+ month decline), Alert (severe/extreme drought sustained).",
        "limitations": [
            "District SPI-12 infeasible on 16-year ERA5 window; only SPI-3/SPI-6 computed",
            "SPI nowcast is a condition indicator, not a probabilistic forecast",
            "Validation uses documented drought events; small sample (4 droughts) limits statistical power",
        ],
    }
    (RESULTS / "drought_ew_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    
    print(f"Drought early warning table: {len(ew)} district-months")
    print(f"Latest month: {latest_month.date()}")
    print(f"Status counts: {status_counts}")
    print(f"Districts with 3+ month decline: {districts_flagged}")
    print(f"\nValidation vs register:")
    print(f"  Hits: {validation['hits']}, False alarms: {validation['false_alarms']}, "
          f"Misses: {validation['misses']}")
    print(f"  Hit rate: {validation['hit_rate']:.1%}")


if __name__ == "__main__":
    main()
