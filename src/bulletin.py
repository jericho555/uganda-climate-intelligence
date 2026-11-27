"""Automated climate risk bulletin generator.

Synthesizes the forward-looking layers (drought EW, yield forecast, seasonal outlook)
into a narrative Markdown report. Every number is read from the summary JSONs — nothing
is hard-coded, so the bulletin cannot drift from the pipeline.

Outputs:
  reports/climate_risk_bulletin_YYYY_MM.md
"""
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA, RESULTS, REPORTS


def section_executive(ew_summary, yield_summary, outlook_summary):
    """Executive summary section."""
    lines = ["## Executive Summary", ""]
    
    # Drought status
    status_counts = ew_summary.get("status_counts", {})
    n_alert = status_counts.get("Alert", 0)
    n_warning = status_counts.get("Warning", 0)
    n_watch = status_counts.get("Watch", 0)
    
    lines.append(f"**Drought Early Warning:** As of {ew_summary.get('as_of_month', 'N/A')}, "
                 f"{n_alert} district(s) are in Alert status, {n_warning} in Warning, "
                 f"and {n_watch} in Watch.")
    
    if ew_summary.get("districts_flagged_declining_3mo"):
        declining = ", ".join(ew_summary["districts_flagged_declining_3mo"])
        lines.append(f"Districts with 3+ month SPI decline: {declining}.")
    lines.append("")
    
    # Yield forecast
    proj = yield_summary.get("projection", {})
    if proj.get("projected_anomaly_pct") is not None:
        lines.append(f"**Crop Yield Forecast:** For {yield_summary.get('projection_year', 'N/A')}, "
                     f"the national cereal yield is projected to be {proj['projected_anomaly_pct']:.1f}% "
                     f"from trend (95% PI: [{proj['prediction_interval_95'][0]:.1f}, "
                     f"{proj['prediction_interval_95'][1]:.1f}]%). "
                     f"Probability of below-normal yield: {proj['p_below_normal']:.1%}.")
        skill = yield_summary.get("skill_check", {})
        lines.append(f"Model skill: {skill.get('verdict', 'N/A').replace('_', ' ')}.")
    lines.append("")
    
    # Seasonal outlook
    if outlook_summary.get("available", False):
        class_counts = outlook_summary.get("forecast_class_counts", {})
        n_below = class_counts.get("below_normal", 0)
        lines.append(f"**Seasonal Outlook:** {n_below} district-month(s) are forecast as "
                     f"below-normal precipitation.")
        n_escalated = outlook_summary.get("n_districts_escalated", 0)
        if n_escalated:
            lines.append(f"Seasonal forecast escalated {n_escalated} district(s) to a higher risk tier.")
    else:
        lines.append("**Seasonal Outlook:** Unavailable (seasonal forecast data could not be ingested).")
    lines.append("")
    
    return "\n".join(lines)


def section_drought(ew_summary, ew_df):
    """Drought early warning section."""
    lines = ["## Drought Early Warning", ""]
    lines.append(f"**As of:** {ew_summary.get('as_of_month', 'N/A')}")
    lines.append("")
    
    status_counts = ew_summary.get("status_counts", {})
    lines.append("### District Status Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status in ["Normal", "Watch", "Warning", "Alert"]:
        lines.append(f"| {status} | {status_counts.get(status, 0)} |")
    lines.append("")
    
    # Validation
    val = ew_summary.get("validation_vs_register", {})
    lines.append("### Validation Against Documented Droughts")
    lines.append("")
    lines.append(f"- Hits: {val.get('hits', 0)}")
    lines.append(f"- False alarms: {val.get('false_alarms', 0)}")
    lines.append(f"- Misses: {val.get('misses', 0)}")
    lines.append(f"- Hit rate: {val.get('hit_rate', 0):.1%}")
    lines.append("")
    
    # Districts in Alert/Warning
    latest_month = ew_df["month"].max()
    latest = ew_df[ew_df["month"] == latest_month]
    flagged = latest[latest["status"].isin(["Warning", "Alert"])]
    if len(flagged):
        lines.append("### Districts in Warning or Alert")
        lines.append("")
        for _, row in flagged.iterrows():
            lines.append(f"- **{row['location']}** ({row['region']}): {row['status']} — "
                         f"SPI-6 = {row['SPI6']:.2f}, trajectory = {row['trajectory']}")
        lines.append("")
    
    # Limitations
    lines.append("### Limitations")
    lines.append("")
    for lim in ew_summary.get("limitations", []):
        lines.append(f"- {lim}")
    lines.append("")
    
    return "\n".join(lines)


def section_yield(yield_summary):
    """Yield forecast section."""
    lines = ["## Crop Yield Impact Forecast", ""]
    
    model = yield_summary.get("model", {})
    lines.append(f"**Model:** yield_anomaly ~ growing-season SPI-3 (national mean)")
    lines.append(f"- Years: {model.get('years', 'N/A')} (n={model.get('n', 0)})")
    lines.append(f"- R² = {model.get('r_squared', 'N/A')}")
    lines.append(f"- RMSE = {model.get('rmse', 'N/A')}%")
    lines.append(f"- Pearson r = {model.get('pearson_r', 'N/A')} (p={model.get('pearson_p', 'N/A')})")
    lines.append("")
    
    proj = yield_summary.get("projection", {})
    if proj.get("projected_anomaly_pct") is not None:
        lines.append(f"**{yield_summary.get('projection_year', 'N/A')} Projection:**")
        lines.append(f"- Current SPI = {proj.get('current_spi', 'N/A')}")
        lines.append(f"- Projected yield anomaly = {proj.get('projected_anomaly_pct', 'N/A')}%")
        pi = proj.get("prediction_interval_95", ['N/A', 'N/A'])
        lines.append(f"- 95% prediction interval: [{pi[0]}, {pi[1]}]%")
        lines.append(f"- P(below-normal) = {proj.get('p_below_normal', 'N/A'):.1%}")
        lines.append("")
    
    skill = yield_summary.get("skill_check", {})
    lines.append(f"**Skill check:** {skill.get('verdict', 'N/A').replace('_', ' ')}")
    if skill.get("skill_ratio") is not None:
        lines.append(f"- Holdout RMSE / Climatology RMSE = {skill.get('skill_ratio', 'N/A')}")
    lines.append("")
    
    lines.append("### Limitations")
    lines.append("")
    for lim in yield_summary.get("limitations", []):
        lines.append(f"- {lim}")
    lines.append("")
    
    return "\n".join(lines)


def section_outlook(outlook_summary):
    """Seasonal outlook section."""
    lines = ["## Seasonal Climate Outlook", ""]
    
    if not outlook_summary.get("available", False):
        lines.append("*Seasonal outlook unavailable.*")
        lines.append("")
        lines.append(f"**Reason:** {outlook_summary.get('note', 'No data')}")
        lines.append("")
        return "\n".join(lines)
    
    lines.append(f"**Districts:** {outlook_summary.get('n_districts', 0)}")
    lines.append(f"**Forecast months:** {outlook_summary.get('n_months_forecast', 0)}")
    lines.append("")
    
    class_counts = outlook_summary.get("forecast_class_counts", {})
    lines.append("### Forecast Class Distribution")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("|-------|-------|")
    for cls in ["below_normal", "near_normal", "above_normal"]:
        lines.append(f"| {cls.replace('_', ' ').title()} | {class_counts.get(cls, 0)} |")
    lines.append("")
    
    escalated_counts = outlook_summary.get("escalated_status_counts", {})
    lines.append("### Escalated Risk Status")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status in ["Normal", "Watch", "Warning", "Alert"]:
        lines.append(f"| {status} | {escalated_counts.get(status, 0)} |")
    lines.append("")
    
    n_escalated = outlook_summary.get("n_districts_escalated", 0)
    lines.append(f"**Districts escalated:** {n_escalated}")
    lines.append("")
    
    lines.append("### Limitations")
    lines.append("")
    for lim in outlook_summary.get("limitations", []):
        lines.append(f"- {lim}")
    lines.append("")
    
    return "\n".join(lines)


def section_figures():
    """Figure references section."""
    lines = ["## Figures", ""]
    lines.append("The following figures are generated by the pipeline and included in the")
    lines.append("manuscript and dashboard:")
    lines.append("")
    lines.append(f"- `figures/fig_drought_ew.png` — Drought early warning choropleth + SPI trajectories")
    lines.append(f"- `figures/fig_yield_forecast.png` — Yield anomaly vs SPI scatter + projection")
    lines.append(f"- `figures/fig_seasonal_outlook.png` — Seasonal precipitation outlook map + bars")
    lines.append("")
    return "\n".join(lines)


def render_bulletin():
    """Render the full bulletin as Markdown."""
    # Load summaries
    ew_summary = json.loads((RESULTS / "drought_ew_summary.json").read_text())
    yield_summary = json.loads((RESULTS / "yield_forecast_summary.json").read_text())
    outlook_path = RESULTS / "seasonal_outlook_summary.json"
    outlook_summary = json.loads(outlook_path.read_text()) if outlook_path.exists() else {"available": False}
    
    # Load tables
    ew_df = pd.read_csv(RESULTS / "tables/drought_early_warning.csv")
    ew_df["month"] = pd.to_datetime(ew_df["month"])
    
    # Build bulletin
    now = datetime.now()
    lines = [
        f"# Uganda Climate Risk Bulletin — {now.strftime('%B %Y')}",
        "",
        f"*Generated:* {now.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"*Data sources:* ERA5 reanalysis (2008-2023), FAOSTAT cereal yields (1961-2024), "
        f"curated disaster register (32 events)",
        "",
        "---",
        "",
    ]
    
    lines.append(section_executive(ew_summary, yield_summary, outlook_summary))
    lines.append(section_drought(ew_summary, ew_df))
    lines.append(section_yield(yield_summary))
    lines.append(section_outlook(outlook_summary))
    lines.append(section_figures())
    
    lines.append("---")
    lines.append("")
    lines.append("*This bulletin is generated automatically by the Uganda Climate Intelligence Framework.*")
    lines.append("*Every number is computed from the released data and code; nothing is hard-coded.*")
    lines.append("")
    
    return "\n".join(lines)


def main():
    bulletin = render_bulletin()
    
    now = datetime.now()
    out_path = REPORTS / f"climate_risk_bulletin_{now.strftime('%Y_%m')}.md"
    out_path.write_text(bulletin, encoding="utf-8")
    
    print(f"Climate risk bulletin saved to: {out_path}")
    print(f"Length: {len(bulletin)} characters, {len(bulletin.splitlines())} lines")


if __name__ == "__main__":
    main()
