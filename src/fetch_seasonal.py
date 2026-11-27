"""Seasonal climate forecast ingestion from Open-Meteo SEAS5 API.

Fetches ECMWF SEAS5 seasonal precipitation and temperature forecasts for each
district centroid. The API is free, no authentication required, CC-BY 4.0.

Graceful degradation: if the API is unreachable or returns no data, the module
writes an empty file and sets a flag so downstream modules report "outlook
unavailable" rather than failing.

Outputs:
  data/seasonal_forecast_raw.csv   location x month x {forecast_precip, forecast_temp}
"""
import sys
import time
import json
import warnings
from pathlib import Path
import urllib.request
import urllib.error
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from config import DATA


def fetch_point(lat, lon, retries=5, backoff=2.0):
    """Fetch seasonal forecast for a single point with retry logic.
    
    Returns: dict with monthly precipitation and temperature forecasts, or None on failure.
    """
    # Open-Meteo seasonal API endpoint (ECMWF SEAS5)
    # Variables: precipitation_sum, temperature_2m_mean
    # Lead time: up to 7 months
    url = (
        f"{C.SEASONAL_API_URL}?"
        f"latitude={lat}&longitude={lon}"
        f"&models={C.SEASONAL_MODELS}"
        f"&daily=precipitation_sum,temperature_2m_mean"
        f"&timezone=Africa%2FKampala"
    )
    
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "UgandaClimateIntel/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            # Parse the response
            if "daily" not in data:
                return None
            
            daily = data["daily"]
            dates = pd.to_datetime(daily["time"])
            precip = daily.get("precipitation_sum", [])
            temp = daily.get("temperature_2m_mean", [])
            
            # Aggregate to monthly
            df = pd.DataFrame({
                "date": dates,
                "precip": precip,
                "temp": temp,
            })
            df["month"] = df["date"].dt.to_period("M")
            monthly = df.groupby("month").agg(
                precip_mm=("precip", "sum"),
                temp_c=("temp", "mean")
            ).reset_index()
            
            return monthly.to_dict("records")
        
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
            else:
                print(f"  Warning: seasonal API fetch failed after {retries} attempts: {e}")
                return None
        except Exception as e:
            print(f"  Warning: unexpected error fetching seasonal forecast: {e}")
            return None


def main():
    """Fetch seasonal forecasts for all district centroids."""
    rows = []
    failed = []
    
    print(f"Fetching seasonal forecasts for {len(C.LOCATIONS)} districts...")
    for region, name, lat, lon, elev in C.LOCATIONS:
        print(f"  {name} ({lat:.3f}, {lon:.3f})...", end=" ")
        result = fetch_point(lat, lon)
        if result is None:
            print("FAILED")
            failed.append(name)
        else:
            print(f"OK ({len(result)} months)")
            for m in result:
                rows.append({
                    "location": name,
                    "region": region,
                    "month": str(m["month"]),
                    "forecast_precip_mm": round(float(m["precip_mm"]), 2) if pd.notna(m["precip_mm"]) else None,
                    "forecast_temp_c": round(float(m["temp_c"]), 2) if pd.notna(m["temp_c"]) else None,
                })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(DATA / "seasonal_forecast_raw.csv", index=False)
        print(f"\nSeasonal forecast saved: {len(df)} rows, {len(rows) // len(C.LOCATIONS)} months/district")
    else:
        # Write empty file with header
        pd.DataFrame(columns=["location", "region", "month", "forecast_precip_mm", "forecast_temp_c"]).to_csv(
            DATA / "seasonal_forecast_raw.csv", index=False
        )
        print("\nNo seasonal forecast data available; wrote empty file")
    
    if failed:
        print(f"Failed districts: {', '.join(failed)}")
    
    # Write status flag
    status = {
        "fetched": len(rows) > 0,
        "n_districts": len(C.LOCATIONS) - len(failed),
        "failed_districts": failed,
        "note": "Seasonal forecast ingestion. If fetched=False, downstream modules will report 'outlook unavailable'.",
    }
    (DATA / "seasonal_fetch_status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
