"""Fetch ERA5 daily reanalysis data for the Ugandan locations via Open-Meteo.

Writes one tidy CSV per location to data/raw/ and a combined long-format
frame to data/era5_daily.csv. Fully reproducible and openly licensed
(ERA5 via Open-Meteo, CC-BY 4.0).
"""
import sys
import time
import json
import urllib.parse
import urllib.request
import urllib.error
import ssl
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

CTX = ssl.create_default_context()
BASE = "https://archive-api.open-meteo.com/v1/era5"
# Cache is keyed by the requested range so a widened window forces a fresh fetch
# instead of silently reusing a narrower earlier download.
RAWDIR = C.DATA / "raw" / f"{C.START_DATE}_{C.END_DATE}"


def _get(url, tries=8, sleep=5):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research pipeline)"})
            with urllib.request.urlopen(req, timeout=180, context=CTX) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            # 429 / 5xx -> back off (capped); other 4xx -> fail fast
            if e.code == 429 or 500 <= e.code < 600:
                wait = min(sleep * (i + 1), 45)
                print(f"    [{e.code}] backing off {wait}s ...")
                time.sleep(wait)
                continue
            raise
        except Exception as e:  # noqa
            last = e
            time.sleep(min(sleep * (i + 1), 30))
    raise RuntimeError(f"fetch failed after {tries} tries: {last!r}")


def fetch_location(name, lat, lon, region, elev, start, end):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(C.DAILY_VARS),
        "timezone": C.TIMEZONE,
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    data = _get(url)
    daily = data["daily"]
    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    for c in C.DAILY_VARS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "location", name)
    df.insert(1, "region", region)
    df["lat"] = lat
    df["lon"] = lon
    df["elevation_m"] = elev
    units = data.get("daily_units", {})
    RAWDIR.mkdir(parents=True, exist_ok=True)
    return df, units


def _year_chunks(start_str, end_str, span=10):
    """Split [start,end] into <=`span`-year blocks so no single request is too
    large (the archive times out on ~75-year pulls)."""
    y0 = int(start_str[:4]); y1 = int(end_str[:4])
    y = y0
    while y <= y1:
        cs = f"{y}-01-01"
        ce_year = min(y + span - 1, y1)
        ce = f"{ce_year}-12-31"
        yield cs, ce
        y = ce_year + 1


def main():
    RAWDIR.mkdir(parents=True, exist_ok=True)
    chunks = list(_year_chunks(C.START_DATE, C.END_DATE, span=10))
    frames = []
    meta = {}
    for region, name, lat, lon, elev in C.LOCATIONS:
        cframes = []
        for ci, (cs, ce) in enumerate(chunks):
            out = RAWDIR / f"{name}__{cs[:4]}_{ce[:4]}.csv"
            if out.exists():
                cdf = pd.read_csv(out, parse_dates=["date"])
            else:
                try:
                    cdf, units = fetch_location(name, lat, lon, region, elev, cs, ce)
                    meta[name] = units
                    if len(cdf):
                        cdf.to_csv(out, index=False)  # persist chunk for resume
                    time.sleep(3.0)  # be polite to the free API (avoid 429)
                except Exception as e:
                    print(f"[skip] {name} {cs[:4]}-{ce[:4]}: {e!r}")
                    cdf = pd.DataFrame()
            cframes.append(cdf)
        loc = pd.concat([c for c in cframes if len(c)], ignore_index=True)
        if len(loc):
            loc = loc.sort_values("date").drop_duplicates(subset=["date"], keep="first")
            loc.to_csv(RAWDIR / f"{name}.csv", index=False)
            print(f"[loc] {name}: {len(loc)} rows  {loc.date.min().date()}..{loc.date.max().date()}")
            frames.append(loc)
        time.sleep(1.5)

    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(C.DATA / "era5_daily.csv", index=False)
    with open(C.DATA / "era5_units.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nTOTAL rows={len(allf)}  locations={allf.location.nunique()}  "
          f"date range={allf.date.min().date()}..{allf.date.max().date()}")
    print("NaN %% by feature:")
    print((allf[C.DAILY_VARS].isna().mean() * 100).round(2).to_string())


if __name__ == "__main__":
    main()
