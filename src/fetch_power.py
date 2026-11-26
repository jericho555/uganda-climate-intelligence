"""Fetch an INDEPENDENT, rate-limit-free reanalysis record via NASA POWER.

Why this module exists
----------------------
Our primary daily climate (ERA5 via Open-Meteo) starts in 2008 and the Meteostat
in-situ record has a genuine reporting collapse for the 1990-2007 window (the
Ugandan stations simply stopped filing daily data, confirmed empty on NCEI GHCN).
NOAA PSL (CHIRPS/GPCC/ERA5) is netCDF-only and hard-throttled here, and this
environment cannot decode netCDF (no xarray/netCDF4/gdal).

NASA POWER (MERRA-2 based, White et al. 2023) is openly licensed (CC-0/open),
serves PLAIN JSON with no API key, is not aggressively 429-throttled, and reaches
back to 1984. It reliably supplies, for every district centroid:
  * T2M                air temperature at 2 m (deg C)      -> Tier-1 / anomaly
  * RH2M               near-surface relative humidity (%)   -> Tier-1 (was missing)
  * ALLSKY_SFC_SW_DWN  surface downwelling shortwave (kWh/m^2/day) -> Tier-1 solar
This CLOSES the 1990-2007 multivariate climate gap for temperature, humidity and
radiation with real data (precipitation is not exposed in POWER's reachable
community here, so we do NOT claim to fill the rainfall gap).

Output: data/power_daily.csv  (location, region, date, t2m_c, rh2m_pct, sw_kwh_m2)
"""
import sys, json, time, argparse, urllib.request, urllib.error, ssl
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

CTX = ssl.create_default_context()
OUT = C.DATA / "power_daily.csv"
BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "T2M,RH2M,ALLSKY_SFC_SW_DWN"
# dark period + a small overlap on both ends for later splicing / cross-validation
START, END = "19880101", "20071231"


def _get(url, tries=6):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(10 * (i + 1), 60))     # back off hard, then retry
                continue
            if e.code in (400, 404, 422):             # bad request: do not retry
                raise RuntimeError(f"{url} -> {e.code}: "
                                   f"{e.read().decode('utf-8','replace')[:160]}")
            time.sleep(min(6 * (i + 1), 30))
        except Exception as e:                         # noqa network/timeout
            last = repr(e)[:160]
            time.sleep(min(6 * (i + 1), 30))
    raise RuntimeError(f"exhausted retries: {last}")


def fetch_one(name, lat, lon):
    url = (f"{BASE}?parameters={PARAMS}&start={START}&end={END}"
           f"&community=RE&format=JSON&latitude={lat}&longitude={lon}")
    js = _get(url)
    p = js["properties"]["parameter"]
    t2m, rh, sw = p.get("T2M", {}), p.get("RH2M", {}), p.get("ALLSKY_SFC_SW_DWN", {})
    rows = []
    for d in t2m:                                     # d = YYYYMMDD
        def clean(v):
            return None if (v is None or v == -999.0) else float(v)
        rows.append(dict(location=name, date=f"{d[:4]}-{d[4:6]}-{d[6:8]}",
                         t2m_c=clean(t2m.get(d)), rh2m_pct=clean(rh.get(d)),
                         sw_kwh_m2=clean(sw.get(d))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N locations (smoke test)")
    ap.add_argument("--append", action="store_true", help="append to existing CSV")
    args = ap.parse_args()

    locs = C.LOCATIONS[:args.limit] if args.limit else C.LOCATIONS
    print(f"NASA POWER daily fetch: {len(locs)} districts, {START}..{END}", flush=True)
    all_rows, region_of = [], {l[1]: l[0] for l in C.LOCATIONS}
    for i, (region, name, lat, lon, elev) in enumerate(locs, 1):
        rows = fetch_one(name, lat, lon)
        for r in rows:
            r["region"] = region
        all_rows += rows
        got = sum(1 for r in rows if r["t2m_c"] is not None)
        print(f"  [{i:2d}/{len(locs)}] {name:11s} {len(rows)} days (t2m ok={got})", flush=True)
        time.sleep(2)                                  # be gentle on the endpoint

    df = pd.DataFrame(all_rows, columns=["location", "region", "date",
                                         "t2m_c", "rh2m_pct", "sw_kwh_m2"])
    df["date"] = pd.to_datetime(df["date"])
    if args.append and OUT.exists():
        df = pd.concat([pd.read_csv(OUT, parse_dates=["date"]), df], ignore_index=True)
        df = df.drop_duplicates(subset=["location", "date"]).sort_values(["location", "date"])
    df.to_csv(OUT, index=False)

    yrs = sorted(df.date.dt.year.unique())
    print(f"\nsaved {len(df)} rows -> {OUT.name}")
    print(f"  districts={df.location.nunique()}  years={yrs[0]}..{yrs[-1]} ({len(yrs)} yr)")
    for c in ("t2m_c", "rh2m_pct", "sw_kwh_m2"):
        nn = df[c].notna().sum()
        print(f"  {c:10s} non-null {nn:6d}  mean={df[c].mean():.2f}")


if __name__ == "__main__":
    main()
