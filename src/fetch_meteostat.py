"""Fetch the LONG in-situ Ugandan daily record from Meteostat bulk (CC-BY 4.0).

Why: ERA5 via Open-Meteo is free-tier rate-limited and starts at 1950; Meteostat
bulk serves ~1920s-2026 daily observations per station, tiny gz files, no key and
no throttling. This both widens the temporal range dramatically and provides the
independent station data for ERA5 cross-validation (a stated limitation).

Outputs:
  data/meteostat_stations.csv   Uganda station metadata (id,name,lat,lon,elev)
  data/meteostat_daily.csv      long daily frame (station,name,region_date,tavg,tmin,tmax,prcp)
"""
import sys, json, gzip, io, time, urllib.request, urllib.error, ssl
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

CTX = ssl.create_default_context()
ME = C.DATA / "meteostat"
ME.mkdir(parents=True, exist_ok=True)
STATION_CSV = C.DATA / "meteostat_stations.csv"
DAILY_CSV = C.DATA / "meteostat_daily.csv"
DAILY_COLS = ["date", "tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspm", "wspd", "tsun", "tslg"]


def _get(url, timeout=180, tries=6):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research pipeline)"})
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                raise
            time.sleep(min(4 * (i + 1), 30))
        except Exception as e:  # noqa
            last = e
            time.sleep(min(4 * (i + 1), 30))
    raise RuntimeError(f"failed {url}: {last!r}")


def uganda_stations():
    if STATION_CSV.exists():
        return pd.read_csv(STATION_CSV)
    print("downloading meteostat station metadata (~22MB gz) ...", flush=True)
    raw = _get("https://bulk.meteostat.net/v2/stations/full.json.gz", timeout=600)
    obj = json.loads(gzip.decompress(raw).decode("utf-8"))
    rows = []
    for s in obj:
        if str(s.get("country", "")).upper() != "UG":
            continue
        loc = s.get("location", {}) or {}
        nm = s.get("name", {})
        name = nm.get("en") if isinstance(nm, dict) else str(nm)
        rows.append(dict(station=str(s["id"]), name=name,
                         lat=loc.get("latitude"), lon=loc.get("longitude"),
                         elev=loc.get("elevation")))
    df = pd.DataFrame(rows).sort_values("station").reset_index(drop=True)
    df.to_csv(STATION_CSV, index=False)
    print("Uganda stations:", len(df))
    return df


def fetch_daily(sid):
    out = ME / f"{sid}.csv"
    if out.exists():
        return pd.read_csv(out, parse_dates=["date"])
    url = f"https://bulk.meteostat.net/v2/daily/{sid}.csv.gz"
    try:
        raw = _get(url, timeout=180)
    except urllib.error.HTTPError:
        return None
    txt = gzip.decompress(raw).decode("utf-8")
    df = pd.read_csv(io.StringIO(txt), header=None, names=DAILY_COLS)
    if not len(df):
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "tavg", "tmin", "tmax", "prcp"]]
    df.insert(0, "station", str(sid))
    df.to_csv(out, index=False)
    return df


def main():
    st = uganda_stations()
    name_by = dict(zip(st.station.astype(str), st.name))
    frames, ok = [], 0
    for sid in st.station.astype(str):
        d = fetch_daily(sid)
        if d is None or not len(d):
            continue
        d["name"] = name_by.get(sid, sid)
        d["lat"] = float(st.loc[st.station.astype(str) == sid, "lat"].iloc[0])
        d["lon"] = float(st.loc[st.station.astype(str) == sid, "lon"].iloc[0])
        frames.append(d)
        ok += 1
        print(f"[ok] {sid} {d['name']}: {len(d)} rows {d.date.min().date()}..{d.date.max().date()}", flush=True)
        time.sleep(0.5)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(DAILY_CSV, index=False)
    print(f"\nMETEOSTAT  stations={ok}/{len(st)}  rows={len(allf)}  "
          f"range={allf.date.min().date()}..{allf.date.max().date()}")
    print("years covered:", allf.date.dt.year.nunique())


if __name__ == "__main__":
    main()
