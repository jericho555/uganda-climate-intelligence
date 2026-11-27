"""Fetch the World Bank climate-portal indicator series for Uganda.

Why this script exists
----------------------
The framework asks for validation against the World Bank Climate Portal. The
portal site itself (climateknowledgeportal.worldbank.org) refuses non-browser
clients -- it answers this environment with HTTP 403 from its edge WAF, on both
the country page and its /api paths. The same indicator programme publishes its
country series through ``api.worldbank.org``, which IS reachable here, so the
portal content is taken from the API that serves it rather than scraped from the
site, and every indicator is stored with the metadata the API reports for it
(source, sourceNote, lastUpdated) so the provenance is auditable later.

Indicators (Uganda, 1990-2025)
-----------------------------
EN.CLC.SPEI.XD   national Standardised Precipitation-Evapotranspiration Index,
                 annual. An independent drought index: it was not derived from
                 the ERA5 series this paper's detectors run on.
AG.LND.PRCP.MM   long-term average annual precipitation (mm/yr), WDI
                 (FAO-derived). A level check on the reanalysis climatology.
EN.CLC.MDAT.ZS   population affected by droughts, floods and extreme
                 temperatures (% , average 1990-2009), EM-DAT-derived.
CC.FLD.TOTL.ZS   share of the population exposed to floods (CCDR).
CC.TNET.FLD.ZS   share of the transport network exposed to floods (CCDR).
CC.THHZ.RANK.RF / CC.THHZ.RANK.UF
                 ThinkHazard! natural-hazard level for river / urban flood.

Outputs
-------
data/wb_climate.csv        year, wb_spei  (the annual index we can correlate)
data/wb_climate_meta.json  indicator -> metadata + every non-null value served
"""
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

CTX = ssl.create_default_context()
OUT_CSV = C.DATA / "wb_climate.csv"
OUT_META = C.DATA / "wb_climate_meta.json"

# A plain "Mozilla/5.0" UA is what the edge accepts; the research scripts in this
# repo that use a shorter token get rate-limited on the agricultural series.
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "application/json"}

INDICATORS = {
    "wb_spei": "EN.CLC.SPEI.XD",
    "wb_precip_normal_mm": "AG.LND.PRCP.MM",
    "wb_pop_affected_drought_flood_pct": "EN.CLC.MDAT.ZS",
    "wb_pop_exposed_flood_pct": "CC.FLD.TOTL.ZS",
    "wb_transport_exposed_flood_pct": "CC.TNET.FLD.ZS",
    "wb_thinkhazard_river_flood_level": "CC.THHZ.RANK.RF",
    "wb_thinkhazard_urban_flood_level": "CC.THHZ.RANK.UF",
}


def wb_get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                print(f"  HTTP {e.code} (permanent, skip): {url[-60:]}")
                return None
            print(f"  HTTP {e.code}: retry {i}")
        except Exception as e:  # noqa
            print(f"  retry {i}: {type(e).__name__}")
        time.sleep(min(4 * (i + 1), 15))
    return None


def fetch_series(code, date="1990:2025"):
    env = wb_get(f"https://api.worldbank.org/v2/country/UGA/indicator/{code}"
                 f"?format=json&date={date}&per_page=200")
    if not env or not isinstance(env, list) or len(env) < 2 or not env[1]:
        return None, None
    rows = env[1]
    meta = {}
    first = rows[0].get("indicator") or {}
    for k in ("id", "name", "unit", "source", "sourceNote", "lastUpdated",
              "observationsValue"):
        if first.get(k) not in (None, ""):
            meta[k] = first[k]
    meta["topics"] = [t.get("value") for t in (first.get("topics") or [])
                      if t.get("value")]
    vals = {}
    for r in rows:
        if r.get("value") is not None:
            vals[int(r["date"])] = float(r["value"])
    return meta, vals


def main():
    print("fetching World Bank climate-portal indicators (UGA) ...", flush=True)
    meta_all, series = {}, {}
    for name, code in INDICATORS.items():
        meta, vals = fetch_series(code)
        if not vals:
            print(f"  {code} ({name}): NO DATA")
            meta_all[code] = dict(name=name, metadata=meta or {}, values={})
            continue
        print(f"  {code} ({name}): {len(vals)} yrs "
              f"{min(vals)}..{max(vals)}  [{meta.get('unit', '')}]")
        meta_all[code] = dict(name=name, metadata=meta or {},
                              values={str(k): v for k, v in sorted(vals.items())})
        if name == "wb_spei":
            series = vals

    if not series:
        raise SystemExit("no SPEI series returned -- refusing to fabricate")

    df = pd.DataFrame({"year": sorted(series),
                       "wb_spei": [series[y] for y in sorted(series)]})
    df.to_csv(OUT_CSV, index=False)
    OUT_META.write_text(json.dumps(meta_all, indent=2))
    print(f"saved {len(df)} yrs -> {OUT_CSV.name}")
    print(df.describe().loc[["min", "mean", "max"]].round(3).to_string())
    driest = df.nsmallest(5, "wb_spei")
    wettest = df.nlargest(5, "wb_spei")
    print("driest 5 yrs :", list(zip(driest.year, driest.wb_spei.round(2))))
    print("wettest 5 yrs:", list(zip(wettest.year, wettest.wb_spei.round(2))))


if __name__ == "__main__":
    main()
