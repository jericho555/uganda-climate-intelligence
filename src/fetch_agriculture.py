"""Fetch REAL Ugandan agricultural production data (World Bank Open Data, CC BY 4.0).

The World Bank series are FAOSTAT-derived national annual figures and are the
most robust openly-licensable agricultural backbone reachable from this
environment (FAOSTAT's API is not cleanly scriptable; Open-Meteo/soil-moisture
is rate-limited). We pull:
  * AG.PRD.CREL.MT  cereal production (metric tonnes)
  * AG.LND.CREL.HA  land under cereal production (hectares)
and derive cereal yield (kg/ha) = 1000 * production_t / area_ha.

Cereals (maize-dominant in Uganda) are the smallholder staple and the crop most
sensitive to the rain-fed growing seasons, so this is the right impact variable
to couple against the drought indices.

Output: data/uganda_crop_yield.csv (year, cereal_prod_t, cereal_area_ha,
        cereal_yield_kgha)
"""
import sys, json, time, urllib.request, urllib.error, ssl
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

CTX = ssl.create_default_context()
OUT = C.DATA / "uganda_crop_yield.csv"
BASE = "https://api.worldbank.org/v2/country/UGA/indicator"


def _wb(indicator, lo=1961, hi=2024, tries=8):
    url = f"{BASE}/{indicator}?format=json&date={lo}:{hi}&per_page=100"
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
            with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
                js = json.loads(r.read().decode("utf-8"))
            rows = js[1] if len(js) > 1 and js[1] else []
            return {int(d["date"]): d["value"] for d in rows if d["value"] is not None}
        except urllib.error.HTTPError as e:
            last = e
            time.sleep(min(8 * (i + 1), 45))   # WB anonymous rate-limit: back off hard
        except Exception as e:                  # noqa
            last = e
            time.sleep(min(6 * (i + 1), 30))
    raise RuntimeError(f"failed {indicator}: {last!r}")


def main():
    print("fetching World Bank cereal production ...", flush=True)
    prod = _wb("AG.PRD.CREL.MT")
    time.sleep(3)
    print("fetching World Bank cereal area ...", flush=True)
    area = _wb("AG.LND.CREL.HA")

    years = sorted(set(prod) & set(area))
    rows = []
    for y in years:
        p, a = float(prod[y]), float(area[y])
        if a > 0:
            rows.append(dict(year=y, cereal_prod_t=p, cereal_area_ha=a,
                             cereal_yield_kgha=round(1000.0 * p / a, 1)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} years  {df.year.min()}..{df.year.max()} -> {OUT.name}")
    print("  yield (kg/ha): "
          f"{df.cereal_yield_kgha.iloc[0]:.0f} ({df.year.iloc[0]}) -> "
          f"{df.cereal_yield_kgha.iloc[-1]:.0f} ({df.year.iloc[-1]})")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
