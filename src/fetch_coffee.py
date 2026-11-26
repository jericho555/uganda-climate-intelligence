"""Fetch REAL Ugandan coffee data (Our World in Data, FAOSTAT-derived, CC BY 4.0).

Same provenance class as the cereal series (World Bank / FAOSTAT), but for
Uganda's signature export crop. OWID grapher CSVs are the reachable, key-free,
stable endpoint for FAOSTAT crop statistics in this environment (the FAOSTAT
JSON API is 404 / key-gated here, and UN Comtrade times out).

Primary series : Green coffee - Production (tonnes), 1961-2024.
Best-effort     : Green coffee - Area harvested (ha) -> yield = 1000*P/A (kg/ha).

Output: data/uganda_coffee.csv (year, coffee_prod_t[, coffee_area_ha,
        coffee_yield_kgha]).
"""
import sys, ssl, time, urllib.request, urllib.error
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

CTX = ssl.create_default_context()
UA = {"User-Agent": "Mozilla/5.0 (research)", "Accept": "text/csv,*/*"}
OUT = C.DATA / "uganda_coffee.csv"


def owid(slug, country="UGA", tries=6):
    url = (f"https://ourworldindata.org/grapher/{slug}.csv"
           f"?v=1&csvType=full&country={country}")
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
                txt = r.read().decode("utf-8", "replace")
            from io import StringIO
            df = pd.read_csv(StringIO(txt))
            # csvType=full returns every entity; keep Uganda only
            if "Code" in df.columns:
                df = df[df["Code"] == country]
            return df
        except urllib.error.HTTPError as e:
            print(f"  {slug}: HTTP {e.code} (skip)")
            return None
        except Exception as e:  # noqa
            print(f"  {slug}: retry {i} {type(e).__name__}")
            time.sleep(min(5 * (i + 1), 20))
    return None


def tidy(df, valuecol):
    d = df[["Year", valuecol]].dropna().copy()
    d["Year"] = d["Year"].astype(int)
    d[valuecol] = pd.to_numeric(d[valuecol], errors="coerce")
    return d.dropna().set_index("Year")[valuecol]


def main():
    print("fetching OWID coffee-bean-production (UGA) ...", flush=True)
    prod = owid("coffee-bean-production")
    if prod is None or prod.empty:
        raise SystemExit("production series unreachable -- refusing to fabricate")
    pcol = [c for c in prod.columns if "Production" in c][0]
    pser = tidy(prod, pcol)
    print(f"  production: {len(pser)} yrs {pser.index.min()}..{pser.index.max()} "
          f"({pcol})")

    # best-effort area harvested
    aser = None
    for slug in ["coffee-bean-area-harvested", "coffee-area-harvested",
                 "green-coffee-area-harvested"]:
        a = owid(slug)
        if a is not None and not a.empty:
            acol = [c for c in a.columns if "Area" in c or "harvested" in c.lower()]
            if acol:
                aser = tidy(a, acol[0])
                print(f"  area (from {slug}): {len(aser)} yrs")
                break

    years = sorted(pser.index)
    rows = []
    for y in years:
        r = dict(year=y, coffee_prod_t=float(pser[y]))
        if aser is not None and y in aser.index and aser[y] > 0:
            r["coffee_area_ha"] = float(aser[y])
            r["coffee_yield_kgha"] = round(1000.0 * r["coffee_prod_t"] / aser[y], 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} rows -> {OUT.name}  cols={list(df.columns)}")
    print(df.head(3).to_string(index=False))
    print(df.tail(6).to_string(index=False))
    # quick sanity: coffee production min/max, latest vs 1990s
    print(f"prod range: {df.coffee_prod_t.min():.0f}..{df.coffee_prod_t.max():.0f} t")


if __name__ == "__main__":
    main()
