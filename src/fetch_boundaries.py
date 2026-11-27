"""Fetch Uganda DISTRICT polygons so the maps can be true choropleths.

Why this module exists
----------------------
The pipeline's spatial units are 19 named points (src/config.py LOCATIONS):
Open-Meteo/ERA5 grid extractions at district headquarters. Until now every map
plotted those points as circles, which is honest but weak -- it implies no
areal extent, so a reader cannot see which district a value belongs to or how
much of the country the analysis actually covers.

Source: the Uganda Common Operational Dataset (COD-AB) published by OCHA on the
Humanitarian Data Exchange, "Uganda - Subnational Administrative Boundaries"
v01, admin level 2 = 135 districts, geometry valid 2020-08-24, licensed
CC BY-IGO (attribution to an intergovernmental organisation). Chosen over
geoBoundaries gbOpen UGA ADM2, which encodes 2006-era COUNTIES (151 units,
e.g. "Dodoth", "Kyadondo") rather than the districts this paper names.

What it writes (all committed, all small):
  data/district_boundaries.geojson   135 simplified district polygons
  data/region_boundaries.geojson     4 regional outlines (map context)
  data/district_lookup.csv           config name -> district, PCODE, region
  data/boundaries_provenance.json    URL, licence, version, sha256, geometry stats

The raw 37 MB archive is never written into data/ (only data/raw/, which is
gitignored); the derived subset is what the figures consume.

Usage:  python src/fetch_boundaries.py [--from-zip PATH] [--force]
"""
import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

DATASET = "cod-ab-uga"
DATASET_URL = "https://data.humdata.org/dataset/" + DATASET
CKAN = "https://data.humdata.org/api/3/action/package_show?id=" + DATASET
UA = {"User-Agent": "Mozilla/5.0 (research; UCIF pipeline)"}
# sha256 of uga_admin_boundaries.geojson.zip as retrieved from HDX on 2026-11-27.
# A mismatch is reported, not fatal: OCHA re-issues COD files.
PINNED_SHA = "5d2cc90a25954dd1d085ef6707782df34f0c5a77d63f7ab7c96a2eec31b7a0d1"
KEEP2 = ("adm2_name", "adm2_pcode", "adm1_name", "adm1_pcode",
         "area_sqkm", "center_lat", "center_lon", "valid_on", "version")
KEEP1 = ("adm1_name", "adm1_pcode", "area_sqkm", "valid_on", "version")
MAX_BYTES = 3_000_000
TOLERANCES = (0.003, 0.005, 0.008, 0.012)


def log(msg):
    print(msg, flush=True)


def _open(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                  timeout=300)


def resource_url():
    """Locate the GeoJSON resource for the Uganda COD dataset."""
    d = json.loads(_open(CKAN).read().decode("utf-8", "replace"))
    if not d.get("success"):
        raise RuntimeError("CKAN package_show failed: " + str(d)[:200])
    pkg = d["result"]
    for r in pkg["resources"]:
        if str(r.get("format", "")).lower() == "geojson":
            return r["url"], pkg
    raise RuntimeError("no GeoJSON resource listed for " + DATASET)


def download(dest):
    url, pkg = resource_url()
    log("fetching " + url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _open(url) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    tmp.rename(dest)
    return url, pkg


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_member(zip_path, member):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as fh:
            return json.loads(fh.read().decode("utf-8", "replace"))


# --------------------------------------------------------------------------
# Geometry helpers. shapely is used when present; the fallbacks are exact
# enough for a 0.003 deg simplification and a point-membership test.
# --------------------------------------------------------------------------
def _dp(ring, tol):
    """Douglas-Peucker on a [[lon, lat], ...] ring (pure python)."""
    if len(ring) < 3:
        return ring
    keep = [False] * len(ring)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]

    def perp(i, a, b):
        ax, ay = a
        bx, by = b
        px, py = ring[i]
        dx, dy = bx - ax, by - ay
        n = (dx * dx + dy * dy) ** 0.5
        if n == 0:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        return abs(dy * px - dx * py + bx * ay - by * ax) / n

    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        dmax, idx = -1.0, lo
        for j in range(lo + 1, hi):
            d = perp(j, ring[lo], ring[hi])
            if d > dmax:
                dmax, idx = d, j
        if dmax > tol:
            keep[idx] = True
            stack.append((lo, idx))
            stack.append((idx, hi))
    return [p for p, k in zip(ring, keep) if k]


def _rings(geom):
    """Yield rings (as lists of [lon, lat]) from a GeoJSON geometry."""
    t = geom["type"]
    if t == "Polygon":
        yield from geom["coordinates"]
    elif t == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield from poly


def _poly_groups(geom):
    """Yield ring-groups, one per polygon in the geometry."""
    if geom["type"] == "Polygon":
        yield geom["coordinates"]
    elif geom["type"] == "MultiPolygon":
        for poly in geom["coordinates"]:
            yield poly


def _rebuild(geom, tol, ndigits=4):
    """Simplify each ring, keep polygon grouping, drop invisible micro-holes."""
    groups = []
    for poly in _poly_groups(geom):
        rings = []
        for r in poly:
            s = _dp(list(r), tol)
            if len(s) < 4:
                continue
            rings.append([[round(x, ndigits), round(y, ndigits)] for x, y in s])
        if not rings:
            continue
        ext = max(rings, key=_area)
        holes = [r for r in rings if r is not ext and _area(r) > 1e-6]
        groups.append([ext] + holes)
    if not groups:
        return None
    if len(groups) == 1 and len(groups[0]) == 1:
        return {"type": "Polygon", "coordinates": groups[0]}
    return {"type": "MultiPolygon", "coordinates": groups}


def _area(ring):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _in_ring(lon, lat, ring):
    """Even-odd ray casting."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            xint = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < xint:
                inside = not inside
        j = i
    return inside


def contains(geom, lon, lat):
    """Point membership for Polygon/MultiPolygon, exterior rings only."""
    if geom["type"] == "Polygon":
        polys = [geom["coordinates"]]
    else:
        polys = geom["coordinates"]
    for poly in polys:
        if not poly:
            continue
        if not _in_ring(lon, lat, poly[0]):
            continue
        if any(_in_ring(lon, lat, hole) for hole in poly[1:]):
            continue
        return True
    return False


# --------------------------------------------------------------------------
def simplify(fc, tol):
    feats = []
    dropped = 0
    n_before = n_after = 0
    for f in fc["features"]:
        g = f.get("geometry")
        if not g:
            dropped += 1
            continue
        n_before += sum(len(r) for r in _rings(g))
        newg = _rebuild(g, tol)
        if newg is None:
            dropped += 1
            continue
        n_after += sum(len(r) for r in _rings(newg))
        feats.append({"type": "Feature", "geometry": newg,
                      "properties": f["properties"]})
    return {"type": "FeatureCollection", "features": feats}, \
        {"vertices_before": n_before, "vertices_after": n_after,
         "features_dropped": dropped}


def trim(fc, keep):
    out = []
    for f in fc["features"]:
        p = {k: f["properties"].get(k) for k in keep}
        out.append({"type": "Feature", "geometry": f["geometry"], "properties": p})
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": out}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"))
    return path.stat().st_size


def coverage():
    """How much of Uganda the analysed districts actually cover.

    Areal shares are computed from the committed district areas so the
    manuscript can state coverage honestly instead of implying a national
    census. Also records the ERA5 grid-cell footprint for reference.
    """
    districts = json.loads((C.DATA / "district_boundaries.geojson")
                           .read_text(encoding="utf-8"))["features"]
    lk = {}
    with open(C.DATA / "district_lookup.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lk[row["adm2_name"]] = row
    total = sum(float(f["properties"].get("area_sqkm") or 0) for f in districts)

    def share(names):
        return sum(float(d["properties"].get("area_sqkm") or 0)
                   for d in districts
                   if d["properties"]["adm2_name"] in names)

    analysed = set(lk)
    loc2adm = {row["location"]: row["adm2_name"] for row in lk.values()}
    era5 = set()
    da = C.RESULTS / "tables" / "district_anomaly.csv"
    if da.exists():
        with open(da, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("location") in loc2adm:
                    era5.add(loc2adm[row["location"]])
    per_region = {}
    for f in districts:
        r = str(f["properties"].get("adm1_name"))
        n = f["properties"]["adm2_name"]
        d = per_region.setdefault(r, {"districts": 0, "area_sqkm": 0.0,
                                      "analysed": 0, "analysed_area_sqkm": 0.0})
        d["districts"] += 1
        d["area_sqkm"] += float(f["properties"].get("area_sqkm") or 0)
        if n in analysed:
            d["analysed"] += 1
            d["analysed_area_sqkm"] += float(f["properties"].get("area_sqkm") or 0)
    for d in per_region.values():
        d["analysed_area_share"] = (round(d["analysed_area_sqkm"] / d["area_sqkm"], 4)
                                    if d["area_sqkm"] else None)

    # ERA5 archival grid: 0.25 deg x 0.25 deg (Hersbach et al. 2020).
    deg_lat = 110.574
    lat_mid = 1.3
    cell = round(deg_lat * 0.25 * (111.32 * 0.25 * math.cos(math.radians(lat_mid))), 1)

    obj = {
        "districts_total": len(districts),
        "districts_analysed": len(analysed),
        "districts_era5_panel": len(era5),
        "national_area_sqkm_from_adm2": round(total, 1),
        "analysed_area_sqkm": round(share(analysed), 1),
        "analysed_area_share": round(share(analysed) / total, 4),
        "era5_area_share": round(share(era5) / total, 4) if era5 else None,
        "era5_cell_area_sqkm_at_1p3N": cell,
        "per_region": per_region,
        "geometry_source": "OCHA Uganda COD v01 admin2 (CC BY-IGO), see "
                           "data/boundaries_provenance.json",
    }
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    write_json(C.RESULTS / "district_coverage.json", obj)
    log("wrote results/district_coverage.json: %d/%d districts, %.1f%% of the "
        "area in the COD admin-2 sum" % (len(analysed), len(districts),
                                         100 * obj["analysed_area_share"]))
    return obj


# --------------------------------------------------------------------------
def main(force=False, from_zip=None):
    raw = C.DATA / "raw" / "uga_admin_boundaries.geojson.zip"
    out_gj = C.DATA / "district_boundaries.geojson"
    out_rg = C.DATA / "region_boundaries.geojson"
    out_lk = C.DATA / "district_lookup.csv"
    out_pv = C.DATA / "boundaries_provenance.json"
    if out_gj.exists() and out_lk.exists() and not force:
        log("boundaries already present (%d bytes); --force to rebuild"
            % out_gj.stat().st_size)
        coverage()
        return

    raw.parent.mkdir(parents=True, exist_ok=True)
    url = DATASET_URL
    cache = None
    if from_zip:
        src = Path(from_zip)
        if not src.exists():
            raise SystemExit("--from-zip not found: %s" % src)
        shutil.copyfile(src, raw)
        cache = raw.name          # canonical name, not the caller's path
        pkg = {}
        log("using cached archive " + str(src))
    elif raw.exists() and not force:
        url, pkg = resource_url()
        log("reusing cached download " + str(raw))
    else:
        url, pkg = download(raw)

    digest = sha256(raw)
    if digest != PINNED_SHA:
        log("NOTE: archive sha256 differs from the pinned value; OCHA may have "
            "re-issued the COD file.\n  pinned: %s\n  actual: %s" % (PINNED_SHA, digest))

    names = zipfile.ZipFile(raw).namelist()
    want = {"adm2": None, "adm1": None}
    for n in names:
        low = n.lower()
        if low.endswith("admin2.geojson"):
            want["adm2"] = n
        elif low.endswith("admin1.geojson"):
            want["adm1"] = n
    if not want["adm2"]:
        raise SystemExit("no admin2 layer in archive: %s" % names[:10])

    adm2 = read_member(raw, want["adm2"])
    log("admin2 features: %d (properties %d)"
        % (len(adm2["features"]), len(adm2["features"][0]["properties"])))

    # --- simplify until the committed file is small enough ------------------
    chosen = None
    for tol in TOLERANCES:
        sim, stats = simplify(adm2, tol)
        size = len(json.dumps(trim(sim, KEEP2), separators=(",", ":")).encode())
        log("  tolerance %.3f deg -> %d features, %d vertices, %d KB"
            % (tol, len(sim["features"]), stats["vertices_after"], size // 1024))
        chosen = (tol, sim, stats, size)
        if size <= MAX_BYTES:
            break
    tol, sim, stats, size = chosen

    districts = trim(sim, KEEP2)
    n_bytes = write_json(out_gj, districts)
    log("wrote %s (%.0f KB, %d districts)"
        % (out_gj.relative_to(C.ROOT), n_bytes / 1024, len(districts["features"])))

    regions = None
    if want["adm1"]:
        r1 = simplify(read_member(raw, want["adm1"]), max(tol, 0.005))[0]
        n_bytes_r = write_json(out_rg, trim(r1, KEEP1))
        regions = len(r1["features"])
        log("wrote %s (%.0f KB, %d regions)"
            % (out_rg.relative_to(C.ROOT), n_bytes_r / 1024, regions))

    # --- join the pipeline's own points to the polygons ---------------------
    rows, misses = [], []
    for region, name, lat, lon, elev in C.LOCATIONS:
        hit = None
        for f in districts["features"]:
            if contains(f["geometry"], lon, lat):
                hit = f["properties"]
                break
        if hit is None:
            misses.append(name)
            rows.append({"location": name, "region_config": region,
                         "lat": lat, "lon": lon, "adm2_name": "",
                         "adm2_pcode": "", "adm1_name": "", "area_sqkm": "",
                         "matched": 0})
            continue
        rows.append({"location": name, "region_config": region,
                     "lat": lat, "lon": lon,
                     "adm2_name": hit.get("adm2_name"),
                     "adm2_pcode": hit.get("adm2_pcode"),
                     "adm1_name": hit.get("adm1_name"),
                     "area_sqkm": hit.get("area_sqkm"), "matched": 1})
        if str(hit.get("adm1_name")) != region:
            log("  NOTE region label differs: %s config=%s COD=%s"
                % (name, region, hit.get("adm1_name")))

    out_lk.parent.mkdir(parents=True, exist_ok=True)
    with open(out_lk, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    matched = sum(r["matched"] for r in rows)
    log("lookup: %d/%d config locations inside a district polygon"
        % (matched, len(rows)))
    if misses:
        raise SystemExit("unmatched locations: %s" % misses)

    # A cached rebuild must still record what the register says about itself.
    if not pkg:
        try:
            _, pkg = resource_url()
        except Exception as exc:
            log("NOTE: HDX metadata unavailable (%r); provenance will carry "
                "only the fields read from the archive" % (exc,))

    prov = {
        "dataset": pkg.get("title") or "Uganda - Subnational Administrative Boundaries",
        "hdx_id": DATASET,
        "hdx_last_modified": pkg.get("last_modified"),
        "hdx_license_id": pkg.get("license_id"),
        "publisher": "OCHA (Common Operational Dataset, UNOSAT mapping)",
        "url": url,
        "local_cache_archive": cache,
        "license": "CC BY-IGO (Creative Commons Attribution for Intergovernmental Organisations)",
        "license_url": "https://humanitarianopendata.osdapp.org/licence/",
        "version": districts["features"][0]["properties"].get("version"),
        "valid_on": districts["features"][0]["properties"].get("valid_on"),
        "retrieved": str(date.today()),
        "archive_sha256": digest,
        "pinned_sha256": PINNED_SHA,
        "levels": {"adm1_regions": regions, "adm2_districts": len(districts["features"])},
        "simplification": {"method": "Douglas-Peucker, exterior rings, "
                                      "coordinates rounded to 4 dp",
                           "tolerance_deg": tol,
                           "max_area_kept_deg2": 1e-6, **stats},
        "join": {"locations": len(rows), "matched": matched,
                 "method": "point-in-polygon on the config centroid"},
        "note": "The 2020-08-24 district geography post-dates parts of the "
                "2008-2023 analysis window; several districts in this file were "
                "created by split-offs after 2010. Point values are assigned to "
                "the district that contains the point today.",
    }
    write_json(out_pv, prov)
    log("wrote %s" % out_pv.relative_to(C.ROOT))
    coverage()
    log("done. boundary artifacts are in data/ and are committed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="rebuild even if present")
    ap.add_argument("--from-zip", default=None,
                    help="use a local copy of uga_admin_boundaries.geojson.zip")
    a = ap.parse_args()
    main(force=a.force, from_zip=a.from_zip)
