"""Pre-submission release gate: verify the bundle, not just the code.

Usage::

    python src/verify_release.py [--skip-dashboard]

Exit status is non-zero if any check fails, and ``run_all.py`` runs this as its
last stage so a broken bundle aborts the pipeline instead of printing a
``[done ...]`` line.

Every check here is a claim the manuscript or the README makes about its own
artefacts -- that the figures are mirrored and 300 dpi, that no reference is
orphaned, that the local-attribution matrix agrees with the rows it explains,
that the district arithmetic on the maps adds up. Each one is asserted rather
than printed as a warning, because a warning in a log is exactly how a stale
figure or a broken cross-reference reaches a submission.
"""
from __future__ import annotations

import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                     # noqa: E402

FAILS: list[str] = []
N = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global N
    N += 1
    print(("PASS  " if ok else "FAIL  ") + name
          + (f"   [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def png_dpi(path: Path):
    """Return (x_dpi, y_dpi) from a PNG pHYs chunk, or None if absent.

    Read straight out of the file header so the check needs no imaging library:
    matplotlib writes pHYs before IDAT, so scanning stops there.
    """
    b = path.read_bytes()
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    i = 8
    while i + 8 <= len(b):
        ln = struct.unpack(">I", b[i:i + 4])[0]
        typ = b[i + 4:i + 8].decode("latin1")
        if typ == "pHYs":
            x, y, unit = struct.unpack(">IIB", b[i + 8:i + 17])
            return (round(x / 39.3701), round(y / 39.3701)) if unit == 1 else None
        if typ == "IDAT":
            return None
        i += 12 + ln
    return None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def column_values(path: Path, col: str):
    """One column of a small CSV, without pulling in pandas."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return [r[col] for r in csv.DictReader(fh)]


# --------------------------------------------------------------------------- #
def check_figures() -> None:
    a = {p.name: p for p in C.FIGURES.glob("*.png")}
    b = {p.name: p for p in (C.MANUSCRIPT / "figures").glob("*.png")}
    check("figure folders hold the same set of PNGs",
          set(a) == set(b), f"{len(a)} vs {len(b)}")
    same = [k for k in set(a) & set(b)
            if a[k].read_bytes() != b[k].read_bytes()]
    check("mirrored figures are byte-identical", not same,
          "drifted: " + ", ".join(sorted(same)) if same else f"{len(a)} files")
    low = []
    for k, p in sorted(b.items()):
        d = png_dpi(p)
        if d is None or min(d) < C.FIG_DPI:
            low.append(f"{k}={d}")
    check(f"every manuscript figure is >= {C.FIG_DPI} dpi", not low,
          ", ".join(low) if low else f"{len(b)} files")


def check_manuscript() -> None:
    tex_path = C.MANUSCRIPT / "main.tex"
    tex = read_text(tex_path)
    begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", tex))
    ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", tex))
    unbalanced = {k: (begins[k], ends[k]) for k in set(begins) | set(ends)
                  if begins[k] != ends[k]}
    check("LaTeX environments balanced", not unbalanced, str(unbalanced))

    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:ref|eqref|Cref|cref|autoref)\{([^}]+)\}", tex))
    orphan = sorted(r for r in refs if r not in labels)
    check("every cross-reference resolves", not orphan,
          "missing: " + ", ".join(orphan) if orphan else f"{len(refs)} refs")

    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    absent = [f for f in figs if not (C.MANUSCRIPT / f).exists()]
    check("every \\includegraphics target is in the bundle", not absent,
          "absent: " + ", ".join(absent) if absent else f"{len(figs)} figures")

    bib = re.search(r"\\bibliography\{([^}]*)\}", tex)
    bibs = [b.strip() + ".bib" for b in bib.group(1).split(",")] if bib else []
    check("every declared .bib file is present",
          all((C.MANUSCRIPT / b).exists() for b in bibs), ", ".join(bibs))

    entries = read_text(C.MANUSCRIPT / "references.bib")
    keys = set(re.findall(r"@\w+\{([^,\s]+),", entries))
    cited = set(re.findall(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    cited = {c.strip() for grp in cited for c in grp.split(",")}
    check("no citation points at a missing bib entry",
          not (cited - keys), "unknown: " + ", ".join(sorted(cited - keys)))

    bad = [i + 1 for i, l in enumerate(tex.splitlines())
           if any(ch in l for ch in ("\u2014", "\u2013", "\u2212", "\u00a0"))]
    check("no unicode dashes/nbsp left in the source", not bad,
          f"line(s) {bad[:6]}" if bad else "")


def check_artefacts() -> None:
    """Claims the paper makes that are cheap to falsify against the JSON."""
    ev_p = C.RESULTS / "shap_expected_value.json"
    if ev_p.exists():
        ev = json.loads(read_text(ev_p))
        check("persisted local attributions are additive for the surrogate",
              ev["additivity_max_abs_error"] < 1e-9,
              f"max error {ev['additivity_max_abs_error']:.1e}")
        import numpy as np
        sv = np.load(C.RESULTS / "shap_values.npy")
        check("attribution matrix covers the persisted rows",
              sv.shape[0] == ev["n_rows"],
              f"{sv.shape[0]} rows vs n_rows {ev['n_rows']}")
        out = C.RESULTS / "shap_row_outputs.npy"
        check("surrogate outputs stored alongside the attributions",
              out.exists() and np.load(out).shape[0] == ev["n_rows"])
    else:
        check("local-attribution artefact exists", False,
              "run src/_explain_fast.py")

    ews_p = C.RESULTS / "early_warning_summary.json"
    ews = json.loads(read_text(ews_p)) if ews_p.exists() else {}
    cu = ews.get("carry_uncertainty") or {}
    check("nowcast carries a measured uncertainty envelope",
          bool(cu.get("all_district_months")) and bool(cu.get("from_high_or_extreme")))

    digest = json.loads(read_text(C.RESULTS / "results_digest.json"))
    check("results digest exports the nowcast summary",
          digest.get("early_warning_summary") is not None)

    geo = json.loads(read_text(C.DATA / "district_boundaries.geojson"))
    cov = json.loads(read_text(C.RESULTS / "district_coverage.json"))
    feats = len(geo["features"])
    total = cov["districts_total"]
    panel = cov["districts_era5_panel"]
    sites = cov["districts_analysed"]
    check("geometry count matches the coverage artefact", feats == total,
          f"{feats} polygons vs districts_total {total}")
    # The choropleth legend promises three classes -- scored, site-without-score,
    # never observed. Recompute them the way the map does and make sure they
    # partition the districts, so "no score" can never be painted as "no data".
    lookup = dict(zip(column_values(C.DATA / "district_lookup.csv", "location"),
                      column_values(C.DATA / "district_lookup.csv", "adm2_name")))
    scored = {lookup[s] for s in
              column_values(C.RESULTS / "tables" / "district_anomaly.csv",
                            "location") if s in lookup}
    site_names = set(lookup.values())
    gap = site_names - scored
    check("map class: scored districts", len(scored) == panel,
          f"{len(scored)} shaded vs districts_era5_panel {panel}")
    check("map class: site districts", len(site_names) == sites,
          f"{len(site_names)} sites vs districts_analysed {sites}")
    check("map class: site without a panel score", len(gap) == sites - panel,
          f"{len(gap)}: " + ", ".join(sorted(gap)))
    check("map classes partition the districts",
          panel + len(gap) + (total - len(site_names | scored)) == total,
          f"{panel} + {len(gap)} + {total - len(site_names | scored)} = {total}")


def check_dashboard() -> bool:
    """Render every tab in-process and require a clean run. Returns False if
    the optional web stack is not installed."""
    for vend in (C.ROOT / "._webpkgs",):
        if vend.exists() and str(vend) not in sys.path:
            sys.path.insert(0, str(vend))
    # An ambient ~/.streamlit/config.toml with logger.level="debug" otherwise
    # buries this report in cache traces, so DEBUG records are suppressed for the
    # duration of the render and restored immediately afterwards.
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:
        print("SKIP    dashboard (streamlit not importable)")
        return False
    import logging
    logging.disable(logging.DEBUG)
    try:
        at = AppTest.from_file(str(C.ROOT / "dashboard" / "app.py"),
                               default_timeout=300)
        at.run()
    finally:
        logging.disable(logging.NOTSET)
    check("dashboard renders every tab without an exception",
          not at.exception and not at.error,
          "; ".join(str(e.value)[:200] for e in at.exception) if at.exception else
          f"{len(at.tabs)} tabs, {len(at.metric)} metrics")
    labels = [getattr(t, "label", "") for t in at.tabs]
    check("dashboard exposes the ten panels the paper describes",
          len(labels) == 10, ", ".join(labels))
    texts = "\n".join(str(getattr(m, "value", "")) for m in at.info)
    texts += "\n".join(str(getattr(c, "value", "")) for c in at.caption)
    check("no tab reports a missing artefact",
          "missing. Run" not in texts and "not built yet" not in texts)
    return True


def main() -> None:
    print("== figures ==")
    check_figures()
    print("== manuscript ==")
    check_manuscript()
    print("== artefacts ==")
    check_artefacts()
    if "--skip-dashboard" not in sys.argv:
        print("== dashboard ==")
        check_dashboard()
    print(f"\n{N - len(FAILS)}/{N} checks passed")
    if FAILS:
        # Raises SystemExit, so run_all.py stops loudly rather than reporting a
        # broken bundle as a completed stage.
        raise SystemExit("RELEASE CHECKS FAILED: " + "; ".join(FAILS))


if __name__ == "__main__":
    main()
