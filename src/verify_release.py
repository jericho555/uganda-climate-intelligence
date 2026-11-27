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


def check_numbers() -> None:
    """Assertions that the numbers quoted in the text are the released ones.

    The manuscript promises no value is hard-coded. That is only true if the
    headline figures still match the artefacts they came from, so the counts,
    spans, test statistics, conformal widths and correlations quoted in the text
    are re-derived here. This is the check that catches a re-run that moved a
    number without anyone editing the prose.
    """
    tex = read_text(C.MANUSCRIPT / "main.tex")

    def latex_int(v: int) -> str:
        return f"{v:,}".replace(",", "{,}")

    # ---- benchmark size and the ranking test built on it -----------------
    bench = C.RESULTS / "benchmark.csv"
    check("benchmark table artefact exists", bench.exists())
    if not bench.exists():
        return
    rows = column_values(bench, "detector")
    n = len(rows)
    fr = json.loads(read_text(C.RESULTS / "friedman.json"))
    check("friedman detector count matches the benchmark rows",
          fr["n_detectors"] == n, f"csv {n} vs json {fr['n_detectors']}")
    check("the text states the benchmarked detector count",
          f"benchmark of {n} detectors" in tex
          and f"over {n} detectors" in tex
          and f"All {n} methods" in tex,
          f"expected '{n} detectors' / 'All {n} methods'")
    added = {"LoOP", "TemporalIF"}
    check("the two detectors added at revision are benchmarked and named",
          added <= set(rows) and all(d in tex for d in added),
          "missing from csv: " + ", ".join(sorted(added - set(rows))))
    roc = dict(zip(rows, column_values(bench, "roc_auc")))
    pr = dict(zip(rows, column_values(bench, "pr_auc")))
    nonseq = [r for r in rows if r not in ("LSTMAE", "TransformerAE")]
    check("non-sequence detector count implied by the text is right",
          f"the {len(nonseq)}\nnon-sequence methods" in tex
          or f"the {len(nonseq)} cross-sectional methods" in tex,
          f"{len(nonseq)} non-sequence of {n}")
    lo, hi = min(float(roc[r]) for r in nonseq), max(float(roc[r]) for r in nonseq)
    check("the ROC-AUC span quoted in the text is the table's",
          f"{lo:.3f}--{hi:.3f}" in tex, f"{lo:.3f}-{hi:.3f}")
    plo, phi = min(float(pr[r]) for r in nonseq), max(float(pr[r]) for r in nonseq)
    check("the PR-AUC span quoted in the text is the table's",
          f"{plo:.3f}--{phi:.3f}" in tex, f"{plo:.3f}-{phi:.3f}")
    chi2, p = fr["friedman_chi2"], fr["friedman_p"]
    check("the Friedman chi-square in the text is the artefact's",
          f"$\\chi^2={chi2:.2f}$" in tex, f"{chi2:.2f}")
    pm, _, pe = f"{p:.1e}".partition("e")
    check("the Friedman p-value in the text is the artefact's",
          f"p={pm}\\times10^{{{int(pe)}}}" in tex, f"{p:.1e}")

    # ---- split-conformal prediction intervals ---------------------------
    ews = json.loads(read_text(C.RESULTS / "early_warning_summary.json"))
    pi = ews.get("prediction_intervals") or {}
    mdl = pi.get("models") or {}
    check("the nowcast summary publishes fitted prediction intervals",
          {"persistence", "ridge"} <= set(mdl),
          "keys: " + ", ".join(sorted(mdl)))
    if {"persistence", "ridge"} <= set(mdl):
        check("the transition counts quoted in the text are the conformal ones",
              all(latex_int(pi[k]) in tex for k in
                  ("n_transitions", "n_calibration", "n_test")),
              f"{pi['n_transitions']}/{pi['n_calibration']}/{pi['n_test']}")
        per, rid = mdl["persistence"], mdl["ridge"]
        check("both conformal RMSE values appear in the text",
              f"RMSE {per['rmse_test']}" in tex and f"{rid['rmse_test']}" in tex,
              f"{per['rmse_test']} / {rid['rmse_test']}")
        widths = [f"$\\pm{mdl[m][lvl]['conformal_quantile']}$"
                  for m in mdl for lvl in ("level_90", "level_80")]
        check("every conformal width quoted in the text is the fitted one",
              all(w in tex for w in widths), ", ".join(widths))
        covs = [f"{100 * mdl[m][lvl]['coverage_test']:.1f}\\%"
                for m in mdl for lvl in ("level_90", "level_80")]
        check("every measured conformal coverage figure is in the text",
              all(c in tex for c in covs), ", ".join(covs))
        band = per["level_90"]["coverage_by_starting_band"]
        check("per-band persistence coverage quoted in the text is the artefact's",
              all(f"{100 * band[b]['coverage']:.1f}\\%" in tex for b in band),
              ", ".join(f"{b}={band[b]['coverage']}" for b in band))
        check("the verdict does not oversell the better-fitting model's band",
              "conservative" in (pi.get("verdict") or "").lower(),
              (pi.get("verdict") or "")[:60])

    # ---- World Bank climate-portal cross-check --------------------------
    for rel in ["wb_climate.csv", "wb_climate_meta.json"]:
        check(f"World Bank input present: data/{rel}", (C.DATA / rel).exists())
    wb_p = C.RESULTS / "wb_climate_check.json"
    check("World Bank cross-check artefact exists", wb_p.exists())
    if wb_p.exists():
        wb = json.loads(read_text(wb_p))
        ag = wb["agreement_vs_ours_spei6_mean"]
        check("the World Bank SPEI correlation quoted in the text is the measured one",
              f"$r={ag['pearson_r']:.3f}$" in tex, f"r={ag['pearson_r']}")
        check("its significance is quoted as measured",
              f"$p={ag['pearson_p']:.2f}$" in tex, f"p={ag['pearson_p']:.2f}")
        check("the abstract rounds the same correlation",
              f"$r={ag['pearson_r']:.2f}$" in tex, f"r={ag['pearson_r']:.2f}")
        dry = wb["dry_year_check"]
        check("the Bank's severe-drought years named in the text are the artefact's",
              all(str(y) in tex for y in dry["wb_years_le_minus_1"]),
              str(dry["wb_years_le_minus_1"]))
        spei = (json.loads(read_text(C.DATA / "wb_climate_meta.json"))
                .get("EN.CLC.SPEI.XD", {}).get("values", {}))
        check("no register drought year is dry on the Bank's national index",
              spei and all(spei.get(str(y)) is not None and spei[str(y)] > 0
                           for y in dry["register_drought_years"]),
              str(dry["register_drought_years"]))
        pc = wb["precip_normal_check"]
        check("the precipitation scale check quoted in the text is the measured one",
              latex_int(int(round(pc["era5_mean_annual_mm"]))) in tex
              or latex_int(pc["era5_mean_annual_mm"]) in tex,
              f"{pc['era5_mean_annual_mm']} mm vs normal {pc['wb_normal_mm']}")
    for rel in ["wb_spei_compare.csv", "wb_precip_compare.csv"]:
        check(f"World Bank comparison table present: results/tables/{rel}",
              (C.RESULTS / "tables" / rel).exists())
    check("the World Bank figure is drawn",
          (C.FIGURES / "fig_wb_check.png").exists()
          and "figures/fig_wb_check.png" in tex)
    check("the conformal-interval figure is drawn",
          (C.FIGURES / "fig_intervals.png").exists()
          and "figures/fig_intervals.png" in tex)


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
    check("results digest exports the drought EW summary",
          digest.get("drought_ew_summary") is not None)
    check("results digest exports the yield forecast summary",
          digest.get("yield_forecast_summary") is not None)
    check("results digest exports the seasonal outlook summary",
          digest.get("seasonal_outlook_summary") is not None)

    # ---- forward-looking early-warning artefacts -------------------------
    for tbl in ["drought_early_warning.csv", "yield_forecast.csv",
                "seasonal_outlook.csv"]:
        check(f"early-warning table present: results/tables/{tbl}",
              (C.RESULTS / "tables" / tbl).exists())
    for js in ["drought_ew_summary.json", "yield_forecast_summary.json",
               "seasonal_outlook_summary.json"]:
        check(f"early-warning summary present: results/{js}",
              (C.RESULTS / js).exists())
    for fig in ["fig_drought_ew.png", "fig_yield_forecast.png",
                "fig_seasonal_outlook.png"]:
        check(f"early-warning figure present: figures/{fig}",
              (C.FIGURES / fig).exists())
    # Bulletin
    bulletins = list(C.REPORTS.glob("climate_risk_bulletin_*.md"))
    check("automated climate risk bulletin exists in reports/",
          len(bulletins) >= 1,
          f"{len(bulletins)} bulletin(s)")

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
    check("dashboard exposes the fourteen panels the paper describes",
          len(labels) == 14, ", ".join(labels))
    texts = "\n".join(str(getattr(m, "value", "")) for m in at.info)
    texts += "\n".join(str(getattr(c, "value", "")) for c in at.caption)
    texts += "\n".join(str(getattr(w, "value", "")) for w in at.warning)
    check("no tab reports a missing artefact",
          "missing. Run" not in texts and "not built yet" not in texts)

    # The rendered page must say what the artefacts say.  ``app.py`` used to
    # carry hand-typed counts ("24 detectors", "chi2(23)", "five ... 1.22x")
    # that a re-run silently invalidated, so the numbers on screen are compared
    # with the files, and the stale literals are refused outright.
    src = read_text(C.ROOT / "dashboard" / "app.py")
    body = "\n".join(l for l in src.splitlines()
                     if not l.lstrip().startswith("#"))
    bench_n = len(column_values(C.RESULTS / "benchmark.csv", "detector"))
    stale = [s for s in (f"{bench_n} detectors", f"{bench_n}-detector",
                         "\u03c7\u00b2(23)", "Full 24-row", "five cross-sectional",
                         "1.22")
             if s in body]
    check("no stale hand-typed count survives in the dashboard prose",
          not stale, ", ".join(stale))
    check("dashboard takes the detector count from the artefact",
          "N_DET = len(_BENCH0)" in body and "{N_DET} detectors" in body
          and "{len(bench)} detectors" in body,
          f"benchmark.csv holds {bench_n} rows")
    for need in ("wb_climate_check.json", "wb_spei_compare.csv",
                 "wb_precip_compare.csv", "nowcast_interval.csv",
                 "prediction_intervals", "fig_wb_check.png",
                 "fig_intervals.png"):
        check(f"dashboard renders the {need} artefact", need in body)
    check("dashboard renders the conformal verdict and the World Bank finding",
          "Verdict as written by the pipeline" in texts
          and "does not validate the index" in texts)
    return True


def main() -> None:
    print("== figures ==")
    check_figures()
    print("== manuscript ==")
    check_manuscript()
    print("== numbers in the text vs the artefacts ==")
    check_numbers()
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
