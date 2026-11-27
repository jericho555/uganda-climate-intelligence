# Changelog

All notable changes to the Uganda Climate Intelligence Framework (UCIF).

## [Unreleased]

### Added
- **District geometry (`src/fetch_boundaries.py`):** the study area is now
  resolved into real polygons from the OCHA Uganda Common Operational Dataset
  (`cod-ab-uga`, 135 admin-2 districts, licence CC BY-IGO, archive pinned by
  sha256). Committed artefacts: `data/district_boundaries.geojson` (217 KB
  topology-preserving subset, 434,427 → 10,869 vertices),
  `data/region_boundaries.geojson`, `data/district_lookup.csv` (all 19 sites fall
  inside exactly one district; Luweero→Luwero, FortPortal→Kabarole) and
  `data/boundaries_provenance.json`. `results/district_coverage.json` quantifies
  what the panel represents: 19 of 135 districts, 13.7% of national district area
  (ERA5 sub-panel 16 districts, 11.0%).
- **Region-label sensitivity (`src/region_label_check.py`):** the boundary
  register places Masindi in the Western region while `src/config.py` registers
  it as Northern. Because event exposure is defined by region
  (`preprocess.add_labels`), the effect is measured on persisted artefacts rather
  than argued: 1,036 of 93,504 district-days would change label (6.16% of the
  current positives), 21 of 32 events change exposure, and the count of districts
  affected by any one event shifts by at most one. Committed as
  `results/region_label_sensitivity.json`. No published number was re-derived.
- **Reference-index check (`src/check_ref_index.py`):** the refuted baseline
  draft is now cited as what it is — an author-held unpublished technical draft.
  Nine answered queries to Crossref, OpenAlex and arXiv (plus one rate-limited
  Semantic Scholar attempt, recorded as unanswered) find no record of it;
  `results/masaba2023_index_check.json` also lists the closest indexed Ugandan
  weather records and their authors as a positive control. The script exits
  rather than emit a negative verdict when a required index does not answer.
- **Torch sequence detectors (RQ1, benchmark now 24):** a CPU-only `torch 2.4.1`
  distribution is now vendored into the workspace (`pip --target=._torchpkgs`),
  enabling a genuine LSTM encoder–decoder autoencoder and a Transformer encoder
  autoencoder over 14-day windows. These are the first detectors to roughly
  double the size-matched random null (F1 0.13–0.14, precision ≈2× base rate),
  though global ranking stays near chance (ROC-AUC 0.51–0.53): an honest
  refinement of the central claim.
- **Measured random null (`src/random_null.py`):** the size-matched random-flag
  baseline is now computed from `results/test_scores.npz` (20,000 Monte-Carlo
  draws at the 5% alert budget, plus per-year nulls, a PR-AUC null and
  Benjamini–Hochberg-controlled one-sided tests) and released as
  `results/random_null.json`. Wired into `run_all.py` after `detect`.
- **Map figures (`src/figures_maps.py`):** study-area reference map, district
  risk map panel (anomaly rate with Gi* hotspot, precipitation trend, UCASI
  nowcast), gap-fill trend maps and a null-comparison panel. All are drawn from
  persisted artefacts.
- **Interactive dashboard:** a Streamlit app (`dashboard/`) exposing the
  benchmark, daily anomaly stream, spatial hotspots, UCASI severity, trends and
  risk nowcast, bound to the persisted artefacts. Launched via
  `python dashboard/run_dashboard.py` with a vendored `._webpkgs` install.

### Changed
- **Title:** now *A National Climate Anomaly Intelligence System for Uganda:
  Ground-Truth Validation, Risk Nowcasting, and Long-Term Climate Change
  Assessment*, naming the three things the paper actually delivers and the
  system-level claim the evidence supports. The subtitle no longer enumerates
  method counts (24 detectors, three substrates), which read as a capability
  list; "ground-truth-honest" is retained in the prose where the argument is
  made. Updated in `main.tex` (title and running title), `COVER_LETTER.md`,
  `README.md`, `references.bib`, `CITATION.cff` and `.zenodo.json`.
- **Every map is now a choropleth:** `fig_study_area`, `fig_hotspot_map`,
  `fig_risk_maps` and `fig_gapfill_maps` shade district polygons instead of
  plotting circles, with an explicit no-data class (116 districts the pipeline
  never observes are grey, never zero), hatching for non-significant districts
  and a thick outline for the one Gi* hotspot. `figures.py:fig_map` delegates to
  the shared geometry layer so no figure in the package implies a radius of
  influence. Figure footers carry the boundary attribution.
- **`masaba2023` reclassified in `references.bib`** from `@misc` with a
  "details to be supplied by the author" placeholder to `@unpublished`, and the
  Abstract, Introduction, Related work, Results and Discussion now identify it as
  our own unpublished draft instead of implying a citable prior benchmark. The
  cover letter's unverifiable "most cited such result for Uganda" claim was
  removed; the Highlights bullet now says whose baseline is being refuted.
- **Related work rebuilt around the threads that carried no evidence:** the
  section grew from 763 to about 1,490 words and gained four new paragraphs
  (operational warning practice and the missing verification standard, applied
  machine learning for hazard monitoring, disaster registers as data products,
  reproducible environmental software and decision support) plus an explicit
  "gap and position" close. Eleven verified sources were added (58 bib entries,
  30 from 2020+), including two from the target journal (Janssen et al. 2020 on
  code sharing; Zulkafli et al. 2017 on user-driven decision support), Guzzetti
  et al. 2020 on geographical landslide early warning, a companion Bududa
  assessment (Namwano et al. 2026), two applied hazard-ML studies (Hadi et al.
  2024; Alemu and Zimale 2025), EM-DAT methodology and geocoding critiques
  (Delforge et al. 2025; Teber et al. 2025), and the verification/value anchors
  (Gneiting et al. 2007; Thompson and Brier 1955; Reichstein et al. 2019).
  Every statement about a cited study is grounded in its Crossref record or
  published abstract; nothing was written from memory.
- **Unearned novelty claims scoped:** the Introduction no longer says "the first
  such national system", and the benchmark contribution is now qualified "to our
  knowledge". A Discussion paragraph adds the comparison the protocol invites but
  could not honestly run (a rainfall-threshold baseline), naming the missing
  inventory as the reason. The cover letter gained the same positioning, and
  "a novel composite severity index" lost the word "novel". No table was added:
  a new table before the data table would renumber Table 1, which is referenced
  inside a generated figure caption that cannot be regenerated here.
- **Retitled around the evidence, not the capability list (EMS):**
  *A ground-truth-honest climate-anomaly detection system for Uganda: a
  24-detector held-out benchmark, district risk nowcasting, and century-scale
  climatology from three observation substrates*. The previous title advertised
  "Detection ... Impact Analysis and Explainability", which are the three
  components whose measured verdict is chance-level ranking, confounded impact
  linkage and detector-only attribution, and its four-noun subtitle carried no
  finding. Propagated to `manuscript/main.tex` (title, header comment and the
  abstract's opening claim), `COVER_LETTER.md`, `README.md` and the
  `references.bib` header. No metric, table or figure value changed.
- **Manuscript reframed as a national system (EMS positioning):** retitled to
  *A National Climate Anomaly Intelligence System for Uganda*; the six
  capabilities (anomaly detection, spatial intelligence, district risk nowcast,
  agricultural impact, explainability, interactive dashboard) are now stated
  explicitly, with a new subsection on the decision-support dashboard and an
  added contribution item. Framing verified to preserve the honest
  nowcast/impact results — no metric, table or figure value changed.

### Fixed
- **Packaging metadata out of sync:** `CITATION.cff` and `.zenodo.json` still
  quoted the original conference-paper title ("A Climate Intelligence Framework
  for Detecting, Validating and Assessing Weather Anomalies in Uganda") while
  shipping the retitled manuscript. Both now quote the current manuscript title;
  the Zenodo keyword list gains `early warning` and `risk nowcasting`.
- **Bibliography integrity (critical):** six cited works could not be verified
  against any indexed record and were removed — `rietsch2020`, `gunderson2022`,
  `boyd2022`, `wilkinson2023`, `khan2021` and the `white2023power` attribution.
  They are replaced by verified sources: Schmidl, Wenig & Papenbrock (2022)
  PVLDB 15(9):1779–1797; Mugagga, Kakembo & Buyinza (2012) Catena 90; Yue &
  Wang (2004) Water Resour. Manag. 18(3):201–218; and a landing-page dataset
  record for NASA POWER. Metadata errors corrected on the real entries
  (`zhao2019` author list, `chandola2009` title, `ord1995` → Getis & Ord 1992,
  `hirsch1984` dropped). Added the missing primary references for every method
  actually used (generalised ESD, Isolation Forest, LOF, one-class SVM,
  PELT, SHAP, scikit-learn, PyOD). The `masaba2023` entry is marked as pending
  author-supplied publication details.
- **Unsupported random-null value (critical):** the previously quoted
  size-matched random null of F1 ≈ 0.096 had no generating script anywhere in the
  repository. Recomputed from the held-out labels it is **0.069 [0.059, 0.079]**
  (analytic check 0.0690 at base rate 0.11155 and γ = 0.05; PR-AUC null 0.112
  [0.109, 0.116]). Consequence for the wording of the central claim, now applied
  consistently in `manuscript/main.tex`, `README.md`, `dashboard/` and the cover
  letter: five cross-sectional detectors remain nominally above the equal-budget
  null but by at most 1.22× (BH-significant: OneClassSVM, LOF, ECOD, COPOD, PCA),
  and only the two sequence autoencoders roughly double it. The dashboard reads
  the value from the artefact instead of hard-coding it, and the manuscript
  discloses that the 68.8% event coverage comes from the transductive
  operational surface and is therefore an upper bound.
- **Citation recency:** the reference list now favours current sources where the
  claim is not foundational (Blázquez-García 2021; Schmidl 2022; Gleixner 2020;
  Namwano 2024; WMO EW4All 2023; EM-DAT documentation; Venning 2025; Lam 2023),
  retaining older citations only for the methods themselves (STL, Sen, Mann–Kendall,
  LOF, PCA, SPI/SPEI, Getis–Ord, PELT, SHAP). All newly added entries were checked
  against the publisher record; unverifiable candidates were rejected rather than
  cited approximately.

## [1.0.0] - 2026-11-26

First public, citable release of the reproducible pipeline and manuscript.

### Added
- **Detection benchmark (RQ1):** held-out evaluation of unsupervised anomaly
  detectors against an external disaster register, with bootstrap CIs and a
  size-matched random null showing apparent coverage is a window-size artefact.
- **Spatial risk (RQ2):** Global Moran's I and Getis–Ord Gi* district hotspots.
- **Explainability (RQ3):** SHAP + permutation importance for the detection signal.
- **Climatology (RQ4):** STL seasonal strength, Mann–Kendall + Sen's slope,
  PELT change points, multi-seed F1 stability.
- **Century in-situ record:** Meteostat bulk daily series (12 stations, 1926–2026)
  driving long trends and an ERA5↔station cross-validation (r = 0.95–0.999).
- **Independent gap-fill (NASA POWER / MERRA-2):** daily T2M / RH / shortwave for
  19 districts over 1988–2007, closing the 1990–2007 in-situ reporting collapse;
  19/19 districts warm and 15/19 dry significantly.
- **Derived drought indices (SPI/SPEI):** gamma / Pearson-III indices that
  independently confirm 3 of 4 documented register droughts.
- **Agricultural impact linkage:** 64-year World Bank/FAOSTAT cereal yield vs
  growing-season drought, reported as an honest two-sided result.
- **UCASI** severity index and **RQ5** event-recall / lead-time analysis.
- Reproducibility scaffolding: `run_all.py`, `requirements.txt`, `LICENSE` (MIT),
  `CITATION.cff`, `.zenodo.json`, `.gitignore`, this changelog.

### Notes
- Every reported number is computed by the pipeline; no values are fabricated.
- Large raw/derived data files are gitignored and regenerated by `run_all.py`.
- A citable Zenodo DOI is minted by publishing a GitHub release with the Zenodo
  bridge enabled (see README "Citing & archiving").
