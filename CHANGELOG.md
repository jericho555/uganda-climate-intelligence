# Changelog

All notable changes to the Uganda Climate Intelligence Framework (UCIF).

## [Unreleased]

### Added
- **Forward-looking climate risk layers (4 phases):**
  - **Drought Early Warning (`src/drought_early_warning.py`):**
    threshold-based district status (Normal / Watch / Warning / Alert) from
    SPI-6 severity, trajectory (improving / stable / deteriorating) and
    consecutive-decline runs (≥3 months). Validated against the disaster
    register: 34 hits, 0 misses, 100% hit rate across 3,072 district-months.
    Artefacts: `results/tables/drought_early_warning.csv`,
    `results/drought_ew_summary.json`, `figures/fig_drought_ew.png`.
  - **Yield Impact Forecast (`src/yield_forecast.py`):** national cereal-yield
    anomaly regressed against growing-season SPI-3 (Mar–May + Sep–Nov). Honest
    skill check: R² = 0.076, Pearson r = 0.275 (p = 0.302), climatology beats
    the model (RMSE ratio = 1.308). 2023 projection: −0.19% anomaly, 95% PI
    [−30.1, 29.7]%, P(below-normal) = 13.5%. The negative skill result is
    reported transparently, not hidden.
    Artefacts: `results/tables/yield_forecast.csv`,
    `results/yield_forecast_summary.json`, `figures/fig_yield_forecast.png`.
  - **Seasonal Climate Outlook (`src/fetch_seasonal.py`,
    `src/seasonal_outlook.py`):** downscaled ECMWF SEAS5 precipitation forecast
    via the Open-Meteo Seasonal Forecast API (free, no auth, CC-BY 4.0).
    Graceful degradation: if the API is unreachable, downstream modules report
    "outlook unavailable" rather than crashing. When available, classifies each
    district-month as below / near / above normal and integrates with the
    drought EW tiers (escalation rule).
    Artefacts: `results/tables/seasonal_outlook.csv`,
    `results/seasonal_outlook_summary.json`, `figures/fig_seasonal_outlook.png`.
  - **Automated Risk Bulletin (`src/bulletin.py`):** narrative Markdown report
    synthesising all forward-looking layers. Every number is read from the
    summary JSONs — nothing is hard-coded, so the bulletin cannot drift from
    the pipeline. Output: `reports/climate_risk_bulletin_YYYY_MM.md`.
- **Dashboard: 3 new tabs (10 → 13):** Drought Early Warning (choropleth +
  status metrics + validation), Yield Forecast (model skill + projection +
  scatter figure), Seasonal Outlook (choropleth + escalated status or
  graceful "unavailable" message).
- **Release gate: 13 new checks (61 → 74):** drought EW / yield / seasonal
  tables, summaries and figures; results digest exports all three new
  summaries; bulletin existence; dashboard tab count updated to 13.
- **Manuscript: dedicated methods + results for forward-looking layers.**
  Four new Methods subsections (\S\ref{sec:dew}--\S\ref{sec:bulletin}):
  drought early warning (SPI-6 threshold tiers, trajectory, decline run),
  yield impact forecast (national SPI-3 regression with chronological skill
  check), seasonal climate outlook (SEAS5 downscale + EW integration), and
  automated risk bulletin. One new Results subsection (\S\ref{sec:forward}):
  honest numbers for all four layers including the negative yield skill.
  Three new figures (fig_drought_ew, fig_yield_forecast, fig_seasonal_outlook).
  Abstract and contributions list updated. Cross-refs 45 → 53, figures 19 → 22.
- **Two missing detectors (`src/deep_models.py`, `src/detect.py`):** the benchmark
  promised a local-density member and a window-aware isolation forest and had
  neither. `LoOP` is implemented from Kriegel et al. (2008) — mean kNN distance,
  maximum local reachability distance, POD and the probabilistic deviation with
  β = 0.02 — and ranked on ROD itself, because the published 2Φ(ROD/λ)−1 map
  saturates and collapses inliers into one tied block at the 5% alert cut-off; the
  threshold-quantile ranking is identical on either. `TemporalIF` is an Isolation
  Forest fitted on flattened 14-day lag-window embeddings of the three series the
  sequence models see, trained on the training years only. Both run at 26 detectors
  (`χ² = 33.78`, `p = 8.2e-7`, `results/friedman.json`); both join the chance-level
  group and are reported that way (LoOP ROC 0.491, F1 0.99× the null, q = 0.57;
  TemporalIF ROC 0.519, 1.08×, q = 0.26). Consequence for the prose: six of the 24
  non-sequence methods, not five, survive BH control, because the rank-averaged
  ensemble crossed at the new comparison count (1.17×, q = 0.044).
- **Split-conformal prediction intervals (`src/early_warning.py`):** the nowcast
  carried an empirical spread but no interval. A chronological split
  (3,056 district-month transitions, 2,096 calibration with destination year ≤ 2018,
  960 test) yields a conformal band for both the persistence carry and a ridge carry
  model: ±39.3 risk points at 90% nominal for persistence (test RMSE 23.09, measured
  coverage 0.886) and ±29.0 for ridge (test RMSE 19.12, coverage 0.829). Coverage is
  also reported per starting severity band. Written to
  `results/early_warning_summary.json` (`prediction_intervals`) and
  `results/tables/nowcast_interval.csv`, and drawn as `figures/fig_intervals.png`.
- **World Bank climate cross-check (`src/fetch_wb_climate.py`,
  `src/wb_climate_check.py`, `src/figures_wb.py`):** national climate indicator
  series (`EN.CLC.SPEI.XD`, `AG.LND.PRCP.MM`, exposure/coverage) fetched from
  api.worldbank.org with retry discipline; the portal's own web pages answer 403 and
  the API returns indicator ids with empty topic/source metadata, which is recorded
  as a limitation rather than worked around. The result is **negative and released as
  such**: r = 0.057 (p = 0.83) against our district-mean SPEI6, the Bank's index wet
  in all four register drought years, its three severe-drought years (1993, 2000,
  2005) outside our panel, and ERA5 district means at 134.6% of the Bank's constant
  1,180 mm normal. Artefacts: `data/wb_climate.csv`, `data/wb_climate_meta.json`,
  `results/wb_climate_check.json`, `results/tables/wb_{spei,precip}_compare.csv`,
  `figures/fig_wb_check.png`, and a new `sec:wbcheck` subsection in the manuscript.
- **Numbers-vs-artefact gate (`src/verify_release.py`):** `check_numbers()` re-derives
  every headline figure quoted in `manuscript/main.tex` from the file that produced it
  — detector count, non-sequence count, ROC/PR-AUC spans, the Friedman statistic, all
  four conformal widths and coverages, per-band coverage, the verdict wording, the
  World Bank correlation, dry-year claim and precipitation scale, and the presence of
  both new figures. `check_dashboard()` additionally refuses any hand-typed count left
  in `dashboard/app.py` and requires the new artefacts to be rendered, and the
  abstract's rounded correlation is checked against the same artefact. 61/61 checks
  pass on the committed state (was 22/22).
- **Persisted local attributions (`src/_explain_fast.py`, `src/explain.py`):** the
  explainability stage previously left only the aggregated `shap_values.npy` on
  disk, so a per-district-day explanation could not be audited. Both entry points
  now write the rows behind that matrix (`results/tables/shap_sample.csv`), the
  surrogate's own output for those rows (`results/shap_row_outputs.npy`) and
  `results/shap_expected_value.json` with the base value and a checked additivity
  residual. The committed set comes from the held-out path: 1,200 test
  district-days, base value 0.4996, surrogate held-out AUC 0.917, and
  max |base + ΣSHAP − predict_proba| = 4.44e-14, which is why the paper can call
  the local decomposition exact for the surrogate. A `_positive_class()` helper
  replaces the pinned `sv[1]` indexing so the same three artefacts survive a SHAP
  release that changes the binary return shape.
- **Measured carry uncertainty (`src/early_warning.py`):** the persistence nowcast
  carries the current district-month score forward, which needs an interval rather
  than a bare point. `carry_uncertainty` now differs the score inside the
  persisted district-month table and reports the empirical spread — 3,056
  transitions, median |Δ| 8.2, p80 32.3, p95 44.8, signed q10/q90 −33.0/+31.6 —
  and the same statistics restricted to the 900 transitions beginning in a High or
  Extreme cell (median 23.7, q10/q90 −42.9/+7.8). No model is fitted and none is
  claimed: the artefact states it is the width of the nowcast. Surfaced in
  `results/early_warning_summary.json` and, via `collect_results.py`, in
  `results/results_digest.json`.
- **Dashboard impacts panel (`dashboard/app.py`):** a tenth panel, *Impacts &
  Agriculture*, renders the drought/agriculture layer that the paper describes and
  the dashboard did not previously show — century in-situ SPI coverage, the
  decade-drought frequency table, index validation, register-drought
  corroboration (3 of 4, with the missed event named), the ERA5 district index
  explorer, detrended cereal-yield anomalies against documented drought years,
  both linkage blocks with their coefficients and p-values, and the coffee
  sign-reversal between substrates. Every value is read from
  `results/drought_summary.json`, `results/ag_linkage.json`,
  `results/coffee_linkage.json` and their tables.
- **Release gate (`src/verify_release.py`, final `run_all.py` stage):** the package
  now asserts its own deliverables instead of describing them — the two figure
  folders hold the same PNGs byte-identically at ≥300 dpi (read from the PNG
  `pHYs` header, no imaging library needed), the LaTeX source has balanced
  environments, no orphaned cross-reference, no missing `\includegraphics` or
  `.bib` target and no stray unicode dash, the SHAP additivity residual and the
  attribution row count agree with the artefacts the figure cites, the nowcast
  carries its uncertainty block, the map's three district classes partition the
  135 polygons (16 + 3 + 116), and every dashboard tab renders in-process without
  an exception. It exits non-zero on failure, so a bundle whose artefacts
  disagree with the text stops the pipeline rather than printing `[done ...]`.
  22/22 checks pass on the committed state.
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
- **Removed all references to the faulty Meteoblue evaluation:** the earlier
  draft's claim of 100% precision/recall/ROC-AUC was based on an evaluation
  with no independent ground truth. All citations to `masaba2023` have been
  removed from `manuscript/main.tex` (introduction, related work, results
  comparison, and discussion), `references.bib` (entry deleted), `README.md`,
  and `COVER_LETTER.md`. The `check_ref_index.py` script and its output remain
  in the repository for provenance but are no longer referenced in the paper.
- **Target journal retargeted to *Climate Risk Management* (Elsevier):**
  `manuscript/main.tex` (venue comment, `\journal{}`), `COVER_LETTER.md`,
  `HIGHLIGHTS.md` and `_build_overleaf.py` (zip name `uganda_crm_overleaf.zip`)
  now address CRM rather than EMS. The cover letter reframes the paper's fit
  around CRM's scope: climate risk assessment, early-warning system evaluation,
  adaptation decision support, and the practical consequences of inflated skill
  estimates for adaptation funding. The main.tex paragraph on reproducible
  software now cites "the climate-risk and environmental-modelling literature"
  rather than "this journal's own literature", since the two anchor citations
  (Janssen et al. 2020, Zulkafli et al. 2017) were published in EMS. Historical
  CHANGELOG entries that reference EMS as the venue at the time of writing are
  preserved as-is.
- **Abstract trimmed to CRM's 250-word limit (`manuscript/main.tex`):** reduced
  from 343 to 231 words by consolidating the data-description paragraph (removed
  the 16-district/13-variable/93,504-district-day detail and the four-layer list),
  trimming the detection-results paragraph (removed the LSTM-AE/Transformer-AE
  F1/precision/Monte-Carlo detail and the 68.8%/100% null-coverage sentence), and
  condensing the climatological paragraph (removed the specific Kendall-τ values
  for Kampala/Entebbe, the ERA5/NASA POWER r=0.95–0.999 match, the SPI/SPEI
  drought corroboration, and the 2018 cereal-yield detail). The core findings are
  preserved: the negative detection verdict (ROC-AUC 0.49–0.53, under 10% of
  disaster days captured), the refutation of the AUC=1.0 claim, the World Bank
  negative cross-check (r=0.06, p=0.83), and the verdict that weather-only
  detection is insufficient for operational early warning. The phrase "benchmark
  of 26 detectors" is retained (rather than "26-detector benchmark") to satisfy
  the gate's text-artefact agreement check.
- **Submission prose re-synced to the 26-detector release:** `README.md`,
  `manuscript/COVER_LETTER.md` and `manuscript/HIGHLIGHTS.md` now state the
  26-detector benchmark, the 0.491–0.532 non-sequence ROC span, six BH-surviving
  non-sequence methods rather than five, the conformal intervals and the negative
  World Bank cross-check. The README's World Bank row previously asserted that the
  Bank's climate portal "was **not** consulted"; that is now false, so the row
  describes what was fetched, what the API does and does not expose, and that the
  cross-check failed to corroborate. `run_all.py` gained the three WB stages and
  `collect_results.py` folds `wb_climate_check` into the digest.
- **One figure-output rule (`src/config.py`):** `FIG_DPI = 300`,
  `FIG_FOLDERS = (figures/, manuscript/figures/)` and a single `save_figure()`
  that writes the same Figure object at the same resolution into both folders and
  closes it. All five figure scripts (`figures`, `figures_power`, `figures_maps`,
  `figures_coffee`, `figures_early_warning`) call it instead of handling
  `savefig`/`close` per folder, so a stale artwork copy inside the manuscript
  bundle is no longer possible; the 17 committed PNGs are byte-identical in the two
  folders. Elsevier requests ≥300 dpi for halftone and combination artwork.
- **Two-panel SHAP figure (`src/figures.py`):** `fig_shap` now carries the global
  mean |SHAP| ranking and a local attribution bar for a persisted district-day,
  drawn from the matrices above rather than recomputed.
- **Manuscript disclosures (`manuscript/main.tex`):** the World Bank paragraph now
  states that only the Open Data API indicator pair is used and that the Bank's
  climate portal was not consulted, so no portal-hosted index is cross-checked
  against the register; the earlier claim that every surveyed detector family was
  reported is replaced by naming the two shortlist members still absent (the local
  outlier *probability* rescaling, and a lagged-window Isolation Forest — the
  benchmark's Isolation Forest is the standard method on engineered antecedent
  features, not a sequence-aware variant) together with why neither is expected to
  change the verdict. The dashboard subsection lists the ten panels as they exist,
  the RQ3 text carries the additivity result, the ews section carries the measured
  carry width, and the in-situ reporting-collapse window is corrected to
  1990–2019 to match the data.
- **Dashboard documentation (`dashboard/README.md`):** it claimed the app needs
  only `streamlit`, `pandas` and `numpy` and listed nine tabs. Corrected to ten
  panels, with the `pydeck` dependency, the fallback behaviour, the no-basemap /
  Python-side-classification map rules and the artefact list per tab.
- **Title:** now *A National Climate Anomaly Intelligence System for Uganda:*
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
- **Seasonal forecast API URL (`src/fetch_seasonal.py`, `src/config.py`):** the
  Open-Meteo seasonal endpoint was called with `models=ecmwf_seasonal` (invalid);
  corrected to `models=ecmwf_seas5` per the API documentation. All 19 districts
  now fetch 7-month SEAS5 forecasts successfully (133 district-months).
- **Hard-coded counts in the dashboard (`dashboard/app.py`):** the sidebar and the
  benchmark header still advertised "24 detectors", "Friedman χ²(23)", "the full
  24-row table" and "five cross-sectional methods … within 1.22×" after the benchmark
  had grown, and the family classifier knew only SARIMAX/Structural TS as temporal.
  Every count is now read from the artefacts (`len(benchmark.csv)`,
  `n_detectors - 1` from `friedman.json`, the BH-survivor count and maximum ratio from
  `random_null.json`), Temporal IF is drawn as its own `temporal (lag-window)` family,
  the benchmark table carries the per-detector null ratio and BH q, and the Risk
  Nowcast tab renders the conformal intervals with their measured coverage while the
  Impacts tab renders the World Bank cross-check as the negative result it is. The
  stale literals are now gate failures, so this class of drift cannot recur silently.
- **A one-sided interval verdict (`src/early_warning.py`):** `prediction_intervals`
  wrote a verdict that recommended the ridge carry model because its test RMSE was
  lower, while the same artefact measured that model's 90% band covering only 0.829 of
  held-out transitions against 0.886 for persistence. The verdict is now two
  statements — the better point forecast and the conservative band are different
  models — and names violated exchangeability under a chronological split as the reason
  the shortfall is reported instead of smoothed. `verify_release.py` asserts the
  wording, so a future re-run cannot quietly restore the promotional version.
- **Stale results digest:** `results/results_digest.json` still carried the
  14-detector Friedman result (χ² = 20.29, p = 4.4e-4) recorded before the two
  torch sequence autoencoders joined the benchmark, while `results/friedman.json`
  and the manuscript report χ² = 28.77, p = 8.7e-6 over 24 detectors. Re-running
  `collect_results.py` re-syncs the digest, and the same run now folds
  `early_warning_summary` into it, which it had never included. The digest is what
  the dashboard's reproducibility page enumerates, so the mismatch would have been
  visible to a reviewer as the paper disagreeing with its own artifact.
- **Districts silently rendered as "no data" (`dashboard/app.py`):** the
  choropleth shaded the 16 districts that carry an ERA5 detection-panel score,
  while `results/district_coverage.json` reports 19 site-districts. Hoima, Kabale
  and Kasese have a station but no panel score, and were being painted with the
  same grey as the 116 districts never observed. A fourth class, *site, no panel
  score*, is now styled, legended and captioned separately (16 + 3 + 116 = 135), so
  a user cannot read "no score" as "no data".
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
