# Cover Letter

**To:** The Editors, *Environmental Modelling & Software* (Elsevier)
**From:** Jeremiah Masaba, Department of Computer Science, Makerere University, Kampala, Uganda (masabjericho@gmail.com)
**Manuscript title:** *A Climate Intelligence Framework for Detecting, Validating and Assessing Weather Anomalies in Uganda*

Dear Editors,

We submit an original research article that we believe fits the aims of
*Environmental Modelling & Software*: it delivers an open, reproducible
environmental-modeling pipeline and, in doing so, confronts a widespread
evaluation failure in the climate machine-learning literature.

Unsupervised weather-anomaly detection is frequently marketed as a disaster
early-warning tool, and studies on African climate records have reported
near-perfect metrics. The most cited such result for Uganda — precision, recall
and ROC-AUC all equal to 1.0 — was produced without any independent ground
truth. In this work we rebuild the analysis on openly licensed ERA5 reanalysis
(16 districts, 13 daily variables, 16 years), pair it with a curated register of
32 documented climate disasters used strictly as held-out validation, and run a
17-detector benchmark with chronological splits, bootstrap confidence intervals,
Friedman/Nemenyi testing and — crucially — a size-matched random null.

Our central finding is deliberately uncomfortable and, we argue, publishable
*because* it is honest: no generic detector beats chance against real disaster
windows (held-out ROC-AUC 0.49–0.51; best F1 = 0.083; best PR-AUC = 0.117 vs a
base rate of 0.1115), and apparent event coverage (68.8%) is fully explained by a
random null (100%). This refutes the earlier AUC = 1.0 claim and quantifies
*why* ground-truth-honest evaluation matters — a message squarely aligned with
the journal's focus on the responsible modelling and software practice.

The framework nonetheless produces defensible environmental intelligence: a
significant, near-universal warming signal (e.g., Kampala T_max τ = 0.42,
p < 1e-16) with a coherent warming–drying–rising-evapotranspiration contrast at
a detected Getis–Ord hotspot (Mbale, G_i* = 3.50), and a novel composite severity
index (UCASI) whose extreme tail independently coincides with the documented
2010 eastern floods, 2022 central landslide and 2018 drought. SHAP explains the
detection signal, and we give a candid operational assessment of feasibility.

Every figure and table is generated programmatically from the released code and
artefacts; no value is hard-coded. The work is original, unpublished, not under
consideration elsewhere, and the author declares no competing interest.

Thank you for considering our manuscript.

Sincerely,
Jeremiah Masaba
On behalf of the author
