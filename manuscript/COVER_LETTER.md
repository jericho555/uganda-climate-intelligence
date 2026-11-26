# Cover Letter

**To:** The Editors, *Environmental Modelling & Software* (Elsevier)
**From:** Jeremiah Masaba, Department of Computer Science, Makerere University, Kampala, Uganda (masabjericho@gmail.com)
**Manuscript title:** *A ground-truth-honest climate-anomaly detection system for Uganda: a 24-detector held-out benchmark, district risk nowcasting, and century-scale climatology from three observation substrates*

Dear Editors,

We submit an original research article that we believe fits the aims of
*Environmental Modelling & Software*: it delivers an open, reproducible national
climate-anomaly system: a leakage-safe pipeline plus an interactive
decision-support dashboard. In doing so it confronts a widespread
evaluation failure in the climate machine-learning literature.

Unsupervised weather-anomaly detection is frequently marketed as a disaster
early-warning tool, and studies on African climate records have reported
near-perfect metrics. The most cited such result for Uganda, reporting precision,
recall and ROC-AUC all equal to 1.0, was produced without any independent ground
truth. In this work we rebuild the analysis on openly licensed ERA5 reanalysis
(16 districts, 13 daily variables, 16 years), pair it with a curated register of
32 documented climate disasters used strictly as held-out validation, and run a
24-detector benchmark with chronological splits, bootstrap confidence intervals,
Friedman/Nemenyi testing and a measured size-matched random null.

Our central finding is deliberately uncomfortable and, we argue, publishable
*because* it is honest. Against real disaster windows, no cross-sectional detector
ranks above chance (held-out ROC-AUC 0.49–0.53; best F1 = 0.084; best PR-AUC =
0.121 vs a base rate of 0.1115). Scored against the size-matched random alert
budget, which we now compute and release rather than assume (F1 = 0.069 [0.059,
0.079]), five cross-sectional methods stay within 1.22× of it, and only the two
deep sequence autoencoders (LSTM-AE, Transformer-AE) roughly double it
(F1 ≈ 0.13–0.14) while still ranking near chance on ROC-AUC and capturing under
10% of disaster days. Apparent event coverage (68.8%) is reproduced by a random
null at 100% and is disclosed in the text as an upper bound computed on a
transductive surface. This refutes the earlier AUC = 1.0 claim and quantifies why
ground-truth-honest evaluation matters: a message aligned with the journal's
focus on responsible modelling and software practice.

The system nonetheless produces defensible environmental intelligence: a
significant, near-universal warming signal (e.g., Kampala T_max τ = 0.42,
p < 1e-16) with a coherent warming–drying–rising-evapotranspiration contrast at
a detected Getis–Ord hotspot (Mbale, G_i* = 3.50), and a novel composite severity
index (UCASI) whose extreme tail independently coincides with the documented
2010 eastern floods, 2022 central landslide and 2018 drought. SHAP explains the
detection signal, and we give a candid operational assessment of feasibility, exposed
through an interactive, read-only decision-support dashboard released with the code.

Every figure and table is generated programmatically from the released code and
artefacts; no value is hard-coded. The work is original, unpublished, not under
consideration elsewhere, and the author declares no competing interest.

Thank you for considering our manuscript.

Sincerely,
Jeremiah Masaba
On behalf of the author
