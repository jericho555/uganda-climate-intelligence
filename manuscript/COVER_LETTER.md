# Cover Letter

**To:** The Editors, *Climate Risk Management* (Elsevier)
**From:** Jeremiah Masaba, Department of Computer Science, Makerere University, Kampala, Uganda (masabjericho@gmail.com)
**Manuscript title:** *A National Climate Anomaly Intelligence System for Uganda: Ground-Truth Validation, Risk Nowcasting, and Long-Term Climate Change Assessment*

Dear Editors,

We submit an original research article that we believe fits the aims of
*Climate Risk Management*: it delivers a ground-truth-honest evaluation of
unsupervised anomaly detection as an early-warning tool for climate disasters
in Uganda, and it reframes the finding — deliberately negative for the
detection layer — as a contribution to climate risk assessment and adaptation
decision support. In doing so it confronts a widespread evaluation failure in
the climate machine-learning literature that directly affects how scarce
adaptation resources are allocated.

Unsupervised weather-anomaly detection is frequently marketed as a disaster
early-warning tool, and studies on African climate records have reported
near-perfect metrics — but often against the same record used to train the
model, with no independent ground truth. In this work we rebuild the analysis
on openly licensed ERA5 reanalysis (16 districts, 13 daily variables, 16
years), pair it with a curated register of 32 documented climate disasters
used strictly as held-out validation, and run a 26-detector benchmark with
chronological splits, bootstrap confidence intervals, Friedman/Nemenyi
testing and a measured size-matched random null. That benchmark deliberately
includes the detectors a reviewer would expect to see: LoOP, and an isolation
forest refitted on fourteen-day lag windows. Neither changes the verdict, and
we report them as measured.

Our central finding is deliberately uncomfortable and, we argue, publishable
*because* it is honest. Against real disaster windows, no cross-sectional detector
ranks above chance (held-out ROC-AUC 0.49–0.53; best F1 = 0.084; best PR-AUC =
0.121 vs a base rate of 0.1115). Scored against the size-matched random alert
budget, which we now compute and release rather than assume (F1 = 0.069 [0.059,
0.079]), six of the twenty-four non-sequence methods remain above it after
Benjamini-Hochberg control and by at most 1.22×, and only the two
deep sequence autoencoders (LSTM-AE, Transformer-AE) roughly double it
(F1 ≈ 0.13–0.14) while still ranking near chance on ROC-AUC and capturing under
10% of disaster days. Apparent event coverage (68.8%) is reproduced by a random
null at 100% and is disclosed in the text as an upper bound computed on a
transductive surface. The result quantifies why ground-truth-honest evaluation
matters: a message aligned with
the journal's focus on responsible climate risk assessment and the practical
consequences of inflated skill estimates for adaptation funding and warning
chains.

The system nonetheless produces defensible environmental intelligence for risk
management: a significant, near-universal warming signal (e.g., Kampala T_max
τ = 0.42, p < 1e-16) with a coherent warming–drying–rising-evapotranspiration
contrast at a detected Getis–Ord hotspot (Mbale, G_i* = 3.50), and a composite
severity index (UCASI) whose extreme tail independently coincides with the
documented 2010 eastern floods, 2022 central landslide and 2018 drought. Every
map in the paper is a district choropleth drawn on the OCHA Uganda common
operational dataset (135 districts, CC BY-IGO), so the 13.7% of districts that
the panel really observes is shown as coverage rather than implied. SHAP explains
the detection signal, and we give a candid operational assessment of feasibility,
exposed through an interactive, read-only decision-support dashboard released
with the code.

Every figure and table is generated programmatically from the released code and
artefacts; no value is hard-coded, and the release script re-derives every headline
number quoted in the text from the file that produced it. Where the system makes a
forward claim it carries an interval: the district-month nowcast is wrapped in
split-conformal prediction intervals whose *measured* held-out coverage is reported,
including the shortfall caused by the chronological split. Where an external check was
possible we ran it and print the negative result: the World Bank's national SPEI series
is uncorrelated with our district index (r = 0.06, p = 0.83) and wet in every year the
register records drought, which we report as a provenance finding rather than as
corroboration. The related-work section positions the work
against two literatures relevant to climate risk management: the
landslide early-warning reviews, which note that no accepted standard exists for
verifying a warning system's skill, and the environmental-modelling evidence on code
sharing and on decision support built around a single implicit user. We adopt the
second as a requirement and state plainly where our dashboard does not yet meet
it. The work is original, unpublished, not under consideration elsewhere, and the
author declares no competing interest.

Thank you for considering our manuscript.

Sincerely,
Jeremiah Masaba
On behalf of the author
