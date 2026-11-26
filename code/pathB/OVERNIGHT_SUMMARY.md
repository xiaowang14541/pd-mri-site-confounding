# Path B overnight run — summary (2026-09-02 → 03)

## What was asked
Run Phase A marker exploration; if a marker clears the gate, run Phase B on Tao Wu +
NEUROCON; produce a finished article.

## What happened

### 1. Data assembled
- 3 discovery same-scanner cohorts (ds000245, ds001907, ds005892): 69 PD / 76 HC
- Downloaded + pipeline-processed Tao Wu (20/20) and NEUROCON (27/16) identically
  -> data/radiomics_v3_taowu_neurocon.csv (83/83 ok, reg Dice 0.67-0.75)
- OpenNeuro full-catalogue search (1875 datasets): NO other usable same-scanner
  structural PD+HC dataset exists. SoCal SKIP (ds005263/265/266) are young-controls-only.

### 2. Phase A (discovery, 38 age/sex-matched pairs)  -> pathB/phaseA_results.txt
- ICV-corrected subcortical volumes: pooled d +0.05..-0.26, perm p 0.14-0.47,
  LOCO AUC 0.44-0.58. Null.
- 232 radiomic texture features: min FDR-BH 0.98, ZERO significant. The 1 feature at
  LOCO AUC 0.66 has p=0.13, d=0.16 (best-of-232 noise).
- Caudate first-order minimum (the prior 'finding'): d -0.33 unmatched -> -0.10 after
  sex-exact matching. Erosion 1-2 vox removes 33-64%. Volumetry d=+0.03, AUC 0.50.
- Multivariate panels: LOCO AUC 0.38-0.50.
- GATE: not met (nothing is a credible marker; automated 'pass' flag fired only on the
  naive best-of-232 LOCO rule and is not substantive).

### 3. Phase B / 5-cohort robustness (64 matched pairs)  -> pathB/phaseB_results.txt
- Volumes: pooled d -0.09..+0.07, perm p 0.32-0.46, LOCO AUC 0.50-0.57. Null.
- 232-feature screen: min FDR-BH 0.54, ZERO significant.
- HC-HC between-site |d| mean 3.5 (max 7.9) vs within-scanner PD-HC |d| mean 0.45 -> 7.7x.
- VERDICT: the null holds across all 5 single-scanner cohorts. -> Path A.

### 4. Deliverable: the paper (Path A, cautionary/methodological)
- pathB/manuscript.md  (full IMRaD, ~4200 words, 4 tables, 4 figure captions)
- pathB/PD_site_confounding_manuscript.docx  (figures embedded)
- pathB/PD_site_confounding_manuscript.html
- pathB/figures/fig1-4 (pdf + png)
Title: 'Site-label confounding, not disease signal, explains high-accuracy deep-learning
classification of Parkinson's disease from aggregated public structural MRI'

Core result: DenseNet AUC 0.977 (dataset-stratified CV) vs 0.19 (leave-one-dataset-out),
same net same scans; no site-independent structural marker beats chance on 5 scanners;
the one surviving feature is a caudate/ventricle boundary artifact; power analysis shows
open data (~75 matched/group) cannot detect the d~=0.2-0.35 a real PD T1 effect implies.

## What is NOT done / needs you
- Author list, affiliations, funding, ethics statement, cover letter.
- Target journal decision (suggest: a neuroimaging-methods or reproducibility venue —
  e.g. Imaging Neuroscience, NeuroImage:Reports, GigaScience, or a brief-report track).
- Related-work citations: the manuscript argues the point but the reference list is a
  stub (§ Introduction / Discussion mark where PPMI-based and survey citations go).
- Optional: re-run the DenseNet LODO with a 3D model for a reviewer-proof robustness line.
- Decision: submit as-is (Path A) or shelve. Path B is exhausted for free data.

## Files / scripts (all under pathB/)
audit_power.py  phaseA.py  phaseB.py  extract_new_cohorts.py  make_figs.py
PREREGISTRATION.md  phaseA_gate.json
