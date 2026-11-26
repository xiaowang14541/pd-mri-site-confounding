# Pre-registration — Path B
## Does a pre-specified structural-MRI marker separate PD from HC when scanner site is controlled by construction?

Status: DRAFT v0.1 — 2026-09-02
Owner: wzw
Freeze target: before any Tao Wu / NEUROCON label is inspected (see §7)

---

## 0. Background and motivation (why this study exists)

An earlier version of this project reported AUC ≈ 0.977 for a DenseNet121 PD-vs-HC
classifier trained on aggregated OpenNeuro T1w MRI. A confound audit (July–Aug 2026)
showed that result is explained by **site/scanner confounding**: three of the source
datasets are single-diagnosis (ds003592 all-HC; ds004392, ds004471 all-PD), so a model
can separate the classes without learning anything about disease. When the analysis is
restricted to subjects whose PD and HC come from the **same scanner**, discrimination
falls to AUC 0.43–0.65 and is mostly non-significant; cross-dataset HC-vs-HC contrasts
have larger effect sizes (|d| 2–11) than any PD-vs-HC contrast; and the one candidate
feature that survived (caudate `firstorder_Minimum`) behaves like a CSF partial-volume
artifact at the caudate/ventricle border (killed by 1–2 voxel ROI erosion; absent in
proper volumetry).

This pre-registration defines a **single confirmatory test** of whether *any* genuine,
site-independent structural signal remains, run on data that played no part in
generating the hypothesis.

---

## 1. Question and hypotheses

**Primary question.** In within-scanner, age- and sex-matched PD/HC samples, does a
single pre-specified structural-MRI marker M discriminate PD from HC better than chance,
and does the effect replicate in a held-out confirmatory cohort?

- **H1 (primary).** Pooled within-cohort effect of M is in the pre-specified direction
  and stronger than a within-cohort label-permutation null, at one-sided α = 0.005,
  **in the confirmatory set** (Tao Wu + NEUROCON).
- **H0.** M carries no within-scanner PD/HC information beyond chance.

**Decision rule.** H1 is accepted only if BOTH hold in the confirmatory set:
1. one-sided permutation p < 0.005, AND
2. pooled matched AUC with 95% bootstrap CI lower bound > 0.55.

If H1 is rejected, the study is written up as a **registered null result** (Path A
merges here — the confound analysis + this null become the paper).

---

## 2. Design

- Observational, cross-sectional, secondary analysis of public data.
- **Unit of analysis:** subject (one T1w scan per subject; earliest session if >1).
- **Confound control by construction:** only cohorts where PD and HC were scanned on the
  same scanner/protocol enter the primary analysis. Scanner == cohort here, so a
  within-cohort contrast is scanner-free.
- **Additional confound control:** 1:1 age+sex matching within each cohort (§4), plus a
  pre-specified ANCOVA sensitivity analysis (§6).
- **Replication:** leave-one-cohort-out (LOCO) across the discovery cohorts, and a
  single held-out confirmatory set never used in specification.

---

## 3. Data

### 3.1 Discovery cohorts (used to LOCK M and the pipeline — §7 Phase A)
| cohort | scanner note | nPD | nHC | known imbalance |
|---|---|---|---|---|
| ds000245  | same scanner | 15 | 29 | PD ~7 yr older (d=+1.2) |
| ds001907  | same scanner | 21 | 25 | HC 72% male vs PD 43% |
| ds005892  | same scanner | 33 | 22 | PD 67% male vs HC 32% |

### 3.2 Confirmatory cohorts (LOCKED OUT until Phase B — §7)
| cohort | source | nPD | nHC | note |
|---|---|---|---|---|
| Tao Wu   | FCP/INDI (NITRC) | ~20 | ~20 | age-matched by original design |
| NEUROCON | FCP/INDI (NITRC) | ~27 | ~16 | age-matched by original design |

### 3.3 Explicitly EXCLUDED from all primary/confirmatory analyses
ds003592 (all-HC), ds004392 (all-PD), ds004471 (all-PD): single-diagnosis, scanner
collinear with label. May appear ONLY in a clearly labelled "direction-of-effect
corroboration" supplement, never in the primary statistic or its null.

### 3.4 Inclusion / exclusion (subject level)
- Include: T1w structural scan present; age and sex available; PD or HC label.
- Exclude: no usable T1w; failed registration QC (§5); age missing and unrecoverable.
- For ds005892, PD = {PD-NC, PD-MCI}; HC = {Control}.
- All exclusions logged with reason in `pathB/exclusions.csv` before unblinding labels
  in Phase B.

---

## 4. Matching

- Within each cohort, greedy 1:1 nearest-age matching, **sex-exact**, age tolerance
  ±3 years (fallback ±5 y only if it recovers >20% more pairs; decision logged).
- Match order randomised; 200 random orders; the matched set carried forward is the
  one whose mean |age gap| is the median across orders (deterministic seed = 20260902).
- Unmatched subjects are dropped from the primary analysis (kept for the §6 ANCOVA).
- Expected yield (from audit): discovery ≈ 38 sex-exact pairs; confirmatory ≈ 30–40.

---

## 5. Preprocessing and QC (frozen)

Reuse the existing harmonized pipeline (`extract_v3_harmonized.py`) unchanged:
- N4 bias correction → affine + SyN registration to MNI152 2 mm.
- Brain mask = MNI152 brain mask warped to native space (identical definition across
  cohorts) — this is the fix for the earlier cross-dataset z-score inconsistency.
- Intensity z-scored inside that common mask.
- Primary measurements taken in the common MNI 2 mm grid.

**Registration QC (pre-specified, label-blind):**
- Compute `reg_dice` (warped-template brain vs subject brain mask).
- Exclude scans with `reg_dice` < (median − 3·MAD of the pooled distribution).
- Visual QC of all excluded + a 10% random sample; rater blind to diagnosis.
- Anisotropic acquisitions (voxel z/x ratio > 1.2) flagged; primary analysis uses the
  2 mm-grid values which are on a common grid regardless.

---

## 6. The marker M and the statistical model

### 6.1 M is chosen in Phase A and LOCKED. Candidate families (pick exactly one):
- **M-vol:** a single subcortical volume (putamen or caudate), from the warped-atlas
  volumetry, adjusted for total intracranial volume. Rationale: closest to a
  mechanistically interpretable atrophy measure; prior DBM check was near-null, so this
  is a conservative choice with a real chance of H0.
- **M-panel:** a fixed 5-feature linear score over subcortical volumes + first-order
  intensity (putamen, caudate, pallidum, thalamus), coefficients fit ONCE on the pooled
  discovery matched set, then frozen. No feature selection after locking.
- **M-tex:** a single first-order texture statistic in a single ROI, but ONLY if it
  survives 1-voxel ROI erosion in Phase A (the caudate `Minimum` artifact is excluded a
  priori).
- **M-cnn:** 3D CNN (or 2.5D) with domain-adversarial training, evaluated strictly LOCO;
  one architecture, one hyperparameter set, fixed seed. Chosen only if Phase A shows a
  non-trivial LOCO AUC (> 0.62) on the discovery cohorts.

The Phase-A selection memo (which family, why, and the exact frozen definition) is
committed to git with a timestamp BEFORE any confirmatory data is touched.

### 6.2 Primary statistic
Pooled within-cohort standardized effect of M:
`d_pooled = mean over cohorts of Cohen's d(M | PD vs HC) on the matched set`,
direction pre-specified from Phase A.

### 6.3 Null and p-value
Stratified permutation: shuffle PD/HC labels **within each cohort**, preserving cohort
sizes; 20,000 permutations; recompute `d_pooled`; one-sided p = fraction of null ≥/≤
observed (in the pre-specified direction). No cross-cohort comparison enters statistic
or null (this is the fix for the earlier uninformative "shift vs null" test).

### 6.4 Secondary / sensitivity (all pre-specified, reported regardless of outcome)
1. Per-cohort forest plot of d (matched), with bootstrap CIs.
2. Pooled matched AUC + 95% bootstrap CI (2,000 resamples, cohort-stratified).
3. LOCO: fit/threshold on k−1 cohorts, evaluate on the held-out one; report per-fold AUC.
4. **ANCOVA instead of matching:** per-cohort `M ~ label + age + sex` on the full
   (unmatched) cohort; Stouffer-combine the label p-values.
5. **Erosion robustness** (if M is texture/intensity): effect must persist at 1-voxel
   ROI erosion (|d| retains ≥ 60% of its un-eroded value).
6. **Variant/degrees-of-freedom disclosure:** if any tuning parameter had >1 value
   considered in Phase A, report best-of-k permutation p as well.

---

## 7. Analysis phases and what is frozen when

**Phase A — specification (discovery cohorts only: ds000245, ds001907, ds005892)**
- Allowed: choose M family + exact definition; set matching params; set QC thresholds;
  inspect discovery-cohort effects, LOCO AUC, erosion behaviour.
- Output: `pathB/PHASE_A_SELECTION.md` + frozen `pipeline.py` + this file with §6.1
  resolved. Git tag `pathB-frozen`.
- Tao Wu / NEUROCON: downloaded, preprocessed, QC'd — but **labels not merged, effects
  not computed**. A colleague or a hash-committed script holds the label file.

**Phase B — confirmation (Tao Wu + NEUROCON)**
- Run the frozen `pipeline.py` exactly once. No changes to M, matching, QC, or stats.
- Apply the §1 decision rule.
- Any deviation from this document is reported as a labelled deviation with rationale.

**Phase C — full-sample report**
- Re-run everything on all five cohorts pooled (discovery + confirmatory) for the final
  effect-size estimate and forest plot. This is descriptive, not confirmatory.

---

## 8. Power (from the 2026-09-02 audit)

| set | matched n/group | detectable d @ α.005/80% | implied single-feature AUC |
|---|---|---|---|
| discovery (sex-exact ±3y) | ~38 | ~0.84 | ~0.72 |
| discovery + confirmatory  | ~70–80 | ~0.55–0.60 | ~0.65 |

Honest expectation: powered only for a **moderate-to-large** effect. If the true
within-scanner PD/HC structural effect is d ≈ 0.2–0.35 (AUC 0.56–0.60), this study will
most likely return H0 even if a real effect exists. A null is therefore a publishable,
pre-registered outcome, not a failed study.

---

## 9. Outcomes and what each means

| confirmatory result | interpretation | paper |
|---|---|---|
| p<0.005 AND AUC CI-lo >0.55, effect survives §6.4 | a small site-independent structural signature exists | Path B positive |
| p<0.005 but fails AUC or sensitivity checks | fragile / artifact-suspect | reported as inconclusive; lean Path A |
| p ≥ 0.005 | no detectable within-scanner signal at this power | Path A: registered null + the confound dissection |

---

## 10. Deviations log
(append-only; each entry: date, what changed, why, who approved)
- —
