# Site- and Cohort-Associated Confounding in Multi-Cohort PD T1-Weighted MRI Classification

Reproducible code, derived non-image feature tables, fold splits, random seeds, and summary results for:
**Frontiers in Neuroinformatics (Manuscript ID 2031561)** — *Site- and cohort-associated confounding inflates pooled Parkinson's disease classification from multi-cohort T1-weighted MRI*.

## 1. Public Datasets (Open-Source Links)
All raw T1-weighted MRI datasets analysed in this study are publicly accessible:
- **OpenNeuro ds000245** (Mixed PD/HC, n=44): https://openneuro.org/datasets/ds000245
- **OpenNeuro ds001907** (Mixed PD/HC, n=46): https://openneuro.org/datasets/ds001907
- **OpenNeuro ds005892** (Mixed PD/HC, n=55): https://openneuro.org/datasets/ds005892
- **OpenNeuro ds003592** (Controls only, n=90): https://openneuro.org/datasets/ds003592
- **OpenNeuro ds004392** (PD only, n=58): https://openneuro.org/datasets/ds004392
- **OpenNeuro ds004471** (PD only, n=40): https://openneuro.org/datasets/ds004471
- **INDI Tao Wu & NEUROCON** (External single-scanner mixed PD/HC, n=40 & n=43): http://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html

## 2. Open-Source Software & Atlases
- SimpleITK (`https://simpleitk.org`), PyRadiomics (`https://pyradiomics.readthedocs.io`), DIPY (`https://dipy.org`), Nilearn (`https://nilearn.github.io`), neuroCombat (`https://github.com/Jfortin1/neuroCombat`), scikit-learn (`https://scikit-learn.org`), statsmodels (`https://www.statsmodels.org`), PyTorch / torchvision (`https://pytorch.org`).
- Templates / Atlases: MNI152 2 mm template and Automated Anatomical Labelling (AAL) parcellation retrieved via `nilearn.datasets`.

## 3. Repository Structure
- `code/`: Feature extraction (`extract_v3_harmonized.py`), DBM (`analyze_dbm_volume.py`), CSR (`csr_analysis.py`), HC-vs-HC null model (`null_model_v3.py`), 5-cohort external check (`five_cohorts.py`, `pathB/robustness_5cohort.py`), boundary erosion & mechanism (`erosion_experiment.py`, `erosion_analysis.py`, `site_sensitivity_by_statistic.py`, `prespecified_mechanism_test.py`), deep learning (`src/`), and full revision benchmark & figure scripts (`code/revision_scripts/`).
- `derived_data/`: Extracted radiomics, DBM, erosion, foundation-model embedding CSVs, `splits/`, and `revision_metrics.json`.
- `results_summaries/`: JSON/CSV validation metrics and revised summary tables (`revision_tables/`).

## 4. Quick Reproduction (from Derived Features)
```bash
pip install -r requirements.txt
python code/revision_scripts/run_revision_analyses.py
python code/revision_scripts/make_revision_figures.py
```
