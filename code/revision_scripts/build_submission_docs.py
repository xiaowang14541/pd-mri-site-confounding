#!/usr/bin/env python3
"""
Build:
1. Manuscript_Revised_Clean.docx (+ .md)
2. Manuscript_Revised_Marked.docx (new/revised text & tables highlighted in blue)
3. Supplementary_Material_Revised.docx
for Frontiers in Neuroinformatics Manuscript ID 2031561.
"""
import json
import os
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE = "/master/wang_zi_wei/cursor/pd_classification"
OUT_DIR = os.path.join(BASE, "frontiers_revision_2031561")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

with open(os.path.join(OUT_DIR, "revision_metrics.json")) as f:
    M = json.load(f)

BLUE_REV = RGBColor(0, 70, 180)
DARK_GREY = RGBColor(60, 60, 60)


def set_cell_shading(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)


def set_table_borders(table):
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6" if edge in ("top", "bottom") else "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "333333" if edge in ("top", "bottom") else "CCCCCC")
        borders.append(el)
    for edge in ("left", "right", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        borders.append(el)
    tblPr.append(borders)


def configure_doc(doc):
    for sec in doc.sections:
        sec.top_margin = Inches(0.9)
        sec.bottom_margin = Inches(0.9)
        sec.left_margin = Inches(0.9)
        sec.right_margin = Inches(0.9)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style.paragraph_format.line_spacing = 1.15
    style.paragraph_format.space_after = Pt(6)


def add_heading_p(doc, text, level=1, marked_rev=False, marked_mode=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15 if level == 0 else (13 if level == 1 else 11.5))
    if marked_mode and marked_rev:
        r.font.color.rgb = BLUE_REV
        r.underline = True
    return p


def add_seg_p(doc, segments, marked_mode=False, italic_all=False, bold_all=False, align=None, space_after=6):
    """segments: list of (text, is_revised_bool)"""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if align is not None:
        p.alignment = align
    for txt, is_rev in segments:
        r = p.add_run(txt)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)
        if italic_all:
            r.italic = True
        if bold_all:
            r.bold = True
        if marked_mode and is_rev:
            r.font.color.rgb = BLUE_REV
            r.underline = True
            r.bold = True
    return p


def add_plain_p(doc, text, is_rev=False, marked_mode=False, italic_all=False, bold_all=False, font_size=11):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(font_size)
    if italic_all:
        r.italic = True
    if bold_all:
        r.bold = True
    if marked_mode and is_rev:
        r.font.color.rgb = BLUE_REV
        r.underline = True
        r.bold = True
    return p


def add_styled_table(doc, headers, rows, marked_mode=False, highlight_all=False, highlight_cols=None, col_font_pt=8.5):
    highlight_cols = highlight_cols or set()
    tbl = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl)
    for j, h in enumerate(headers):
        cell = tbl.rows[0].cells[j]
        set_cell_shading(cell, "E8EEF5" if (marked_mode and (highlight_all or j in highlight_cols)) else "F2F2F2")
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(str(h))
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(col_font_pt)
        if marked_mode and (highlight_all or j in highlight_cols):
            r.font.color.rgb = BLUE_REV
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.rows[i + 1].cells[j]
            if marked_mode and (highlight_all or j in highlight_cols):
                set_cell_shading(cell, "F3F7FC")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.add_run(str(val))
            r.font.name = "Times New Roman"
            r.font.size = Pt(col_font_pt)
            if marked_mode and (highlight_all or j in highlight_cols):
                r.font.color.rgb = BLUE_REV
    return tbl


def build_manuscript(marked_mode=False):
    doc = Document()
    configure_doc(doc)

    vc = M["validation_cis"]
    mx = M["mixed_diagnosis_only"]
    mc = vc["matched_caudate_minimum"]

    if marked_mode:
        p_note = doc.add_paragraph()
        r_n = p_note.add_run(
            "MARKED-UP REVISION COPY (Manuscript ID 2031561): All newly added or substantively revised text, "
            "updated confidence intervals, corrected figure numbering, and new/expanded tables addressing "
            "Reviewer 1 and Reviewer 2 are highlighted in bold blue underlined text."
        )
        r_n.bold = True
        r_n.font.size = Pt(10)
        r_n.font.color.rgb = BLUE_REV

    # Title & Front Matter
    add_heading_p(doc, "Site- and cohort-associated confounding inflates pooled Parkinson's disease classification from multi-cohort T1-weighted MRI", level=0, marked_rev=True, marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Running title: ", False),
        ("Cohort confounding in pooled PD MRI classification", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Article type: ", False),
        ("Original Research · Frontiers in Neuroinformatics", True),
    ], marked_mode=marked_mode)
    add_plain_p(doc, "Authors: Wang Xiao1, Yang Yang1, Cao Ying1, Liang Yi-na1, Wang Mengjiao1, Hu Haifeng1*", bold_all=False)
    add_plain_p(doc, "1 The Second Affiliated Hospital of Qiqihar Medical University, Qiqihar, Heilongjiang 161003, China")
    add_plain_p(doc, "* Correspondence: Hu Haifeng, mubai822@163.com")
    add_plain_p(doc, "Keywords: Parkinson's disease; structural MRI; radiomics; site effects; cohort confounding; external validation; reproducibility")

    # Abstract
    add_heading_p(doc, "Abstract", level=1)
    add_seg_p(doc, [
        ("Background. Machine-learning studies of Parkinson's disease (PD) using T1-weighted MRI often report areas under the curve (AUC) above 0.90 from pooled multi-cohort data. We asked how much of that performance reflects between-cohort confounding — encompassing scanner hardware, acquisition protocol, voxel geometry, demographic composition, recruitment differences, and prior preprocessing — rather than disease.", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Methods. Six public OpenNeuro datasets (333 subjects; 167 PD, 166 controls) were analysed; three contain only one diagnostic group. Pooled stratified cross-validation was compared with leave-one-dataset-out (LODO) validation across five feature families, and stratified bootstrap or Hanley–McNeil 95% confidence intervals (CIs) were computed for all validation metrics. ", False),
        ("To eliminate structural collinearity between site and diagnosis, we additionally restricted radiomics classification exclusively to mixed-diagnosis cohorts (three OpenNeuro mixed cohorts, n = 145; and all five single-scanner mixed cohorts adding Tao Wu and NEUROCON, n = 228). ", True),
        ("A null model of between-cohort variation was built from healthy-versus-healthy contrasts, and per-cohort registration Dice overlap and imaging acquisition/preprocessing metrics were audited across all eight cohorts. One candidate radiomics feature was followed through age- and sex-matched designs with stratified permutation testing and then checked in the two external single-scanner cohorts.", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Results. Pooled cross-validation (232 scans, four datasets) gave a fold mean AUC of 0.977 ± 0.013 (out-of-fold AUC 0.983, 95% CI 0.969–0.993). LODO on the same 232 scans gave a pooled AUC of 0.187 (95% CI 0.129–0.249), with a concave-up receiver-operating-characteristic (ROC) curve driven by training-fold label prior shift and site-signature inversion; extended LODO adding ds001907 (278 scans) gave 0.465 (95% CI 0.397–0.532), and direct transfer between the two mixed-diagnosis OpenNeuro cohorts was below chance (0.358 [0.198–0.518] and 0.248 [0.104–0.392]). ", True),
        ("When single-diagnosis cohorts were excluded completely, leave-one-cohort-out AUCs remained at or below chance across classifiers (0.338–0.392 on the three OpenNeuro mixed cohorts; 0.456–0.466 across all five single-scanner cohorts), even under transductive within-cohort centring (0.355–0.422 and 0.404–0.544, all 95% CIs including 0.50). ", True),
        ("Contrasts between healthy control groups of different cohorts, which contain no disease difference, spanned |d| = 0.29–4.87 (median 2.47) and equalled or exceeded the within-cohort PD–control effect (mean d = −0.37). Caudate first-order Minimum passed age adjustment and permutation testing (matched pooled AUC 0.592, 95% CI 0.446–0.698, p = 0.035), ", True),
        ("but failed our measurement-stability and external checks: its bootstrap CI included zero, its sex effect reversed between cohorts, its apparent signal resided in the single darkest boundary voxel, it reversed sign in Tao Wu (d = +0.579) and NEUROCON (d = +0.320), and its mechanistic prediction was refuted in an independent cohort. Thus, it is disqualified as a biomarker.", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Conclusion. Pooled accuracy in multi-cohort T1-weighted MRI studies is inflated by cohort-level confounding. We recommend routine reporting of mixed-cohort transfer, cohort-level registration/acquisition metrics, and a healthy-versus-healthy null model before claiming imaging biomarkers.", True),
    ], marked_mode=marked_mode)

    # 1. Introduction
    add_heading_p(doc, "1. Introduction", level=1)
    add_plain_p(doc, (
        "Parkinson's disease (PD) is the second most common neurodegenerative disorder, affecting approximately 1–2% of people "
        "over 65 years of age (Poewe et al., 2017). Diagnosis remains primarily clinical and rests on the presence of bradykinesia "
        "together with rest tremor or rigidity, supported by response to dopaminergic therapy. Clinicopathological series report 53% "
        "diagnostic accuracy in medication-responsive disease of less than five years' duration and 26% in untreated patients, rising "
        "above 85% only with longer duration (Adler et al., 2014); pooled accuracy against pathological gold standard across all settings "
        "is 80.6% (Rizzo et al., 2016). This has motivated an extensive search for objective imaging markers that could support diagnosis "
        "before motor signs are unambiguous."
    ))
    add_plain_p(doc, (
        "The structural target is well specified. A single-centre volumetric MRI study reports caudate reductions of 11% and putamen "
        "reductions of 8.1% in PD relative to controls (Pitcher et al., 2012), which we take as the reference target effect for the "
        "feasibility calculations below; expressed as a standardised difference in our measurement units this corresponds to d ≈ 0.585. "
        "Longitudinal meta-analysis confirms progressive striatal volume loss in PD without fixing a single cross-sectional percentage "
        "(He et al., 2020). An effect of that size is detectable in principle, but it is small relative to the between-subject variability "
        "of any single imaging measurement, and small relative to the differences that multi-cohort acquisition and processing can introduce."
    ))
    add_plain_p(doc, (
        "The pathological target is also poorly matched to the modality. The defining lesion of PD is loss of dopaminergic neurons in the "
        "substantia nigra pars compacta, which is not visible on conventional T1-weighted images; the sequences developed to see it directly "
        "are neuromelanin-sensitive imaging, quantitative susceptibility mapping and nigrosome-1 visualisation (Jin et al., 2019; Hartono et al., "
        "2023; Ariz et al., 2024; He et al., 2023), none of which is part of a routine T1 protocol. Any T1-based classifier is therefore working "
        "on downstream, non-specific consequences of that lesion."
    ))
    add_seg_p(doc, [
        ("Machine-learning analyses of MRI have nevertheless reported high discrimination between PD and healthy controls, with several studies exceeding an AUC of 0.90 (Camacho et al., 2023; Shi et al., 2022). Because assembling sufficient sample sizes usually requires pooling data acquired at different centres, and because multi-site neuroimaging carries substantial non-biological and cohort-level variation — arising jointly from scanner hardware, acquisition protocol, voxel geometry, demographic composition, recruitment criteria, disease characteristics, and the state of prior processing — the interpretation of such figures is not straightforward. ", True),
        ("Yamashita et al. (2019), using travelling subjects scanned at multiple sites, showed that measurement bias between sites can exceed disorder-related effects, and that subject classification can be achieved from site information alone. Consistent with this, a large multi-site study of major depressive disorder found balanced accuracy falling from approximately 62% to 52% once site harmonization was applied (Belov et al., 2024). Systematic misestimation of machine-learning performance in small neuroimaging samples has been documented directly (Flint et al., 2021). Harmonization methods such as ComBat correct the mean and variance of each feature but do not, on their own, guarantee that a classifier trained on one set of cohorts transfers to another; recent work shows that removing site effects from covariance, not only from location and scale, is required for multivariate methods (Chen et al., 2022; Bayer et al., 2022).", False),
    ], marked_mode=marked_mode)
    add_plain_p(doc, (
        "Despite this accumulating evidence, most published PD MRI classification studies still report a single pooled cross-validation "
        "figure without an accompanying estimate of how large a difference cohort membership alone can produce in the same measurement "
        "and the same pipeline. The question facing a reader is therefore not whether cohort effects exist — that is established — but "
        "whether a specific reported effect is large enough to be distinguished from them. This sits within a wider problem: predictive-modelling "
        "pipelines carry substantial analytical flexibility, internal validation tends to return optimistic estimates, and external validation "
        "is frequently omitted (Cabitza et al., 2021; la Roi-Teeuw et al., 2024; Snell et al., 2021). Reporting completeness against established "
        "guidelines remains poor in this literature (Heus et al., 2018), and reproducibility of published deep-learning imaging work is limited "
        "(Moassefi et al., 2023; Colliot et al., 2023)."
    ))
    add_seg_p(doc, [
        ("We state our position relative to that literature explicitly. That site effects can exceed disorder effects is not a new finding, and was demonstrated more rigorously than we can demonstrate it here: Yamashita et al. (2019) used travelling subjects, the same people scanned at multiple sites, which separates pure scanner measurement bias from differences between the populations recruited at each site. We have no travelling subjects, and our between-cohort estimates therefore reflect broader cohort-level confounding — intertwining scanner, protocol, demographics, clinical characteristics, and preprocessing state — rather than scanner hardware alone. What we add is not the finding but its operationalisation within a single study: a null distribution of cohort effects, computed from the same feature and the same pipeline as the disease effect, against which a specific claim can be placed. We also report what happens when that judgement is applied to a candidate of our own.", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("The present study had four aims. First, to quantify, across five representative feature families and both mixed-and-single-diagnosis as well as strictly mixed-diagnosis-only cohort subsets, the gap between pooled cross-validation, leave-one-dataset-out (LODO) validation, and direct cross-cohort transfer. Second, to construct an explicit null model for the magnitude of cohort-level confounding by contrasting healthy control groups across cohorts, and to audit registration quality and imaging QC metrics across all cohorts. Third, to follow one promising candidate feature through a complete validation chain — and show why it ultimately fails as a biomarker — in order to identify which checks are informative and which are not. Fourth, to determine the mechanism by which that candidate produced an apparent effect. In the course of this work we also encountered five pipeline defects that yielded superficially valid but incorrect results; because these are difficult to detect from a published methods section, we summarise them in Supplementary Section S3.", True),
    ], marked_mode=marked_mode)

    # 2. Methods
    add_heading_p(doc, "2. Methods", level=1)
    add_heading_p(doc, "2.1 Cohorts", level=2)
    add_seg_p(doc, [
        ("Six publicly available datasets were obtained from OpenNeuro (Table 1; Supplementary Table S3). Three contain both diagnostic groups (ds000245, ds001907, ds005892) and therefore permit a within-cohort, same-scanner comparison. Three contain a single diagnostic group (ds003592, healthy controls only; ds004392 and ds004471, PD only), where acquisition cohort is by construction collinear with diagnosis. Different analyses used different subsets of these datasets. The pooled cross-validation, the primary LODO and the Models Genesis analyses used 232 scans from four datasets (ds000245, ds003592, ds004392, ds004471; 113 PD, 119 HC). The extended LODO and the BrainSegFounder analysis added ds001907, giving 278 scans from five datasets (134 PD, 144 HC); ds005892 did not enter the deep-learning analyses. ", False),
        ("Two further single-scanner datasets containing both diagnostic groups — Tao Wu (20 PD / 20 HC) and NEUROCON (27 PD / 16 HC), distributed through the International Neuroimaging Data-Sharing Initiative (INDI; Badea et al., 2017) — were processed through the identical pipeline and used as an external single-scanner check of the within-scanner and mixed-cohort analyses (Sections 3.2 and 3.5; Supplementary Table S4).", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Expanded demographic distributions — including sample sizes, age mean ± standard deviation (SD), age median and interquartile range (IQR), age range, and sex counts — are reported across all eight cohorts in Table 1. In ds000245, the PD group is 7.0 years older than controls (70.7 ± 4.8 vs 63.7 ± 6.3 years, p = 0.0002). Note that in an earlier internal tabulation, participant identifier collisions ('sub-XXX' identifiers shared between ds004471 and ds003592) caused ds004471 age and sex and ds004392 sex to be misassigned; using the exact per-dataset BIDS participants.tsv records confirms that ds004471 PD patients have mean age 60.2 ± 6.8 years (median 60.5 [IQR 56.0–65.2], range 38–70; 65.0% male [26/40]) and ds004392 PD patients have mean age 70.0 ± 7.5 years (n = 57 with recorded age; 67.2% male [39/58]).", True),
        (" ds000245 additionally comprises three original subgroups: PD with olfactory dysfunction, idiopathic olfactory dysfunction without parkinsonism, and healthy controls; the latter two were combined as controls in the primary analysis and analysed separately in a sensitivity check.", False),
    ], marked_mode=marked_mode)

    # Table 1
    add_plain_p(doc, "Table 1. Expanded cohort characteristics across all six OpenNeuro datasets and two external INDI cohorts (mean ± SD, median [IQR], range, and sex).", is_rev=True, marked_mode=marked_mode, bold_all=True)
    t1_df = pd.read_csv(os.path.join(TAB_DIR, "table1_expanded.csv"))
    t1_headers = ["Dataset", "Design", "n (PD/HC)", "PD Age Mean ± SD", "PD Age Med [IQR]", "PD Age Range", "PD Male %", "HC Age Mean ± SD", "HC Age Med [IQR]", "HC Age Range", "HC Male %"]
    t1_rows = []
    for _, r in t1_df.iterrows():
        t1_rows.append([
            r["Dataset"], r["Design"], f"{r['n']} ({r['PD']}/{r['HC']})",
            r["PD_Age_Mean_SD"], r["PD_Age_Median_IQR"], r["PD_Age_Range"], r["PD_Male_pct"],
            r["HC_Age_Mean_SD"], r["HC_Age_Median_IQR"], r["HC_Age_Range"], r["HC_Male_pct"],
        ])
    t1_rows.append(["Total (OpenNeuro)", "6 cohorts", "333 (167/166)", "—", "—", "—", "—", "—", "—", "—", "—"])
    t1_rows.append(["Total (All 8 cohorts)", "6 OpenNeuro + 2 INDI", "416 (214/202)", "—", "—", "—", "—", "—", "—", "—", "—"])
    add_styled_table(doc, t1_headers, t1_rows, marked_mode=marked_mode, highlight_cols={4, 5, 8, 9}, col_font_pt=7.5)

    add_heading_p(doc, "2.2 Preprocessing and Registration Metrics", level=2, marked_rev=True, marked_mode=marked_mode)
    add_plain_p(doc, (
        "Images were resampled to 1 mm isotropic voxels, intensity-normalized and spatially standardized. Because the datasets are distributed "
        "in heterogeneous states of prior processing (Supplementary Section S3, defect 5), two normalization variants were compared: (A) z-scoring "
        "over voxels above the 5th intensity percentile, which is sensitive to how much non-brain tissue each dataset retains; and (B) z-scoring "
        "within an MNI152 brain mask propagated to each subject through that subject's own registration, which places all cohorts on a common "
        "definition of brain tissue. Variant B was used for all primary analyses unless stated otherwise."
    ))
    add_seg_p(doc, [
        ("For region-of-interest analyses, each image was registered to the MNI152 template (2 mm) with a three-stage affine sequence — translation, rigid, then full affine — using a mutual-information metric with 32 histogram bins, a three-level pyramid with 500, 100 and 25 iterations, smoothing sigmas of 3.0, 1.0 and 0.0 voxels, and subsampling factors of 4, 2 and 1. This was followed by symmetric diffeomorphic registration with a cross-correlation metric and 25, 15 and 8 iterations per level, following Avants et al. (2008) as implemented in DIPY (Garyfallidis et al., 2014). Initialisation used centre-of-mass alignment with min–max template normalisation (Supplementary Section S3, defect 2). Automated Anatomical Labelling (AAL) atlas labels for caudate, putamen, pallidum and thalamus were resampled onto the template grid with nearest-neighbour interpolation, combined across hemispheres, and propagated to each subject via inverse transforms. ", False),
        ("Registration quality was quantified for every subject as the Dice overlap between the affine-aligned brain mask and the MNI152 template mask, and automated extraction success (completion of registration and feature extraction without pipeline errors; no scans were excluded via manual visual inspection) plus imaging geometry/preprocessing metrics (native slice thickness, anisotropy indicator, raw background zero-voxel fraction, and Variant-A normalization mask fraction) were audited across all eight cohorts (Table 2). Within each mixed-diagnosis cohort, registration Dice overlap was compared between PD and HC using Cohen's d and two-sided Mann–Whitney U tests to rule out differential registration failure as a source of within-cohort group bias.", True),
    ], marked_mode=marked_mode)
    add_plain_p(doc, (
        "Two further region-level measurements were made. Regions were eroded inward by one and two voxels using binary morphological erosion with "
        "a 6-connected structuring element, and all statistics recomputed on the eroded masks; the number of surviving voxels is reported alongside "
        "each result. Separately, a Euclidean distance transform of each region was used to record how far from the region boundary the ten darkest "
        "and ten brightest voxels lay, in voxel units, where a value of one indicates a voxel on the boundary itself."
    ))

    # Table 2 (New)
    add_plain_p(doc, "Table 2. Automated extraction success, registration Dice overlap, and imaging preprocessing metrics across all eight cohorts.", is_rev=True, marked_mode=marked_mode, bold_all=True)
    t2_df = pd.read_csv(os.path.join(TAB_DIR, "table2_qc_metrics.csv"))
    t2_headers = ["Dataset", "Extraction Success", "Reg. Dice Mean ± SD", "Reg. Dice Med [IQR]", "Reg. Dice Range", "Dice PD vs HC", "Slice Thick. (mm)", "Anisotropic %", "Raw Zero Frac Mean ± SD", "NormFrac-A Mean ± SD"]
    t2_rows = [
        [r["Dataset"], r["Extraction_Success"], r["Dice_Mean_SD"], r["Dice_Median_IQR"], r["Dice_Range"],
         r["Dice_PD_vs_HC"], r["Slice_Thick_mm"], r["Anisotropic_pct"], r["Frac_Zero_Mean_SD"], r["NormFrac_A_Mean_SD"]]
        for _, r in t2_df.iterrows()
    ]
    add_styled_table(doc, t2_headers, t2_rows, marked_mode=marked_mode, highlight_all=True, col_font_pt=7.5)

    add_heading_p(doc, "2.3 Feature families and models", level=2)
    add_plain_p(doc, (
        "Five representative approaches were evaluated, on the subsets of subjects stated in Section 2.1: (i) ImageNet-pretrained DenseNet121 "
        "applied to 2D coronal slices taken from the central third of each volume; (ii) the same network with a domain-adversarial branch penalizing "
        "dataset-identifiable representations; (iii) subcortical radiomics comprising first-order, grey-level co-occurrence matrix and grey-level "
        "run-length matrix features (232 features across four bilateral regions × 58 features, discretized with 64 fixed bins per IBSI guidance; "
        "Zwanenburg et al., 2020); (iv) two pretrained 3D brain foundation models, Models Genesis (Zhou et al., 2019) and a UK Biobank-pretrained "
        "SwinViT (BrainSegFounder); and (v) deformation-based morphometry (log-Jacobian determinant averaged within each atlas region)."
    ))
    add_plain_p(doc, (
        "Network training used a batch size of 16, Adam (learning rate 10^-4, weight decay 0.01), up to 50 epochs (early stopping patience 10), "
        "and spatial augmentation (horizontal flip, ±15° rotation, 0.9–1.1 scaling, random crop), with classifier-head warmup for 10 epochs. "
        "Information barriers ensured that no quantity derived from any held-out cohort entered training, intensity standardisation, age residualisation, "
        "or within-cohort matching (ComBat was used solely for the descriptive Section 3.3 confound-to-signal ratio without train/test splitting)."
    ))

    add_heading_p(doc, "2.4 Validation designs and mixed-diagnosis-only sensitivity analyses", level=2, marked_rev=True, marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Three primary designs were compared: (1) Pooled stratified 5-fold cross-validation stratified on the joint label–dataset key; (2) Leave-one-dataset-out (LODO) validation holding out an entire cohort at each iteration (232 scans, plus domain-adversarial and ROI-cropped variants, and an extended 278-scan LODO adding ds001907); and (3) Within-cohort 1:1 age- and sex-matched analysis (same sex, age difference ≤ 4 years, averaged over 200 random patient orderings). ", False),
        ("To eliminate collinearity between cohort identity and diagnosis (Reviewer 1), we added a fourth validation suite restricted exclusively to mixed-diagnosis cohorts: (a) the three OpenNeuro mixed cohorts (ds000245, ds001907, ds005892; n = 145: 69 PD, 76 HC) and (b) all five single-scanner mixed cohorts including Tao Wu and NEUROCON (n = 228: 116 PD, 112 HC). Using the 232 Variant-B subcortical radiomics features, we evaluated L2-regularised logistic regression (C = 1.0), RBF-kernel SVM, and Random Forest (200 trees) under pooled 5-fold CV (stratified by cohort × diagnosis), leave-one-cohort-out (LOCO) validation, and all pairwise directed cohort-to-cohort transfers. For LOCO and pairwise transfers we evaluated both globally standardised features (scaler fitted strictly on training cohorts) and a transductive within-cohort centred benchmark in which each cohort's features were z-scored using that cohort's own mean and standard deviation to test whether removing marginal cohort shifts rescues cross-cohort discrimination.", True),
    ], marked_mode=marked_mode)

    add_heading_p(doc, "2.5 Site-effect null model", level=2)
    add_plain_p(doc, (
        "To establish how large a difference cohort membership alone can produce, we computed effect sizes between the healthy control groups of "
        "different cohorts, which contain no disease contrast by construction, adjusted for age and its square. Within-cohort PD–HC contrasts, "
        "in which both groups share a scanner and recruitment setting, were identified separately."
    ))

    add_heading_p(doc, "2.6 Confound-to-signal ratio and target effect-size sensitivity", level=2, marked_rev=True, marked_mode=marked_mode)
    add_seg_p(doc, [
        ("As a prospective, label-free feasibility measure we defined the confound-to-signal ratio (CSR) as the between-dataset nuisance effect size, measured in healthy controls only and adjusted for age and its square, divided by a target disease effect size taken from the literature. We chose the 11% caudate volumetric reduction reported in a well-characterised single-centre structural MRI study (Pitcher et al., 2012) — corresponding to d ≈ 0.585 in our measurement units — because longitudinal meta-analysis confirms progressive striatal volume loss in PD (He et al., 2020) while early clinical diagnostic accuracy is modest, making a moderate structural target of d ≈ 0.5–0.6 a realistic upper bound for conventional T1-weighted MRI. ", True),
        ("However, because reported effect sizes vary with disease stage, cohort composition, and imaging metric — and true subtler T1-weighted effects in early or heterogeneous PD may be smaller (d ≈ 0.20–0.40) while larger effects (d ≈ 0.80) represent optimistic benchmarks — we conducted an explicit sensitivity analysis evaluating CSR across d_target ∈ {0.20, 0.30, 0.40, 0.585, 0.80}. A CSR above unity indicates that cohort-level nuisance variation exceeds the disease effect sought.", True),
    ], marked_mode=marked_mode)

    add_heading_p(doc, "2.7 Statistical analysis", level=2)
    add_seg_p(doc, [
        ("Stratified permutation testing (20,000 permutations for pooled effects; 2,000 for matched designs) shuffled diagnostic labels within cohort while repeating feature selection and matching inside each permutation. ", False),
        ("For all reported validation AUCs and accuracies, we computed 95% confidence intervals (CIs): 5,000-resample stratified bootstrap CIs whenever subject-level predictions were available (pooled 5-fold CV OOF, primary LODO, LODO+DANN, LODO+ROI, within-ds000245 CV, mixed-diagnosis-only CV/LOCO/pairwise transfer, and matched pooled AUC), Wilson score 95% CIs for single-diagnosis per-site accuracies, and Hanley–McNeil asymptotic 95% CIs where only summary confusion matrices and AUCs were stored (extended LODO, direct cross-cohort transfer, Models Genesis, and BrainSegFounder).", True),
        (" Principal component analysis (PCA) was applied to the 232 standardized Variant-B radiomic features across all 333 OpenNeuro subjects.", False),
    ], marked_mode=marked_mode)

    # 3. Results
    add_heading_p(doc, "3. Results", level=1)
    add_heading_p(doc, "3.1 Pooled cross-validation versus leave-one-dataset-out and the concave-up ROC", level=2, marked_rev=True, marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Under pooled stratified five-fold cross-validation on 232 scans (ds000245, ds003592, ds004392, ds004471), the DenseNet121 transfer-learning model achieved a fold mean AUC of 0.977 ± 0.013 (out-of-fold pooled AUC 0.983, 95% CI 0.969–0.993; pooled accuracy 0.918, 95% CI 0.884–0.953; fold accuracy 0.904 ± 0.051; F1-score 0.903 ± 0.046; Figure 1a).", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Under LODO validation on the identical 232 scans, the same pipeline collapsed to a pooled AUC of 0.187 (95% CI 0.129–0.249) and a pooled accuracy of 0.224 (95% CI 0.172–0.276; Supplementary Table S3). On the only mixed-diagnosis cohort in this 4-dataset set (ds000245), held-out AUC was 0.529 (95% CI 0.349–0.706; accuracy 0.568, 95% CI 0.432–0.705). In the extended LODO adding ds001907 (278 scans), pooled AUC was 0.465 (95% CI 0.397–0.532; accuracy 0.460, 95% CI 0.403–0.519), with held-out AUCs of 0.625 (95% CI 0.446–0.805) for ds000245 and 0.524 (95% CI 0.355–0.693) for ds001907. Direct transfer between the two mixed-diagnosis OpenNeuro cohorts was below chance in both directions: 0.358 (95% CI 0.198–0.518) for ds000245 → ds001907 and 0.248 (95% CI 0.104–0.392) for ds001907 → ds000245. Neither domain-adversarial training (pooled LODO AUC 0.134, 95% CI 0.088–0.184; ds000245 0.466 [0.292–0.647]), ROI cropping (0.190 [0.135–0.249]; ds000245 0.463 [0.280–0.664]), nor 3D foundation models (Models Genesis LODO AUC 0.330 [0.261–0.399]; BrainSegFounder 0.402 [0.336–0.468]) rescued cross-cohort generalization.", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Mechanism of the concave-up ROC curve (Figure 2a). Unlike a chance classifier that tracks the main diagonal, the pooled LODO ROC curves (AUC 0.187 and 0.134) bow markedly downward into a concave-up arc below the diagonal (note: the corresponding file was mislabelled as Figure 3.TIF in the initial submission and is corrected to Figure 2.TIF here). Per-site probability inspection (Figure 2b) demonstrates that this concave-up geometry reflects systematic ranking inversion produced by the interaction between training-fold label prior shift and cohort-signature memorization. Specifically, when the control-only cohort ds003592 (90 HC) is held out, the remaining training set is 79.6% PD (113/142), and the classifier assigns a mean predicted P(PD) of 0.792 to ds003592, misclassifying 96.7% (87/90; accuracy 3.3%, 95% CI 1.1–9.3%) as PD. Conversely, when the PD-only cohorts ds004392 (58 PD) or ds004471 (40 PD) are held out, the training sets are dominated by controls (31.6% and 38.0% PD, respectively), yielding mean P(PD) of 0.338 and 0.395 and calling only 22.4% (95% CI 13.6–34.7%) and 27.5% (95% CI 16.1–42.8%) PD. Because held-out healthy controls receive systematically higher PD probabilities than held-out PD patients, sweeping the classification threshold traces a concave-up curve strictly beneath the chance diagonal.", True),
    ], marked_mode=marked_mode)

    # 3.2 Mixed-diagnosis-only
    add_heading_p(doc, "3.2 Mixed-diagnosis-only classification without single-diagnosis confounding", level=2, marked_rev=True, marked_mode=marked_mode)
    add_seg_p(doc, [
        ("To test whether performance collapse persists when single-diagnosis cohorts are excluded completely, we evaluated 232 Variant-B subcortical radiomics features across (a) the three OpenNeuro mixed-diagnosis cohorts (n = 145: 69 PD, 76 HC) and (b) all five single-scanner mixed cohorts adding Tao Wu and NEUROCON (n = 228: 116 PD, 112 HC; Figure 1a, Supplementary Table S2). Even within the three OpenNeuro mixed cohorts, pooled 5-fold CV yielded above-chance apparent discrimination for non-linear models that can exploit residual site-by-prevalence differences (34.1% PD in ds000245, 45.7% in ds001907, 60.0% in ds005892): Random Forest AUC 0.665 (95% CI 0.570–0.751) and RBF-SVM 0.621 (95% CI 0.529–0.712), versus logistic regression 0.516 (95% CI 0.420–0.611). Under leave-one-cohort-out (LOCO) validation, however, all models collapsed to or below chance: logistic regression 0.392 (95% CI 0.301–0.484), RBF-SVM 0.346 (95% CI 0.261–0.436), and Random Forest 0.338 (95% CI 0.253–0.427); even transductive within-cohort feature centring did not recover transfer (LOCO AUC 0.355–0.422). Across all five single-scanner cohorts (n = 228), pooled CV gave 0.564–0.592, whereas raw LOCO yielded 0.458 (95% CI 0.383–0.534) for logistic regression, 0.456 (0.380–0.528) for RBF-SVM, and 0.466 (0.391–0.541) for Random Forest, and transductive cohort-centred LOCO yielded 0.404–0.544 (all 95% CIs straddling 0.50). Across all 20 directed pairwise cohort-to-cohort transfers in the five-cohort set (Supplementary Table S2), mean logistic-regression AUC was 0.465 raw (range 0.293–0.623) and 0.471 cohort-centred (range 0.340–0.625); across the 6 pairs among the three OpenNeuro mixed cohorts, mean AUC was 0.386 raw (0.405 cohort-centred).", True),
    ], marked_mode=marked_mode)

    # 3.3 CSR + Sensitivity
    add_heading_p(doc, "3.3 Confound-to-signal ratio and sensitivity to target disease effect size", level=2, marked_rev=True, marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Between-cohort nuisance variation measured exclusively in healthy controls was 2.8 to 7.5 times the reference literature disease effect (d = 0.585) across unharmonized feature families: DBM volumetric median nuisance |d| = 1.63 (CSR = 2.79), subcortical radiomics 1.70 (CSR = 2.90), whole-brain central radiomics 2.25 (CSR = 3.84), and BrainSegFounder 3D deep embeddings 4.40 (CSR = 7.53, with 87.9% of feature variance explained by cohort identity). ComBat reduced subcortical radiomics median nuisance |d| to 0.17 (CSR = 0.29), yet as shown in Section 3.5 did not yield transferable disease discrimination.", False),
        (" In the sensitivity analysis across alternative disease effect assumptions (Supplementary Table S1), every unharmonized feature family retained a median CSR well above 2.0 even under an optimistic target of d = 0.80 (range 2.04–5.50), rising to 4.08–11.00 for d = 0.40 and 8.15–22.00 for a subtle target of d = 0.20. Thus, the qualitative conclusion that unharmonized cohort confounding overwhelms plausible structural PD effects holds across all realistic benchmark choices.", True),
    ], marked_mode=marked_mode)

    # 3.4 Null model & QC
    add_heading_p(doc, "3.4 The site-effect null model, PCA structure, and cohort registration/acquisition audit", level=2, marked_rev=True, marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Using the corrected per-dataset participant mapping and healthy-control-only age residualisation (matching the Section 2.3 information barrier and Table 3), contrasts between healthy control groups from different cohorts spanned |d| = 0.290, 0.461, 0.814, 4.134, 4.751 and 4.865 (median 2.47; Figure 1b; mean |d| = 0.52 excluding ds001907, which was distributed pre-skull-stripped; Supplementary Section S3, defect 5). The three within-cohort PD–HC contrasts were d = −0.539, −0.126 and −0.446 (mean −0.370), indistinguishable from disease-free between-cohort contrasts (Welch t = 0.84, p = 0.428; 83% [5/6] of HC–HC contrasts were at least as large as the mean within-cohort PD effect).", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("In PCA of the 232 subcortical radiomic features across all 333 OpenNeuro subjects (Figure 3), PC1 accounted for 28.9% of total variance, of which 81.1% was explained by cohort identity and only 0.7% by diagnosis within mixed cohorts. Cohort identity was recovered from the top ten PCs with 77% accuracy (majority-class baseline 27%), whereas within-cohort diagnosis after cohort-mean removal achieved cross-validated AUC 0.40. ", False),
        ("Because ds004471 patients have a mean age of 60.2 ± 6.8 years rather than an extreme age offset, the strong separation of ds004471 along PC1 is likely related to its distinct acquisition geometry (100% anisotropic scans at 1.40 mm slice thickness) and lower affine registration Dice overlap (0.678 ± 0.028) compared with isotropic cohorts such as ds003592 (0.744 ± 0.017) or ds005892 (0.717 ± 0.011), although isolating specific hardware versus protocol drivers would require travelling-subject data. Across all eight cohorts (416 scans total), automated registration and feature extraction succeeded in 100% of scans without pipeline errors (no scans were excluded via manual visual inspection), and within-cohort registration Dice did not differ significantly between PD and HC in any mixed cohort (all Mann–Whitney U p ≥ 0.201; Table 2), indicating that within-cohort contrasts are not driven by differential registration failure, whereas between-cohort variance in raw zero-voxel fraction (0.240 in ds000245 vs 0.859 in ds001907) and Variant-A mask fraction (0.701 vs 0.236) reflects substantial protocol and prior-processing heterogeneity.", True),
    ], marked_mode=marked_mode)

    # 3.5 Candidate feature
    add_heading_p(doc, "3.5 A candidate feature fails validation and is disqualified as a biomarker", level=2, marked_rev=True, marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Screening 58 harmonized radiomics features identified caudate first-order Minimum as the strongest candidate. Although it passed several conventional checks — age-residualized cross-cohort AUCs 0.708–0.728, consistent negative direction in ds001907 (matched d = −0.476, 95% CI −1.08 to +0.22) and ds005892 (d = −0.608, 95% CI −1.26 to +0.25), and matched pooled AUC 0.592 (95% bootstrap CI 0.446–0.698; stratified permutation p = 0.035; Figure 4) — ", True),
        ("it failed decisively as a generalizable biomarker: (1) its pooled matched effect d = −0.330 had a 95% bootstrap CI spanning zero (−0.73 to +0.19); (2) in ds000245 it reversed to d = +0.094 (−0.81 to +1.08), while the nested permutation test accounting for 58-feature screening yielded p = 0.177 (vs naive p = 0.006); (3) its sex effect in healthy controls reversed direction across cohorts (d = −0.521 vs +0.844 and +0.681), and age correlations varied from r = +0.085 to +0.599 and −0.339; (4) proper volumetry was null (warped caudate volume d = +0.033, AUC 0.503; DBM p = 0.542); and (5) in the two external single-scanner cohorts (Tao Wu and NEUROCON; Supplementary Table S4), caudate Minimum reversed sign in both (Tao Wu matched d = +0.579, 95% CI +0.18 to +1.07; NEUROCON d = +0.320, 95% CI −0.99 to +0.96; 5-cohort pooled matched d = −0.018, 95% CI −0.39 to +0.31; Figure 4), leaving zero features significant across five cohorts after FDR correction (minimum q = 0.410 under the primary ±4-year matching tolerance; minimum q = 0.544 under a stricter ±3-year sensitivity tolerance). Caudate Minimum is therefore explicitly disqualified as a disease biomarker.", True),
    ], marked_mode=marked_mode)

    # 3.6 Characterisation
    add_heading_p(doc, "3.6 Characterisation: where the apparent effect resides", level=2)
    add_plain_p(doc, (
        "Three descriptive analyses localise the apparent caudate Minimum effect (Figures 5 and 6). First, it is confined to the extreme dark tail "
        "(Figure 6a): within the same caudate ROI, matched effect decayed from d = −0.335 (Minimum) to −0.090 (10th percentile), +0.131 (median), "
        "and +0.170 (mean). Second, it was graded with cerebrospinal-fluid (CSF) adjacency (Figure 6b): d = −0.330 in caudate and −0.197 in thalamus, "
        "versus +0.323 in putamen and +0.097 in pallidum. Third, eroding the caudate ROI inward by 1 voxel (2 mm; 1,666 to 749 voxels) and 2 voxels "
        "(4 mm; 236 voxels) reduced the matched effect from d = −0.335 to −0.225 and −0.120 (a 64% reduction; Figure 6c). CSF-like voxel fraction "
        "did not differ between groups (0.1437 vs 0.1430), and both the 10 darkest and 10 brightest voxels lay directly on the ROI boundary (mean "
        "distance 1.02 and 1.03 voxels vs 1.35 for all ROI voxels; Figure 5e)."
    ))

    # 3.7 Site variance by statistic (Table 3)
    add_heading_p(doc, "3.7 Excess site variance in healthy controls alone", level=2)
    add_plain_p(doc, (
        "Across pairwise age-adjusted healthy-versus-healthy contrasts (Table 3, Figure 6d), caudate Minimum showed a median |d| of 2.47 against "
        "0.90 for caudate Mean — a 2.74-fold excess — whereas in thalamus (ratio 0.53), pallidum (0.41) and putamen (0.29), the Minimum was "
        "more stable across cohorts than the Mean."
    ))
    add_plain_p(doc, "Table 3. Between-site variability by statistic and structure, healthy controls only (median |d| of pairwise cross-cohort contrasts, age-adjusted).", bold_all=True)
    t3_headers = ["ROI", "Minimum", "10th pct", "Median", "Mean", "Minimum / Mean"]
    t3_rows = [
        ["Caudate (borders ventricle)", "2.47", "1.60", "1.10", "0.90", "2.74"],
        ["Thalamus", "0.95", "1.22", "1.81", "1.79", "0.53"],
        ["Pallidum", "0.96", "2.30", "2.24", "2.33", "0.41"],
        ["Putamen", "0.53", "1.35", "1.55", "1.81", "0.29"],
    ]
    add_styled_table(doc, t3_headers, t3_rows, marked_mode=False, col_font_pt=8.5)

    # 3.8 Refutation
    add_heading_p(doc, "3.8 Test and refutation of the proposed boundary-registration account", level=2)
    add_plain_p(doc, (
        "To test whether boundary partial-volume placement explained the pattern, we stated in advance inside the analysis script the prediction "
        "|r(Minimum, registration geometry)| > |r(Mean, registration geometry)| in the independent single-scanner ds003592 control cohort (n = 90). "
        "The prediction failed significantly in the opposite direction: caudate Mean correlated more strongly than Minimum with both registration Dice "
        "(|r| = 0.550 vs 0.210; Steiger z = −2.96, p = 0.0031) and warped mask volume (|r| = 0.581 vs 0.106; z = −4.08, p < 0.0001). Thus, while "
        "the empirical instability pattern in Sections 3.6–3.7 stands, the simple graded mask-placement hypothesis does not explain it."
    ))

    # 4. Discussion
    add_heading_p(doc, "4. Discussion", level=1)
    add_seg_p(doc, [
        ("The central observation is the stark discrepancy between internal pooled cross-validation (fold AUC 0.977 ± 0.013; OOF AUC 0.983 [0.969–0.993]) and external cross-cohort validation (LODO AUC 0.187 [0.129–0.249] on 232 scans; extended LODO 0.465 [0.397–0.532]; mixed-cohort direct transfer 0.248–0.358; mixed-diagnosis-only LOCO 0.338–0.466). ", True),
        ("This collapse persisted across ImageNet-pretrained 2D CNNs, domain-adversarial networks, ROI-cropped inputs, hand-crafted subcortical radiomics, and 3D brain foundation models. Crucially, the concave-up geometry of the LODO ROC curve (Figure 2a) reveals more than random noise: when single-diagnosis cohorts create severe label-prevalence shifts across training folds alongside strong cohort signatures, models actively invert held-out site predictions (calling 96.7% of ds003592 healthy controls PD and 72.5–77.6% of held-out PD patients healthy).", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Addressing single-diagnosis collinearity and cohort-level heterogeneity. When all three single-diagnosis cohorts were removed completely, leaving only mixed-diagnosis single-scanner cohorts (n = 145 OpenNeuro; n = 228 including Tao Wu and NEUROCON), LOCO AUCs remained indistinguishable from or below chance (0.338–0.466 raw; 0.355–0.544 under transductive within-cohort centring), confirming that failure to generalize is not an artefact of single-diagnosis pooling alone. ", True),
        ("Moreover, because our cohorts differ simultaneously in scanner hardware, slice thickness/anisotropy (Table 2), prior skull-stripping state, recruitment criteria, clinical stage, and demographic distributions (both location and spread — SD and IQR in Table 1), our results reflect broader cohort-level confounding rather than scanner physics alone. Variance heterogeneity across cohorts (such as wider age IQRs in neurocon and ds001907 or anisotropic voxel geometry in ds004471) can interact non-linearly with multivariate classifiers to amplify correlation-based shortcuts even when cohort means appear similar. Future validation on large controlled-access multi-site initiatives that recruit both PD and healthy controls under harmonized protocols at every site — such as the Parkinson's Progression Markers Initiative (PPMI) or Parkinson's Disease Biomarkers Program (PDBP) — was not conducted here but will be valuable to quantify how much cross-site bias persists when clinical inclusion criteria and acquisition protocols are standardized.", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("How to apply the healthy-versus-healthy null model in future multi-centre studies. To make this comparator routine and actionable before any diagnostic classifier is trained or tuned, we recommend a four-step protocol: (1) Construct the site-null distribution on healthy controls only — compute pairwise cross-cohort standardized differences (|d|, adjusted for age, age^2, and sex) across all candidate features or latent embeddings; (2) Compute the prospective Confound-to-Signal Ratio (CSR) against an a priori literature target effect size (and inspect sensitivity across d_target ∈ [0.20, 0.80]), flagging feature families where CSR > 1; (3) Audit multivariate leakage prior to classification by applying PCA or a linear site-classifier to healthy controls alone (and checking covariance harmonization beyond location/scale); and (4) Pair internal cross-validation with strict leave-one-cohort-out and mixed-cohort direct transfer evaluations with 95% confidence intervals, never relying on pooled folds containing single-diagnosis sites.", True),
    ], marked_mode=marked_mode)
    add_seg_p(doc, [
        ("Cautious interpretation of candidate markers and study limitations. Our evaluation of caudate first-order Minimum underscores why statistical significance (even after age adjustment, matched pairing, and permutation testing) does not establish biomarker validity. Because the feature failed external replication by reversing sign in both Tao Wu and NEUROCON, exhibited unstable sex effects, vanished under 2–4 mm ROI erosion, and carried 2.74-fold excess healthy-control site variance, it must be regarded as a disqualified cautionary example rather than a clinical PD marker. ", True),
        ("Limitations include the absence of travelling subjects (preventing isolation of pure hardware bias from recruitment/clinical heterogeneity), modest within-cohort matched sample sizes (9–18 pairs per cohort, 0.42–0.59 power for d = 0.585), lack of unified MDS-UPDRS/medication metadata across all eight open datasets, restriction to a single atlas (AAL) and T1-weighted MRI, and unresolved divergence of the thalamus across our two localising analyses.", True),
    ], marked_mode=marked_mode)

    # 5. Conclusion
    add_heading_p(doc, "5. Conclusion", level=1)
    add_seg_p(doc, [
        ("Across six OpenNeuro cohorts and two external single-scanner cohorts (416 subjects total), apparent PD classification accuracy from pooled T1-weighted MRI was inflated by cohort-level confounding rather than disease signal. When single-diagnosis sites were excluded or whole cohorts were held out, transfer performance collapsed to or below chance across deep networks, foundation models, and radiomics, with concave-up LODO ROC curves diagnostic of prior-shift and site-signature inversion. ", True),
        ("Caudate first-order Minimum failed external single-scanner replication and measurement-stability checks and is disqualified as a biomarker. Routine adoption of healthy-versus-healthy null models, mixed-cohort transfer with 95% confidence intervals, and per-cohort QC auditing is inexpensive and far more informative than pooled p-values.", True),
    ], marked_mode=marked_mode)

    # Data & Code Availability Statement
    add_heading_p(doc, "Data and Code Availability Statement", level=1, marked_rev=True, marked_mode=marked_mode)
    add_seg_p(doc, [
        ("All neuroimaging data analysed in this study are publicly available from OpenNeuro (https://openneuro.org/datasets/: ds000245, ds001907, ds003592, ds004392, ds004471, ds005892) and the International Neuroimaging Data-Sharing Initiative (INDI; Tao Wu and NEUROCON cohorts, http://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html; Badea et al., 2017). ", True),
        ("All analysis scripts, preprocessing and registration pipelines, mixed-diagnosis-only classification benchmarks, figure-generation scripts, derived non-image feature tables, cross-validation fold assignments, random seeds, package requirements, and summary metrics are packaged as a self-contained repository in Supplementary Data Sheet 1 (Data_Sheet_1_Code_and_Data.zip) and will be mirrored in a public GitHub repository upon publication ([GitHub repository URL to be inserted at proof stage; full snapshot included in Data Sheet 1]). Open-source tools and atlases used include SimpleITK (https://simpleitk.org), PyRadiomics (https://pyradiomics.readthedocs.io), DIPY (https://dipy.org), Nilearn (https://nilearn.github.io), neuroCombat (https://github.com/Jfortin1/neuroCombat), scikit-learn (https://scikit-learn.org), statsmodels (https://www.statsmodels.org), and the Automated Anatomical Labelling (AAL) and MNI152 templates distributed via Nilearn.", True),
    ], marked_mode=marked_mode)

    # Ethics / Author / Funding / COI / GenAI
    add_heading_p(doc, "Ethics Statement", level=1)
    add_plain_p(doc, (
        "This work is a secondary analysis of eight openly licensed, de-identified datasets distributed through OpenNeuro and INDI. "
        "Ethical approval and informed consent were obtained by the investigators of each original study, and no additional ethical approval was required."
    ))
    add_heading_p(doc, "Author Contributions", level=1)
    add_plain_p(doc, (
        "WX: Conceptualization, Methodology, Software, Formal analysis, Investigation, Writing – original draft, Visualization. "
        "YY: Methodology, Software, Validation, Formal analysis, Writing – review & editing. "
        "CY: Data curation, Investigation, Validation, Writing – review & editing. "
        "LY: Data curation, Investigation, Writing – review & editing. "
        "WM: Formal analysis, Visualization, Writing – review & editing. "
        "HH: Conceptualization, Supervision, Project administration, Writing – review & editing."
    ))
    add_heading_p(doc, "Funding & Conflict of Interest", level=1)
    add_plain_p(doc, "This research received no external funding. The authors declare no conflict of interest.")
    add_heading_p(doc, "Generative AI Statement", level=1)
    add_plain_p(doc, (
        "The authors declare that generative AI was used in the preparation of this manuscript. Claude "
        "(Anthropic; model Claude Sonnet 5.5, accessed through the Claude Code application, https://claude.com) "
        "was used to reformat the manuscript and the Supplementary Material to the journal's requirements; "
        "to write and run the scripts for the principal component analysis, the ROC analysis, the caudate voxel-location "
        "analysis and the redrawing of all figures; to check reported numbers against the analysis logs and result files; "
        "and to verify the reference list against Crossref. In preparing the revised submission, AI coding assistance was "
        "likewise used to automate the supplementary mixed-cohort benchmarks, table formatting, and consistency auditing. "
        "The authors reviewed and verified all AI-assisted scripts and outputs and take full responsibility for the content of the manuscript."
    ))

    # References (preserve all 24 original references)
    add_heading_p(doc, "References", level=1)
    refs = [
        "Adler, C. H., Beach, T. G., Hentz, J. G., Shill, H. A., Caviness, J. N., Driver-Dunckley, E., et al. (2014). Low clinical diagnostic accuracy of early vs advanced Parkinson disease. Neurology 83, 406-412. doi:10.1212/WNL.0000000000000641",
        "Ariz, M., Martínez, M., Alvarez, I., Fernández-Seara, M. A., Castellanos, G., Pastor, P., et al. (2024). Automatic segmentation and quantification of nigrosome-1 neuromelanin and iron in MRI: a candidate biomarker for Parkinson's disease. J. Magn. Reson. Imaging 60, 534–547. doi:10.1002/jmri.29073",
        "Avants, B. B., Epstein, C. L., Grossman, M., and Gee, J. C. (2008). Symmetric diffeomorphic image registration with cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Med. Image Anal. 12, 26–41. doi:10.1016/j.media.2007.06.004",
        "Badea, L., Onu, M., Wu, T., Roceanu, A., and Bajenaru, O. (2017). Exploring the reproducibility of functional connectivity alterations in Parkinson's disease. PLOS ONE 12, e0188196. doi:10.1371/journal.pone.0188196",
        "Bayer, J. M. M., Thompson, P. M., Ching, C. R. K., Liu, M., Chen, A., Panzenhagen, A. C., et al. (2022). Site effects how-to and when: An overview of retrospective techniques to accommodate site effects in multi-site neuroimaging analyses. Frontiers in Neurology 13, 923988. doi:10.3389/fneur.2022.923988",
        "Belov, V., Erwin-Grabner, T., Aghajani, M., Aleman, A., Amod, A. R., Basgoze, Z., et al. (2024). Multi-site benchmark classification of major depressive disorder using machine learning on cortical and subcortical measures. Scientific Reports 14, 1084. doi:10.1038/s41598-023-47934-8",
        "Cabitza, F., Campagner, A., Soares, F., García de Guadiana-Romualdo, L., Challa, F., Sulejmani, A., et al. (2021). The importance of being external: methodological insights for the external validation of machine learning models in medicine. Computer Methods and Programs in Biomedicine 208, 106288. doi:10.1016/j.cmpb.2021.106288",
        "Camacho, M., Wilms, M., Mouches, P., Almgren, H., Souza, R., Camicioli, R., et al. (2023). Explainable classification of Parkinson's disease using deep learning trained on a large multi-center database of T1-weighted MRI datasets. NeuroImage: Clinical 38, 103405. doi:10.1016/j.nicl.2023.103405",
        "Chen, A. A., Beer, J. C., Tustison, N. J., Cook, P. A., Shinohara, R. T., Shou, H., et al. (2022). Mitigating site effects in covariance for machine learning in neuroimaging data. Human Brain Mapping 43, 1179-1195. doi:10.1002/hbm.25688",
        "Chyzhyk, D., Varoquaux, G., Milham, M., and Thirion, B. (2022). How to remove or control confounds in predictive models, with applications to brain biomarkers. GigaScience 11, giac014. doi:10.1093/gigascience/giac014",
        "Colliot, O., Thibeau-Sutre, E., and Burgos, N. (2023). Reproducibility in machine learning for medical imaging. In Machine Learning for Brain Disorders, Neuromethods vol. 197 (New York: Humana), 631–653. doi:10.1007/978-1-0716-3195-9_21",
        "Flint, C., Cearns, M., Opel, N., Redlich, R., Mehler, D. M. A., Emden, D., et al. (2021). Systematic misestimation of machine learning performance in neuroimaging studies of depression. Neuropsychopharmacology 46, 1510-1517. doi:10.1038/s41386-021-01020-7",
        "Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., et al. (2016). Domain-adversarial training of neural networks. J. Mach. Learn. Res. 17, 1–35.",
        "Garyfallidis, E., Brett, M., Amirbekian, B., Rokem, A., van der Walt, S., Descoteaux, M., et al. (2014). Dipy, a library for the analysis of diffusion MRI data. Frontiers in Neuroinformatics 8, 8. doi:10.3389/fninf.2014.00008",
        "Hartono, S., Chen, R. C., Welton, T., Tan, A. S., Lee, W., Teh, P. Y., et al. (2023). Quantitative iron–neuromelanin MRI associates with motor severity in Parkinson's disease and matches radiological disease classification. Frontiers in Aging Neuroscience 15, 1287917. doi:10.3389/fnagi.2023.1287917",
        "He, H., Liang, L., Tang, T., Luo, J., Wang, Y., and Cui, H. (2020). Progressive brain changes in Parkinson's disease: A meta-analysis of structural magnetic resonance imaging studies. Brain Research 1740, 146847. doi:10.1016/j.brainres.2020.146847",
        "He, N., Chen, Y., LeWitt, P. A., Yan, F., and Haacke, E. M. (2023). Application of Neuromelanin MR Imaging in Parkinson Disease. Journal of Magnetic Resonance Imaging 57, 337-352. doi:10.1002/jmri.28414",
        "Heus, P., Damen, J. A. A. G., Pajouheshnia, R., Scholten, R. J. P. M., Reitsma, J. B., Collins, G. S., et al. (2018). Poor reporting of multivariable prediction model studies: towards a targeted implementation strategy of the TRIPOD statement. BMC Medicine 16, 120. doi:10.1186/s12916-018-1099-2",
        "Huang, G., Liu, Z., Van Der Maaten, L., and Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2261-2269. doi:10.1109/CVPR.2017.243",
        "Jin, L., Wang, J., Wang, C., Lian, D., Zhou, Y., Zhang, Y., et al. (2019). Combined Visualization of Nigrosome-1 and Neuromelanin in the Substantia Nigra Using 3T MRI for the Differential Diagnosis of Essential Tremor and de novo Parkinson's Disease. Frontiers in Neurology 10, 100. doi:10.3389/fneur.2019.00100",
        "Johnson, W. E., Li, C., and Rabinovic, A. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. Biostatistics 8, 118-127. doi:10.1093/biostatistics/kxj037",
        "la Roi-Teeuw, H. M., van Royen, F. S., de Hond, A., Zahra, A., de Vries, S., Bartels, R., et al. (2024). Don't be misled: 3 misconceptions about external validation of clinical prediction models. Journal of Clinical Epidemiology 172, 111387. doi:10.1016/j.jclinepi.2024.111387",
        "Moassefi, M., Rouzrokh, P., Conte, G. M., Vahdati, S., Fu, T., Tahmasebi, A., et al. (2023). Reproducibility of Deep Learning Algorithms Developed for Medical Imaging Analysis: A Systematic Review. Journal of Digital Imaging 36, 2306-2312. doi:10.1007/s10278-023-00870-5",
        "Parker, C. S., Veale, T., Bocchetta, M., Slattery, C. F., Malone, I. B., Thomas, D. L., et al. (2021). Not all voxels are created equal: Reducing estimation bias in regional NODDI metrics using tissue-weighted means. NeuroImage 245, 118749. doi:10.1016/j.neuroimage.2021.118749",
        "Pitcher, T. L., Melzer, T. R., MacAskill, M. R., Graham, C. F., Livingston, L., Keenan, R. J., et al. (2012). Reduced striatal volumes in Parkinson's disease: a magnetic resonance imaging study. Translational Neurodegeneration 1, 17. doi:10.1186/2047-9158-1-17",
        "Poewe, W., Seppi, K., Tanner, C. M., Halliday, G. M., Brundin, P., Volkmann, J., et al. (2017). Parkinson disease. Nat. Rev. Dis. Primers 3, 17013. doi:10.1038/nrdp.2017.13",
        "Poirot, M. G., Caan, M. W. A., Ruhe, H. G., Bjørnerud, A., Groote, I., Reneman, L., et al. (2022). Robustness of radiomics to variations in segmentation methods in multimodal brain MRI. Scientific Reports 12, 16712. doi:10.1038/s41598-022-20703-9",
        "Rizzo, G., Copetti, M., Arcuti, S., Martino, D., Fontana, A., and Logroscino, G. (2016). Accuracy of clinical diagnosis of Parkinson disease. Neurology 86, 566-576. doi:10.1212/WNL.0000000000002350",
        "Shi, D., Zhang, H., Wang, G., Wang, S., Yao, X., Li, Y., et al. (2022). Machine Learning for Detecting Parkinson's Disease by Resting-State Functional Magnetic Resonance Imaging: A Multicenter Radiomics Analysis. Frontiers in Aging Neuroscience 14, 806828. doi:10.3389/fnagi.2022.806828",
        "Snell, K. I. E., Archer, L., Ensor, J., Bonnett, L. J., Debray, T. P. A., Phillips, B., et al. (2021). External validation of clinical prediction models: simulation-based sample size calculations were more reliable than rules-of-thumb. Journal of Clinical Epidemiology 135, 79-89. doi:10.1016/j.jclinepi.2021.02.011",
        "Tzourio-Mazoyer, N., Landeau, B., Papathanassiou, D., Crivello, F., Etard, O., Delcroix, N., et al. (2002). Automated Anatomical Labeling of Activations in SPM Using a Macroscopic Anatomical Parcellation of the MNI MRI Single-Subject Brain. NeuroImage 15, 273-289. doi:10.1006/nimg.2001.0978",
        "van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., et al. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research 77, e104-e107. doi:10.1158/0008-5472.CAN-17-0339",
        "Yamashita, A., Yahata, N., Itahashi, T., Lisi, G., Yamada, T., Ichikawa, N., et al. (2019). Harmonization of resting-state functional MRI data across multiple imaging sites via the separation of site differences into sampling bias and measurement bias. PLOS Biology 17, e3000042. doi:10.1371/journal.pbio.3000042",
        "Zhou, Z., Sodha, V., Rahman Siddiquee, M. M., Feng, R., Tajbakhsh, N., Gotway, M. B., et al. (2019). Models Genesis: Generic Autodidactic Models for 3D Medical Image Analysis. Lecture Notes in Computer Science, 384-393. doi:10.1007/978-3-030-32251-9_42",
        "Zwanenburg, A., Vallières, M., Abdalah, M. A., Aerts, H. J. W. L., Andrearczyk, V., Apte, A., et al. (2020). The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping. Radiology 295, 328-338. doi:10.1148/radiol.2020191145",
    ]
    for rf in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(rf)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9.5)

    # Figures & Legends (with embedded PNG previews and corrected Figure 1-6 numbering)
    add_heading_p(doc, "Figure Legends and Figures", level=1)
    fig_legends = [
        ("Figure 1. Apparent accuracy is a property of the validation design, and cohort variation is as large as the disease effect. ",
         "(a) AUC with 95% confidence intervals across pooled cross-validation, leave-one-dataset-out (LODO) validation, direct transfer between mixed-diagnosis cohorts, and the newly added mixed-diagnosis-only radiomics sensitivity check (3 OpenNeuro mixed cohorts, n = 145; 5 single-scanner cohorts, n = 228). (b) Age-adjusted absolute effect sizes (|Cohen's d|, log scale) for caudate first-order Minimum: orange, pairwise healthy-vs-healthy contrasts across cohorts; blue, within-cohort same-scanner PD-vs-HC contrasts."),
        ("Figure 2. Concave-up LODO ROC curve and held-out probability inversion (corrected numbering from former Figure 3.TIF). ",
         "(a) Pooled LODO ROC curves on 232 scans without (AUC = 0.187 [0.129–0.249]) and with domain-adversarial training (AUC = 0.134 [0.088–0.184]), bowing strictly below the chance diagonal in a concave-up arc. (b) Predicted P(PD) per subject when each cohort is held out, annotated with both % called PD and the training-fold PD prevalence, illustrating how prior-shift and cohort-signature memorization invert predictions."),
        ("Figure 3. Cohort, not diagnosis, structures the feature space (corrected numbering from former Figure 4.TIF). ",
         "PCA of 232 subcortical Variant-B radiomic features across 333 OpenNeuro subjects, coloured by cohort (a) and by diagnosis (b; circles = mixed cohorts, triangles = single-diagnosis cohorts). (c) Variance explained (%) by cohort vs within-cohort diagnosis across PC1–PC7."),
        ("Figure 4. Best-case matched estimate for caudate first-order Minimum and external reversal (corrected numbering from former Figure 2.TIF). ",
         "Age- and sex-matched effect (Cohen's d with 95% bootstrap CI) across the three OpenNeuro mixed cohorts (top block; pooled d = −0.330 [−0.73, +0.19], perm p = 0.035) and the two external single-scanner INDI cohorts (bottom block; Tao Wu +0.579 [+0.18, +1.07], NEUROCON +0.320 [−0.99, +0.96]; 5-cohort pooled d = −0.018 [−0.39, +0.31])."),
        ("Figure 5. Where the extreme values of the caudate fall. ",
         "(a) Caudate ROI (AAL atlas) on the MNI152 template with 1- and 2-voxel erosions. (b, c) Spatial counts of the 10 darkest and 10 brightest voxels across 145 mixed-cohort subjects. (d) Example healthy control in MNI space. (e) Per-subject mean Euclidean distance from the ROI boundary (1.02 and 1.03 voxels vs 1.35 for all ROI voxels)."),
        ("Figure 6. Four descriptive analyses localising the apparent caudate effect. ",
         "(a) Tail sensitivity (Minimum −0.335 to Mean +0.170). (b) CSF-interface gradient across subcortical ROIs. (c) Decay under inward boundary erosion (0, 2, 4 mm). (d) Healthy-control between-site median |d| for Mean vs Minimum across structures (caudate Minimum/Mean ratio 2.74)."),
    ]
    for idx_f, (title_part, body_part) in enumerate(fig_legends, start=1):
        add_seg_p(doc, [(title_part, False), (body_part, True)], marked_mode=marked_mode)
        img_p = doc.add_paragraph()
        img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        img_p.add_run().add_picture(os.path.join(FIG_DIR, f"Figure {idx_f}.png"), width=Inches(6.2))

    out_path = os.path.join(OUT_DIR, "Manuscript_Revised_Marked.docx" if marked_mode else "Manuscript_Revised_Clean.docx")
    doc.save(out_path)
    print(f"Saved {out_path}")


def build_supplementary():
    doc = Document()
    configure_doc(doc)
    add_heading_p(doc, "Supplementary Material (Revised): Manuscript ID 2031561", level=0)
    add_plain_p(doc, "Site- and cohort-associated confounding inflates pooled Parkinson's disease classification from multi-cohort T1-weighted MRI", italic_all=True)

    # S1: CSR sensitivity
    add_heading_p(doc, "Supplementary Table S1. Confound-to-signal ratio (CSR) sensitivity across alternative target disease effect sizes", level=1)
    add_plain_p(doc, "Nuisance |d| medians are measured exclusively in age-adjusted healthy controls. Values > 1.0 indicate cohort-level nuisance variation exceeds the assumed disease effect.")
    s1_df = pd.read_csv(os.path.join(TAB_DIR, "csr_sensitivity.csv"))
    add_styled_table(doc, list(s1_df.columns), s1_df.values.tolist(), col_font_pt=8.0)

    # S2: Mixed-diagnosis-only classification & pairwise transfer
    add_heading_p(doc, "Supplementary Table S2. Mixed-diagnosis-only subcortical radiomics classification (single-diagnosis sites excluded)", level=1)
    add_plain_p(doc, "Part A: Pooled 5-fold stratified CV vs Leave-One-Cohort-Out (LOCO) across three OpenNeuro mixed cohorts (n = 145) and all five single-scanner cohorts (n = 228).")
    s2a = pd.read_csv(os.path.join(TAB_DIR, "mixed_diagnosis_only_classification.csv"))
    add_styled_table(doc, list(s2a.columns), s2a.values.tolist(), col_font_pt=7.5)

    add_plain_p(doc, "Part B: All 20 directed pairwise cohort-to-cohort transfers (L2 logistic regression, raw standardized and cohort-centred features).", space_after=4) if False else add_plain_p(doc, "Part B: All directed pairwise cohort-to-cohort transfers (L2 logistic regression, raw standardized and cohort-centred features).")
    s2b = pd.read_csv(os.path.join(TAB_DIR, "mixed_pairwise_transfer.csv"))
    add_styled_table(doc, list(s2b.columns), s2b.values.tolist(), col_font_pt=7.2)

    # S3: Detailed LODO / Extended LODO / Direct Transfer with 95% CIs
    add_heading_p(doc, "Supplementary Table S3. Detailed validation performance with 95% confidence intervals across deep-learning and foundation-model baselines", level=1)
    vc = M["validation_cis"]
    s3_headers = ["Validation Design / Cohort", "n (PD/HC)", "AUC [95% CI]", "Accuracy [95% CI]", "Notes / Train PD%"]
    s3_rows = [
        ["Pooled 5-fold CV DenseNet121 (OOF)", "232 (113/119)", f"{vc['pooled_cv_densenet_oof']['auc']:.3f} [{vc['pooled_cv_densenet_oof']['auc_ci95'][0]:.3f}, {vc['pooled_cv_densenet_oof']['auc_ci95'][1]:.3f}]", f"{vc['pooled_cv_densenet_oof']['acc']:.3f} [{vc['pooled_cv_densenet_oof']['acc_ci95'][0]:.3f}, {vc['pooled_cv_densenet_oof']['acc_ci95'][1]:.3f}]", "Fold mean AUC 0.977 ± 0.013"],
        ["Primary LODO DenseNet121 (pooled)", "232 (113/119)", f"{vc['lodo_densenet_pooled']['auc']:.3f} [{vc['lodo_densenet_pooled']['auc_ci95'][0]:.3f}, {vc['lodo_densenet_pooled']['auc_ci95'][1]:.3f}]", f"{vc['lodo_densenet_pooled']['acc']:.3f} [{vc['lodo_densenet_pooled']['acc_ci95'][0]:.3f}, {vc['lodo_densenet_pooled']['acc_ci95'][1]:.3f}]", "Concave-up ROC inversion"],
    ]
    for ds_n in ["ds000245", "ds003592", "ds004392", "ds004471"]:
        st = vc["lodo_densenet_per_site"][ds_n]
        auc_s = f"{st['auc']:.3f} [{st['auc_ci95'][0]:.3f}, {st['auc_ci95'][1]:.3f}]" if st["auc"] is not None else "n/a (single-dx)"
        s3_rows.append([
            f"  LODO held-out: {ds_n}", f"{st['n']} ({st['n_pd']}/{st['n_hc']})",
            auc_s, f"{st['acc']:.3f} [{st['acc_ci95'][0]:.3f}, {st['acc_ci95'][1]:.3f}]",
            f"Called PD: {st['pct_called_pd']:.1f}% | Train PD: {st['train_pd_pct']:.1f}%"
        ])
    s3_rows.extend([
        ["LODO + DANN (pooled)", "232 (113/119)", f"{vc['lodo_dann_pooled']['auc']:.3f} [{vc['lodo_dann_pooled']['auc_ci95'][0]:.3f}, {vc['lodo_dann_pooled']['auc_ci95'][1]:.3f}]", f"{vc['lodo_dann_pooled']['acc']:.3f} [{vc['lodo_dann_pooled']['acc_ci95'][0]:.3f}, {vc['lodo_dann_pooled']['acc_ci95'][1]:.3f}]", f"ds000245 AUC {vc['lodo_dann_ds000245']['auc']:.3f}"],
        ["LODO + ROI crop (pooled)", "232 (113/119)", f"{vc['lodo_roi_pooled']['auc']:.3f} [{vc['lodo_roi_pooled']['auc_ci95'][0]:.3f}, {vc['lodo_roi_pooled']['auc_ci95'][1]:.3f}]", f"{vc['lodo_roi_pooled']['acc']:.3f} [{vc['lodo_roi_pooled']['acc_ci95'][0]:.3f}, {vc['lodo_roi_pooled']['acc_ci95'][1]:.3f}]", f"ds000245 AUC {vc['lodo_roi_ds000245']['auc']:.3f}"],
        ["Extended LODO DenseNet121 (pooled)", "278 (134/144)", f"{vc['extended_lodo_pooled']['auc']:.3f} [{vc['extended_lodo_pooled']['auc_ci95_hm'][0]:.3f}, {vc['extended_lodo_pooled']['auc_ci95_hm'][1]:.3f}]", f"{vc['extended_lodo_pooled']['acc']:.3f} [{vc['extended_lodo_pooled']['acc_ci95_wilson'][0]:.3f}, {vc['extended_lodo_pooled']['acc_ci95_wilson'][1]:.3f}]", "Hanley–McNeil / Wilson 95% CIs"],
        ["Direct transfer: ds000245 → ds001907", "46 (21/25)", f"{vc['direct_transfer']['ds000245_to_ds001907']['auc']:.3f} [{vc['direct_transfer']['ds000245_to_ds001907']['auc_ci95_hm'][0]:.3f}, {vc['direct_transfer']['ds000245_to_ds001907']['auc_ci95_hm'][1]:.3f}]", f"{vc['direct_transfer']['ds000245_to_ds001907']['acc']:.3f} [{vc['direct_transfer']['ds000245_to_ds001907']['acc_ci95_wilson'][0]:.3f}, {vc['direct_transfer']['ds000245_to_ds001907']['acc_ci95_wilson'][1]:.3f}]", "Mixed-to-mixed transfer"],
        ["Direct transfer: ds001907 → ds000245", "44 (15/29)", f"{vc['direct_transfer']['ds001907_to_ds000245']['auc']:.3f} [{vc['direct_transfer']['ds001907_to_ds000245']['auc_ci95_hm'][0]:.3f}, {vc['direct_transfer']['ds001907_to_ds000245']['auc_ci95_hm'][1]:.3f}]", f"{vc['direct_transfer']['ds001907_to_ds000245']['acc']:.3f} [{vc['direct_transfer']['ds001907_to_ds000245']['acc_ci95_wilson'][0]:.3f}, {vc['direct_transfer']['ds001907_to_ds000245']['acc_ci95_wilson'][1]:.3f}]", "Mixed-to-mixed transfer"],
        ["Models Genesis LODO (LogReg)", "232 (113/119)", f"{vc['models_genesis_lodo_logreg']['auc']:.3f} [{vc['models_genesis_lodo_logreg']['auc_ci95_hm'][0]:.3f}, {vc['models_genesis_lodo_logreg']['auc_ci95_hm'][1]:.3f}]", "—", "3D foundation model"],
        ["BrainSegFounder LODO (LogReg)", "278 (134/144)", f"{vc['brainsegfounder_lodo_logreg']['auc']:.3f} [{vc['brainsegfounder_lodo_logreg']['auc_ci95_hm'][0]:.3f}, {vc['brainsegfounder_lodo_logreg']['auc_ci95_hm'][1]:.3f}]", "—", "UK Biobank SwinViT"],
    ])
    add_styled_table(doc, s3_headers, s3_rows, col_font_pt=7.5)

    # S4: External Single-Scanner Cohorts & Five Pipeline Defects
    add_heading_p(doc, "Supplementary Table S4. External single-scanner replication (Tao Wu & NEUROCON) for caudate first-order Minimum", level=1)
    mc = vc["matched_caudate_minimum"]
    s4_headers = ["Cohort", "Matched Pairs (mean)", "Matched Cohen's d [95% CI]", "Matched AUC"]
    s4_rows = []
    for ds in ["ds000245", "ds001907", "ds005892", "taowu", "neurocon"]:
        r = mc[ds]
        s4_rows.append([ds, f"{r['pairs_mean']:.1f}", f"{r['d']:+.3f} [{r['d_ci95'][0]:+.2f}, {r['d_ci95'][1]:+.2f}]", f"{r['auc_matched']:.3f}"])
    s4_rows.append(["Pooled OpenNeuro (3 cohorts)", f"{mc['pooled_3_openneuro']['pairs_int']}", f"{mc['pooled_3_openneuro']['d']:+.3f} [{mc['pooled_3_openneuro']['d_ci95'][0]:+.2f}, {mc['pooled_3_openneuro']['d_ci95'][1]:+.2f}]", f"{mc['pooled_3_openneuro']['auc_matched']:.3f} [{mc['pooled_3_openneuro']['auc_matched_ci95'][0]:.3f}, {mc['pooled_3_openneuro']['auc_matched_ci95'][1]:.3f}]"])
    s4_rows.append(["Pooled All 5 Cohorts", f"{mc['pooled_5_all']['pairs_int']}", f"{mc['pooled_5_all']['d']:+.3f} [{mc['pooled_5_all']['d_ci95'][0]:+.2f}, {mc['pooled_5_all']['d_ci95'][1]:+.2f}]", "—"])
    add_styled_table(doc, s4_headers, s4_rows, col_font_pt=8.0)

    add_heading_p(doc, "Supplementary Section S3. Summary of five pipeline defects identified during development", level=1)
    defects = [
        "Defect 1 (Information leakage / unnested feature selection): Selecting features on pooled data prior to cross-validation or permutation inflated apparent significance (naive p = 0.006 vs nested permutation p = 0.177).",
        "Defect 2 (Centre-of-mass registration initialisation with z-scored template): Centre-of-mass alignment divides by total template intensity; using zero-mean z-scored templates caused catastrophic misalignment until replaced by non-negative min–max normalisation.",
        "Defect 3 (Participant identifier collisions across datasets): Merging demographic tables on bare 'sub-XXX' IDs without dataset keys caused ds004471 to inherit young control ages from ds003592; resolved by keying strictly on (dataset, participant_id).",
        "Defect 4 (Fixed bin width after z-scoring): Applying a fixed intensity bin width to arbitrary-unit z-scored T1 images collapsed grey-level discretization; resolved by IBSI-compliant fixed bin count (64 bins).",
        "Defect 5 (Heterogeneous prior skull-stripping and background zeros): ds001907 is distributed pre-skull-stripped (85.9% background zeros vs 24.0–40.9% elsewhere), distorting >5th-percentile normalisation (Variant A) and producing |d| = 4.13–4.87 HC–HC offsets under Variant B (even after brain-mask standardisation); mitigated via common MNI152 brain-mask z-scoring (Variant B).",
    ]
    for df_txt in defects:
        add_plain_p(doc, df_txt, font_size=10)

    out_path = os.path.join(OUT_DIR, "Supplementary_Material_Revised.docx")
    doc.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    build_manuscript(marked_mode=False)
    build_manuscript(marked_mode=True)
    build_supplementary()
