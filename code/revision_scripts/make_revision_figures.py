#!/usr/bin/env python3
"""
Generate all 6 revised figures (300 DPI LZW RGB TIFF + PNG previews) for
Frontiers in Neuroinformatics Manuscript ID 2031561.
Corrects original Figure 2/3/4 file-numbering mismatch and adds 95% CIs,
mixed-diagnosis-only benchmarks, concave-up ROC diagnostics, and 5-cohort forest block.
"""
import json
import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

BASE = "/master/wang_zi_wei/cursor/pd_classification"
OUT_DIR = os.path.join(BASE, "frontiers_revision_2031561")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "revision_metrics.json")) as f:
    M = json.load(f)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#7A7A7A"
GOLD = "#E69F00"
SKY = "#56B4E9"


def save_fig(fig, name_no_ext):
    png_path = os.path.join(FIG_DIR, f"{name_no_ext}.png")
    tif_path = os.path.join(FIG_DIR, f"{name_no_ext}.TIF")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    im = Image.open(png_path).convert("RGB")
    im.save(tif_path, compression="tiff_lzw", dpi=(300, 300))
    plt.close(fig)
    print(f"Saved {tif_path} ({im.size})")


# ==============================================================================
# FIGURE 1: Validation design gap (with 95% CIs) + HC-HC site null vs PD-HC
# ==============================================================================
def make_figure_1():
    vc = M["validation_cis"]
    mx = M["mixed_diagnosis_only"]

    rows_top = [
        ("Pooled 5-fold CV, DenseNet121 (232 scans)", vc["pooled_cv_densenet_oof"]["auc"], vc["pooled_cv_densenet_oof"]["auc_ci95"], "pool"),
        ("LODO DenseNet121, pooled (232 scans)", vc["lodo_densenet_pooled"]["auc"], vc["lodo_densenet_pooled"]["auc_ci95"], "held"),
        ("LODO DenseNet121, ds000245 held out", vc["lodo_densenet_per_site"]["ds000245"]["auc"], vc["lodo_densenet_per_site"]["ds000245"]["auc_ci95"], "held"),
        ("Extended LODO DenseNet121, pooled (278)", vc["extended_lodo_pooled"]["auc"], vc["extended_lodo_pooled"]["auc_ci95_hm"], "held"),
        ("Extended LODO, ds000245 held out", vc["extended_lodo_per_fold"][0]["auc"], vc["extended_lodo_per_fold"][0]["auc_ci95_hm"], "held"),
        ("Extended LODO, ds001907 held out", vc["extended_lodo_per_fold"][1]["auc"], vc["extended_lodo_per_fold"][1]["auc_ci95_hm"], "held"),
        ("Transfer ds000245 → ds001907", vc["direct_transfer"]["ds000245_to_ds001907"]["auc"], vc["direct_transfer"]["ds000245_to_ds001907"]["auc_ci95_hm"], "held"),
        ("Transfer ds001907 → ds000245", vc["direct_transfer"]["ds001907_to_ds000245"]["auc"], vc["direct_transfer"]["ds001907_to_ds000245"]["auc_ci95_hm"], "held"),
    ]
    rows_bot = [
        ("Mixed-only (3 cohorts, n=145): Pooled CV (RF)", mx["openneuro_3mixed"]["models"]["rf"]["pooled_cv_raw"]["auc"], mx["openneuro_3mixed"]["models"]["rf"]["pooled_cv_raw"]["auc_ci95"], "mix_pool"),
        ("Mixed-only (3 cohorts, n=145): LOCO (LogReg)", mx["openneuro_3mixed"]["models"]["logreg"]["loco_raw_pooled"]["auc"], mx["openneuro_3mixed"]["models"]["logreg"]["loco_raw_pooled"]["auc_ci95"], "mix_held"),
        ("Mixed-only (5 cohorts, n=228): Pooled CV (SVM)", mx["all_5_single_scanner"]["models"]["svm_rbf"]["pooled_cv_raw"]["auc"], mx["all_5_single_scanner"]["models"]["svm_rbf"]["pooled_cv_raw"]["auc_ci95"], "mix_pool"),
        ("Mixed-only (5 cohorts, n=228): LOCO (LogReg)", mx["all_5_single_scanner"]["models"]["logreg"]["loco_raw_pooled"]["auc"], mx["all_5_single_scanner"]["models"]["logreg"]["loco_raw_pooled"]["auc_ci95"], "mix_held"),
    ]

    y_top = np.array([12.3, 11.3, 10.3, 9.3, 8.3, 7.3, 6.3, 5.3])
    y_bot = np.array([3.0, 2.0, 1.0, 0.0])
    rows = rows_top + rows_bot
    y_pos = np.concatenate([y_top, y_bot])

    fig = plt.figure(figsize=(9.8, 8.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.55, 1.0], hspace=0.42)

    ax = fig.add_subplot(gs[0])

    # Subtle horizontal row guides and AUC = 0.5 / 1.0 references confined to [0, 1]
    for y in y_pos:
        ax.plot([0.0, 1.0], [y, y], color="#EFEFEF", linewidth=0.8, zorder=0)
    ax.plot([1.0, 1.0], [-0.55, 12.8], color="#E2E2E2", linewidth=0.8, zorder=0)
    ax.plot([0.5, 0.5], [-0.55, 12.8], color="#555555", linestyle="--", linewidth=1.0, zorder=1)

    # Shaded section divider banner masking the 0.5 dashed line cleanly
    ax.fill_between([0.0, 1.0], 3.74, 4.62, color="#F3F5F8", edgecolor="#DCE1E8", linewidth=0.7, zorder=2)
    ax.text(0.02, 4.18, "Mixed-diagnosis-only radiomics sensitivity check (single-diagnosis sites excluded)",
            fontsize=8.0, style="italic", color="#4A5568", va="center", ha="left", zorder=3)

    cmap = {"pool": BLUE, "held": ORANGE, "mix_pool": SKY, "mix_held": "#9E3D00"}
    ax.text(1.035, 13.05, "AUC [95% CI]", fontsize=8.3, fontweight="bold", color="#222222", va="center", ha="left")

    for y, (label, auc, ci, kind) in zip(y_pos, rows):
        col = cmap[kind]
        ax.plot([0.5, auc], [y, y], color=col, linewidth=2.0, alpha=0.55, zorder=3)
        ax.errorbar(auc, y, xerr=[[auc - ci[0]], [ci[1] - auc]], fmt="o", color=col,
                    markersize=5.6, elinewidth=1.5, capsize=3.2, zorder=4)
        ax.text(1.035, y, f"{auc:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]",
                va="center", ha="left", fontsize=8.2, color="#222222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_xlim(0.0, 1.33)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    ax.set_ylim(-0.75, 13.45)
    ax.set_xlabel("AUC (with 95% confidence interval)", x=0.5 / 1.33)
    ax.set_title("a  Measured performance depends on validation design (95% CIs)", loc="left", pad=46)
    handles = [
        Line2D([0], [0], marker="o", color=BLUE, label="Pooled CV (4 cohorts, DenseNet121)", markersize=6, lw=1.5),
        Line2D([0], [0], marker="o", color=ORANGE, label="Held-out / transfer (DenseNet121)", markersize=6, lw=1.5),
        Line2D([0], [0], marker="o", color=SKY, label="Pooled CV (mixed-only radiomics)", markersize=6, lw=1.5),
        Line2D([0], [0], marker="o", color="#9E3D00", label="LOCO (mixed-only radiomics)", markersize=6, lw=1.5),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, frameon=True, framealpha=0.95, fontsize=7.8)

    # Panel b: HC-vs-HC contrasts vs within-cohort PD-vs-HC contrasts (from corrected HC-only null model)
    ax2 = fig.add_subplot(gs[1])
    nm = M["hc_null_model_corrected"]
    hc_hc = np.array(nm["hc_hc_abs_sorted"], dtype=float)
    pd_hc = np.abs(np.array([x["d"] for x in nm["within_cohort_pd_hc"]], dtype=float))
    mean_pd = float(np.mean(pd_hc))
    pct_ge = int(round(100.0 * nm["frac_hc_hc_ge_mean_within_pd_hc"]))

    # Deterministic vertical offsets so closely spaced points stay visually separated
    hc_offsets = np.array([0.0, 0.06, -0.06, -0.04, 0.10, -0.10])[:len(hc_hc)]
    pd_sorted_idx = np.argsort(pd_hc)
    pd_offsets = np.zeros_like(pd_hc)
    for k, idx_p in enumerate(pd_sorted_idx):
        pd_offsets[idx_p] = [0.0, -0.06, 0.06][k % 3]

    ax2.scatter(hc_hc, 1.0 + hc_offsets, color=ORANGE, s=46, zorder=4, edgecolors="white", linewidth=0.6)
    ax2.scatter(pd_hc, 0.0 + pd_offsets, color=BLUE, s=46, zorder=4, edgecolors="white", linewidth=0.6)
    ax2.axvline(mean_pd, color=BLUE, linestyle="--", linewidth=1.2, zorder=2)
    ax2.text(mean_pd * 1.12, 0.50, f"mean site-free disease |d| = {mean_pd:.2f}\n({pct_ge}% of site contrasts ≥ this)",
             color=BLUE, fontsize=8.2, va="center",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax2.annotate("ds001907 pairs\n(pre-skull-stripped)",
                 xy=(4.55, 0.84), xytext=(4.55, 0.34),
                 ha="center", va="center", fontsize=7.8, color="#8A3B00",
                 arrowprops=dict(arrowstyle="->", color="#B56530", lw=1.0),
                 bbox=dict(boxstyle="round,pad=0.22", fc="#FFF7F2", ec="#E6C3AE", lw=0.7))
    ax2.set_xscale("log")
    ax2.set_xlim(0.08, 7.8)
    ax2.set_xticks([0.1, 0.3, 1, 3, 5])
    ax2.set_xticklabels(["0.1", "0.3", "1", "3", "5"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["PD vs HC\n(same scanner)", "HC vs HC\n(cohort/site only)"])
    ax2.set_ylim(-0.52, 1.52)
    ax2.set_xlabel("Absolute effect size (|Cohen's d|, age-adjusted, log scale)")
    ax2.set_title("b  Between-cohort differences in healthy controls match or exceed the disease effect", loc="left", pad=8)

    save_fig(fig, "Figure 1")


# ==============================================================================
# FIGURE 2: Concave-up LODO ROC + Per-subject P(PD) & training prevalence
# ==============================================================================
def make_figure_2():
    lodo = np.load(os.path.join(BASE, "results/lodo_20260723_014320/lodo_roc_data.npz"), allow_pickle=True)
    dann = np.load(os.path.join(BASE, "results/lodo_dann_20260723_081305/lodo_roc_data.npz"), allow_pickle=True)
    vc = M["validation_cis"]

    fpr_l, tpr_l, _ = roc_curve(lodo["y_true"], lodo["y_prob"])
    fpr_d, tpr_d, _ = roc_curve(dann["y_true"], dann["y_prob"])

    fig = plt.figure(figsize=(9.2, 4.3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)

    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between([0, 1], [0, 1], [0, 0], color="#FCE8DC", alpha=0.35, zorder=0)
    ax1.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1.1, label="Chance (AUC = 0.500)")
    ci_l = vc["lodo_densenet_pooled"]["auc_ci95"]
    ci_d = vc["lodo_dann_pooled"]["auc_ci95"]
    ax1.plot(fpr_l, tpr_l, color=ORANGE, linewidth=2.2,
             label=f"LODO (AUC = {vc['lodo_densenet_pooled']['auc']:.3f} [{ci_l[0]:.2f}, {ci_l[1]:.2f}])")
    ax1.plot(fpr_d, tpr_d, color=PURPLE, linewidth=2.0,
             label=f"LODO + DANN (AUC = {vc['lodo_dann_pooled']['auc']:.3f} [{ci_d[0]:.2f}, {ci_d[1]:.2f}])")
    ax1.annotate("Concave-up inversion\n(prior-shift + site bias)", xy=(0.72, 0.16), xytext=(0.30, 0.36),
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
                 fontsize=7.8, color="#444444", ha="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.9))
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlabel("False-positive rate")
    ax1.set_ylabel("True-positive rate")
    ax1.set_title("a  ROC, held-out sites (concave-up)", loc="left", pad=8)
    ax1.legend(loc="upper left", frameon=True, fontsize=7.5)

    ax2 = fig.add_subplot(gs[1])
    sites = [
        ("ds003592", "ds003592\n(HC only, n=90)"),
        ("ds000245", "ds000245\n(mixed, n=44)"),
        ("ds004392", "ds004392\n(PD only, n=58)"),
        ("ds004471", "ds004471\n(PD only, n=40)"),
    ]
    ax2.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0)
    rng = np.random.default_rng(12)
    per_site = vc["lodo_densenet_per_site"]

    for idx_s, (ds, xl) in enumerate(sites):
        m = lodo["dataset"] == ds
        yt = lodo["y_true"][m]
        yp = lodo["y_prob"][m]
        xj = idx_s + rng.normal(0, 0.09, size=len(yp))
        hc_m = yt == 0
        pd_m = yt == 1
        if hc_m.any():
            ax2.scatter(xj[hc_m], yp[hc_m], color=GREY, s=18, alpha=0.85, edgecolors="white", linewidth=0.3,
                        label="True HC" if idx_s == 0 else None)
        if pd_m.any():
            ax2.scatter(xj[pd_m], yp[pd_m], color=ORANGE, s=18, alpha=0.85, edgecolors="white", linewidth=0.3,
                        label="True PD" if idx_s == 1 else None)
        info = per_site[ds]
        ax2.text(idx_s, 1.06, f"{info['pct_called_pd']:.0f}% called PD\n(train {info['train_pd_pct']:.0f}% PD)",
                 ha="center", va="bottom", fontsize=7.6, color="#222222")

    ax2.set_xticks(range(len(sites)))
    ax2.set_xticklabels([s[1] for s in sites], fontsize=8)
    ax2.set_ylim(-0.03, 1.24)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylabel("Predicted probability of PD")
    ax2.set_title("b  LODO P(PD) & training-fold PD prevalence", loc="left", pad=8)
    ax2.legend(loc="lower right", frameon=True, fontsize=7.8)

    save_fig(fig, "Figure 2")


# ==============================================================================
# FIGURE 3: PCA of 232 subcortical radiomic features across 333 OpenNeuro subjects
# ==============================================================================
def make_figure_3():
    df = pd.read_csv(os.path.join(BASE, "data/radiomics_v3_harmonized.csv"))
    feat_cols = [c for c in df.columns if c.startswith("tplB_")]
    X = StandardScaler().fit_transform(df[feat_cols].values)
    pca = PCA(n_components=10, random_state=42)
    Z = pca.fit_transform(X)
    ds = df["dataset"].values
    y = df["label"].values

    # Orient PC1 so ds004471 sits on the positive side (matching original visualization)
    if Z[ds == "ds004471", 0].mean() < 0:
        Z[:, 0] *= -1
    if Z[ds == "ds001907", 1].mean() < 0:
        Z[:, 1] *= -1

    fig = plt.figure(figsize=(9.0, 7.2))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 0.95], hspace=0.38, wspace=0.28)

    cohort_colors = {
        "ds000245": BLUE,
        "ds001907": GREEN,
        "ds005892": PURPLE,
        "ds003592": GOLD,
        "ds004392": ORANGE,
        "ds004471": SKY,
    }
    ax1 = fig.add_subplot(gs[0, 0])
    for c, col in cohort_colors.items():
        m = ds == c
        ax1.scatter(Z[m, 0], Z[m, 1], color=col, s=16, alpha=0.8, label=c)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% of variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
    ax1.set_title("a  Coloured by cohort", loc="left", pad=6)
    ax1.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.2, markerscale=1.1)

    ax2 = fig.add_subplot(gs[0, 1])
    mixed_set = np.isin(ds, ["ds000245", "ds001907", "ds005892"])
    for lab, col, lname in [(0, GREY, "HC"), (1, ORANGE, "PD")]:
        m_mix = (y == lab) & mixed_set
        m_sin = (y == lab) & (~mixed_set)
        ax2.scatter(Z[m_mix, 0], Z[m_mix, 1], color=col, marker="o", s=16, alpha=0.8, label=f"{lname} (mixed)")
        ax2.scatter(Z[m_sin, 0], Z[m_sin, 1], color=col, marker="^", s=18, alpha=0.8, label=f"{lname} (single-dx)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("b  Same subjects, coloured by diagnosis", loc="left", pad=6)
    ax2.legend(loc="upper right", ncol=2, frameon=True, fontsize=7.0)

    # Panel c: variance explained for PC1-PC7
    ax3 = fig.add_subplot(gs[1, :])
    coh_r2, diag_r2 = [], []
    mix_mask = mixed_set
    for k in range(7):
        z = Z[:, k]
        tss = np.sum((z - z.mean()) ** 2)
        gm = {c: z[ds == c].mean() for c in np.unique(ds)}
        r2c = np.sum((np.array([gm[c] for c in ds]) - z.mean()) ** 2) / tss
        coh_r2.append(r2c * 100.0)

        zm = z[mix_mask]; dsm = ds[mix_mask]; ym = y[mix_mask]
        res_c = zm - np.array([zm[dsm == c].mean() for c in dsm])
        rss_c = np.sum(res_c ** 2)
        res_b = np.zeros_like(zm)
        for c in np.unique(dsm):
            for lb in (0, 1):
                mm = (dsm == c) & (ym == lb)
                res_b[mm] = zm[mm] - zm[mm].mean()
        r2d = (rss_c - np.sum(res_b ** 2)) / np.sum((zm - zm.mean()) ** 2)
        diag_r2.append(r2d * 100.0)

    x = np.arange(7)
    w = 0.36
    ax3.bar(x - w / 2, coh_r2, width=w, color=GOLD, label="Cohort (six groups)")
    ax3.bar(x + w / 2, diag_r2, width=w, color=BLUE, label="Diagnosis within cohort (three mixed cohorts)")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"PC{k+1}" for k in range(7)])
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Variance explained (%)")
    ax3.set_title("c  What each principal component encodes", loc="left", pad=6)
    ax3.legend(loc="upper right", frameon=True, fontsize=8)

    save_fig(fig, "Figure 3")


# ==============================================================================
# FIGURE 4: Forest plot of matched caudate first-order Minimum (3 OpenNeuro + 2 external)
# ==============================================================================
def make_figure_4():
    mc = M["validation_cis"]["matched_caudate_minimum"]
    rows = [
        ("ds000245 (OpenNeuro, ~12 pairs)", mc["ds000245"]["d"], mc["ds000245"]["d_ci95"], ORANGE),
        ("ds001907 (OpenNeuro, ~13 pairs)", mc["ds001907"]["d"], mc["ds001907"]["d_ci95"], BLUE),
        ("ds005892 (OpenNeuro, ~16 pairs)", mc["ds005892"]["d"], mc["ds005892"]["d_ci95"], BLUE),
        ("Pooled OpenNeuro (~41 pairs, perm p = 0.035)", mc["pooled_3_openneuro"]["d"], mc["pooled_3_openneuro"]["d_ci95"], BLUE),
        ("Tao Wu (INDI external, ~18 pairs)", mc["taowu"]["d"], mc["taowu"]["d_ci95"], ORANGE),
        ("NEUROCON (INDI external, ~10 pairs)", mc["neurocon"]["d"], mc["neurocon"]["d_ci95"], ORANGE),
        ("Pooled all 5 cohorts (~69 pairs)", mc["pooled_5_all"]["d"], mc["pooled_5_all"]["d_ci95"], "#444444"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y_pos = np.arange(len(rows))[::-1]
    ax.axvline(0.0, color="#222222", linewidth=1.1, zorder=1)
    ax.axhline(2.5, color="#CCCCCC", linestyle=":", linewidth=1.2)
    ax.axhline(3.5, color="#E0E0E0", linestyle="--", linewidth=0.9)
    ax.axhline(0.5, color="#E0E0E0", linestyle="--", linewidth=0.9)

    for idx_r, (y, (label, dval, ci, col)) in enumerate(zip(y_pos, rows)):
        is_pool = "Pooled" in label
        fmt = "D" if is_pool else "o"
        ms = 6.8 if is_pool else 5.5
        ax.errorbar(dval, y, xerr=[[dval - ci[0]], [ci[1] - dval]], fmt=fmt, color=col,
                    markersize=ms, elinewidth=1.8, capsize=4, zorder=4)
        ax.text(1.22, y, f"{dval:+.3f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", va="center", ha="left",
                fontsize=8.2, fontweight="bold" if is_pool else "normal", color="#222222")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_xlim(-1.55, 2.15)
    ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("Matched effect (Cohen's d, age- and sex-matched by design)")
    ax.set_title("Matched effect of caudate first-order Minimum across primary and external cohorts", loc="left", pad=8)

    save_fig(fig, "Figure 4")


# ==============================================================================
# FIGURE 5: Caudate ROI and extreme voxel positions (from figdata/)
# ==============================================================================
def make_figure_5():
    ref = np.load(os.path.join(BASE, "figdata/ref.npz"), allow_pickle=True)
    tpl = ref["tpl"]
    caud = ref["caudate"].astype(bool)
    struct = ndi.generate_binary_structure(3, 1)
    er1 = ndi.binary_erosion(caud, structure=struct, iterations=1)
    er2 = ndi.binary_erosion(caud, structure=struct, iterations=2)

    with open(os.path.join(BASE, "figdata/anat_results.json")) as f:
        ar = json.load(f)

    # Bounding box around caudate in axial view (x, y)
    xs, ys, zs = np.where(caud)
    z_slice = int(np.median(zs))
    pad = 8
    x0, x1 = max(0, xs.min() - pad), min(tpl.shape[0], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(tpl.shape[1], ys.max() + pad + 1)

    dark_map = np.zeros(tpl.shape[:2], dtype=float)
    bright_map = np.zeros(tpl.shape[:2], dtype=float)
    for rec in ar:
        for i, j, k in rec["dark"]:
            dark_map[int(i), int(j)] += 1
        for i, j, k in rec["bright"]:
            bright_map[int(i), int(j)] += 1

    # Load one healthy control image
    hc_img = np.load(os.path.join(BASE, "figdata/img_ds000245_0.npz"))["z"].astype(float)
    # Find the ds000245 label=0 subject closest to z_slice or use ar[0]
    rec0 = next(r for r in ar if r["ds"] == "ds000245" and r["label"] == 0)
    z_ctrl = int(round(np.median([pt[2] for pt in rec0["dark"]])))

    d_dark = np.array([r["d_dark"] for r in ar], dtype=float)
    d_bright = np.array([r["d_bright"] for r in ar], dtype=float)
    d_all = float(np.mean([r["d_all"] for r in ar]))

    fig = plt.figure(figsize=(9.2, 6.4))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.1, 1.0], hspace=0.32, wspace=0.22)

    def crop(arr2d):
        return arr2d[x0:x1, y0:y1].T

    caud_proj = np.any(caud, axis=2)

    # Panel a: Caudate ROI + 1 & 2 voxel erosions
    axa = fig.add_subplot(gs[0, 0])
    axa.imshow(crop(tpl[:, :, z_slice]), cmap="gray", origin="lower")
    axa.contour(crop(caud[:, :, z_slice].astype(float)), levels=[0.5], colors=[ORANGE], linewidths=1.5)
    axa.contour(crop(er1[:, :, z_slice].astype(float)), levels=[0.5], colors=[GOLD], linewidths=1.3)
    axa.contour(crop(er2[:, :, z_slice].astype(float)), levels=[0.5], colors=["#F0E442"], linewidths=1.2)
    axa.set_xticks([]); axa.set_yticks([])
    axa.set_title("a  Caudate ROI, axial", loc="left", pad=5)
    h_a = [
        Line2D([0], [0], color=ORANGE, lw=1.6, label="whole ROI"),
        Line2D([0], [0], color=GOLD, lw=1.5, label="eroded 1 voxel"),
        Line2D([0], [0], color="#F0E442", lw=1.4, label="eroded 2 voxels"),
    ]
    axa.legend(handles=h_a, loc="lower left", fontsize=7, frameon=True, framealpha=0.88)

    vmax_cnt = max(dark_map.max(), bright_map.max())
    # Panel b: Darkest 10 voxels
    axb = fig.add_subplot(gs[0, 1])
    axb.imshow(crop(tpl[:, :, z_slice]), cmap="gray", origin="lower")
    dm_c = np.ma.masked_where(crop(dark_map) == 0, crop(dark_map))
    im_b = axb.imshow(dm_c, cmap="magma", origin="lower", vmin=0, vmax=vmax_cnt, alpha=0.88)
    axb.contour(crop(caud_proj.astype(float)), levels=[0.5], colors=[SKY], linewidths=1.1)
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_title("b  Darkest 10 voxels", loc="left", pad=5)

    # Panel c: Brightest 10 voxels
    axc = fig.add_subplot(gs[0, 2])
    axc.imshow(crop(tpl[:, :, z_slice]), cmap="gray", origin="lower")
    bm_c = np.ma.masked_where(crop(bright_map) == 0, crop(bright_map))
    im_c = axc.imshow(bm_c, cmap="magma", origin="lower", vmin=0, vmax=vmax_cnt, alpha=0.88)
    axc.contour(crop(caud_proj.astype(float)), levels=[0.5], colors=[SKY], linewidths=1.1)
    axc.set_xticks([]); axc.set_yticks([])
    axc.set_title("c  Brightest 10 voxels", loc="left", pad=5)
    cbar = fig.colorbar(im_c, ax=axc, fraction=0.046, pad=0.04)
    cbar.set_label("voxel count", fontsize=8)

    # Panel d: One healthy control
    axd = fig.add_subplot(gs[1, 0])
    axd.imshow(crop(hc_img[:, :, z_ctrl]), cmap="gray", origin="lower", vmin=-2.0, vmax=2.5)
    axd.contour(crop(caud[:, :, z_ctrl].astype(float)), levels=[0.5], colors=[ORANGE], linewidths=1.3)
    for pt in rec0["dark"]:
        if abs(pt[2] - z_ctrl) <= 2:
            axd.scatter(pt[0] - x0, pt[1] - y0, facecolors="none", edgecolors=SKY, s=28, linewidths=1.3)
    for pt in rec0["bright"]:
        if abs(pt[2] - z_ctrl) <= 2:
            axd.scatter(pt[0] - x0, pt[1] - y0, facecolors="none", edgecolors=PURPLE, s=28, linewidths=1.3)
    axd.set_xticks([]); axd.set_yticks([])
    axd.set_title("d  One healthy control", loc="left", pad=5)
    h_d = [
        Line2D([0], [0], marker="o", color="w", markeredgecolor=SKY, markerfacecolor="none", markersize=6, label="darkest"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor=PURPLE, markerfacecolor="none", markersize=6, label="brightest"),
    ]
    axd.legend(handles=h_d, loc="lower left", fontsize=7.2, frameon=True, framealpha=0.88)

    # Panel e: Per-subject mean distance from ROI boundary
    axe = fig.add_subplot(gs[1, 1:])
    vp = axe.violinplot([d_dark, d_bright], positions=[0, 1], widths=0.55, showmeans=False, showextrema=False)
    for body, col in zip(vp["bodies"], [SKY, PURPLE]):
        body.set_facecolor(col)
        body.set_alpha(0.45)
    rng = np.random.default_rng(9)
    axe.scatter(rng.normal(0, 0.05, len(d_dark)), d_dark, color=SKY, s=10, alpha=0.65)
    axe.scatter(1 + rng.normal(0, 0.05, len(d_bright)), d_bright, color=PURPLE, s=10, alpha=0.65)
    axe.plot([-0.22, 0.22], [d_dark.mean(), d_dark.mean()], color="black", lw=2.0)
    axe.plot([0.78, 1.22], [d_bright.mean(), d_bright.mean()], color="black", lw=2.0)
    axe.text(0.25, d_dark.mean() + 0.006, f"{d_dark.mean():.2f}", fontsize=8.5, fontweight="bold")
    axe.text(1.25, d_bright.mean() + 0.006, f"{d_bright.mean():.2f}", fontsize=8.5, fontweight="bold")
    axe.axhline(1.0, color="#666666", linestyle=":", linewidth=1.0)
    axe.text(0.5, 0.980, "1 = on the boundary", ha="center", va="top", fontsize=7.6, color="#555555")
    axe.axhline(d_all, color="#444444", linestyle="--", linewidth=1.1)
    axe.text(1.42, d_all + 0.012, f"all ROI voxels: {d_all:.2f}", ha="right", fontsize=8, color="#333333")
    axe.set_xticks([0, 1])
    axe.set_xticklabels(["darkest 10 voxels", "brightest 10 voxels"])
    axe.set_ylim(0.95, 1.42)
    axe.set_ylabel("Distance from ROI boundary (voxels)")
    axe.set_title("e  Per-subject mean distance from the boundary", loc="left", pad=5)

    save_fig(fig, "Figure 5")


# ==============================================================================
# FIGURE 6: Four descriptive analyses localising the apparent effect
# ==============================================================================
def make_figure_6():
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4))
    plt.subplots_adjust(hspace=0.38, wspace=0.34)

    # Panel a: Tail effect
    axa = axes[0, 0]
    stats_lbl = ["Min.\n(1 voxel)", "10th\npercentile", "Median", "Mean"]
    vals_a = [-0.335, -0.090, +0.131, +0.170]
    xa = np.arange(len(stats_lbl))
    axa.axhline(0, color="#333333", lw=1.0)
    axa.fill_between(xa, 0, vals_a, color=SKY, alpha=0.35)
    axa.plot(xa, vals_a, "-o", color=BLUE, lw=2.0, markersize=6)
    for xi, vi in zip(xa, vals_a):
        off = -0.050 if vi < 0 else 0.03
        axa.text(xi, vi + off, f"{vi:+.3f}", ha="center", fontsize=8)
    axa.set_xticks(xa); axa.set_xticklabels(stats_lbl)
    axa.set_ylim(-0.48, 0.28)
    axa.set_ylabel("Matched effect (d)")
    axa.set_title("a  Effect lives in the tail", loc="left", pad=6)

    # Panel b: Graded with CSF interface
    axb = axes[0, 1]
    structs = ["Pallidum\nnone", "Putamen\nnone", "Thalamus\n3rd ventricle", "Caudate\nlat. ventricle"]
    vals_b = [+0.097, +0.323, -0.197, -0.330]
    cols_b = [GREY, GREY, SKY, BLUE]
    yb = np.arange(len(structs))
    axb.axvline(0, color="#333333", lw=1.0)
    for yi, vi, col in zip(yb, vals_b, cols_b):
        axb.plot([0, vi], [yi, yi], color=col, lw=2.2)
        axb.scatter([vi], [yi], color=col, s=45, zorder=4)
        axb.text(vi + (0.03 if vi > 0 else -0.03), yi, f"{vi:+.3f}",
                 va="center", ha="left" if vi > 0 else "right", fontsize=8)
    axb.set_yticks(yb); axb.set_yticklabels(structs)
    axb.set_xlim(-0.58, 0.58)
    axb.set_xlabel("Matched effect (d)")
    axb.set_title("b  Graded with the CSF interface", loc="left", pad=6)

    # Panel c: Removing the boundary
    axc = axes[1, 0]
    eros = ["none\n(1,666 vox)", "2 mm\n(749 vox)", "4 mm\n(236 vox)"]
    vals_c = [-0.335, -0.225, -0.120]
    xc = np.arange(len(eros))
    axc.axhline(0, color="#333333", lw=1.0)
    axc.fill_between(xc, 0, vals_c, color=SKY, alpha=0.35)
    axc.plot(xc, vals_c, "-o", color=BLUE, lw=2.0, markersize=6)
    for xi, vi in zip(xc, vals_c):
        axc.text(xi, vi - 0.04, f"{vi:+.3f}", ha="center", fontsize=8)
    axc.set_xticks(xc); axc.set_xticklabels(eros)
    axc.set_ylim(-0.44, 0.08)
    axc.set_xlabel("Erosion from the boundary")
    axc.set_ylabel("Matched effect (d)")
    axc.set_title("c  Removing the boundary", loc="left", pad=6)

    # Panel d: Site variance by statistic (healthy controls only)
    axd = axes[1, 1]
    lines_d = [
        ("Caudate 2.47", 0.90, 2.47, ORANGE, 2.4),
        ("Thalamus 0.95", 1.79, 0.95, "#999999", 1.6),
        ("Pallidum 0.96", 2.33, 0.96, "#999999", 1.6),
        ("Putamen 0.53", 1.81, 0.53, "#999999", 1.6),
    ]
    for lbl, vmean, vmin, col, lw in lines_d:
        axd.plot([0, 1], [vmean, vmin], color=col, lw=lw, marker="o", markersize=5)
        y_off = 0.075 if "Pallidum" in lbl else (-0.075 if "Thalamus" in lbl else 0.0)
        axd.text(1.06, vmin + y_off, lbl, va="center", fontsize=8.2,
                 color=ORANGE if "Caudate" in lbl else "#555555",
                 fontweight="bold" if "Caudate" in lbl else "normal")
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["Mean\n(centre)", "Minimum\n(1 voxel)"])
    axd.set_xlim(-0.15, 1.65)
    axd.set_ylim(0.0, 2.75)
    axd.set_ylabel("Between-site effect,\nhealthy controls (median |d|)")
    axd.set_title("d  Site variance by statistic", loc="left", pad=6)

    save_fig(fig, "Figure 6")


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
    make_figure_3()
    make_figure_4()
    make_figure_5()
    make_figure_6()
    print("All 6 revised figures generated successfully.")
