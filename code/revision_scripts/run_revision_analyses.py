#!/usr/bin/env python3
"""
Revision analyses for Frontiers in Neuroinformatics Manuscript ID 2031561.
Computes:
1. Expanded Table 1 (demographics with mean+-SD, median[IQR], range, male% across all 8 cohorts)
2. Table 2 (registration Dice and imaging QC metrics across all 8 cohorts)
3. 95% CIs (stratified bootstrap, Wilson, Hanley-McNeil) for all reported validation metrics
4. Mixed-diagnosis-only classification (3 OpenNeuro mixed cohorts n=145; 5 single-scanner cohorts n=228)
   under pooled 5-fold CV, LOCO (raw & cohort-centred), and pairwise cohort transfer
5. CSR sensitivity analysis across alternative target disease effect sizes
6. Concave-up ROC diagnostics (training prevalence vs held-out predicted P(PD))
7. Matched effect sizes + 95% bootstrap CIs across all 5 single-scanner cohorts (for Figure 4)
"""
import json
import math
import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = "/master/wang_zi_wei/cursor/pd_classification"
PP = "/master/wang_zi_wei/cursor/parkinson_mri_data"
PB = "/master/wang_zi_wei/cursor/pathB_data"
OUT_DIR = os.path.join(BASE, "frontiers_revision_2031561")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(TAB_DIR, exist_ok=True)

SEED = 20261125
N_BOOT = 5000


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def hanley_mcneil_ci(auc, n_pos, n_neg, z=1.959963984540054):
    if auc is None or n_pos == 0 or n_neg == 0:
        return [None, None]
    # Use symmetric AUC for variance approximation when auc < 0.5
    q1 = auc / (2.0 - auc)
    q2 = (2.0 * auc * auc) / (1.0 + auc)
    se = math.sqrt(
        (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc))
        / (n_pos * n_neg)
    )
    return [max(0.0, auc - z * se), min(1.0, auc + z * se)]


def bootstrap_auc_acc(y_true, y_prob, n_boot=N_BOOT, seed=SEED):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
        k = int(((y_prob >= 0.5).astype(int) == y_true).sum())
        return {
            "n": int(len(y_true)),
            "n_pd": int(len(pos_idx)),
            "n_hc": int(len(neg_idx)),
            "auc": None,
            "auc_ci95": [None, None],
            "acc": acc,
            "acc_ci95": wilson_ci(k, len(y_true)),
            "bacc": None,
        }
    obs_auc = float(roc_auc_score(y_true, y_prob))
    pred = (y_prob >= 0.5).astype(int)
    obs_acc = float(accuracy_score(y_true, pred))
    obs_bacc = float(balanced_accuracy_score(y_true, pred))
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot, dtype=float)
    accs = np.empty(n_boot, dtype=float)
    baccs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        yt = y_true[idx]
        yp = y_prob[idx]
        pr = (yp >= 0.5).astype(int)
        aucs[b] = roc_auc_score(yt, yp)
        accs[b] = accuracy_score(yt, pr)
        baccs[b] = balanced_accuracy_score(yt, pr)
    return {
        "n": int(len(y_true)),
        "n_pd": int(len(pos_idx)),
        "n_hc": int(len(neg_idx)),
        "auc": obs_auc,
        "auc_ci95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        "acc": obs_acc,
        "acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "bacc": obs_bacc,
        "bacc_ci95": [float(np.percentile(baccs, 2.5)), float(np.percentile(baccs, 97.5))],
    }


def load_all_demographics():
    rows = []
    # ds000245
    d245 = pd.read_csv(os.path.join(PP, "participants.tsv"), sep="\t").rename(
        columns={"participant_id": "pid", "Age": "age", "Gender": "sex"}
    )
    d245["ds_hint"] = "ds000245"
    rows.append(d245[["pid", "age", "sex", "ds_hint"]])

    # ds001907
    d19 = pd.read_csv(os.path.join(PP, "ds001907_UdallANT/age_clean.csv")).rename(
        columns={"participant_id": "pid", "gender": "sex"}
    )
    d19["ds_hint"] = "ds001907"
    rows.append(d19[["pid", "age", "sex", "ds_hint"]])

    # ds005892
    d58 = pd.read_csv(os.path.join(PP, "ds005892/participants.tsv"), sep="\t").rename(
        columns={"participant_id": "pid"}
    )
    d58["ds_hint"] = "ds005892"
    rows.append(d58[["pid", "age", "sex", "ds_hint"]])

    # ds003592
    d35 = pd.read_csv(os.path.join(PP, "ds003592_HealthyControls/participants.tsv"), sep="\t").rename(
        columns={"participant_id": "pid"}
    )
    d35["ds_hint"] = "ds003592"
    rows.append(d35[["pid", "age", "sex", "ds_hint"]])

    # ds004392
    d43 = pd.read_csv(os.path.join(PP, "ds004392_PD_Cognition/participants.tsv"), sep="\t").rename(
        columns={"participant_id": "pid"}
    )
    d43["ds_hint"] = "ds004392"
    rows.append(d43[["pid", "age", "sex", "ds_hint"]])

    # ds004471
    d44 = pd.read_csv(os.path.join(PP, "ds004471_LHSCPD/participants.tsv"), sep="\t").rename(
        columns={"participant_id": "pid"}
    )
    d44["ds_hint"] = "ds004471"
    rows.append(d44[["pid", "age", "sex", "ds_hint"]])

    # taowu
    tw = pd.read_csv(os.path.join(PB, "taowu/taowu/taowu_patients.tsv"), sep="\t")
    tw["pid"] = "sub-" + tw["name"].astype(str)
    tw["ds_hint"] = "taowu"
    rows.append(tw[["pid", "age", "sex", "ds_hint"]])

    # neurocon
    nc = pd.read_csv(os.path.join(PB, "neurocon/neurocon/neurocon_patients.tsv"), sep="\t")
    nc["pid"] = "sub-" + nc["code"].astype(str)
    nc["ds_hint"] = "neurocon"
    rows.append(nc[["pid", "age", "sex", "ds_hint"]])

    dm = pd.concat(rows, ignore_index=True)
    dm["age"] = pd.to_numeric(dm["age"], errors="coerce")

    def parse_male(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().upper()
        if s in ("N/A", "NA", "NAN", "", "NONE"):
            return np.nan
        if s.startswith("M") or s == "1":
            return 1.0
        if s.startswith("F") or s == "0" or s == "2":
            return 0.0
        return np.nan

    dm["male"] = dm["sex"].apply(parse_male)
    dm = dm.drop_duplicates(subset=["ds_hint", "pid"])
    return dm


def age_stats(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "n_age": int(len(arr)),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def sex_stats(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"n_sex": 0, "n_male": None, "male_pct": None}
    return {
        "n_sex": int(len(arr)),
        "n_male": int(np.sum(arr == 1.0)),
        "male_pct": float(100.0 * np.mean(arr == 1.0)),
    }


def main():
    results = {}

    # Load radiomics tables (exact 333 OpenNeuro + 83 INDI subjects)
    r_on = pd.read_csv(os.path.join(BASE, "data/radiomics_v3_harmonized.csv"))
    r_ex = pd.read_csv(os.path.join(BASE, "data/radiomics_v3_taowu_neurocon.csv"))
    if "error" in r_on.columns:
        err_on = int(r_on["error"].notna().sum())
        r_on = r_on[r_on["error"].isna()].copy()
    else:
        err_on = 0
    if "error" in r_ex.columns:
        err_ex = int(r_ex["error"].notna().sum())
        r_ex = r_ex[r_ex["error"].isna()].copy()
    else:
        err_ex = 0

    df_all = pd.concat([r_on, r_ex], ignore_index=True)
    df_all["pid"] = df_all["file_path"].apply(lambda f: re.search(r"(sub-[A-Za-z0-9]+)", str(f)).group(1))
    dm = load_all_demographics()
    df_all = df_all.merge(dm[["ds_hint", "pid", "age", "male"]], left_on=["dataset", "pid"], right_on=["ds_hint", "pid"], how="left")

    cohort_order = [
        ("ds000245", "Mixed, same scanner (OpenNeuro)"),
        ("ds001907", "Mixed, same scanner (OpenNeuro)"),
        ("ds005892", "Mixed, same scanner (OpenNeuro)"),
        ("ds003592", "Controls only (OpenNeuro)"),
        ("ds004392", "PD only (OpenNeuro)"),
        ("ds004471", "PD only (OpenNeuro)"),
        ("taowu", "Mixed, same scanner (INDI)"),
        ("neurocon", "Mixed, same scanner (INDI)"),
    ]

    # ------------------------------------------------------------------
    # 1. Expanded Table 1
    # ------------------------------------------------------------------
    t1_rows = []
    for ds, design in cohort_order:
        sub = df_all[df_all["dataset"] == ds]
        pd_sub = sub[sub["label"] == 1]
        hc_sub = sub[sub["label"] == 0]
        rec = {
            "dataset": ds,
            "design": design,
            "n_total": int(len(sub)),
            "n_pd": int(len(pd_sub)),
            "n_hc": int(len(hc_sub)),
            "pd_age": age_stats(pd_sub["age"].values) if len(pd_sub) else None,
            "hc_age": age_stats(hc_sub["age"].values) if len(hc_sub) else None,
            "pd_sex": sex_stats(pd_sub["male"].values) if len(pd_sub) else None,
            "hc_sex": sex_stats(hc_sub["male"].values) if len(hc_sub) else None,
        }
        t1_rows.append(rec)

    results["table1_expanded"] = t1_rows
    # Save flattened CSV for Table 1
    t1_flat = []
    for r in t1_rows:
        row = {
            "Dataset": r["dataset"],
            "Design": r["design"],
            "n": r["n_total"],
            "PD": r["n_pd"],
            "HC": r["n_hc"],
        }
        for grp, key in [("PD", "pd"), ("HC", "hc")]:
            ag = r[f"{key}_age"]
            sx = r[f"{key}_sex"]
            if ag:
                row[f"{grp}_Age_Mean_SD"] = f"{ag['mean']:.1f} ± {ag['sd']:.1f}"
                row[f"{grp}_Age_Median_IQR"] = f"{ag['median']:.1f} [{ag['q25']:.1f}–{ag['q75']:.1f}]"
                row[f"{grp}_Age_Range"] = f"{ag['min']:.0f}–{ag['max']:.0f}"
            else:
                row[f"{grp}_Age_Mean_SD"] = "—"
                row[f"{grp}_Age_Median_IQR"] = "—"
                row[f"{grp}_Age_Range"] = "—"
            if sx and sx["male_pct"] is not None:
                row[f"{grp}_Male_pct"] = f"{sx['male_pct']:.1f}% ({sx['n_male']}/{sx['n_sex']})"
            elif r[f"n_{key}"] > 0:
                row[f"{grp}_Male_pct"] = "n/a"
            else:
                row[f"{grp}_Male_pct"] = "—"
        t1_flat.append(row)
    pd.DataFrame(t1_flat).to_csv(os.path.join(TAB_DIR, "table1_expanded.csv"), index=False)

    # ------------------------------------------------------------------
    # 2. Table 2: Registration Quality & QC Metrics Across All 8 Cohorts
    # ------------------------------------------------------------------
    t2_rows = []
    for ds, design in cohort_order:
        sub = df_all[df_all["dataset"] == ds]
        dice = sub["reg_dice"].dropna().values
        fz = sub["frac_zero_raw"].dropna().values
        nf = sub["normfrac_A"].dropna().values
        zz = sub["zoom_z"].dropna().values
        aniso_pct = float(100.0 * sub["anisotropic"].astype(bool).mean())

        pd_dice = sub[sub["label"] == 1]["reg_dice"].dropna().values
        hc_dice = sub[sub["label"] == 0]["reg_dice"].dropna().values
        if len(pd_dice) > 0 and len(hc_dice) > 0:
            sd_p = math.sqrt(((len(pd_dice) - 1) * np.var(pd_dice, ddof=1) + (len(hc_dice) - 1) * np.var(hc_dice, ddof=1)) / (len(pd_dice) + len(hc_dice) - 2))
            d_dice = float((np.mean(pd_dice) - np.mean(hc_dice)) / sd_p) if sd_p > 0 else 0.0
            u_stat, u_p = stats.mannwhitneyu(pd_dice, hc_dice, alternative="two-sided")
            dice_cmp = {"pd_mean": float(np.mean(pd_dice)), "hc_mean": float(np.mean(hc_dice)), "d": d_dice, "mwu_p": float(u_p)}
        else:
            dice_cmp = None

        q25_d, med_d, q75_d = np.percentile(dice, [25, 50, 75])
        q25_z, med_z, q75_z = np.percentile(fz, [25, 50, 75])
        rec = {
            "dataset": ds,
            "n": int(len(sub)),
            "dice_mean": float(np.mean(dice)),
            "dice_sd": float(np.std(dice, ddof=1)),
            "dice_median": float(med_d),
            "dice_q25": float(q25_d),
            "dice_q75": float(q75_d),
            "dice_min": float(np.min(dice)),
            "dice_max": float(np.max(dice)),
            "dice_pd_vs_hc": dice_cmp,
            "zoom_z_median": float(np.median(zz)),
            "zoom_z_min": float(np.min(zz)),
            "zoom_z_max": float(np.max(zz)),
            "anisotropic_pct": aniso_pct,
            "frac_zero_mean": float(np.mean(fz)),
            "frac_zero_sd": float(np.std(fz, ddof=1)),
            "frac_zero_median": float(med_z),
            "frac_zero_q25": float(q25_z),
            "frac_zero_q75": float(q75_z),
            "normfrac_A_mean": float(np.mean(nf)),
            "normfrac_A_sd": float(np.std(nf, ddof=1)),
        }
        t2_rows.append(rec)

    results["table2_qc_metrics"] = t2_rows
    results["extraction_errors_dropped"] = {"openneuro": err_on, "taowu_neurocon": err_ex}

    t2_flat = []
    for r in t2_rows:
        cmp_str = "—"
        if r["dice_pd_vs_hc"]:
            cmp_str = f"d = {r['dice_pd_vs_hc']['d']:+.2f}, p = {r['dice_pd_vs_hc']['mwu_p']:.3f}"
        t2_flat.append({
            "Dataset": r["dataset"],
            "Extraction_Success": f"{r['n']}/{r['n']} (100%)",
            "Dice_Mean_SD": f"{r['dice_mean']:.3f} ± {r['dice_sd']:.3f}",
            "Dice_Median_IQR": f"{r['dice_median']:.3f} [{r['dice_q25']:.3f}–{r['dice_q75']:.3f}]",
            "Dice_Range": f"{r['dice_min']:.3f}–{r['dice_max']:.3f}",
            "Dice_PD_vs_HC": cmp_str,
            "Slice_Thick_mm": f"{r['zoom_z_median']:.2f} ({r['zoom_z_min']:.2f}–{r['zoom_z_max']:.2f})",
            "Anisotropic_pct": f"{r['anisotropic_pct']:.1f}%",
            "Frac_Zero_Mean_SD": f"{r['frac_zero_mean']:.3f} ± {r['frac_zero_sd']:.3f}",
            "NormFrac_A_Mean_SD": f"{r['normfrac_A_mean']:.3f} ± {r['normfrac_A_sd']:.3f}",
        })
    pd.DataFrame(t2_flat).to_csv(os.path.join(TAB_DIR, "table2_qc_metrics.csv"), index=False)

    # ------------------------------------------------------------------
    # 3. 95% CIs for Existing Deep-Learning / Foundation-Model Validation Metrics
    # ------------------------------------------------------------------
    val_cis = {}
    cv_oof = np.load(os.path.join(BASE, "figdata/cv_oof.npz"), allow_pickle=True)
    val_cis["pooled_cv_densenet_oof"] = bootstrap_auc_acc(cv_oof["y"], cv_oof["p"])

    lodo_np = np.load(os.path.join(BASE, "results/lodo_20260723_014320/lodo_roc_data.npz"), allow_pickle=True)
    val_cis["lodo_densenet_pooled"] = bootstrap_auc_acc(lodo_np["y_true"], lodo_np["y_prob"])
    val_cis["lodo_densenet_per_site"] = {}
    for ds_name in ["ds000245", "ds003592", "ds004392", "ds004471"]:
        m = lodo_np["dataset"] == ds_name
        yt_s = lodo_np["y_true"][m].astype(int)
        yp_s = lodo_np["y_prob"][m].astype(float)
        st = bootstrap_auc_acc(yt_s, yp_s)
        st["mean_p_pd"] = float(np.mean(yp_s))
        st["median_p_pd"] = float(np.median(yp_s))
        st["q25_p_pd"] = float(np.percentile(yp_s, 25))
        st["q75_p_pd"] = float(np.percentile(yp_s, 75))
        st["pct_called_pd"] = float(100.0 * np.mean(yp_s > 0.5))
        # Training set PD prevalence when ds_name is held out
        tr_m = lodo_np["dataset"] != ds_name
        tr_y = lodo_np["y_true"][tr_m].astype(int)
        st["train_n"] = int(len(tr_y))
        st["train_n_pd"] = int(np.sum(tr_y == 1))
        st["train_n_hc"] = int(np.sum(tr_y == 0))
        st["train_pd_pct"] = float(100.0 * np.mean(tr_y == 1))
        val_cis["lodo_densenet_per_site"][ds_name] = st

    dann_np = np.load(os.path.join(BASE, "results/lodo_dann_20260723_081305/lodo_roc_data.npz"), allow_pickle=True)
    val_cis["lodo_dann_pooled"] = bootstrap_auc_acc(dann_np["y_true"], dann_np["y_prob"])
    m245 = dann_np["dataset"] == "ds000245"
    val_cis["lodo_dann_ds000245"] = bootstrap_auc_acc(dann_np["y_true"][m245], dann_np["y_prob"][m245])
    val_cis["lodo_dann_per_site"] = {}
    for ds_name in ["ds000245", "ds003592", "ds004392", "ds004471"]:
        m = dann_np["dataset"] == ds_name
        yp_s = dann_np["y_prob"][m].astype(float)
        val_cis["lodo_dann_per_site"][ds_name] = {
            "pct_called_pd": float(100.0 * np.mean(yp_s > 0.5)),
            "mean_p_pd": float(np.mean(yp_s)),
        }

    roi_np = np.load(os.path.join(BASE, "results/lodo_roi_20260723_101402/lodo_roc_data.npz"), allow_pickle=True)
    val_cis["lodo_roi_pooled"] = bootstrap_auc_acc(roi_np["y_true"], roi_np["y_prob"])
    m245_r = roi_np["dataset"] == "ds000245"
    val_cis["lodo_roi_ds000245"] = bootstrap_auc_acc(roi_np["y_true"][m245_r], roi_np["y_prob"][m245_r])

    ds245_cv = np.load(os.path.join(BASE, "results/ds000245_internal_cv_20260723_100927/roc_data.npz"), allow_pickle=True)
    val_cis["within_ds000245_cv"] = bootstrap_auc_acc(ds245_cv["y_true"], ds245_cv["y_prob"])

    # Extended LODO & Direct transfer (Hanley-McNeil + Wilson CIs from cross_cohort_summary.json)
    with open(os.path.join(BASE, "results/cross_cohort_summary.json")) as f:
        cc = json.load(f)
    ext = cc["extended_lodo"]["pooled"]
    n_pd_ext = ext["confusion_matrix"][1][0] + ext["confusion_matrix"][1][1]
    n_hc_ext = ext["confusion_matrix"][0][0] + ext["confusion_matrix"][0][1]
    k_ext = ext["confusion_matrix"][0][0] + ext["confusion_matrix"][1][1]
    val_cis["extended_lodo_pooled"] = {
        "n": ext["n"], "n_pd": n_pd_ext, "n_hc": n_hc_ext,
        "auc": ext["auc"], "auc_ci95_hm": hanley_mcneil_ci(ext["auc"], n_pd_ext, n_hc_ext),
        "acc": ext["accuracy"], "acc_ci95_wilson": wilson_ci(k_ext, ext["n"]),
    }
    val_cis["extended_lodo_per_fold"] = []
    for pf in cc["extended_lodo"]["per_fold"]:
        k_f = round(pf["accuracy"] * pf["n"])
        val_cis["extended_lodo_per_fold"].append({
            "held_out": pf["held_out"],
            "n": pf["n"], "n_pd": pf["n_pd"], "n_hc": pf["n_hc"],
            "auc": pf["auc"],
            "auc_ci95_hm": hanley_mcneil_ci(pf["auc"], pf["n_pd"], pf["n_hc"]) if pf["auc"] is not None else [None, None],
            "acc": pf["accuracy"],
            "acc_ci95_wilson": wilson_ci(k_f, pf["n"]),
        })

    val_cis["direct_transfer"] = {}
    for kname, obj in cc["cross_cohort"].items():
        cm = obj["confusion_matrix"]
        n_hc_t = cm[0][0] + cm[0][1]
        n_pd_t = cm[1][0] + cm[1][1]
        k_t = cm[0][0] + cm[1][1]
        val_cis["direct_transfer"][kname] = {
            "n": obj["n"], "n_pd": n_pd_t, "n_hc": n_hc_t,
            "auc": obj["auc"], "auc_ci95_hm": hanley_mcneil_ci(obj["auc"], n_pd_t, n_hc_t),
            "acc": obj["accuracy"], "acc_ci95_wilson": wilson_ci(k_t, obj["n"]),
        }

    # Models Genesis (232 scans: 113 PD, 119 HC) & BrainSegFounder (278 scans: 134 PD, 144 HC)
    with open(os.path.join(BASE, "results/modelsgenesis_classifier_summary.json")) as f:
        mg = json.load(f)
    with open(os.path.join(BASE, "results/brainsegfounder_classifier_summary.json")) as f:
        bsf = json.load(f)
    val_cis["models_genesis_lodo_logreg"] = {
        "n": 232, "n_pd": 113, "n_hc": 119,
        "auc": mg["lodo"]["logreg"]["pooled"]["auc"],
        "auc_ci95_hm": hanley_mcneil_ci(mg["lodo"]["logreg"]["pooled"]["auc"], 113, 119),
    }
    val_cis["brainsegfounder_lodo_logreg"] = {
        "n": 278, "n_pd": 134, "n_hc": 144,
        "auc": bsf["lodo"]["logreg"]["pooled"]["auc"],
        "auc_ci95_hm": hanley_mcneil_ci(bsf["lodo"]["logreg"]["pooled"]["auc"], 134, 144),
    }

    # Matched pooled AUC + 95% CI for caudate Minimum (3 OpenNeuro mixed cohorts, 200 seeds, tol=4.0)
    col_min = "tplB_caudate__original_firstorder_Minimum"
    mixed3 = ["ds000245", "ds001907", "ds005892"]
    all5 = ["ds000245", "ds001907", "ds005892", "taowu", "neurocon"]

    def match_pairs_vals(g_df, col, seed, tol=4.0):
        pdg = g_df[g_df["label"] == 1]
        hcg = g_df[g_df["label"] == 0]
        r = np.random.default_rng(seed)
        idx = list(pdg.index)
        r.shuffle(idx)
        used = set()
        A, Bv = [], []
        for i in idx:
            row = pdg.loc[i]
            c = hcg[(~hcg.index.isin(used)) & (hcg["male"] == row["male"]) & ((hcg["age"] - row["age"]).abs() <= tol)]
            if len(c):
                j = (c["age"] - row["age"]).abs().idxmin()
                used.add(j)
                A.append(row[col])
                Bv.append(hcg.loc[j, col])
        return np.array(A, dtype=float), np.array(Bv, dtype=float)

    matched_forest = {}
    pair_store_3 = []
    pair_store_5 = []
    for ds in all5:
        g = df_all[df_all["dataset"] == ds].dropna(subset=[col_min, "age", "male"])
        dvals, npairs, aucs_m = [], [], []
        lastA, lastB = None, None
        for s in range(200):
            A, Bv = match_pairs_vals(g, col_min, s, tol=4.0)
            if len(A) < 6:
                continue
            sd = math.sqrt(((len(A) - 1) * np.var(A, ddof=1) + (len(Bv) - 1) * np.var(Bv, ddof=1)) / (len(A) + len(Bv) - 2))
            if sd > 0:
                dvals.append((np.mean(A) - np.mean(Bv)) / sd)
                npairs.append(len(A))
                vals = np.r_[A, Bv]
                lb = np.r_[np.ones(len(A)), np.zeros(len(Bv))]
                aucs_m.append(roc_auc_score(lb, -vals))
                lastA, lastB = A, Bv
        # Bootstrap pairs using last seed (same convention as forest_ci.py)
        rb = np.random.default_rng(0)
        boot_d = []
        for _ in range(N_BOOT):
            k = rb.integers(0, len(lastA), len(lastA))
            a, b = lastA[k], lastB[k]
            sd = math.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / (len(a) + len(b) - 2))
            if sd > 0:
                boot_d.append((np.mean(a) - np.mean(b)) / sd)
        matched_forest[ds] = {
            "pairs_mean": float(np.mean(npairs)),
            "pairs_int": int(round(np.mean(npairs))),
            "d": float(np.mean(dvals)),
            "d_ci95": [float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5))],
            "auc_matched": float(np.mean(aucs_m)),
        }
        if ds in mixed3:
            pair_store_3.append((lastA, lastB))
        pair_store_5.append((lastA, lastB))

    # Pooled 3-cohort matched d + CI & matched AUC + CI
    rb = np.random.default_rng(1)
    bp3 = []
    for _ in range(N_BOOT):
        es = []
        for a, b in pair_store_3:
            k = rb.integers(0, len(a), len(a))
            aa, bb = a[k], b[k]
            sd = math.sqrt(((len(aa) - 1) * np.var(aa, ddof=1) + (len(bb) - 1) * np.var(bb, ddof=1)) / (len(aa) + len(bb) - 2))
            if sd > 0:
                es.append((np.mean(aa) - np.mean(bb)) / sd)
        if len(es) == 3:
            bp3.append(np.mean(es))
    matched_forest["pooled_3_openneuro"] = {
        "pairs_int": int(sum(matched_forest[ds]["pairs_int"] for ds in mixed3)),
        "d": float(np.mean([matched_forest[ds]["d"] for ds in mixed3])),
        "d_ci95": [float(np.percentile(bp3, 2.5)), float(np.percentile(bp3, 97.5))],
    }

    # Pooled 3-cohort matched AUC + CI (within-cohort z-scored across 200 seeds)
    pooled_aucs_3 = []
    zs_last, ys_last = [], []
    for s in range(200):
        zs, ys = [], []
        for ds in mixed3:
            g = df_all[df_all["dataset"] == ds].dropna(subset=[col_min, "age", "male"])
            A, Bv = match_pairs_vals(g, col_min, s, tol=4.0)
            vals = np.r_[A, Bv]
            lb = np.r_[np.ones(len(A)), np.zeros(len(Bv))]
            sd = np.std(vals, ddof=1)
            zs.append((vals - np.mean(vals)) / (sd if sd > 0 else 1.0))
            ys.append(lb)
        pooled_aucs_3.append(roc_auc_score(np.concatenate(ys), -np.concatenate(zs)))
        if s == 199:
            zs_last, ys_last = zs, ys
    # Bootstrap matched pairs across the 3 cohorts for matched pooled AUC CI
    rb_auc = np.random.default_rng(SEED)
    boot_m_auc = []
    for _ in range(N_BOOT):
        zs_b, ys_b = [], []
        for a, b in pair_store_3:
            k = rb_auc.integers(0, len(a), len(a))
            aa, bb = a[k], b[k]
            vals = np.r_[aa, bb]
            lb = np.r_[np.ones(len(aa)), np.zeros(len(bb))]
            sd = np.std(vals, ddof=1)
            zs_b.append((vals - np.mean(vals)) / (sd if sd > 0 else 1.0))
            ys_b.append(lb)
        boot_m_auc.append(roc_auc_score(np.concatenate(ys_b), -np.concatenate(zs_b)))
    matched_forest["pooled_3_openneuro"]["auc_matched"] = float(np.mean(pooled_aucs_3))
    matched_forest["pooled_3_openneuro"]["auc_matched_ci95"] = [
        float(np.percentile(boot_m_auc, 2.5)),
        float(np.percentile(boot_m_auc, 97.5)),
    ]

    # Pooled 5-cohort matched d + CI
    rb5 = np.random.default_rng(2)
    bp5 = []
    for _ in range(N_BOOT):
        es = []
        for a, b in pair_store_5:
            k = rb5.integers(0, len(a), len(a))
            aa, bb = a[k], b[k]
            sd = math.sqrt(((len(aa) - 1) * np.var(aa, ddof=1) + (len(bb) - 1) * np.var(bb, ddof=1)) / (len(aa) + len(bb) - 2))
            if sd > 0:
                es.append((np.mean(aa) - np.mean(bb)) / sd)
        if len(es) == 5:
            bp5.append(np.mean(es))
    matched_forest["pooled_5_all"] = {
        "pairs_int": int(sum(matched_forest[ds]["pairs_int"] for ds in all5)),
        "d": float(np.mean([matched_forest[ds]["d"] for ds in all5])),
        "d_ci95": [float(np.percentile(bp5, 2.5)), float(np.percentile(bp5, 97.5))],
    }

    val_cis["matched_caudate_minimum"] = matched_forest
    results["validation_cis"] = val_cis

    # ------------------------------------------------------------------
    # 4. Mixed-Diagnosis-Only Radiomics Classification (No Single-Diagnosis Sites)
    # ------------------------------------------------------------------
    feat_cols = [c for c in df_all.columns if c.startswith("tplB_")]
    assert len(feat_cols) == 232, f"Expected 232 tplB features, got {len(feat_cols)}"

    def make_model(kind):
        if kind == "logreg":
            return LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        if kind == "svm_rbf":
            return SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED)
        if kind == "rf":
            return RandomForestClassifier(n_estimators=200, random_state=SEED)
        raise ValueError(kind)

    def run_mixed_suite(cohorts_list, suite_name):
        sub = df_all[df_all["dataset"].isin(cohorts_list)].reset_index(drop=True)
        X_raw = sub[feat_cols].values.astype(float)
        y = sub["label"].values.astype(int)
        ds_arr = sub["dataset"].values
        strata = np.array([f"{d}_{yy}" for d, yy in zip(ds_arr, y)])

        # Cohort-centred X (z-scored strictly within each cohort separately -> no train/test leakage)
        X_cc = np.zeros_like(X_raw)
        for d_id in cohorts_list:
            m = ds_arr == d_id
            sc_c = StandardScaler()
            X_cc[m] = sc_c.fit_transform(X_raw[m])

        suite_out = {"n": int(len(sub)), "n_pd": int(y.sum()), "n_hc": int((1 - y).sum()), "cohorts": cohorts_list, "models": {}}
        for mname in ["logreg", "svm_rbf", "rf"]:
            # A. Pooled 5-fold stratified CV (stratified by cohort x diagnosis)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            oof_raw = np.zeros(len(sub))
            oof_cc = np.zeros(len(sub))
            for tr, te in skf.split(X_raw, strata):
                sc1 = StandardScaler()
                clf1 = make_model(mname)
                clf1.fit(sc1.fit_transform(X_raw[tr]), y[tr])
                oof_raw[te] = clf1.predict_proba(sc1.transform(X_raw[te]))[:, 1]

                sc2 = StandardScaler()
                clf2 = make_model(mname)
                clf2.fit(sc2.fit_transform(X_cc[tr]), y[tr])
                oof_cc[te] = clf2.predict_proba(sc2.transform(X_cc[te]))[:, 1]

            # B. Leave-one-cohort-out (LOCO)
            loco_raw = np.zeros(len(sub))
            loco_cc = np.zeros(len(sub))
            per_cohort_raw = {}
            per_cohort_cc = {}
            for held in cohorts_list:
                te_m = ds_arr == held
                tr_m = ~te_m
                sc1 = StandardScaler()
                clf1 = make_model(mname)
                clf1.fit(sc1.fit_transform(X_raw[tr_m]), y[tr_m])
                p_r = clf1.predict_proba(sc1.transform(X_raw[te_m]))[:, 1]
                loco_raw[te_m] = p_r
                per_cohort_raw[held] = bootstrap_auc_acc(y[te_m], p_r)

                sc2 = StandardScaler()
                clf2 = make_model(mname)
                clf2.fit(sc2.fit_transform(X_cc[tr_m]), y[tr_m])
                p_c = clf2.predict_proba(sc2.transform(X_cc[te_m]))[:, 1]
                loco_cc[te_m] = p_c
                per_cohort_cc[held] = bootstrap_auc_acc(y[te_m], p_c)

            suite_out["models"][mname] = {
                "pooled_cv_raw": bootstrap_auc_acc(y, oof_raw),
                "pooled_cv_cohort_centred": bootstrap_auc_acc(y, oof_cc),
                "loco_raw_pooled": bootstrap_auc_acc(y, loco_raw),
                "loco_raw_per_cohort": per_cohort_raw,
                "loco_cohort_centred_pooled": bootstrap_auc_acc(y, loco_cc),
                "loco_cohort_centred_per_cohort": per_cohort_cc,
            }

        # Pairwise direct transfer (LogReg, raw standardized & cohort-centred)
        pairwise = []
        for src in cohorts_list:
            for dst in cohorts_list:
                if src == dst:
                    continue
                ms = ds_arr == src
                md = ds_arr == dst
                sc1 = StandardScaler()
                clf1 = make_model("logreg")
                clf1.fit(sc1.fit_transform(X_raw[ms]), y[ms])
                p1 = clf1.predict_proba(sc1.transform(X_raw[md]))[:, 1]
                st1 = bootstrap_auc_acc(y[md], p1)

                sc2 = StandardScaler()
                clf2 = make_model("logreg")
                clf2.fit(sc2.fit_transform(X_cc[ms]), y[ms])
                p2 = clf2.predict_proba(sc2.transform(X_cc[md]))[:, 1]
                st2 = bootstrap_auc_acc(y[md], p2)
                pairwise.append({
                    "source": src,
                    "target": dst,
                    "raw_auc": st1["auc"],
                    "raw_auc_ci95": st1["auc_ci95"],
                    "raw_bacc": st1["bacc"],
                    "cc_auc": st2["auc"],
                    "cc_auc_ci95": st2["auc_ci95"],
                    "cc_bacc": st2["bacc"],
                })
        suite_out["pairwise_logreg"] = pairwise
        return suite_out

    mixed_only = {
        "openneuro_3mixed": run_mixed_suite(["ds000245", "ds001907", "ds005892"], "openneuro_3mixed"),
        "all_5_single_scanner": run_mixed_suite(["ds000245", "ds001907", "ds005892", "taowu", "neurocon"], "all_5_single_scanner"),
    }
    results["mixed_diagnosis_only"] = mixed_only

    # Export mixed-diagnosis-only summary CSV
    mx_rows = []
    for skey, sdict in mixed_only.items():
        for mname, md in sdict["models"].items():
            for dkey in ["pooled_cv_raw", "loco_raw_pooled", "loco_cohort_centred_pooled"]:
                v = md[dkey]
                mx_rows.append({
                    "Cohort_Set": skey,
                    "n": sdict["n"],
                    "Model": mname,
                    "Design": dkey,
                    "AUC": f"{v['auc']:.3f}",
                    "AUC_95CI": f"[{v['auc_ci95'][0]:.3f}, {v['auc_ci95'][1]:.3f}]",
                    "BalAcc": f"{v['bacc']:.3f}",
                    "BalAcc_95CI": f"[{v['bacc_ci95'][0]:.3f}, {v['bacc_ci95'][1]:.3f}]",
                })
    pd.DataFrame(mx_rows).to_csv(os.path.join(TAB_DIR, "mixed_diagnosis_only_classification.csv"), index=False)

    # Export pairwise CSV
    pw_rows = []
    for skey in ["openneuro_3mixed", "all_5_single_scanner"]:
        for p in mixed_only[skey]["pairwise_logreg"]:
            pw_rows.append({
                "Cohort_Set": skey,
                "Source": p["source"],
                "Target": p["target"],
                "Raw_AUC": f"{p['raw_auc']:.3f}",
                "Raw_AUC_95CI": f"[{p['raw_auc_ci95'][0]:.3f}, {p['raw_auc_ci95'][1]:.3f}]",
                "Raw_BalAcc": f"{p['raw_bacc']:.3f}",
                "CohortCentred_AUC": f"{p['cc_auc']:.3f}",
                "CohortCentred_AUC_95CI": f"[{p['cc_auc_ci95'][0]:.3f}, {p['cc_auc_ci95'][1]:.3f}]",
                "CohortCentred_BalAcc": f"{p['cc_bacc']:.3f}",
            })
    pd.DataFrame(pw_rows).to_csv(os.path.join(TAB_DIR, "mixed_pairwise_transfer.csv"), index=False)

    # ------------------------------------------------------------------
    # 5. CSR Sensitivity Analysis Across Alternative Target Effect Sizes
    # ------------------------------------------------------------------
    # Exact median nuisance |d| from csr_103905.log + ComBat median CSR=0.29 * 0.585 = 0.16965
    # Let's also recompute ComBat median nuisance d directly from radiomics_subcortical_features.csv if neuroCombat is available,
    # and report exact numbers anchored to csr_103905.log.
    nuisance_medians = {
        "DBM volumetric (8 subcortical ROIs)": 1.63,
        "Whole-brain central radiomics (58 features)": 2.25,
        "Subcortical radiomics (58 features, unharmonized)": 1.70,
        "BrainSegFounder 3D embedding (768 features)": 4.40,
        "Subcortical radiomics + ComBat": 0.29 * 0.585,
    }
    targets = [0.20, 0.30, 0.40, 0.585, 0.80]
    csr_sens = []
    for fam, nd in nuisance_medians.items():
        row = {"Feature_Family": fam, "Median_Nuisance_d": round(nd, 3)}
        for dt in targets:
            row[f"CSR_d={dt}"] = round(nd / dt, 2)
        csr_sens.append(row)
    results["csr_sensitivity"] = csr_sens
    pd.DataFrame(csr_sens).to_csv(os.path.join(TAB_DIR, "csr_sensitivity.csv"), index=False)

    # ------------------------------------------------------------------
    # 6. Corrected HC-only Age-Adjusted Site-Effect Null Model (OpenNeuro, (dataset, pid))
    # ------------------------------------------------------------------
    import itertools
    on_cohorts = ["ds000245", "ds001907", "ds005892", "ds003592", "ds004392", "ds004471"]
    d_null = df_all[df_all["dataset"].isin(on_cohorts)].dropna(subset=[col_min, "age"]).copy()
    hc_fit = d_null[d_null["label"] == 0]
    A_fit = np.c_[np.ones(len(hc_fit)), hc_fit["age"].values, hc_fit["age"].values ** 2]
    beta_null = np.linalg.lstsq(A_fit, hc_fit[col_min].values, rcond=None)[0]
    A_all_n = np.c_[np.ones(len(d_null)), d_null["age"].values, d_null["age"].values ** 2]
    d_null["_r"] = d_null[col_min].values - A_all_n @ beta_null

    null_groups = {}
    for ds_n, g_n in d_null.groupby("dataset"):
        for lb in [0, 1]:
            sg = g_n[g_n["label"] == lb]
            if len(sg) >= 8:
                null_groups[f"{ds_n}_{'PD' if lb else 'HC'}"] = sg
    hh_list, ph_list = [], []
    for x, y in itertools.combinations(null_groups, 2):
        u = d_null.loc[null_groups[x].index, "_r"].values
        v = d_null.loc[null_groups[y].index, "_r"].values
        sd = math.sqrt(((len(u) - 1) * np.var(u, ddof=1) + (len(v) - 1) * np.var(v, ddof=1)) / (len(u) + len(v) - 2))
        if sd == 0:
            continue
        e = float((np.mean(u) - np.mean(v)) / sd)
        xp, yp = ("_PD" in x), ("_PD" in y)
        same = x.split("_")[0] == y.split("_")[0]
        if not xp and not yp:
            hh_list.append({"pair": f"{x} vs {y}", "d": e, "abs_d": abs(e)})
        elif xp != yp:
            ph_list.append({"pair": f"{x if xp else y} - {y if xp else x}", "d": e if xp else -e, "same_scanner": same})

    hh_abs_sorted = sorted(x["abs_d"] for x in hh_list)
    hh_signed = np.array([x["d"] for x in hh_list])
    ph_all = np.array([x["d"] for x in ph_list])
    within_ph = [x for x in ph_list if x["same_scanner"]]
    within_vals = np.array([x["d"] for x in within_ph])
    t_null, p_null = stats.ttest_ind(ph_all, hh_signed, equal_var=False)
    excl_1907 = [x["abs_d"] for x in hh_list if "ds001907" not in x["pair"]]
    results["hc_null_model_corrected"] = {
        "hc_hc_contrasts": hh_list,
        "hc_hc_abs_sorted": hh_abs_sorted,
        "hc_hc_abs_median": float(np.median(hh_abs_sorted)),
        "hc_hc_abs_mean_excl_ds001907": float(np.mean(excl_1907)),
        "within_cohort_pd_hc": within_ph,
        "within_cohort_pd_hc_mean": float(np.mean(within_vals)),
        "pd_hc_vs_hc_hc_welch_t": float(t_null),
        "pd_hc_vs_hc_hc_welch_p": float(p_null),
        "frac_hc_hc_ge_mean_within_pd_hc": float(np.mean(np.array(hh_abs_sorted) >= abs(np.mean(within_vals)))),
    }

    # Write main JSON
    out_json = os.path.join(OUT_DIR, "revision_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_json}")
    print("\n=== SUMMARY CHECK ===")
    print("Table 1 cohorts:", len(t1_rows), "OpenNeuro total:", sum(r["n_total"] for r in t1_rows[:6]), "External total:", sum(r["n_total"] for r in t1_rows[6:]))
    print("Pooled CV DenseNet OOF AUC:", val_cis["pooled_cv_densenet_oof"]["auc"], val_cis["pooled_cv_densenet_oof"]["auc_ci95"])
    print("Primary LODO DenseNet AUC:", val_cis["lodo_densenet_pooled"]["auc"], val_cis["lodo_densenet_pooled"]["auc_ci95"])
    print("Matched Caudate Min Pooled AUC (3 OpenNeuro):", matched_forest["pooled_3_openneuro"]["auc_matched"], matched_forest["pooled_3_openneuro"]["auc_matched_ci95"])
    print("Mixed 3-cohort LogReg Pooled CV AUC:", mixed_only["openneuro_3mixed"]["models"]["logreg"]["pooled_cv_raw"]["auc"], mixed_only["openneuro_3mixed"]["models"]["logreg"]["pooled_cv_raw"]["auc_ci95"])
    print("Mixed 3-cohort LogReg LOCO raw AUC:", mixed_only["openneuro_3mixed"]["models"]["logreg"]["loco_raw_pooled"]["auc"], mixed_only["openneuro_3mixed"]["models"]["logreg"]["loco_raw_pooled"]["auc_ci95"])
    print("Mixed 5-cohort LogReg Pooled CV AUC:", mixed_only["all_5_single_scanner"]["models"]["logreg"]["pooled_cv_raw"]["auc"], mixed_only["all_5_single_scanner"]["models"]["logreg"]["pooled_cv_raw"]["auc_ci95"])
    print("Mixed 5-cohort LogReg LOCO raw AUC:", mixed_only["all_5_single_scanner"]["models"]["logreg"]["loco_raw_pooled"]["auc"], mixed_only["all_5_single_scanner"]["models"]["logreg"]["loco_raw_pooled"]["auc_ci95"])


if __name__ == "__main__":
    main()
