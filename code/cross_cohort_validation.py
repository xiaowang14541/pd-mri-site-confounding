"""
Genuine cross-cohort validation using ds001907 (25 HC + 21 PD, same scanner) as a
second, fully independent, both-classes cohort alongside ds000245 (29 HC + 15 PD).

Notably: in ds000245, PD is OLDER than HC (70.7 vs 63.9). In ds001907, PD is YOUNGER
than HC (62.1 vs 66.2) -- the age-diagnosis relationship is REVERSED between cohorts.
This gives a sharp test: if a classifier's apparent signal is really an age detector
tuned to one cohort's age-diagnosis direction, it should fail (or even invert) on the
other cohort, where that direction reverses. A genuine PD imaging biomarker, by
contrast, should transfer regardless of each cohort's particular age structure.

Analyses:
1. Age alone, cross-cohort (train age-threshold direction on one cohort, test on other)
2. Radiomics, direct cross-cohort (train on all of cohort A, test on all of cohort B)
3. Extended LODO across all 5 datasets now available (ds000245, ds001907, ds003592,
   ds004392, ds004471) -- 2 of 5 are now both-classes, changing the leakage structure.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, confusion_matrix,
                              precision_score, recall_score, f1_score)
from neuroCombat import neuroCombat

COMBINED = "/master/wang_zi_wei/cursor/pd_classification/data/radiomics_features_with_ds001907.csv"
AGE_DS000245 = "/master/wang_zi_wei/cursor/pd_classification/data/demographics.csv"
AGE_DS001907 = "/master/wang_zi_wei/cursor/parkinson_mri_data/ds001907_UdallANT/age_clean.csv"
OUT_JSON = "/master/wang_zi_wei/cursor/pd_classification/results/cross_cohort_summary.json"


import re
def sid(fp):
    m = re.search(r"(sub-[A-Za-z0-9]+)", str(fp))
    return m.group(1) if m else None


def load():
    df = pd.read_csv(COMBINED)
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"])
    df["participant_id"] = df["file_path"].apply(sid)
    meta = ["file_path", "label", "dataset", "participant_id"]
    feat_cols = [c for c in df.columns if c not in meta]

    age245 = pd.read_csv(AGE_DS000245)
    age245 = age245[age245.dataset == "ds000245"][["participant_id", "age"]]
    age1907 = pd.read_csv(AGE_DS001907)[["participant_id", "age"]]
    ages = pd.concat([age245, age1907], ignore_index=True)

    df = df.merge(ages, on="participant_id", how="left")
    return df, feat_cols


def pooled_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true); y_pred = np.array(y_pred); y_prob = np.array(y_prob)
    cm = confusion_matrix(y_true, y_pred)
    out = {"n": int(len(y_true)), "accuracy": float(accuracy_score(y_true, y_pred))}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["auc"] = None
    tn, fp, fn, tp = cm.ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    out["confusion_matrix"] = cm.tolist()
    return out


def analysis_1_age_cross_cohort(df):
    print("\n" + "="*64, flush=True)
    print("(1) AGE ALONE, cross-cohort (opposite age-diagnosis direction test)", flush=True)
    print("="*64, flush=True)
    d245 = df[df.dataset == "ds000245"].dropna(subset=["age"])
    d1907 = df[df.dataset == "ds001907"].dropna(subset=["age"])
    print(f"ds000245: PD age={d245[d245.label==1].age.mean():.1f}, HC age={d245[d245.label==0].age.mean():.1f}", flush=True)
    print(f"ds001907: PD age={d1907[d1907.label==1].age.mean():.1f}, HC age={d1907[d1907.label==0].age.mean():.1f}", flush=True)

    for train_name, train_df, test_name, test_df in [
        ("ds000245", d245, "ds001907", d1907),
        ("ds001907", d1907, "ds000245", d245),
    ]:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_df[["age"]].values, train_df["label"].values)
        prob = clf.predict_proba(test_df[["age"]].values)[:, 1]
        auc = roc_auc_score(test_df["label"].values, prob)
        print(f"  Train age-model on {train_name} -> test on {test_name}: AUC = {auc:.3f}", flush=True)


def analysis_2_radiomics_cross_cohort(df, feat_cols):
    print("\n" + "="*64, flush=True)
    print("(2) RADIOMICS, direct cross-cohort (train all of A, test all of B)", flush=True)
    print("="*64, flush=True)
    d245 = df[df.dataset == "ds000245"]
    d1907 = df[df.dataset == "ds001907"]
    results = {}
    for train_name, train_df, test_name, test_df in [
        ("ds000245", d245, "ds001907", d1907),
        ("ds001907", d1907, "ds000245", d245),
    ]:
        sc = StandardScaler().fit(train_df[feat_cols].values)
        Xtr = sc.transform(train_df[feat_cols].values)
        Xte = sc.transform(test_df[feat_cols].values)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, train_df["label"].values)
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)
        m = pooled_metrics(test_df["label"].values, pred, prob)
        results[f"{train_name}_to_{test_name}"] = m
        print(f"  Train on {train_name} (n={len(train_df)}) -> test on {test_name} (n={len(test_df)}): "
              f"AUC={m['auc']:.3f} acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} spec={m['specificity']:.3f}", flush=True)
    return results


def analysis_3_extended_lodo(df, feat_cols):
    print("\n" + "="*64, flush=True)
    print("(3) EXTENDED LODO across all 5 datasets (2 of 5 now both-classes)", flush=True)
    print("="*64, flush=True)
    datasets = sorted(df["dataset"].unique())
    print(f"Datasets: {datasets}", flush=True)

    data = df[feat_cols].values.T
    covars = pd.DataFrame({"batch": df["dataset"].values})
    res = neuroCombat(dat=data, covars=covars, batch_col="batch")
    df_h = df.copy(); df_h[feat_cols] = res["data"].T

    all_true, all_pred, all_prob = [], [], []
    per_fold = []
    for held_out in datasets:
        train_df = df_h[df_h.dataset != held_out]
        val_df = df_h[df_h.dataset == held_out]
        sc = StandardScaler().fit(train_df[feat_cols].values)
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(train_df[feat_cols].values), train_df["label"].values)
        prob = clf.predict_proba(sc.transform(val_df[feat_cols].values))[:, 1]
        pred = (prob >= 0.5).astype(int)
        y = val_df["label"].values
        all_true.extend(y); all_pred.extend(pred); all_prob.extend(prob)
        fm = {"held_out": held_out, "n": int(len(y)), "n_pd": int(y.sum()), "n_hc": int(len(y) - y.sum()),
              "accuracy": float(accuracy_score(y, pred))}
        if y.sum() > 0 and (len(y) - y.sum()) > 0:
            fm["auc"] = float(roc_auc_score(y, prob))
        else:
            fm["auc"] = None
        per_fold.append(fm)
        print(f"  Held-out {held_out}: {fm}", flush=True)

    pooled = pooled_metrics(all_true, all_pred, all_prob)
    print(f"  POOLED (5-dataset LODO): {pooled}", flush=True)
    return {"per_fold": per_fold, "pooled": pooled}


def main():
    df, feat_cols = load()
    print(f"Combined n={len(df)}, datasets={sorted(df.dataset.unique())}", flush=True)

    analysis_1_age_cross_cohort(df)
    r2 = analysis_2_radiomics_cross_cohort(df, feat_cols)
    r3 = analysis_3_extended_lodo(df, feat_cols)

    with open(OUT_JSON, "w") as f:
        json.dump({"cross_cohort": r2, "extended_lodo": r3}, f, indent=2)
    print(f"\nSaved to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
