"""
Demographic confound audit across the 4 datasets.

Central question: is age (the single biggest known confound in PD-vs-HC MRI studies,
since PD patients are typically older than controls) imbalanced between PD and HC,
both within ds000245 (the one mixed-class dataset that gave a "genuine" nested-CV
signal) and across the pooled datasets? If so, any classifier's apparent PD signal
may actually be an age detector.
"""
import os
import numpy as np
import pandas as pd

BASE = "/master/wang_zi_wei/cursor/parkinson_mri_data"


def load_ds000245():
    # participants.tsv at BASE root; ODP*=PD, ODN*/CTL*=HC
    p = pd.read_csv(os.path.join(BASE, "participants.tsv"), sep="\t")
    p.columns = [c.strip().lower() for c in p.columns]
    def lab(sid):
        if "ODP" in sid: return 1
        if "ODN" in sid or "CTL" in sid: return 0
        return np.nan
    p["label"] = p["participant_id"].apply(lab)
    p["dataset"] = "ds000245"
    return p[["participant_id", "age", "label", "dataset"]].dropna(subset=["label"])


def load_simple(folder, dataset, label):
    p = pd.read_csv(os.path.join(BASE, folder, "participants.tsv"), sep="\t")
    p.columns = [c.strip().lower() for c in p.columns]
    p["age"] = pd.to_numeric(p["age"], errors="coerce")
    p["label"] = label
    p["dataset"] = dataset
    return p[["participant_id", "age", "label", "dataset"]]


def summarize(df, name):
    print(f"\n=== {name} ===", flush=True)
    for lab, grp in df.groupby("label"):
        ages = grp["age"].dropna()
        cls = "PD" if lab == 1 else "HC"
        if len(ages):
            print(f"  {cls}: n={len(grp)}, age mean={ages.mean():.1f}, sd={ages.std():.1f}, "
                  f"median={ages.median():.1f}, range={ages.min():.0f}-{ages.max():.0f}", flush=True)
        else:
            print(f"  {cls}: n={len(grp)}, age=missing", flush=True)


def main():
    d245 = load_ds000245()
    d3592 = load_simple("ds003592_HealthyControls", "ds003592", 0)
    d4392 = load_simple("ds004392_PD_Cognition", "ds004392", 1)
    d4471 = load_simple("ds004471_LHSCPD", "ds004471", 1)

    for df, name in [(d245, "ds000245 (mixed PD/HC)"), (d3592, "ds003592 (all HC)"),
                     (d4392, "ds004392 (all PD)"), (d4471, "ds004471 (all PD)")]:
        summarize(df, name)

    # The critical test: within ds000245, is age different between PD and HC?
    print("\n" + "="*60, flush=True)
    print("CRITICAL: age PD-vs-HC within ds000245 (the 'genuine signal' dataset)", flush=True)
    pd_ages = d245[d245.label == 1]["age"].dropna()
    hc_ages = d245[d245.label == 0]["age"].dropna()
    print(f"  PD age: mean={pd_ages.mean():.1f} (n={len(pd_ages)})", flush=True)
    print(f"  HC age: mean={hc_ages.mean():.1f} (n={len(hc_ages)})", flush=True)
    print(f"  Difference: {pd_ages.mean() - hc_ages.mean():.1f} years", flush=True)
    try:
        from scipy.stats import mannwhitneyu, ttest_ind
        u, pu = mannwhitneyu(pd_ages, hc_ages, alternative="two-sided")
        t, pt = ttest_ind(pd_ages, hc_ages, equal_var=False)
        print(f"  Mann-Whitney U p={pu:.4f}; Welch t-test p={pt:.4f}", flush=True)
        # How well does age ALONE classify PD vs HC in ds000245?
        from sklearn.metrics import roc_auc_score
        all_ages = pd.concat([pd_ages, hc_ages]).values
        all_labels = [1]*len(pd_ages) + [0]*len(hc_ages)
        age_auc = roc_auc_score(all_labels, all_ages)
        print(f"  *** AGE ALONE as a PD classifier in ds000245: AUC = {max(age_auc, 1-age_auc):.3f} ***", flush=True)
    except Exception as e:
        print("  stats error:", e, flush=True)

    # Pooled cross-dataset age structure
    print("\n" + "="*60, flush=True)
    print("Pooled age by dataset (shows cross-site age structure = the LODO confound):", flush=True)
    allp = pd.concat([d245, d3592, d4392, d4471])
    for ds, grp in allp.groupby("dataset"):
        ages = grp["age"].dropna()
        print(f"  {ds}: age mean={ages.mean():.1f}, sd={ages.std():.1f} (label={grp['label'].unique()})", flush=True)

    allp.to_csv("/master/wang_zi_wei/cursor/pd_classification/data/demographics.csv", index=False)
    print("\nSaved demographics to data/demographics.csv", flush=True)


if __name__ == "__main__":
    main()
