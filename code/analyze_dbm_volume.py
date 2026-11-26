"""
Analysis of DBM volumetric features, with the literature effect size as the reference.

Unlike the texture-based analyses, this one has a specific quantitative prediction to
test: meta-analytic caudate volume reduction of ~11% in PD (Cohen d ~0.85-1.10),
expected AUC 0.72-0.78, for which we have 74-92% power in ds000245 alone and ~97-100%
pooled across the two both-classes cohorts.

Reports, in order of scientific priority:
1. Registration QC (must pass before anything else is interpretable)
2. Direct univariate test on caudate log-Jacobian: group means, % difference,
   Cohen's d, t-test/Mann-Whitney, and observed AUC -- per cohort and pooled.
   Compared against the literature prediction.
3. Multivariate classification (all ROIs) with the usual honest protocols:
   within-cohort CV, cross-cohort, age-residualized, LODO.
"""
import numpy as np
import pandas as pd
import re
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CSV = "/master/wang_zi_wei/cursor/pd_classification/data/dbm_volume_features.csv"
AGE245 = "/master/wang_zi_wei/cursor/pd_classification/data/demographics.csv"
AGE1907 = "/master/wang_zi_wei/cursor/parkinson_mri_data/ds001907_UdallANT/age_clean.csv"

def sid(fp):
    m = re.search(r'(sub-[A-Za-z0-9]+)', str(fp))
    return m.group(1) if m else None

def load():
    df = pd.read_csv(CSV)
    nerr = 0
    if 'error' in df.columns:
        nerr = int(df['error'].notna().sum())
        df = df[df['error'].isna()].drop(columns=[c for c in ('error','tb') if c in df.columns])
    df['participant_id'] = df['file_path'].apply(sid)
    a1 = pd.read_csv(AGE245); a1 = a1[a1.dataset=='ds000245'][['participant_id','age']]
    a2 = pd.read_csv(AGE1907)[['participant_id','age']]
    df = df.merge(pd.concat([a1,a2]), on='participant_id', how='left')
    return df, nerr

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    s = np.sqrt(((nx-1)*np.var(x,ddof=1) + (ny-1)*np.var(y,ddof=1)) / (nx+ny-2))
    return (np.mean(x)-np.mean(y)) / (s + 1e-12)

def univariate(df, feat, name):
    pd_v = df[df.label==1][feat].dropna().values
    hc_v = df[df.label==0][feat].dropna().values
    if len(pd_v) < 3 or len(hc_v) < 3:
        return None
    # log-Jacobian: difference in log space == log volume ratio
    pct = (np.exp(np.mean(pd_v) - np.mean(hc_v)) - 1) * 100
    d = cohen_d(pd_v, hc_v)
    t, pt = stats.ttest_ind(pd_v, hc_v, equal_var=False)
    u, pu = stats.mannwhitneyu(pd_v, hc_v, alternative='two-sided')
    y = np.r_[np.ones(len(pd_v)), np.zeros(len(hc_v))]
    s = np.r_[pd_v, hc_v]
    auc = roc_auc_score(y, s); auc = max(auc, 1-auc)
    print(f"  {name:26s} n={len(pd_v)}PD/{len(hc_v)}HC  PD-HC={pct:+6.2f}%  d={d:+.2f}  "
          f"p_t={pt:.4f} p_mw={pu:.4f}  AUC={auc:.3f}")
    return dict(pct=pct, d=d, p=pu, auc=auc)

def cv_auc(X, y, age=None, resid=False, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pr, tt = [], []
    for tri, vai in skf.split(X, y):
        sc = StandardScaler().fit(X[tri]); Xtr = sc.transform(X[tri]); Xva = sc.transform(X[vai])
        if resid and age is not None:
            for j in range(Xtr.shape[1]):
                lr = LinearRegression().fit(age[tri].reshape(-1,1), Xtr[:,j])
                Xtr[:,j] -= lr.predict(age[tri].reshape(-1,1))
                Xva[:,j] -= lr.predict(age[vai].reshape(-1,1))
        c = LogisticRegression(max_iter=5000, class_weight='balanced').fit(Xtr, y[tri])
        pr.extend(c.predict_proba(Xva)[:,1]); tt.extend(y[vai])
    return roc_auc_score(tt, pr)

def main():
    df, nerr = load()
    print(f"Loaded n={len(df)} ({nerr} extraction errors dropped)")
    print("\n=== 1. REGISTRATION QC ===")
    d = df['reg_dice']
    print(f"  dice: mean={d.mean():.3f} sd={d.std():.3f} min={d.min():.3f} max={d.max():.3f}")
    bad = int((d < 0.4).sum())
    print(f"  subjects with dice<0.40 (suspect): {bad}")
    for ds, g in df.groupby('dataset'):
        print(f"    {ds}: dice {g.reg_dice.mean():.3f} +/- {g.reg_dice.std():.3f} (n={len(g)})")
    df = df[df.reg_dice >= 0.4].copy()
    print(f"  -> keeping n={len(df)} with acceptable registration")

    jac = [c for c in df.columns if c.startswith('jacnl_')]
    d245 = df[df.dataset=='ds000245']; d1907 = df[df.dataset=='ds001907']
    both = df[df.dataset.isin(['ds000245','ds001907'])]

    print("\n=== 2. UNIVARIATE: caudate & other ROI volume (log-Jacobian) ===")
    print("  Literature prediction: caudate -11% (d~0.85-1.10, AUC 0.72-0.78), putamen -8.1%")
    for label, sub in [('ds000245 (15PD/29HC)', d245), ('ds001907 (21PD/25HC)', d1907),
                       ('POOLED both cohorts', both)]:
        print(f"\n  --- {label} ---")
        for f in ['jacnl_caudate_L','jacnl_caudate_R','jacnl_putamen_L','jacnl_putamen_R',
                  'jacnl_pallidum_L','jacnl_pallidum_R','jacnl_thalamus_L','jacnl_thalamus_R']:
            if f in df.columns:
                univariate(sub, f, f.replace('jacnl_',''))
        # bilateral mean caudate
        sub = sub.copy()
        sub['caudate_mean'] = sub[['jacnl_caudate_L','jacnl_caudate_R']].mean(axis=1)
        sub['putamen_mean'] = sub[['jacnl_putamen_L','jacnl_putamen_R']].mean(axis=1)
        univariate(sub, 'caudate_mean', 'CAUDATE bilateral mean')
        univariate(sub, 'putamen_mean', 'PUTAMEN bilateral mean')

    print("\n=== 3. MULTIVARIATE (all 8 ROI log-Jacobians) ===")
    for label, sub in [('ds000245', d245), ('ds001907', d1907), ('pooled both', both)]:
        s = sub.dropna(subset=jac+['age'])
        if len(s) < 20: continue
        X = s[jac].values; y = s.label.values.astype(int); age = s.age.values.astype(float)
        print(f"  {label:12s} n={len(s)}: CV AUC={cv_auc(X,y):.3f}   "
              f"age-residualized AUC={cv_auc(X,y,age,resid=True):.3f}   "
              f"age-alone AUC={max(roc_auc_score(y,age),1-roc_auc_score(y,age)):.3f}")

    print("\n=== 4. CROSS-COHORT (the decisive test) ===")
    for tr, te, tn, ten in [(d245,d1907,'ds000245','ds001907'), (d1907,d245,'ds001907','ds000245')]:
        tr2 = tr.dropna(subset=jac); te2 = te.dropna(subset=jac)
        sc = StandardScaler().fit(tr2[jac].values)
        c = LogisticRegression(max_iter=5000, class_weight='balanced').fit(sc.transform(tr2[jac].values), tr2.label.values)
        p = c.predict_proba(sc.transform(te2[jac].values))[:,1]
        print(f"  train {tn} -> test {ten}: AUC={roc_auc_score(te2.label.values,p):.3f}")

    print("\n=== 5. LODO (all 5 datasets) ===")
    dfa = df.dropna(subset=jac)
    at, ap = [], []
    for ho in sorted(dfa.dataset.unique()):
        tr = dfa[dfa.dataset!=ho]; va = dfa[dfa.dataset==ho]
        sc = StandardScaler().fit(tr[jac].values)
        c = LogisticRegression(max_iter=5000, class_weight='balanced').fit(sc.transform(tr[jac].values), tr.label.values)
        p = c.predict_proba(sc.transform(va[jac].values))[:,1]
        y = va.label.values
        at.extend(y); ap.extend(p)
        auc = f"{roc_auc_score(y,p):.3f}" if y.sum()>0 and (len(y)-y.sum())>0 else "NA(single-class)"
        print(f"  held-out {ho}: n={len(y)} AUC={auc}")
    print(f"  POOLED LODO AUC={roc_auc_score(at,ap):.3f}")

if __name__ == "__main__":
    main()
