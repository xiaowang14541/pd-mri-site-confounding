"""Confound-to-signal ratio (CSR) across feature families.

Uses the single-diagnosis datasets as instruments: among HEALTHY CONTROLS ONLY,
diagnosis is held constant by construction, so any between-dataset difference in a
feature is nuisance (scanner/protocol/site) rather than disease. Age is regressed out
(linear + quadratic) so the residual between-dataset effect is nuisance-specific.

CSR = (between-dataset Cohen's d among HC) / (target disease effect d).
Target disease effect: meta-analytic caudate -11% = ln(1.11)/SD_observed = 0.585.
CSR > 1 means the nuisance effect exceeds the effect the study is trying to detect.
"""
import numpy as np, pandas as pd, re, itertools, sys
import statsmodels.formula.api as smf

DISEASE_D = 0.585
B = '/master/wang_zi_wei/cursor/pd_classification/'

a1 = pd.read_csv(B+'data/demographics.csv'); a1 = a1[a1.dataset=='ds000245'][['participant_id','age']].rename(columns={'participant_id':'pid'})
a2 = pd.read_csv('/master/wang_zi_wei/cursor/parkinson_mri_data/ds001907_UdallANT/age_clean.csv')[['participant_id','age']].rename(columns={'participant_id':'pid'})
a3 = pd.read_csv('/master/wang_zi_wei/cursor/parkinson_mri_data/ds003592_HealthyControls/participants.tsv', sep='\t').rename(columns={'participant_id':'pid'})[['pid','age']]
ages = pd.concat([a1,a2,a3]).drop_duplicates('pid')
ages['age'] = pd.to_numeric(ages['age'], errors='coerce')

def csr_table(path, prefix, label, max_feats=None):
    try:
        df = pd.read_csv(B+path)
    except Exception as e:
        print(label, 'load failed:', e); return None
    if 'error' in df.columns: df = df[df['error'].isna()]
    df['pid'] = df['file_path'].apply(lambda f: re.search(r'(sub-[A-Za-z0-9]+)', str(f)).group(1))
    df = df.merge(ages, on='pid', how='left')
    feats = [c for c in df.columns if c.startswith(prefix)]
    if max_feats: feats = feats[:max_feats]
    hc = df[df.label==0].dropna(subset=['age']).copy()
    ds_all = sorted(hc.dataset.unique())
    res = []
    for f in feats:
        s = hc.dropna(subset=[f])
        if len(s) < 30 or s[f].std() == 0: continue
        y = s[f].values.astype(float)
        A = np.c_[np.ones(len(s)), s.age.values, s.age.values**2]
        try:
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        except Exception:
            continue
        resid = y - A@beta
        s = s.assign(resid=resid)
        ds_here = [d for d in ds_all if (s.dataset==d).sum() >= 8]
        pd_ = []
        for a, b in itertools.combinations(ds_here, 2):
            x = s[s.dataset==a]['resid'].values; z = s[s.dataset==b]['resid'].values
            sd = np.sqrt(((len(x)-1)*x.var(ddof=1) + (len(z)-1)*z.var(ddof=1))/(len(x)+len(z)-2))
            if sd > 0: pd_.append(abs(x.mean()-z.mean())/sd)
        if not pd_: continue
        # variance explained by dataset (one-way ANOVA R2)
        grand = resid.mean(); sst = ((resid-grand)**2).sum()
        ssb = sum(len(s[s.dataset==d]) * (s[s.dataset==d]['resid'].mean()-grand)**2 for d in ds_here)
        r2 = 100*ssb/sst if sst > 0 else np.nan
        res.append(dict(feat=f, d_mean=np.mean(pd_), d_max=max(pd_), r2=r2))
    t = pd.DataFrame(res)
    if not len(t): print(label, ': no usable features'); return None
    t['csr'] = t.d_mean/DISEASE_D
    print(f'--- {label} ---')
    print(f'  {len(t)} features | {len(hc)} HC across {len(ds_all)} datasets')
    print(f'  nuisance d (mean over dataset pairs): median={t.d_mean.median():.2f}  IQR {t.d_mean.quantile(.25):.2f}-{t.d_mean.quantile(.75):.2f}  max={t.d_mean.max():.2f}')
    print(f'  CSR (nuisance_d / disease_d): median={t.csr.median():.2f}x   90th pct={t.csr.quantile(.9):.2f}x')
    n_ok = int((t.csr < 1).sum())
    print(f'  features with CSR<1 (usable: nuisance < disease effect): {n_ok}/{len(t)} = {100*n_ok/len(t):.0f}%')
    print(f'  variance explained by dataset among HC (age-adjusted): median R2={t.r2.median():.1f}%  max={t.r2.max():.1f}%')
    if n_ok:
        print('  most scanner-robust features:')
        for _, r in t.nsmallest(min(5,n_ok),'csr').iterrows():
            print(f'     {r.feat[:48]:48s} d={r.d_mean:.2f} CSR={r.csr:.2f} R2={r.r2:.0f}%')
    print()
    return t

print('=== CONFOUND-TO-SIGNAL RATIO (CSR) ACROSS FEATURE FAMILIES ===')
print(f'Healthy controls only (diagnosis constant). Age regressed out (linear+quadratic).')
print(f'Target disease effect: meta-analytic caudate -11% = d {DISEASE_D} in our measurement units.')
print()
csr_table('data/dbm_volume_features.csv','jacnl_','DBM volumetric (Jacobian), 8 subcortical ROIs')
csr_table('data/radiomics_features.csv','original_','Radiomics texture, whole-brain central ROI (58 feats)')
csr_table('data/radiomics_subcortical_features.csv','original_','Radiomics texture, literature subcortical ROI (58 feats)')
csr_table('data/brainsegfounder_features.csv','bsf_','BrainSegFounder deep embedding (768 feats)')
