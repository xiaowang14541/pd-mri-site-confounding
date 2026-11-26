"""Pooled matched-design test -- the legitimate way to make a real effect show up.

Per-cohort matching on sex (exact) and age (<=4y) removes the confounds BY DESIGN rather
than by model. Matched effects were ds000245 +0.233, ds001907 -0.629, ds005892 -0.637.
Individually each has only 12-17 pairs, so power for d~0.63 is roughly 0.55-0.65 -- too
low to expect significance from any single cohort.

This pools them, and guards against three things:
  * matching-seed luck        -> repeat the whole matching over 200 random seeds
  * label structure           -> stratified permutation null, labels shuffled WITHIN cohort,
                                 with matching redone inside every permutation
  * variant choice            -> A/B/C all reported
ds000245 is kept in throughout. A version excluding it is printed too, clearly labelled,
because dropping the one cohort that disagrees would be cherry-picking if presented alone.
"""
import numpy as np, pandas as pd, re, warnings
warnings.filterwarnings('ignore')

B='/master/wang_zi_wei/cursor/pd_classification/'
MIXED=['ds000245','ds001907','ds005892']
NSEED=200; NPERM=2000
rng=np.random.default_rng(23)

df=pd.read_csv(B+'data/radiomics_v3_harmonized.csv')
if 'error' in df.columns: df=df[df['error'].isna()]
df['pid']=df['file_path'].apply(lambda f: re.search(r'(sub-[A-Za-z0-9]+)',str(f)).group(1))
dem=[]
t=pd.read_csv('/master/wang_zi_wei/cursor/parkinson_mri_data/participants.tsv',sep='\t')
dem.append(t.rename(columns={'participant_id':'pid','Age':'age','Gender':'sex'})[['pid','age','sex']])
t=pd.read_csv('/master/wang_zi_wei/cursor/parkinson_mri_data/ds001907_UdallANT/age_clean.csv')
dem.append(t.rename(columns={'participant_id':'pid','gender':'sex'})[['pid','age','sex']])
t=pd.read_csv('/master/wang_zi_wei/cursor/parkinson_mri_data/ds005892/participants.tsv',sep='\t')
dem.append(t.rename(columns={'participant_id':'pid'})[['pid','age','sex']])
dm=pd.concat(dem).drop_duplicates('pid')
dm['age']=pd.to_numeric(dm['age'],errors='coerce')
dm['male']=dm['sex'].astype(str).str.upper().str[0].map({'M':1,'F':0})
base=df.merge(dm[['pid','age','male']],on='pid',how='left')

def matched_d(vals, lab, age, male, seed, tol=4.0):
    """1:1 sex-exact, age-tolerant matching; returns Cohen d of matched PD vs HC."""
    r=np.random.default_rng(seed)
    pidx=np.where(lab==1)[0]; hidx=list(np.where(lab==0)[0])
    r.shuffle(pidx)
    a=[];b=[]
    for i in pidx:
        best=None; bd=None
        for j in hidx:
            if male[j]!=male[i]: continue
            dd=abs(age[j]-age[i])
            if dd<=tol and (bd is None or dd<bd): best=j; bd=dd
        if best is not None:
            hidx.remove(best); a.append(vals[i]); b.append(vals[best])
    if len(a)<6: return np.nan
    a=np.array(a); b=np.array(b)
    sd=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    if sd==0: return np.nan
    return (a.mean()-b.mean())/sd, len(a)

def pooled_stat(sub, col, labels, seeds, cohorts):
    per={}
    for ds in cohorts:
        g=sub[sub.dataset==ds]
        v=g[col].values; ag=g.age.values; ml=g.male.values; y=labels[ds]
        ds_vals=[]
        for s in seeds:
            r=matched_d(v,y,ag,ml,s)
            if isinstance(r,tuple) and np.isfinite(r[0]): ds_vals.append(r[0])
        if not ds_vals: return np.nan, {}
        per[ds]=float(np.mean(ds_vals))
    return float(np.mean(list(per.values()))), per

seeds=list(range(NSEED))
for cohorts,tag in [(MIXED,'ALL THREE cohorts'),
                    (['ds001907','ds005892'],'EXCLUDING ds000245 (labelled: cherry-picked subset)')]:
    print('='*78); print(tag); print('='*78)
    for V in ['A','B','C']:
        col=f'tpl{V}_caudate__original_firstorder_Minimum'
        if col not in base.columns: continue
        sub=base.dropna(subset=[col,'age','male'])
        real={ds:sub[sub.dataset==ds].label.values.astype(int) for ds in cohorts}
        obs,per=pooled_stat(sub,col,real,seeds,cohorts)
        if not np.isfinite(obs): print(f'  variant {V}: matching failed'); continue
        pseeds=list(range(20))
        null=np.empty(NPERM)
        for i in range(NPERM):
            perm={ds:rng.permutation(real[ds]) for ds in cohorts}
            s,_=pooled_stat(sub,col,perm,pseeds,cohorts)
            null[i]=s if np.isfinite(s) else 0.0
        p=(null<=obs).mean()
        print(f'  variant {V}: per-cohort={ {k:round(v,3) for k,v in per.items()} }  '
              f'pooled={obs:+.3f}  null 5th={np.percentile(null,5):+.3f}  p={p:.4f}')
    print()
