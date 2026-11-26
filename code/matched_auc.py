"""AUC for the pooled matched analysis (the p=0.0265 result).

Averaged over the same 200 matching seeds used for the effect size, so the number does
not depend on one lucky pairing.

Pooling needs care: cohort baselines differ a lot (ds001907 sits at -2.67 while the rest
are near -1.5), so concatenating raw values would let the cohort offset drive the AUC.
Values are therefore z-scored WITHIN each cohort's matched sample before pooling, which
measures only the PD-vs-HC separation and not the between-cohort offset.

Permutation p is for the pooled AUC, labels shuffled within cohort, matching redone.
"""
import numpy as np, pandas as pd, re, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score

B='/master/wang_zi_wei/cursor/pd_classification/'
MIXED=['ds000245','ds001907','ds005892']
NSEED=200; NPERM=2000
rng=np.random.default_rng(31)

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

def match_pairs(lab, age, male, seed, tol=4.0):
    r=np.random.default_rng(seed)
    pidx=np.where(lab==1)[0].copy(); r.shuffle(pidx)
    hidx=list(np.where(lab==0)[0]); out=[]
    for i in pidx:
        best=None; bd=None
        for j in hidx:
            if male[j]!=male[i]: continue
            dd=abs(age[j]-age[i])
            if dd<=tol and (bd is None or dd<bd): best=j; bd=dd
        if best is not None:
            hidx.remove(best); out.append((i,best))
    return out

def run(col, labels, seeds):
    per={ds:[] for ds in MIXED}; pooled=[]
    for s in seeds:
        zs=[]; ys=[]
        ok=True
        for ds in MIXED:
            g=base[base.dataset==ds].dropna(subset=[col,'age','male'])
            v=g[col].values; y=labels[ds]; ag=g.age.values; ml=g.male.values
            pr=match_pairs(y,ag,ml,s)
            if len(pr)<6: ok=False; break
            a=v[[p[0] for p in pr]]; b=v[[p[1] for p in pr]]
            vals=np.r_[a,b]; lb=np.r_[np.ones(len(a)),np.zeros(len(b))]
            per[ds].append(roc_auc_score(lb,-vals))
            sd=vals.std(ddof=1)
            zs.append((vals-vals.mean())/(sd if sd>0 else 1)); ys.append(lb)
        if ok:
            pooled.append(roc_auc_score(np.concatenate(ys), -np.concatenate(zs)))
    return per, pooled

for V in ['B','C','A']:
    col=f'tpl{V}_caudate__original_firstorder_Minimum'
    if col not in base.columns: continue
    real={ds:base[base.dataset==ds].dropna(subset=[col,'age','male']).label.values.astype(int) for ds in MIXED}
    per,pooled=run(col, real, range(NSEED))
    print(f'\nvariant {V}')
    for ds in MIXED:
        if per[ds]: print(f'   {ds:<11} matched AUC = {np.mean(per[ds]):.3f}  (sd across seeds {np.std(per[ds]):.3f})')
    obs=float(np.mean(pooled))
    print(f'   POOLED (within-cohort z-scored) AUC = {obs:.3f}   sd {np.std(pooled):.3f}')
    if V in ('B',):
        null=np.empty(NPERM)
        for i in range(NPERM):
            perm={ds:rng.permutation(real[ds]) for ds in MIXED}
            _,pl=run(col, perm, range(12))
            null[i]=float(np.mean(pl)) if pl else 0.5
        print(f'   permutation: null mean={null.mean():.3f} 95th={np.percentile(null,95):.3f}'
              f'  -> p={(null>=obs).mean():.4f}')
