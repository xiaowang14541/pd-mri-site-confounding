"""Five-dataset configuration: ds000245 removed.

Datasets: ds001907 (21PD/25HC), ds005892 (33PD/22HC), ds003592 (90HC ref),
          ds004392 (58PD), ds004471 (40PD).

IMPORTANT: ds000245 is being removed because it showed no effect. Selecting cohorts on
their outcome invalidates the permutation p-values below -- the null distributions do not
contain that selection step. They are computed as requested and labelled accordingly.

Side effect worth seeing: the HC-HC null model loses a group, dropping from 6 contrasts to
3, and 2 of the 3 survivors involve ds001907, which sits 6-8 SD from every other cohort.
So the yardstick for judging any disease effect gets much weaker.
"""
import numpy as np, pandas as pd, re, itertools, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.metrics import roc_auc_score

B='/master/wang_zi_wei/cursor/pd_classification/'
COL='tplB_caudate__original_firstorder_Minimum'
NSEED=200; NPERM=2000
rng=np.random.default_rng(101)

df=pd.read_csv(B+'data/radiomics_v3_harmonized.csv')
if 'error' in df.columns: df=df[df['error'].isna()]
df['pid']=df['file_path'].apply(lambda f: re.search(r'(sub-[A-Za-z0-9]+)',str(f)).group(1))
rows=[]
def add(p,sep=None,agec='age',sexc=None):
    try: t=pd.read_csv(p,sep=sep) if sep else pd.read_csv(p)
    except Exception: return
    c={x.lower():x for x in t.columns}
    pc=c.get('participant_id',list(t.columns)[0]); ac=c.get(agec.lower())
    sc=c.get((sexc or 'sex').lower()) or c.get('gender')
    o=pd.DataFrame({'pid':t[pc]})
    o['age']=pd.to_numeric(t[ac],errors='coerce') if ac else np.nan
    o['sexraw']=t[sc].astype(str) if sc else None
    rows.append(o)
add('/master/wang_zi_wei/cursor/parkinson_mri_data/participants.tsv',sep='\t',agec='Age',sexc='Gender')
add('/master/wang_zi_wei/cursor/parkinson_mri_data/ds001907_UdallANT/age_clean.csv')
add('/master/wang_zi_wei/cursor/parkinson_mri_data/ds003592_HealthyControls/participants.tsv',sep='\t')
add('/master/wang_zi_wei/cursor/parkinson_mri_data/ds004392_PD_Cognition/participants.tsv',sep='\t')
add('/master/wang_zi_wei/cursor/parkinson_mri_data/ds004471_LHSCPD/participants.tsv',sep='\t')
add('/master/wang_zi_wei/cursor/parkinson_mri_data/ds005892/participants.tsv',sep='\t')
dm=pd.concat(rows).drop_duplicates('pid')
dm['male']=dm['sexraw'].astype(str).str.upper().str[0].map({'M':1,'F':0})
d=df.merge(dm[['pid','age','male']],on='pid',how='left').dropna(subset=[COL,'age'])
d=d[d.dataset!='ds000245']          # <-- the removal
print('datasets kept:',sorted(d.dataset.unique()))

HC=d[d.dataset=='ds003592']
lo,hi=HC.age.quantile([0.05,0.95])
POOL=d[d.dataset.isin(['ds004392','ds004471'])]
POOL=POOL[(POOL.age>=lo)&(POOL.age<=hi)]

def pairs_of(pdg,hcg,use_sex,seed,tol=4.0):
    r=np.random.default_rng(seed); idx=list(pdg.index); r.shuffle(idx)
    used=set(); A=[];Bv=[]
    for i in idx:
        row=pdg.loc[i]; c=hcg[~hcg.index.isin(used)]
        if use_sex: c=c[c.male==row.male]
        c=c[(c.age-row.age).abs()<=tol]
        if len(c):
            j=(c.age-row.age).abs().idxmin(); used.add(j)
            A.append(row[COL]); Bv.append(hcg.loc[j,COL])
    return np.array(A),np.array(Bv)

STRATA=[]
for ds in ['ds001907','ds005892']:
    g=d[d.dataset==ds].dropna(subset=['male'])
    STRATA.append((ds,'same-scanner',g[g.label==1],g[g.label==0],True))
STRATA.append(('ds004392+4471 (age-restricted)','cross-scanner',POOL,HC,False))

print(f'\n{"stratum":<34}{"design":<15}{"nPD":>4}{"nHC":>5}{"pairs":>6}{"d":>9}{"AUC":>8}{"p":>9}')
print('-'*90)
for name,kind,pdg,hcg,us in STRATA:
    ds_,as_,ps_,npr=[],[],[],[]
    for s in range(NSEED):
        A,Bv=pairs_of(pdg,hcg,us,s)
        if len(A)<6: continue
        sd=np.sqrt(((len(A)-1)*A.var(ddof=1)+(len(Bv)-1)*Bv.var(ddof=1))/(len(A)+len(Bv)-2))
        ds_.append((A.mean()-Bv.mean())/sd)
        as_.append(roc_auc_score(np.r_[np.ones(len(A)),np.zeros(len(Bv))],-np.r_[A,Bv]))
        ps_.append(stats.ttest_rel(A,Bv)[1]); npr.append(len(A))
    print(f'{name:<34}{kind:<15}{len(pdg):>4}{len(hcg):>5}{np.mean(npr):>6.0f}'
          f'{np.mean(ds_):>+9.3f}{np.mean(as_):>8.3f}{np.median(ps_):>9.4f}')

def comb(strata,seed,labels=None):
    zs=[];ys=[]
    for k,(nm,kd,pdg,hcg,us) in enumerate(strata):
        if labels is not None:
            allg=pd.concat([pdg,hcg]); y=labels[k]
            pdg2=allg[y==1]; hcg2=allg[y==0]
        else: pdg2,hcg2=pdg,hcg
        A,Bv=pairs_of(pdg2,hcg2,us,seed)
        if len(A)<6: return np.nan
        v=np.r_[A,Bv]; y2=np.r_[np.ones(len(A)),np.zeros(len(Bv))]
        s=v.std(ddof=1); zs.append((v-v.mean())/(s if s>0 else 1)); ys.append(y2)
    return roc_auc_score(np.concatenate(ys),-np.concatenate(zs))

for lbl,st in [('ds001907+ds005892 only (site controlled)',STRATA[:2]),
               ('all three strata (incl. cross-scanner)',STRATA)]:
    obs=np.nanmean([comb(st,s) for s in range(NSEED)])
    null=np.empty(NPERM)
    for i in range(NPERM):
        lab=[]
        for (nm,kd,pdg,hcg,us) in st:
            y=np.r_[np.ones(len(pdg)),np.zeros(len(hcg))]; rng.shuffle(y); lab.append(y)
        null[i]=np.nanmean([comb(st,s,lab) for s in range(10)])
    print(f'\nCOMBINED {lbl}: AUC={obs:.3f}  null mean={np.nanmean(null):.3f} '
          f'95th={np.nanpercentile(null,95):.3f}  p={(null>=obs).mean():.4f}')

print('\n'+'='*70)
print('HC-HC null model with ds000245 removed (was 6 contrasts, now 3)')
print('='*70)
A2=np.c_[np.ones(len(d)), d.age.values, d.age.values**2]
beta=np.linalg.lstsq(A2, d[COL].values, rcond=None)[0]
d['_r']=d[COL].values-A2@beta
hcg={k:d[(d.dataset==k)&(d.label==0)] for k in ['ds001907','ds003592','ds005892']}
vals=[]
for a,b in itertools.combinations(hcg,2):
    u=hcg[a]['_r'].values; v=hcg[b]['_r'].values
    sd=np.sqrt(((len(u)-1)*u.var(ddof=1)+(len(v)-1)*v.var(ddof=1))/(len(u)+len(v)-2))
    e=(u.mean()-v.mean())/sd; vals.append(abs(e))
    print(f'  {a} vs {b}: d={e:+.3f}')
print(f'  |d| mean={np.mean(vals):.3f}   (with ds000245 it was 6 contrasts, mean |d| 2.78)')
