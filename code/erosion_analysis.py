"""Analyse the erosion experiment: does the caudate effect die when the ventricular rim
is removed, and do the darkest voxels sit on that rim?

Matched design as everywhere else (sex-exact, age<=4y, 200 seeds), 3 same-scanner cohorts.
"""
import numpy as np, pandas as pd, re, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from scipy import stats

B='/master/wang_zi_wei/cursor/pd_classification/'
MIXED=['ds000245','ds001907','ds005892']
NSEED=200

df=pd.read_csv(B+'data/erosion_features.csv')
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
add('/master/wang_zi_wei/cursor/parkinson_mri_data/ds005892/participants.tsv',sep='\t')
dm=pd.concat(rows).drop_duplicates('pid')
dm['male']=dm['sexraw'].astype(str).str.upper().str[0].map({'M':1,'F':0})
d=df.merge(dm[['pid','age','male']],on='pid',how='left')
print(f'n={len(d)}  {d.dataset.value_counts().to_dict()}')

def matched(col):
    pd_all=[];au_all=[]
    for ds in MIXED:
        g=d[d.dataset==ds].dropna(subset=[col,'age','male'])
        if len(g)<12: return np.nan,np.nan
        pdg=g[g.label==1]; hcg=g[g.label==0]
        dd=[];aa=[]
        for s in range(NSEED):
            r=np.random.default_rng(s); idx=list(pdg.index); r.shuffle(idx)
            used=set(); A=[];Bv=[]
            for i in idx:
                row=pdg.loc[i]
                c=hcg[(~hcg.index.isin(used))&(hcg.male==row.male)&((hcg.age-row.age).abs()<=4)]
                if len(c):
                    j=(c.age-row.age).abs().idxmin(); used.add(j)
                    A.append(row[col]); Bv.append(hcg.loc[j,col])
            if len(A)<6: continue
            A=np.array(A);Bv=np.array(Bv)
            sd=np.sqrt(((len(A)-1)*A.var(ddof=1)+(len(Bv)-1)*Bv.var(ddof=1))/(len(A)+len(Bv)-2))
            if sd==0: continue
            dd.append((A.mean()-Bv.mean())/sd)
            aa.append(roc_auc_score(np.r_[np.ones(len(A)),np.zeros(len(Bv))],-np.r_[A,Bv]))
        if not dd: return np.nan,np.nan
        pd_all.append(np.mean(dd)); au_all.append(np.mean(aa))
    return float(np.mean(pd_all)),float(np.mean(au_all))

print('\n'+'='*72)
print('TEST 1: EROSION -- does the effect survive removal of the ventricular rim?')
print('='*72)
print(f'{"ROI":<10}{"erosion":<12}{"voxels":>8}{"matched d":>12}{"AUC":>8}')
for roi in ['caudate','putamen']:
    for k,lab in [(0,'none'),(1,'1 vox (2mm)'),(2,'2 vox (4mm)')]:
        col=f'{roi}_e{k}__Minimum'
        if col not in d.columns: continue
        nv=d[f'{roi}_e{k}__n'].median()
        dd,aa=matched(col)
        print(f'{roi:<10}{lab:<12}{nv:>8.0f}{dd:>+12.3f}{aa:>8.3f}')

print('\n'+'='*72)
print('TEST 2: WHERE ARE THE DARKEST VOXELS? (distance from ROI rim, in voxels)')
print('='*72)
print(f'{"ROI":<10}{"darkest 10":>12}{"all voxels":>12}{"brightest 10":>14}')
for roi in ['caudate','putamen']:
    c1,c2,c3=f'{roi}__dist_darkest10',f'{roi}__dist_all_mean',f'{roi}__dist_brightest10'
    if c1 not in d.columns: continue
    print(f'{roi:<10}{d[c1].mean():>12.2f}{d[c2].mean():>12.2f}{d[c3].mean():>14.2f}')
print('  (1.0 = sitting on the rim; larger = deeper inside the structure)')

print('\n'+'='*72)
print('TEST 3: CSF-like voxel fraction inside the ROI, PD vs HC')
print('='*72)
for roi in ['caudate','putamen']:
    for thr in ['1.0','1.5']:
        col=f'{roi}_e0__csffrac{thr}'
        if col not in d.columns: continue
        dd,aa=matched(col)
        m=d.groupby('label')[col].mean()
        print(f'  {roi:<9} frac(z<-{thr}): HC={m.get(0,np.nan):.4f} PD={m.get(1,np.nan):.4f}  matched d={dd:+.3f} AUC={aa:.3f}')

print('\n'+'='*72)
print('TEST 4: statistic gradient within each erosion level (caudate)')
print('='*72)
print(f'{"erosion":<12}{"Minimum":>10}{"p10":>10}{"Median":>10}{"Mean":>10}')
for k,lab in [(0,'none'),(1,'1 vox'),(2,'2 vox')]:
    vals=[]
    for st in ['Minimum','p10','Median','Mean']:
        col=f'caudate_e{k}__{st}'
        vals.append(matched(col)[0] if col in d.columns else np.nan)
    print(f'{lab:<12}'+''.join(f'{v:>+10.3f}' if v==v else f'{"--":>10}' for v in vals))
