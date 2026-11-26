"""What the single-diagnosis cohorts CAN establish: the null model.

ds003592 (90 HC) cannot give a PD-vs-HC contrast on its own, but it can be compared with
the HC of every mixed cohort. Those HC-vs-HC contrasts contain zero disease and measure
pure site effect. They are the yardstick any PD-HC effect has to clear.

Groups available (variant B, age+age^2 adjusted):
  HC: ds000245_HC, ds001907_HC, ds003592, ds005892_HC          -> 6 HC-HC contrasts
  PD: ds000245_PD, ds001907_PD, ds004392, ds004471, ds005892_PD -> 20 PD-HC contrasts
Three of the PD-HC contrasts are within-dataset, so scanner-free; they are marked.

Also uses ds003592's 90 controls -- by far the largest single-scanner control sample -- to
estimate the feature's age and sex relationship more precisely than any mixed cohort can.
"""
import numpy as np, pandas as pd, re, itertools, warnings
warnings.filterwarnings('ignore')
from scipy import stats

B='/master/wang_zi_wei/cursor/pd_classification/'
COL='tplB_caudate__original_firstorder_Minimum'

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

A=np.c_[np.ones(len(d)), d.age.values, d.age.values**2]
beta=np.linalg.lstsq(A, d[COL].values, rcond=None)[0]
d['_r']=d[COL].values - A@beta

groups={}
for ds,g in d.groupby('dataset'):
    for lab in [0,1]:
        s=g[g.label==lab]
        if len(s)>=8: groups[f'{ds}_{"PD" if lab else "HC"}']=s
print('groups:',list(groups))

hh=[];ph=[]
for x,y in itertools.combinations(groups,2):
    u=d.loc[groups[x].index,'_r'].values; v=d.loc[groups[y].index,'_r'].values
    sd=np.sqrt(((len(u)-1)*u.var(ddof=1)+(len(v)-1)*v.var(ddof=1))/(len(u)+len(v)-2))
    if sd==0: continue
    e=(u.mean()-v.mean())/sd
    xp='PD' in x; yp='PD' in y
    same=x.split('_')[0]==y.split('_')[0]
    if not xp and not yp: hh.append((f'{x} vs {y}',e))
    elif xp!=yp:
        ph.append((f'{x if xp else y} - {y if xp else x}', e if xp else -e, same))

print('\n'+'='*70)
print('HC-vs-HC contrasts (zero disease, pure site) -- THE NULL MODEL')
print('='*70)
for k,v in sorted(hh,key=lambda t:t[1]): print(f'  {k:<44} d={v:+.3f}')
hv=np.array([v for _,v in hh])
print(f'  mean={hv.mean():+.3f}  sd={hv.std(ddof=1):.3f}  |d| range {np.abs(hv).min():.2f}-{np.abs(hv).max():.2f}')

print('\n'+'='*70)
print('PD-vs-HC contrasts')
print('='*70)
for k,v,s in sorted(ph,key=lambda t:t[1]):
    print(f'  {k:<44} d={v:+.3f}  {"[same scanner]" if s else ""}')
pv=np.array([v for _,v,_ in ph]); wi=np.array([v for _,v,s in ph if s])
print(f'  mean={pv.mean():+.3f}  sd={pv.std(ddof=1):.3f}   n={len(pv)}')
print(f'  scanner-free subset: mean={wi.mean():+.3f}  values={[round(x,3) for x in wi]}')

t,p=stats.ttest_ind(pv,hv,equal_var=False)
print(f'\n  PD-HC vs HC-HC shift: t={t:.2f}  p={p:.4f}')
t2,p2=stats.ttest_ind(wi,hv,equal_var=False)
print(f'  scanner-free PD-HC vs HC-HC: t={t2:.2f}  p={p2:.4f}')
frac=(np.abs(hv)>=abs(wi.mean())).mean()
print(f'  fraction of disease-free site contrasts as large as the scanner-free PD effect '
      f'({wi.mean():+.3f}): {frac:.2f}')

print('\n'+'='*70)
print('ds003592 (n=90, largest single-scanner control sample): age and sex relationship')
print('='*70)
h=d[d.dataset=='ds003592']
r,pp=stats.pearsonr(h.age.values,h[COL].values)
print(f'  feature vs age: r={r:+.3f}  p={pp:.4f}  (n={len(h)})')
if h.male.notna().sum()>10:
    m=h[h.male==1][COL].values; f=h[h.male==0][COL].values
    sd=np.sqrt(((len(m)-1)*m.var(ddof=1)+(len(f)-1)*f.var(ddof=1))/(len(m)+len(f)-2))
    print(f'  male vs female: d={(m.mean()-f.mean())/sd:+.3f}  (n male={len(m)}, female={len(f)})')
print('  for comparison, age relationship in the smaller mixed-cohort HC groups:')
for ds in ['ds000245','ds001907','ds005892']:
    g=d[(d.dataset==ds)&(d.label==0)]
    rr,ppp=stats.pearsonr(g.age.values,g[COL].values)
    print(f'    {ds} HC (n={len(g)}): r={rr:+.3f} p={ppp:.3f}')
