"""Does the SAME statistic that produced the spurious disease effect also carry the most
site variance? Measured in HEALTHY CONTROLS ONLY, so no disease contribution is possible.

If extreme-value features report boundary voxels rather than tissue, then across cohorts
they should be far more variable than statistics describing the centre of the intensity
distribution. This links the two findings of the paper: the statistic that is most
site-sensitive is the same one that yielded an apparent PD effect.

For each statistic, all pairwise healthy-versus-healthy contrasts between the four cohorts
with >=8 controls, age- and age^2-adjusted. Reported as mean |d| and range.
Also reported for the four subcortical structures separately, to test whether the excess
site sensitivity of extreme-value statistics is itself concentrated in ventricle-adjacent
regions.
"""
import numpy as np, pandas as pd, re, itertools, warnings
warnings.filterwarnings('ignore')

B='/master/wang_zi_wei/cursor/pd_classification/'
df=pd.read_csv(B+'data/radiomics_v3_harmonized.csv')
if 'error' in df.columns: df=df[df['error'].isna()]
df['pid']=df['file_path'].apply(lambda f: re.search(r'(sub-[A-Za-z0-9]+)',str(f)).group(1))
rows=[]
def add(p,sep=None,agec='age'):
    try: t=pd.read_csv(p,sep=sep) if sep else pd.read_csv(p)
    except Exception: return
    c={x.lower():x for x in t.columns}
    pc=c.get('participant_id',list(t.columns)[0]); ac=c.get(agec.lower())
    o=pd.DataFrame({'pid':t[pc]}); o['age']=pd.to_numeric(t[ac],errors='coerce') if ac else np.nan
    rows.append(o)
for p,s in [('/master/wang_zi_wei/cursor/parkinson_mri_data/participants.tsv','\t'),
            ('/master/wang_zi_wei/cursor/parkinson_mri_data/ds001907_UdallANT/age_clean.csv',None),
            ('/master/wang_zi_wei/cursor/parkinson_mri_data/ds003592_HealthyControls/participants.tsv','\t'),
            ('/master/wang_zi_wei/cursor/parkinson_mri_data/ds005892/participants.tsv','\t')]:
    add(p,s,'Age' if 'parkinson_mri_data/participants' in p else 'age')
dm=pd.concat(rows).drop_duplicates('pid')
d=df.merge(dm[['pid','age']],on='pid',how='left').dropna(subset=['age'])
HC=d[d.label==0]
print('HC per cohort:',HC.dataset.value_counts().to_dict())

STATS=[('firstorder_Minimum','Minimum (1 voxel, extreme)'),
       ('firstorder_Maximum','Maximum (1 voxel, extreme)'),
       ('firstorder_Range','Range (both extremes)'),
       ('firstorder_10Percentile','10th percentile (robust tail)'),
       ('firstorder_90Percentile','90th percentile (robust tail)'),
       ('firstorder_Median','Median (centre)'),
       ('firstorder_Mean','Mean (centre)')]

def site_effects(col):
    sub=HC.dropna(subset=[col])
    if len(sub)<40: return None
    A=np.c_[np.ones(len(sub)), sub.age.values, sub.age.values**2]
    beta=np.linalg.lstsq(A, sub[col].values, rcond=None)[0]
    r=sub[col].values-A@beta
    sub=sub.assign(_r=r)
    groups={k:g for k,g in sub.groupby('dataset') if len(g)>=8}
    out=[]
    for a,b in itertools.combinations(groups,2):
        u=groups[a]['_r'].values; v=groups[b]['_r'].values
        sd=np.sqrt(((len(u)-1)*u.var(ddof=1)+(len(v)-1)*v.var(ddof=1))/(len(u)+len(v)-2))
        if sd>0: out.append(abs((u.mean()-v.mean())/sd))
    return np.array(out) if out else None

print('\n'+'='*78)
print('SITE EFFECT BY STATISTIC -- healthy controls only, caudate')
print('  (all pairwise cross-cohort HC-vs-HC contrasts, age-adjusted)')
print('='*78)
print(f'{"statistic":<32}{"n contrasts":>12}{"mean |d|":>10}{"median |d|":>12}{"max |d|":>10}')
ref=None
for st,desc in STATS:
    e=site_effects(f'tplB_caudate__original_{st}')
    if e is None: continue
    if st=='firstorder_Mean': ref=np.median(e)
    print(f'{desc:<32}{len(e):>12}{e.mean():>10.2f}{np.median(e):>12.2f}{e.max():>10.2f}')
if ref:
    print(f'\n  ratio of median site effect, Minimum vs Mean:')
    em=site_effects('tplB_caudate__original_firstorder_Minimum')
    print(f'    {np.median(em)/ref:.2f}x')

print('\n'+'='*78)
print('IS THE EXCESS SITE SENSITIVITY CONCENTRATED IN VENTRICLE-ADJACENT REGIONS?')
print('  median |d| of HC-vs-HC contrasts, by ROI and statistic')
print('='*78)
print(f'{"ROI":<12}{"Minimum":>10}{"10Pct":>10}{"Median":>10}{"Mean":>10}{"Min/Mean":>10}')
for roi in ['caudate','thalamus','putamen','pallidum']:
    vals=[]
    for st in ['firstorder_Minimum','firstorder_10Percentile','firstorder_Median','firstorder_Mean']:
        e=site_effects(f'tplB_{roi}__original_{st}')
        vals.append(np.median(e) if e is not None else np.nan)
    ratio=vals[0]/vals[3] if vals[3] and vals[3]==vals[3] and vals[3]>0 else np.nan
    print(f'{roi:<12}'+''.join(f'{v:>10.2f}' if v==v else f'{"--":>10}' for v in vals)
          +(f'{ratio:>10.2f}' if ratio==ratio else f'{"--":>10}'))
print('\n  (higher Min/Mean ratio = the extreme-value statistic carries disproportionate site variance)')
