"""Pre-specified independent test of the boundary-sampling mechanism.

The mechanism was hypothesised after seeing the results in the three mixed-diagnosis
cohorts, so the four supporting tests reported there are not independent of it. This
script states a prediction the mechanism makes about data that played no part in
generating it, and tests it.

SETTING: ds003592 alone -- 90 healthy controls, one scanner, one diagnosis. No disease
contrast and no between-site comparison enters this analysis, so neither the label
structure nor the site structure that produced the original observation is present.

PREDICTION (stated before running): if the Minimum reads a voxel on the region boundary,
its value must depend on where registration placed the mask, whereas the Mean, which
averages thousands of interior voxels, must not. Therefore
   |r(Minimum, registration quality)|  >  |r(Mean, registration quality)|
   |r(Minimum, warped mask volume)|    >  |r(Mean, warped mask volume)|
and both gaps should be larger in the caudate, which abuts the lateral ventricle, than
in the putamen, which does not.

FALSIFICATION: if the Minimum is no more coupled to registration geometry than the Mean,
or if the caudate shows no excess over the putamen, the mechanism is not supported.

Significance of the difference between two dependent correlations sharing one variable
is tested with Steiger's z.
"""
import numpy as np, pandas as pd, re, warnings
warnings.filterwarnings('ignore')
from scipy import stats

B='/master/wang_zi_wei/cursor/pd_classification/'
df=pd.read_csv(B+'data/radiomics_v3_harmonized.csv')
if 'error' in df.columns: df=df[df['error'].isna()]
d=df[df.dataset=='ds003592'].copy()
print(f'ds003592: n={len(d)}  (all healthy controls, single scanner)')

def steiger(r_xy, r_xz, r_yz, n):
    """Steiger's z for r_xy vs r_xz when y and z are correlated (r_yz)."""
    zf=lambda r: 0.5*np.log((1+r)/(1-r))
    rm2=(r_xy**2+r_xz**2)/2
    f=(1-r_yz)/(2*(1-rm2))
    h=(1-f*rm2)/(1-rm2)
    z=(zf(r_xy)-zf(r_xz))*np.sqrt((n-3)/(2*(1-r_yz)*h))
    return z, 2*(1-stats.norm.cdf(abs(z)))

for geo,geoname in [('reg_dice','registration quality (Dice)'),
                    ('vol_mm3_caudate','warped caudate mask volume'),
                    ('vol_mm3_putamen','warped putamen mask volume')]:
    if geo not in d.columns: continue
    roi='caudate' if 'caudate' in geo or geo=='reg_dice' else 'putamen'
    print('\n'+'='*72)
    print(f'geometry variable: {geoname}')
    print('='*72)
    print(f'{"ROI":<10}{"statistic":<14}{"r with geometry":>17}{"p":>9}')
    for r_ in ['caudate','putamen']:
        rows={}
        for st in ['firstorder_Minimum','firstorder_Mean','firstorder_10Percentile']:
            col=f'tplB_{r_}__original_{st}'
            if col not in d.columns: continue
            sub=d.dropna(subset=[col,geo])
            r,p=stats.pearsonr(sub[col].values, sub[geo].values)
            rows[st]=(r,p,len(sub))
            print(f'{r_:<10}{st.replace("firstorder_",""):<14}{r:>+17.3f}{p:>9.4f}')
        if 'firstorder_Minimum' in rows and 'firstorder_Mean' in rows:
            sub=d.dropna(subset=[f'tplB_{r_}__original_firstorder_Minimum',
                                 f'tplB_{r_}__original_firstorder_Mean',geo])
            r_mn_geo=rows['firstorder_Minimum'][0]; r_mean_geo=rows['firstorder_Mean'][0]
            r_mn_mean=stats.pearsonr(sub[f'tplB_{r_}__original_firstorder_Minimum'].values,
                                     sub[f'tplB_{r_}__original_firstorder_Mean'].values)[0]
            z,pz=steiger(abs(r_mn_geo),abs(r_mean_geo),r_mn_mean,len(sub))
            print(f'{"":<10}-> |r(Min)| - |r(Mean)| = {abs(r_mn_geo)-abs(r_mean_geo):+.3f}'
                  f'   Steiger z={z:+.2f}  p={pz:.4f}')

print('\n'+'='*72)
print('PRE-SPECIFIED VERDICT')
print('='*72)
c_min=stats.pearsonr(d['tplB_caudate__original_firstorder_Minimum'],d['reg_dice'])[0]
c_mean=stats.pearsonr(d['tplB_caudate__original_firstorder_Mean'],d['reg_dice'])[0]
p_min=stats.pearsonr(d['tplB_putamen__original_firstorder_Minimum'],d['reg_dice'])[0]
p_mean=stats.pearsonr(d['tplB_putamen__original_firstorder_Mean'],d['reg_dice'])[0]
print(f'  caudate: |r(Min,Dice)|={abs(c_min):.3f}  |r(Mean,Dice)|={abs(c_mean):.3f}'
      f'  gap={abs(c_min)-abs(c_mean):+.3f}')
print(f'  putamen: |r(Min,Dice)|={abs(p_min):.3f}  |r(Mean,Dice)|={abs(p_mean):.3f}'
      f'  gap={abs(p_min)-abs(p_mean):+.3f}')
print(f'  caudate gap exceeds putamen gap: '
      f'{(abs(c_min)-abs(c_mean)) > (abs(p_min)-abs(p_mean))}')
