"""
Gen-2 manuscript robustness addendum: repeat the key within-scanner analyses of
Sections 3.3-3.6 adding two further single-scanner cohorts (Tao Wu, NEUROCON) that
were not part of the original OpenNeuro set.

Matches the manuscript's spec: 1:1 sex-exact matching, age tolerance +-4 y, nearest
control, 200 random match orders averaged; caudate first-order Minimum on the
harmonised (variant B) feature; DBM/warped-atlas caudate volume for the volumetry check.

Writes: robustness_5cohort_results.txt
"""
import numpy as np, pandas as pd, re, warnings, json
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

B = '/master/wang_zi_wei/cursor/pd_classification/'
PP = '/master/wang_zi_wei/cursor/parkinson_mri_data/'
PB = '/master/wang_zi_wei/cursor/pathB_data/'
COL = 'tplB_caudate__original_firstorder_Minimum'
AGE_TOL = 4.0
NSEED = 200
NPERM = 20000
rng = np.random.default_rng(20260902)
OUT = open(B + 'pathB/robustness_5cohort_results.txt', 'w')
def p(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s+'\n'); OUT.flush()

# ---- demographics for all 5 same-scanner cohorts --------------------------
def demo():
    rows = []
    d = pd.read_csv(PP+'participants.tsv', sep='\t').rename(columns={'participant_id':'pid','Age':'age','Gender':'sex'}); rows.append(d[['pid','age','sex']])
    d = pd.read_csv(PP+'ds001907_UdallANT/age_clean.csv').rename(columns={'participant_id':'pid','gender':'sex'}); rows.append(d[['pid','age','sex']])
    d = pd.read_csv(PP+'ds005892/participants.tsv', sep='\t').rename(columns={'participant_id':'pid'}); rows.append(d[['pid','age','sex']])
    tw = pd.read_csv(PB+'taowu/taowu/taowu_patients.tsv', sep='\t'); tw['pid']='sub-'+tw['name'].astype(str); rows.append(tw[['pid','age','sex']])
    nc = pd.read_csv(PB+'neurocon/neurocon/neurocon_patients.tsv', sep='\t'); nc['pid']='sub-'+nc['code'].astype(str); rows.append(nc[['pid','age','sex']])
    dm = pd.concat(rows).drop_duplicates('pid')
    dm['age'] = pd.to_numeric(dm['age'], errors='coerce')
    dm['male'] = dm['sex'].astype(str).str.upper().str[0].map({'M':1,'F':0})
    return dm[['pid','age','male']]

r1 = pd.read_csv(B+'data/radiomics_v3_harmonized.csv')
r2 = pd.read_csv(B+'data/radiomics_v3_taowu_neurocon.csv')
rad = pd.concat([r1, r2], ignore_index=True)
if 'error' in rad.columns: rad = rad[rad['error'].isna()]
rad['pid'] = rad['file_path'].apply(lambda f: re.search(r'(sub-[A-Za-z0-9]+)', str(f)).group(1))
ALL5 = ['ds000245','ds001907','ds005892','taowu','neurocon']
rad = rad[rad.dataset.isin(ALL5)].reset_index(drop=True).merge(demo(), on='pid', how='left')
rad['volc_caudate'] = rad['vol_mm3_caudate'] / rad['brain_vol_native_mm3']

p('='*80); p('5-COHORT ROBUSTNESS  (manuscript spec: sex-exact, age +-4y, 200 seeds)'); p('='*80)
for ds in ALL5:
    g = rad[rad.dataset==ds]
    aP, aH = g[g.label==1].age, g[g.label==0].age
    p(f'  {ds:<10} PD={ (g.label==1).sum():>3} HC={ (g.label==0).sum():>3}  '
      f'agePD={aP.mean():.1f} ageHC={aH.mean():.1f}  '
      f'%mPD={100*g[g.label==1].male.mean():.0f} %mHC={100*g[g.label==0].male.mean():.0f}  '
      f'dice={g.reg_dice.mean():.3f}')

# ---- matched effect per cohort (caudate Minimum) --------------------------
def matched_d(g, col, seed):
    r = np.random.default_rng(seed)
    pdg = g[g.label==1].dropna(subset=['age','male',col]); hcg = g[g.label==0].dropna(subset=['age','male',col])
    idx = list(pdg.index); r.shuffle(idx); used=set(); A=[];Bv=[]
    for i in idx:
        row = pdg.loc[i]; c = hcg[~hcg.index.isin(used) & (hcg.male==row.male)]
        c = c[(c.age-row.age).abs()<=AGE_TOL]
        if len(c):
            j=(c.age-row.age).abs().idxmin(); used.add(j)
            A.append(row[col]); Bv.append(hcg.loc[j,col])
    if len(A) < 4: return np.nan, np.nan, 0
    A=np.array(A); Bv=np.array(Bv)
    sd=np.sqrt(((len(A)-1)*A.var(ddof=1)+(len(Bv)-1)*Bv.var(ddof=1))/(len(A)+len(Bv)-2))
    d = (A.mean()-Bv.mean())/sd if sd>0 else np.nan
    try: auc = roc_auc_score(np.r_[np.ones(len(A)),np.zeros(len(Bv))], -np.r_[A,Bv])
    except Exception: auc = np.nan
    return d, auc, len(A)

p('\n--- caudate first-order Minimum, matched effect per cohort ---')
per = {}
for ds in ALL5:
    g = rad[rad.dataset==ds]
    ds_, au_, np_ = zip(*[matched_d(g, COL, s) for s in range(NSEED)])
    per[ds] = (np.nanmean(ds_), np.nanmean(au_), int(np.median(np_)))
    p(f'  {ds:<10} d={per[ds][0]:+.3f}  AUC={per[ds][1]:.3f}  pairs={per[ds][2]}')
new = [per[k][0] for k in ['taowu','neurocon']]
p(f'\n  original 3 within-cohort d: -0.378, -0.651, -0.447  (mean -0.492)')
p(f'  + taowu={per["taowu"][0]:+.3f}  + neurocon={per["neurocon"][0]:+.3f}')
alld = [-0.378,-0.651,-0.447, per['taowu'][0], per['neurocon'][0]]
p(f'  5-cohort mean within-scanner PD-HC d = {np.mean(alld):+.3f}')

# ---- HC-HC between-site null over 5 cohorts (age-adjusted) ---------------
p('\n--- HC-HC between-site |d| for caudate Minimum (age+age^2 adjusted), 5 cohorts ---')
hc = {}
for ds in ALL5:
    g = rad[(rad.dataset==ds) & (rad.label==0)].dropna(subset=[COL,'age'])
    if len(g) >= 8:
        # residualise on age + age^2
        X = np.c_[np.ones(len(g)), g.age.values, g.age.values**2]
        b, *_ = np.linalg.lstsq(X, g[COL].values, rcond=None)
        hc[ds] = g[COL].values - X@b + g[COL].mean()
ks = list(hc); hh=[]
for i in range(len(ks)):
    for j in range(i+1,len(ks)):
        a,b = hc[ks[i]], hc[ks[j]]
        sd=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        if sd>0: hh.append(abs((a.mean()-b.mean())/sd))
p(f'  cohorts with >=8 HC: {ks}')
p(f'  n contrasts={len(hh)}  |d| = {[round(x,2) for x in sorted(hh)]}')
p(f'  mean |d|={np.mean(hh):.2f}  median={np.median(hh):.2f}  max={np.max(hh):.2f}')
p(f'  ratio (mean HC-HC |d|) / (mean within-scanner PD-HC |d|) = {np.mean(hh)/abs(np.mean(alld)):.1f}x')

# ---- volumetry check on the 2 new cohorts ------------------------------
p('\n--- warped caudate volume (ICV-corrected), matched, new cohorts ---')
for ds in ['taowu','neurocon']:
    g = rad[rad.dataset==ds]
    dvals = [matched_d(g, 'volc_caudate', s)[0] for s in range(NSEED)]
    p(f'  {ds:<10} matched d(volume) = {np.nanmean(dvals):+.3f}')

# ---- 232-feature + 4-volume screen, 5 cohorts, FDR --------------------
p('\n--- full radiomic screen, 5 cohorts, matched (sex-exact +-4y) ---')
tcols = [c for c in rad.columns if c.startswith('tplB_') and '__original_' in c and not rad[c].isna().any()]
def pooled_matched_d(col):
    ds_d=[]
    for ds in ALL5:
        g = rad[rad.dataset==ds]
        dd = np.nanmean([matched_d(g, col, s)[0] for s in range(40)])
        if np.isfinite(dd): ds_d.append(dd)
    return np.mean(ds_d) if ds_d else np.nan
# fast screen: paired t on one matched set per cohort (seed 0), pooled
def screen_p(col):
    diffs=[]
    for ds in ALL5:
        g = rad[rad.dataset==ds]
        r = np.random.default_rng(0)
        pdg=g[g.label==1].dropna(subset=['age','male',col]); hcg=g[g.label==0].dropna(subset=['age','male',col])
        idx=list(pdg.index); r.shuffle(idx); used=set()
        for i in idx:
            row=pdg.loc[i]; c=hcg[~hcg.index.isin(used)&(hcg.male==row.male)]; c=c[(c.age-row.age).abs()<=AGE_TOL]
            if len(c):
                j=(c.age-row.age).abs().idxmin(); used.add(j); diffs.append(row[col]-hcg.loc[j,col])
    diffs=np.array(diffs,float)
    return stats.ttest_1samp(diffs,0).pvalue if len(diffs)>10 else np.nan
ps = [(c, screen_p(c)) for c in tcols]
ps = [(c,x) for c,x in ps if np.isfinite(x)]
fdr = multipletests([x for _,x in ps], method='fdr_bh')[1]
order = np.argsort([x for _,x in ps])
p(f'  n features screened: {len(ps)}')
p(f'  min raw p = {min(x for _,x in ps):.4f}   min FDR-BH = {fdr.min():.3f}   # FDR<0.05 = {(fdr<0.05).sum()}')
for k in order[:6]:
    p(f'    {ps[k][0]:<52} p={ps[k][1]:.4f}  FDR={fdr[k]:.3f}')

json.dump({'per_cohort_caudateMin': {k:[round(v,3) for v in per[k]] for k in per},
           'within_scanner_PD_HC_d_5cohort_mean': float(np.mean(alld)),
           'HC_HC_between_site_d': {'n':len(hh),'mean':float(np.mean(hh)),'median':float(np.median(hh)),'max':float(np.max(hh))},
           'ratio': float(np.mean(hh)/abs(np.mean(alld))),
           'screen_min_fdr': float(fdr.min()), 'screen_n_sig': int((fdr<0.05).sum())},
          open(B+'pathB/robustness_5cohort.json','w'), indent=2)
OUT.close()
