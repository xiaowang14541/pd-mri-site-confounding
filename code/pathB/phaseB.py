"""
Path B / 5-cohort robustness.  Runs after pathB/extract_new_cohorts.py finishes.

Adds Tao Wu + NEUROCON (same pipeline) to the three discovery cohorts and repeats the
key within-scanner analyses with FIVE single-scanner cohorts:
  - per-cohort age/sex-matched AUC for the pre-specified markers
  - pooled within-cohort Cohen d + within-cohort permutation p
  - leave-one-cohort-out AUC (5 folds)
  - the full 232-feature screen with FDR (does anything clear now?)
  - HC-vs-HC between-site null (5 HC groups -> 10 contrasts) vs within-scanner PD-HC

Output: pathB/phaseB_results.txt
"""
import numpy as np, pandas as pd, re, warnings, json, glob
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

B = '/master/wang_zi_wei/cursor/pd_classification/'
PP = '/master/wang_zi_wei/cursor/parkinson_mri_data/'
PB = '/master/wang_zi_wei/cursor/pathB_data/'
DISC = ['ds000245', 'ds001907', 'ds005892']
NEW = ['taowu', 'neurocon']
ALL5 = DISC + NEW
SEED = 20260902
AGE_TOL = 3.0
NPERM = 10000
rng = np.random.default_rng(SEED)

OUT = open(B + 'pathB/phaseB_results.txt', 'w')
def p(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()

# ---------------------------------------------------------------- demographics
def demo_disc():
    rows = []
    d = pd.read_csv(PP + 'participants.tsv', sep='\t').rename(
        columns={'participant_id': 'pid', 'Age': 'age', 'Gender': 'sex'})
    rows.append(d[['pid', 'age', 'sex']])
    d = pd.read_csv(PP + 'ds001907_UdallANT/age_clean.csv').rename(
        columns={'participant_id': 'pid', 'gender': 'sex'})
    rows.append(d[['pid', 'age', 'sex']])
    d = pd.read_csv(PP + 'ds005892/participants.tsv', sep='\t').rename(columns={'participant_id': 'pid'})
    rows.append(d[['pid', 'age', 'sex']])
    return pd.concat(rows)

def demo_new():
    rows = []
    tw = pd.read_csv(PB + 'taowu/taowu/taowu_patients.tsv', sep='\t')
    tw['pid'] = 'sub-' + tw['name'].astype(str)
    rows.append(tw[['pid', 'age', 'sex']])
    nc = pd.read_csv(PB + 'neurocon/neurocon/neurocon_patients.tsv', sep='\t')
    nc['pid'] = 'sub-' + nc['code'].astype(str)
    rows.append(nc[['pid', 'age', 'sex']])
    return pd.concat(rows)

dm = pd.concat([demo_disc(), demo_new()]).drop_duplicates('pid')
dm['age'] = pd.to_numeric(dm['age'], errors='coerce')
dm['male'] = dm['sex'].astype(str).str.upper().str[0].map({'M': 1, 'F': 0})
dm = dm[['pid', 'age', 'male']]

# ---------------------------------------------------------------- features
r1 = pd.read_csv(B + 'data/radiomics_v3_harmonized.csv')
r2 = pd.read_csv(B + 'data/radiomics_v3_taowu_neurocon.csv')
rad = pd.concat([r1, r2], ignore_index=True)
if 'error' in rad.columns:
    nerr = rad['error'].notna().sum()
    rad = rad[rad['error'].isna()]
else:
    nerr = 0
rad['pid'] = rad['file_path'].apply(lambda f: re.search(r'(sub-[A-Za-z0-9]+)', str(f)).group(1))
rad = rad[rad.dataset.isin(ALL5)].reset_index(drop=True)
rad = rad.merge(dm, on='pid', how='left')
for roi in ['caudate', 'putamen', 'pallidum', 'thalamus']:
    rad[f'volc_{roi}'] = rad[f'vol_mm3_{roi}'] / rad['brain_vol_native_mm3']

p('=' * 92)
p('PHASE B / 5-COHORT ROBUSTNESS', ALL5, ' (extraction errors dropped:', nerr, ')')
p('=' * 92)
for ds in ALL5:
    g = rad[rad.dataset == ds]
    p(f'  {ds:<10} nPD={(g.label==1).sum():>3} nHC={(g.label==0).sum():>3}  '
      f'age NA={g.age.isna().sum()}  sex NA={g.male.isna().sum()}  '
      f'dice mean={g.reg_dice.mean():.3f}')

# ---------------------------------------------------------------- matching
def matched_index(g, seed):
    r = np.random.default_rng(seed)
    pdg = g[g.label == 1].dropna(subset=['age', 'male'])
    hcg = g[g.label == 0].dropna(subset=['age', 'male'])
    idx = list(pdg.index); r.shuffle(idx)
    used = set(); pr = []
    for i in idx:
        row = pdg.loc[i]
        c = hcg[~hcg.index.isin(used) & (hcg.male == row.male)]
        c = c[(c.age - row.age).abs() <= AGE_TOL]
        if len(c):
            j = (c.age - row.age).abs().idxmin(); used.add(j); pr.append((int(i), int(j)))
    return pr

def best_matched(g):
    cand = [matched_index(g, s) for s in range(200)]
    med = int(np.median([len(x) for x in cand]))
    best, bg = None, 1e9
    for pr in cand:
        if len(pr) != med:
            continue
        gp = np.mean([abs(g.loc[i, 'age'] - g.loc[j, 'age']) for i, j in pr])
        if gp < bg:
            bg, best = gp, pr
    return best or []

MATCH = {ds: best_matched(rad[rad.dataset == ds]) for ds in ALL5}
for ds in ALL5:
    pr = MATCH[ds]
    gp = np.mean([abs(rad.loc[i, 'age'] - rad.loc[j, 'age']) for i, j in pr]) if pr else float('nan')
    p(f'  matched {ds:<10}: {len(pr):>2} pairs  mean|age gap|={gp:.2f}')
NPAIR = sum(len(v) for v in MATCH.values())
p(f'  POOLED matched pairs (5 cohorts): {NPAIR}')

USE = [ds for ds in ALL5 if len(MATCH[ds]) >= 4]

def mrows(dss):
    recs = []
    for ds in dss:
        for i, j in MATCH[ds]:
            recs.append((ds, 1, i)); recs.append((ds, 0, j))
    return pd.DataFrame(recs, columns=['dataset', 'y', 'ridx'])

def cval(fr, col):
    return fr.ridx.map(rad[col]).values

def pooled_d_arr(ds_a, y_a, x_a):
    o = []
    for ds in np.unique(ds_a):
        m = ds_a == ds
        a = x_a[m & (y_a == 1)]; b = x_a[m & (y_a == 0)]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        sd = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        if sd == 0:
            continue
        o.append((a.mean()-b.mean())/sd)
    return float(np.mean(o)) if o else np.nan

def pooled_d(col):
    fr = mrows(USE); return pooled_d_arr(fr.dataset.values, fr.y.values, cval(fr, col))

def perm_p(col, obs):
    fr = mrows(USE); ds_a, y_a, x_a = fr.dataset.values, fr.y.values.copy(), cval(fr, col)
    direction = -1 if (obs == obs and obs < 0) else 1
    null = np.empty(NPERM)
    for k in range(NPERM):
        yp = y_a.copy()
        for ds in USE:
            m = ds_a == ds; yp[m] = rng.permutation(yp[m])
        null[k] = pooled_d_arr(ds_a, yp, x_a)
    null = null[np.isfinite(null)]
    return (null <= obs).mean() if direction < 0 else (null >= obs).mean()

def loco_auc(col):
    y_all, s_all = [], []
    for held in USE:
        tr = mrows([c for c in USE if c != held]); te = mrows([held])
        xtr, xte = cval(tr, col), cval(te, col)
        if not np.isfinite(xtr).all() or not np.isfinite(xte).all():
            return np.nan
        d_tr = pooled_d_arr(tr.dataset.values, tr.y.values, xtr)
        sgn = np.sign(d_tr) if (d_tr == d_tr and d_tr != 0) else 1.0
        mu, sd = xtr.mean(), xtr.std()
        if sd == 0 or not np.isfinite(sd):
            return np.nan
        s_all += list(-sgn*(xte-mu)/sd); y_all += list(te.y.values)
    try:
        return roc_auc_score(y_all, s_all)
    except Exception:
        return np.nan

def per_cohort_auc(col):
    out = {}
    for ds in USE:
        fr = mrows([ds]); x = cval(fr, col); y = fr.y.values
        if np.isfinite(x).all():
            d = pooled_d_arr(fr.dataset.values, y, x)
            sgn = np.sign(d) if d == d and d != 0 else 1
            try:
                out[ds] = roc_auc_score(y, -sgn*x)
            except Exception:
                out[ds] = np.nan
    return out

p('\n' + '=' * 92)
p('PRE-SPECIFIED MARKERS on 5 cohorts (age/sex-matched)')
p('=' * 92)
markers = ['volc_caudate', 'volc_putamen', 'volc_pallidum', 'volc_thalamus']
p(f'{"marker":<20}{"pooled d":>10}{"perm p":>9}{"LOCO AUC":>10}   per-cohort AUC')
for col in markers:
    d = pooled_d(col); pp_ = perm_p(col, d); a = loco_auc(col)
    pc = per_cohort_auc(col)
    p(f'{col:<20}{d:>+10.3f}{pp_:>9.4f}{a:>10.3f}   ' +
      ' '.join(f'{k.split("ds")[-1][:6]}={v:.2f}' for k, v in pc.items()))

p('\n' + '=' * 92)
p('232-FEATURE RADIOMIC SCREEN on 5 cohorts (matched); FDR-BH')
p('=' * 92)
tcols = [c for c in rad.columns if c.startswith('tplB_') and '__original_' in c and not rad[c].isna().any()]
scr = []
for col in tcols:
    diffs = []
    for ds in USE:
        for i, j in MATCH[ds]:
            diffs.append(rad.loc[i, col] - rad.loc[j, col])
    diffs = np.array(diffs, float)
    if not np.isfinite(diffs).all() or len(diffs) < 12:
        continue
    scr.append((col, pooled_d(col), stats.ttest_1samp(diffs, 0).pvalue))
scr.sort(key=lambda x: x[2])
fdr = multipletests([s[2] for s in scr], method='fdr_bh')[1] if scr else np.array([])
p(f'{"feature":<52}{"d":>8}{"p":>11}{"FDR":>8}{"LOCO AUC":>10}')
for k in range(min(12, len(scr))):
    col, d, pv = scr[k]
    p(f'{col:<52}{d:>+8.3f}{pv:>11.2e}{fdr[k]:>8.3f}{loco_auc(col):>10.3f}')
p(f'\n  min FDR-BH over {len(scr)} features: {np.min(fdr) if len(fdr) else float("nan"):.3f}'
  f'   # FDR<0.05: {int((fdr<0.05).sum()) if len(fdr) else 0}')

p('\n' + '=' * 92)
p('HC-vs-HC BETWEEN-SITE NULL vs WITHIN-SCANNER PD-HC  (feature: tplB_caudate firstorder_Minimum)')
p('=' * 92)
COL = 'tplB_caudate__original_firstorder_Minimum'
if COL in rad.columns:
    hc = {ds: rad[(rad.dataset == ds) & (rad.label == 0)][COL].dropna().values for ds in ALL5}
    hc = {k: v for k, v in hc.items() if len(v) >= 8}
    ks = list(hc)
    hh = []
    for i in range(len(ks)):
        for j in range(i+1, len(ks)):
            a, b = hc[ks[i]], hc[ks[j]]
            sd = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
            if sd > 0:
                hh.append(abs((a.mean()-b.mean())/sd))
    p(f'  HC-HC between-site |d|: n={len(hh)}  mean={np.mean(hh):.2f}  '
      f'median={np.median(hh):.2f}  max={np.max(hh):.2f}')
    ww = []
    for ds in ALL5:
        g = rad[rad.dataset == ds]
        a = g[g.label == 1][COL].dropna().values; b = g[g.label == 0][COL].dropna().values
        if len(a) >= 5 and len(b) >= 5:
            sd = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
            if sd > 0:
                ww.append(abs((a.mean()-b.mean())/sd))
    p(f'  within-scanner PD-HC |d|:  n={len(ww)}  mean={np.mean(ww):.2f}  '
      f'values={[round(x,2) for x in ww]}')
    p(f'  ratio between-site / within-scanner (means): {np.mean(hh)/np.mean(ww):.1f}x')

p('\n' + '=' * 92)
p('VERDICT')
p('=' * 92)
best = max([loco_auc(c) for c in markers if loco_auc(c) == loco_auc(c)] +
           [np.nanmax([loco_auc(s[0]) for s in scr[:12]]) if scr else 0])
p(f'  best pre-specified LOCO AUC (5 cohorts): {best:.3f}')
p(f'  radiomic screen FDR-significant features: '
  f'{int((fdr<0.05).sum()) if len(fdr) else 0} / {len(scr)}')
p('  -> Path A null holds across 5 single-scanner cohorts' if best < 0.60 and
  (len(fdr) == 0 or (fdr < 0.05).sum() == 0) else '  -> re-examine: something cleared')
OUT.close()
