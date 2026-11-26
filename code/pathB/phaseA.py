"""
Path B - Phase A exploration (discovery cohorts only: ds000245, ds001907, ds005892).

Decide whether ANY candidate structural marker clears the pre-registration gate --
pooled within-cohort matched LOCO AUC >= 0.60 AND (for texture) survives ROI erosion
(keeps >=60% |d| at 1-voxel erosion) -- on the three same-scanner cohorts.
If nothing clears, Phase B is skipped and the project goes Path A (registered null).

All inputs are already-extracted CSVs; no new preprocessing.
Outputs: pathB/phaseA_results.txt , pathB/phaseA_gate.json
"""
import numpy as np, pandas as pd, re, warnings, json
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

B = '/master/wang_zi_wei/cursor/pd_classification/'
PP = '/master/wang_zi_wei/cursor/parkinson_mri_data/'
MIXED = ['ds000245', 'ds001907', 'ds005892']
SEED = 20260902
AGE_TOL = 3.0
NPERM = 10000
rng = np.random.default_rng(SEED)

OUT = open(B + 'pathB/phaseA_results.txt', 'w')
def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); OUT.write(s + '\n'); OUT.flush()

# ------------------------------------------------------------------ demographics
def load_demo():
    rows = []
    d245 = pd.read_csv(PP + 'participants.tsv', sep='\t').rename(
        columns={'participant_id': 'pid', 'Age': 'age', 'Gender': 'sex'})
    rows.append(d245[['pid', 'age', 'sex']])
    d19 = pd.read_csv(PP + 'ds001907_UdallANT/age_clean.csv').rename(
        columns={'participant_id': 'pid', 'gender': 'sex'})
    rows.append(d19[['pid', 'age', 'sex']])
    d58 = pd.read_csv(PP + 'ds005892/participants.tsv', sep='\t').rename(
        columns={'participant_id': 'pid'})
    rows.append(d58[['pid', 'age', 'sex']])
    dm = pd.concat(rows).drop_duplicates('pid')
    dm['age'] = pd.to_numeric(dm['age'], errors='coerce')
    dm['male'] = dm['sex'].astype(str).str.upper().str[0].map({'M': 1, 'F': 0})
    return dm[['pid', 'age', 'male']]

# ------------------------------------------------------------------ feature frame
rad = pd.read_csv(B + 'data/radiomics_v3_harmonized.csv')
if 'error' in rad.columns:
    rad = rad[rad['error'].isna()]
rad['pid'] = rad['file_path'].apply(lambda f: re.search(r'(sub-[A-Za-z0-9]+)', str(f)).group(1))
rad = rad[rad.dataset.isin(MIXED)].reset_index(drop=True)
rad = rad.merge(load_demo(), on='pid', how='left')
for roi in ['caudate', 'putamen', 'pallidum', 'thalamus']:
    rad[f'volc_{roi}'] = rad[f'vol_mm3_{roi}'] / rad['brain_vol_native_mm3']

ero = pd.read_csv(B + 'data/erosion_features.csv')
ero['pid'] = ero['file_path'].apply(lambda f: re.search(r'(sub-[A-Za-z0-9]+)', str(f)).group(1))
ero = ero[ero.dataset.isin(MIXED)][[c for c in ero.columns
        if c not in ('file_path', 'label', 'dataset', 'group')]]
rad = rad.merge(ero, on='pid', how='left')

p('=' * 92)
p('PHASE A  -- discovery cohorts', MIXED, ' seed', SEED,
  ' age tol +-', AGE_TOL, 'y  sex-exact  perms', NPERM)
p('=' * 92)
for ds in MIXED:
    g = rad[rad.dataset == ds]
    p(f'  {ds}: nPD={(g.label==1).sum()} nHC={(g.label==0).sum()}  '
      f'age NA={g.age.isna().sum()}  sex NA={g.male.isna().sum()}')

# ------------------------------------------------------------------ matching
def matched_index(g, seed):
    r = np.random.default_rng(seed)
    pdg = g[g.label == 1].dropna(subset=['age', 'male'])
    hcg = g[g.label == 0].dropna(subset=['age', 'male'])
    idx = list(pdg.index); r.shuffle(idx)
    used = set(); pairs = []
    for i in idx:
        row = pdg.loc[i]
        c = hcg[~hcg.index.isin(used) & (hcg.male == row.male)]
        c = c[(c.age - row.age).abs() <= AGE_TOL]
        if len(c):
            j = (c.age - row.age).abs().idxmin()
            used.add(j); pairs.append((int(i), int(j)))
    return pairs

def best_matched(g):
    cand = [matched_index(g, s) for s in range(200)]
    med = int(np.median([len(x) for x in cand]))
    best, bestgap = None, 1e9
    for pr in cand:
        if len(pr) != med:
            continue
        gap = np.mean([abs(g.loc[i, 'age'] - g.loc[j, 'age']) for i, j in pr])
        if gap < bestgap:
            bestgap, best = gap, pr
    return best

MATCH = {ds: best_matched(rad[rad.dataset == ds]) for ds in MIXED}
for ds in MIXED:
    pr = MATCH[ds]
    gap = np.mean([abs(rad.loc[i, 'age'] - rad.loc[j, 'age']) for i, j in pr])
    p(f'  matched {ds}: {len(pr)} pairs  mean|age gap|={gap:.2f}')
NPAIR = sum(len(v) for v in MATCH.values())
p(f'  POOLED matched pairs: {NPAIR}')

# ------------------------------------------------------------------ core helpers
def mrows(dss):
    recs = []
    for ds in dss:
        for i, j in MATCH[ds]:
            recs.append((ds, 1, i)); recs.append((ds, 0, j))
    return pd.DataFrame(recs, columns=['dataset', 'y', 'ridx'])

def col_of(fr, col):
    return fr.ridx.map(rad[col]).values

def pooled_d_arr(ds_arr, y_arr, x_arr):
    out = []
    for ds in np.unique(ds_arr):
        m = ds_arr == ds
        a = x_arr[m & (y_arr == 1)]; b = x_arr[m & (y_arr == 0)]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            return np.nan
        sd = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
        if sd == 0:
            return np.nan
        out.append((a.mean() - b.mean()) / sd)
    return float(np.mean(out))

def pooled_d(col):
    fr = mrows(MIXED)
    return pooled_d_arr(fr.dataset.values, fr.y.values, col_of(fr, col))

def perm_p(col, obs):
    fr = mrows(MIXED)
    ds_a, y_a, x_a = fr.dataset.values, fr.y.values.copy(), col_of(fr, col)
    direction = -1 if (obs == obs and obs < 0) else 1
    null = np.empty(NPERM)
    for k in range(NPERM):
        yp = y_a.copy()
        for ds in MIXED:
            m = ds_a == ds
            yp[m] = rng.permutation(yp[m])
        null[k] = pooled_d_arr(ds_a, yp, x_a)
    null = null[np.isfinite(null)]
    return ((null <= obs).mean() if direction < 0 else (null >= obs).mean())

def loco_auc(col):
    y_all, s_all = [], []
    for held in MIXED:
        tr = mrows([c for c in MIXED if c != held]); te = mrows([held])
        xtr = col_of(tr, col); xte = col_of(te, col)
        if not np.isfinite(xtr).all() or not np.isfinite(xte).all():
            return np.nan
        d_tr = pooled_d_arr(tr.dataset.values, tr.y.values, xtr)
        sgn = np.sign(d_tr) if (d_tr == d_tr and d_tr != 0) else 1.0
        mu, sd = xtr.mean(), xtr.std()
        if sd == 0 or not np.isfinite(sd):
            return np.nan
        s_all += list(-sgn * (xte - mu) / sd); y_all += list(te.y.values)
    try:
        return roc_auc_score(y_all, s_all)
    except Exception:
        return np.nan

def loco_panel(cols, penalty='l2', C=1.0, solver='lbfgs'):
    y_all, s_all = [], []
    for held in MIXED:
        tr = mrows([c for c in MIXED if c != held]); te = mrows([held])
        Xtr = np.column_stack([col_of(tr, c) for c in cols])
        Xte = np.column_stack([col_of(te, c) for c in cols])
        if not np.isfinite(Xtr).all() or not np.isfinite(Xte).all():
            return np.nan
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=3000).fit(
            sc.transform(Xtr), tr.y.values)
        s_all += list(lr.predict_proba(sc.transform(Xte))[:, 1]); y_all += list(te.y.values)
    try:
        return roc_auc_score(y_all, s_all)
    except Exception:
        return np.nan

def paired_diffs(col):
    d = []
    for ds in MIXED:
        for i, j in MATCH[ds]:
            d.append(rad.loc[i, col] - rad.loc[j, col])
    return np.array(d, float)

# ================================================================= CANDIDATES
p('\n' + '=' * 92)
p('CANDIDATE 1 -- ICV-corrected subcortical volumes')
p('=' * 92)
p(f'{"marker":<24}{"pooled d":>10}{"LOCO AUC":>10}{"perm p":>10}')
vol_cols = [f'volc_{r}' for r in ['caudate', 'putamen', 'pallidum', 'thalamus']]
vol_loco = {}
for col in vol_cols:
    d = pooled_d(col); a = loco_auc(col); vol_loco[col] = a
    p(f'{col:<24}{d:>+10.3f}{a:>10.3f}{perm_p(col, d):>10.4f}')
vol_panel = loco_panel(vol_cols)
p(f'{"PANEL 4 vols (L2-logit)":<24}{"":>10}{vol_panel:>10.3f}')

p('\n' + '=' * 92)
p('CANDIDATE 2 -- single texture / first-order features: screen all clean tplB')
p('  232 candidate tests; report Bonferroni (p<2.2e-4) and FDR-BH')
p('=' * 92)
tcols = [c for c in rad.columns if c.startswith('tplB_') and '__original_' in c
         and not rad[c].isna().any()]
scr = []
for col in tcols:
    d = paired_diffs(col)
    if not np.isfinite(d).all() or len(d) < 10:
        continue
    tp = stats.ttest_1samp(d, 0).pvalue
    scr.append((col, pooled_d(col), tp))
scr.sort(key=lambda x: x[2])
fdr = multipletests([s[2] for s in scr], method='fdr_bh')[1] if scr else np.array([])
p(f'{"feature":<50}{"d":>8}{"approx p":>11}{"FDR":>8}{"LOCO AUC":>10}{"  erosion keep@e1":>18}')
top_tex = []
for k in range(min(15, len(scr))):
    col, d, tp = scr[k]
    a = loco_auc(col)
    ek = ''
    m = re.match(r'tplB_(caudate|putamen)__original_firstorder_(\w+)$', col)
    if m and f'{m.group(1)}_e0__{m.group(2)}' in rad.columns:
        d0 = pooled_d(f'{m.group(1)}_e0__{m.group(2)}')
        d1 = pooled_d(f'{m.group(1)}_e1__{m.group(2)}')
        if np.isfinite(d0) and d0 != 0 and np.isfinite(d1):
            ek = f'{d1/d0:.0%} (d0{d0:+.2f} d1{d1:+.2f})'
    top_tex.append((col, d, tp, float(fdr[k]), a, ek))
    p(f'{col:<50}{d:>+8.3f}{tp:>11.2e}{fdr[k]:>8.3f}{a:>10.3f}{ek:>18}')
min_fdr = float(np.min(fdr)) if len(fdr) else float('nan')
n_sig = int((fdr < 0.05).sum()) if len(fdr) else 0
p(f'\n  min FDR-BH over {len(scr)} features: {min_fdr:.3f}   # with FDR<0.05: {n_sig}')

p('\n' + '=' * 92)
p('CANDIDATE 3 -- caudate / putamen Minimum erosion series  (gate: keep >=60% |d| at e1)')
p('=' * 92)
for roi in ['caudate', 'putamen']:
    ds_ = [pooled_d(f'{roi}_{e}__Minimum') if f'{roi}_{e}__Minimum' in rad.columns else np.nan
           for e in ['e0', 'e1', 'e2']]
    a0 = loco_auc(f'{roi}_e0__Minimum') if f'{roi}_e0__Minimum' in rad.columns else np.nan
    a1 = loco_auc(f'{roi}_e1__Minimum') if f'{roi}_e1__Minimum' in rad.columns else np.nan
    keep = (ds_[1] / ds_[0]) if (np.isfinite(ds_[0]) and ds_[0] != 0) else np.nan
    p(f'  {roi:<9} d e0={ds_[0]:+.3f} e1={ds_[1]:+.3f} e2={ds_[2]:+.3f}  '
      f'LOCO AUC e0={a0:.3f} e1={a1:.3f}   keep(e1/e0)={keep:.0%}')

p('\n' + '=' * 92)
p('CANDIDATE 4 -- multivariate panels, LOCO (n~38 -- expect overfit; report honestly)')
p('=' * 92)
p(f'  L2-logit  all 4 volumes                 LOCO AUC = {vol_panel:.3f}')
p(f'  L2-logit  volumes + caudate/putamen Min LOCO AUC = '
  f'{loco_panel(vol_cols + [c for c in ["caudate_e0__Minimum","putamen_e0__Minimum"] if c in rad.columns]):.3f}')
good_tex = [c for c in tcols]
p(f'  L1-logit  all {len(good_tex)} texture feats (C=.05) LOCO AUC = '
  f'{loco_panel(good_tex, penalty="l1", C=0.05, solver="liblinear"):.3f}')

# ================================================================= GATE
best_single_vol = max([v for v in vol_loco.values() if v == v] + [0])
best_tex_loco = max([t[4] for t in top_tex if t[4] == t[4]] + [0])
caud_e0 = loco_auc('caudate_e0__Minimum') if 'caudate_e0__Minimum' in rad.columns else None
gate = {
    'pooled_matched_pairs': NPAIR,
    'vol_single_loco_auc': vol_loco,
    'vol_panel_loco_auc': vol_panel,
    'best_texture_min_fdr': min_fdr,
    'n_texture_fdr_lt_05': n_sig,
    'best_texture_loco_auc': best_tex_loco,
    'caudate_min_e0_loco_auc': caud_e0,
    'l1_texture_panel_loco_auc': loco_panel(good_tex, penalty='l1', C=0.05, solver='liblinear'),
}
best_overall = max([x for x in [best_single_vol, vol_panel, best_tex_loco,
                                caud_e0 or 0, gate['l1_texture_panel_loco_auc']] if x == x])
gate['best_overall_loco_auc'] = best_overall
gate['gate_passed'] = bool(best_overall >= 0.60)
json.dump(gate, open(B + 'pathB/phaseA_gate.json', 'w'), indent=2)
p('\n' + '=' * 92)
p('PHASE A DECISION GATE')
p('=' * 92)
p(json.dumps(gate, indent=2))
p(f'\n  BEST LOCO AUC among all candidates: {best_overall:.3f}')
p(f'  -> {"GATE PASSED: proceed to Phase B" if gate["gate_passed"] else "GATE FAILED: go Path A (registered null)"}')
OUT.close()
