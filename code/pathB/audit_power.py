"""
Path B - Task 1: same-scanner sample audit + power analysis.

Same-scanner cohorts (PD and HC scanned on one scanner, so site is controlled
by construction): ds000245, ds001907, ds005892.

Outputs:
  1. per-cohort n PD / n HC, age & sex distributions, PD-HC age/sex imbalance
  2. age(+sex) matched pair counts at tolerance 2/3/5 yr
  3. detectable effect (Cohen d, and AUC) at 80% / 90% power for alpha .05 and .005
     - per cohort and pooled
  4. required n/group to detect d = 0.2 / 0.35 / 0.5
"""
import numpy as np, pandas as pd, re, warnings, os
warnings.filterwarnings('ignore')
from scipy import stats

B = '/master/wang_zi_wei/cursor/pd_classification/'
PP = '/master/wang_zi_wei/cursor/parkinson_mri_data/'
MIXED = ['ds000245', 'ds001907', 'ds005892']

# ---- build demographics (pid, age, male, label, dataset) --------------------
df = pd.read_csv(B + 'data/radiomics_v3_harmonized.csv')
if 'error' in df.columns:
    df = df[df['error'].isna()]
df['pid'] = df['file_path'].apply(lambda f: re.search(r'(sub-[A-Za-z0-9]+)', str(f)).group(1))

rows = []
def add(p, sep=None, agec='age', sexc='sex'):
    try:
        t = pd.read_csv(p, sep=sep) if sep else pd.read_csv(p)
    except Exception as e:
        print('  (skip', p, e, ')'); return
    c = {x.lower(): x for x in t.columns}
    pc = c.get('participant_id', list(t.columns)[0])
    ac = c.get(agec.lower())
    sc = c.get(sexc.lower()) or c.get('gender')
    o = pd.DataFrame({'pid': t[pc].astype(str)})
    o['age'] = pd.to_numeric(t[ac], errors='coerce') if ac else np.nan
    o['sexraw'] = t[sc].astype(str) if sc else None
    rows.append(o)

add(PP + 'participants.tsv', sep='\t', agec='Age', sexc='Gender')
add(PP + 'ds001907_UdallANT/age_clean.csv')
add(PP + 'ds005892/participants.tsv', sep='\t')
dm = pd.concat(rows).drop_duplicates('pid')
dm['male'] = dm['sexraw'].astype(str).str.upper().str[0].map({'M': 1, 'F': 0})
d = df.merge(dm[['pid', 'age', 'male']], on='pid', how='left')

# fall back to demographics.csv for any missing age
demo = pd.read_csv(B + 'data/demographics.csv')
demo = demo.rename(columns={'participant_id': 'pid'})
d = d.merge(demo[['pid', 'age']].rename(columns={'age': 'age_demo'}), on='pid', how='left')
d['age'] = d['age'].fillna(d['age_demo'])

def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / sp

def d_to_auc(dv):
    return stats.norm.cdf(dv / np.sqrt(2))

def detectable_d(n1, n2, alpha, power=0.80):
    """smallest |d| a two-sample t-test can detect at given power (approx, normal)."""
    from scipy.optimize import brentq
    za = stats.norm.ppf(1 - alpha/2)
    zb = stats.norm.ppf(power)
    ne = 1.0 / (1.0/n1 + 1.0/n2)          # effective per-group n
    return (za + zb) / np.sqrt(ne)

def required_n(dv, alpha, power=0.80):
    za = stats.norm.ppf(1 - alpha/2)
    zb = stats.norm.ppf(power)
    return ((za + zb) / dv)**2            # per group, balanced

# ---- 1. per-cohort audit ---------------------------------------------------
print('='*92)
print('SAME-SCANNER COHORT AUDIT  (age in years; sex = %male)')
print('='*92)
print(f'{"cohort":<12}{"nPD":>5}{"nHC":>5}{"agePD":>8}{"ageHC":>8}{"dAge":>7}'
      f'{"pAge":>8}{"%mPD":>7}{"%mHC":>7}{"pSex":>8}{"age_NA":>8}')
audit = {}
for ds in MIXED:
    g = d[d.dataset == ds]
    pdg, hcg = g[g.label == 1], g[g.label == 0]
    aP, aH = pdg['age'].dropna(), hcg['age'].dropna()
    dage = cohens_d(aP.values, aH.values)
    pv = stats.ttest_ind(aP, aH, equal_var=False).pvalue
    mP, mH = pdg['male'].dropna(), hcg['male'].dropna()
    try:
        tab = [[ (pdg.male==1).sum(), (pdg.male==0).sum() ],
               [ (hcg.male==1).sum(), (hcg.male==0).sum() ]]
        psex = stats.fisher_exact(tab)[1]
    except Exception:
        psex = np.nan
    na = g['age'].isna().sum()
    print(f'{ds:<12}{len(pdg):>5}{len(hcg):>5}{aP.mean():>8.1f}{aH.mean():>8.1f}'
          f'{dage:>+7.2f}{pv:>8.3f}{100*mP.mean():>7.0f}{100*mH.mean():>7.0f}{psex:>8.3f}{na:>8}')
    audit[ds] = dict(nPD=len(pdg), nHC=len(hcg))

# ---- 2. matched pair counts ---------------------------------------------------
def match_pairs(pdg, hcg, tol, use_sex, seed=0):
    r = np.random.default_rng(seed)
    pidx = list(pdg.index); r.shuffle(pidx)
    used = set(); k = 0
    for i in pidx:
        row = pdg.loc[i]
        c = hcg[~hcg.index.isin(used)].dropna(subset=['age'])
        if use_sex and not np.isnan(row.get('male', np.nan)):
            c = c[c.male == row.male]
        c = c[(c.age - row.age).abs() <= tol]
        if len(c):
            j = (c.age - row.age).abs().idxmin()
            used.add(j); k += 1
    return k

print()
print('='*92)
print('AGE(+SEX)-MATCHED PAIR COUNTS  (median over 200 random match orders)')
print('='*92)
print(f'{"cohort":<12}{"tol=2y":>10}{"tol=3y":>10}{"tol=5y":>10}'
      f'{"+sex 2y":>10}{"+sex 3y":>10}{"+sex 5y":>10}')
pooled = {2: 0, 3: 0, 5: 0, 's2': 0, 's3': 0, 's5': 0}
for ds in MIXED:
    g = d[d.dataset == ds].dropna(subset=['age'])
    pdg, hcg = g[g.label == 1], g[g.label == 0]
    res = {}
    for tol in (2, 3, 5):
        res[tol] = int(np.median([match_pairs(pdg, hcg, tol, False, s) for s in range(200)]))
        res['s%d' % tol] = int(np.median([match_pairs(pdg, hcg, tol, True, s) for s in range(200)]))
    for k in pooled:
        pooled[k] += res[k]
    print(f'{ds:<12}{res[2]:>10}{res[3]:>10}{res[5]:>10}{res["s2"]:>10}{res["s3"]:>10}{res["s5"]:>10}')
print(f'{"POOLED":<12}{pooled[2]:>10}{pooled[3]:>10}{pooled[5]:>10}'
      f'{pooled["s2"]:>10}{pooled["s3"]:>10}{pooled["s5"]:>10}')

# ---- 3. detectable effect --------------------------------------------------
print()
print('='*92)
print('DETECTABLE EFFECT  (two-group test; d and implied single-feature AUC)')
print('='*92)
print(f'{"cohort":<26}{"n1/n2":>12}{"  a=.05 80%":>14}{"a=.05 90%":>13}'
      f'{"a=.005 80%":>13}{"a=.005 90%":>13}')
def line(name, n1, n2):
    vals = []
    for al, po in [(.05,.8),(.05,.9),(.005,.8),(.005,.9)]:
        dd = detectable_d(n1, n2, al, po)
        vals.append(f'{dd:.2f}/{d_to_auc(dd):.2f}')
    print(f'{name:<26}{f"{n1}/{n2}":>12}' + ''.join(f'{v:>13}' for v in vals))
for ds in MIXED:
    line(ds + ' (unmatched)', audit[ds]['nPD'], audit[ds]['nHC'])
line('POOLED unmatched', sum(a['nPD'] for a in audit.values()),
     sum(a['nHC'] for a in audit.values()))
line('POOLED matched +sex 3y', pooled['s3'], pooled['s3'])
line('POOLED matched age 3y', pooled[3], pooled[3])

# ---- 4. required n --------------------------------------------------------
print()
print('='*92)
print('REQUIRED n PER GROUP  (balanced two-group test)')
print('='*92)
print(f'{"target effect":<28}{"a=.05 80%":>12}{"a=.05 90%":>12}{"a=.005 80%":>13}{"a=.005 90%":>13}')
for dv, lbl in [(0.20, 'd=0.20  (AUC 0.56)'),
                (0.35, 'd=0.35  (AUC 0.60)'),
                (0.50, 'd=0.50  (AUC 0.64)'),
                (0.80, 'd=0.80  (AUC 0.71)')]:
    r = [required_n(dv, al, po) for al, po in [(.05,.8),(.05,.9),(.005,.8),(.005,.9)]]
    print(f'{lbl:<28}' + ''.join(f'{int(np.ceil(x)):>12}' for x in r[:2])
          + ''.join(f'{int(np.ceil(x)):>13}' for x in r[2:]))

print()
print('NOTE: "AUC" here is the single-feature separation implied by a normal-'
      'model Cohen d.  A multivariate classifier can do better, but LODO / '
      'replication AUC on truly unconfounded data is the number that matters.')
