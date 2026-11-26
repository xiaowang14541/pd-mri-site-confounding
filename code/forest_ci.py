"""Per-cohort matched effect with 95% CI, for the forest plot.
Averaged over 200 matching seeds; the CI is taken from the paired differences pooled
across seeds via the mean within-seed standard error, which keeps the pair structure.
"""
import numpy as np, pandas as pd, re, warnings
warnings.filterwarnings('ignore')
from scipy import stats

B='/master/wang_zi_wei/cursor/pd_classification/'
COL='tplB_caudate__original_firstorder_Minimum'
MIXED=['ds000245','ds001907','ds005892']

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
add('/master/wang_zi_wei/cursor/parkinson_mri_data/ds005892/participants.tsv',sep='\t')
dm=pd.concat(rows).drop_duplicates('pid')
dm['male']=dm['sexraw'].astype(str).str.upper().str[0].map({'M':1,'F':0})
d=df.merge(dm[['pid','age','male']],on='pid',how='left')

print(f'{"cohort":<11}{"pairs":>6}{"d":>9}{"lo95":>9}{"hi95":>9}')
ds_list=[]; pair_store=[]
for ds in MIXED:
    g=d[d.dataset==ds].dropna(subset=[COL,'age','male'])
    pdg=g[g.label==1]; hcg=g[g.label==0]
    dvals=[]; npairs=[]
    for s in range(200):
        r=np.random.default_rng(s); idx=list(pdg.index); r.shuffle(idx)
        used=set(); A=[];Bv=[]
        for i in idx:
            row=pdg.loc[i]
            c=hcg[(~hcg.index.isin(used))&(hcg.male==row.male)&((hcg.age-row.age).abs()<=4)]
            if len(c):
                j=(c.age-row.age).abs().idxmin(); used.add(j)
                A.append(row[COL]); Bv.append(hcg.loc[j,COL])
        if len(A)<6: continue
        A=np.array(A);Bv=np.array(Bv)
        sd=np.sqrt(((len(A)-1)*A.var(ddof=1)+(len(Bv)-1)*Bv.var(ddof=1))/(len(A)+len(Bv)-2))
        if sd>0:
            dvals.append((A.mean()-Bv.mean())/sd); npairs.append(len(A))
            lastA,lastB=A,Bv
    dm_=np.mean(dvals); n=int(np.mean(npairs))
    # bootstrap over matched PAIRS (resample pairs, recompute d) -- makes no assumption
    # about the paired variance structure, unlike the independent-sample SE formula
    A0,B0=np.array(lastA),np.array(lastB)
    boot=[]
    rb=np.random.default_rng(0)
    for _ in range(5000):
        k=rb.integers(0,len(A0),len(A0))
        a,b=A0[k],B0[k]
        sd=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        if sd>0: boot.append((a.mean()-b.mean())/sd)
    lo,hi=np.percentile(boot,[2.5,97.5])
    print(f'{ds:<11}{n:>6}{dm_:>+9.3f}{lo:>+9.3f}{hi:>+9.3f}')
    ds_list.append((ds,n,dm_,lo,hi)); pair_store.append((lastA,lastB))

# pooled: inverse-variance weighted
# pooled: bootstrap the mean of the three cohort effects, resampling pairs within each
allpairs=[(np.array(a),np.array(b)) for a,b in pair_store]
rb=np.random.default_rng(1); bp=[]
for _ in range(5000):
    es=[]
    for a,b in allpairs:
        k=rb.integers(0,len(a),len(a)); aa,bb=a[k],b[k]
        sd=np.sqrt(((len(aa)-1)*aa.var(ddof=1)+(len(bb)-1)*bb.var(ddof=1))/(len(aa)+len(bb)-2))
        if sd>0: es.append((aa.mean()-bb.mean())/sd)
    if len(es)==3: bp.append(np.mean(es))
lo_p,hi_p=np.percentile(bp,[2.5,97.5])
print(f'{"POOLED":<11}{sum(n for _,n,_,_,_ in ds_list):>6}'
      f'{np.mean([dd for _,_,dd,_,_ in ds_list]):>+9.3f}{lo_p:>+9.3f}{hi_p:>+9.3f}')
