"""
Extract the harmonized radiomics + volumes for the Phase-B confirmatory cohorts
(Tao Wu, NEUROCON) using the EXACT same pipeline as extract_v3_harmonized.py.

Reuses process_one / get_ref unchanged so features are comparable to the discovery set.
Output: data/radiomics_v3_taowu_neurocon.csv
"""
import os, sys, glob
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, '/master/wang_zi_wei/cursor/pd_classification/src')
sys.path.insert(0, '/master/wang_zi_wei/cursor/pd_classification')
from extract_v3_harmonized import process_one, get_ref

ROOTS = {
    'taowu':   '/master/wang_zi_wei/cursor/pathB_data/taowu/taowu',
    'neurocon': '/master/wang_zi_wei/cursor/pathB_data/neurocon/neurocon',
}
OUT_CSV = '/master/wang_zi_wei/cursor/pd_classification/data/radiomics_v3_taowu_neurocon.csv'
N_WORKERS = 20


def tasks():
    out = []
    for ds, root in ROOTS.items():
        for sd in sorted(glob.glob(os.path.join(root, 'sub-*'))):
            if not os.path.isdir(sd):
                continue
            pid = os.path.basename(sd)
            t1 = os.path.join(sd, 'anat', f'{pid}_T1w.nii.gz')
            if not os.path.exists(t1):
                cand = glob.glob(os.path.join(sd, 'anat', '*T1w.nii.gz'))
                if not cand:
                    print('  no T1 for', pid, flush=True); continue
                t1 = cand[0]
            label = 0 if 'control' in pid.lower() else 1
            out.append((t1, label, 'control' if label == 0 else 'patient', ds))
    return out


def main():
    ts = tasks()
    print(f'Total {len(ts)} subjects', flush=True)
    for ds in ROOTS:
        n = sum(1 for t in ts if t[3] == ds)
        npd = sum(1 for t in ts if t[3] == ds and t[1] == 1)
        print(f'  {ds}: {n}  (PD {npd} / HC {n-npd})', flush=True)
    get_ref()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(process_one, t): t for t in ts}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); rows.append(r)
            if 'error' in r:
                print(f'[{i}] ERR {r["file_path"]}: {r["error"]}', flush=True)
            elif i % 10 == 0 or i == len(ts):
                print(f'[{i}/{len(ts)}] dice={r.get("reg_dice",0):.3f} '
                      f'caud={r.get("vol_mm3_caudate",0):.0f} '
                      f'brain={r.get("brain_vol_native_mm3",0)/1000:.0f}mL', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    nerr = df['error'].notna().sum() if 'error' in df.columns else 0
    print(f'\nDone {len(df)-nerr}/{len(df)} ok -> {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
