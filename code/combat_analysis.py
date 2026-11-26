#!/usr/bin/env python3
"""
ComBat harmonization sensitivity analysis.
Q: Does PD classification AUC survive after scanner batch effects are removed?
If AUC drops to ~0.5 after ComBat → confound fully explains performance.
If AUC stays high → genuine pathological signal exists.
"""
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from neuroCombat import neuroCombat
import json, warnings
warnings.filterwarnings('ignore')

DATA_CSV   = '/master/wang_zi_wei/cursor/pd_classification/data/splits/data_info.csv'
SPLITS_JSON = '/master/wang_zi_wei/cursor/pd_classification/data/splits/cv_splits.json'
TARGET_SHAPE = (64, 77, 64)
N_PCA = 100
RESULTS_DIR = '/master/wang_zi_wei/cursor/pd_classification/results/combat_analysis'

import os; os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data_info.csv...")
df = pd.read_csv(DATA_CSV).reset_index(drop=True)
print(f"Total: {len(df)} subjects | datasets: {df['dataset'].value_counts().to_dict()}")

def load_flatten(fpath):
    img  = nib.load(fpath)
    data = img.get_fdata(dtype=np.float32)
    m, s = data.mean(), data.std()
    if s > 1e-6:
        data = (data - m) / s
    factors = [t / o for t, o in zip(TARGET_SHAPE, data.shape)]
    return zoom(data, factors, order=1).flatten()

print("Loading MRI features (this may take a few minutes)...")
features, labels, datasets = [], [], []
failed = []
for _, row in df.iterrows():
    try:
        feat = load_flatten(row['file_path'])
        features.append(feat)
        labels.append(int(row['label']))
        datasets.append(row['dataset'])
    except Exception as e:
        failed.append(row['subject_id'])
        print(f"  SKIP {row['subject_id']}: {e}")

X = np.array(features, dtype=np.float32)   # (N, voxels)
y = np.array(labels)
print(f"Loaded: {X.shape}, failed: {len(failed)}")

# Dataset → integer batch code
ds_map  = {d: i for i, d in enumerate(sorted(set(datasets)))}
batches = np.array([ds_map[d] for d in datasets])
print(f"Dataset mapping: {ds_map}")

# ── ComBat harmonization ─────────────────────────────────────────────────────
print("\nApplying ComBat harmonization...")
# neuroCombat expects dat = (features × samples)
covars_df = pd.DataFrame({'batch': batches, 'label': y})
combat_out = neuroCombat(
    dat       = X.T,           # (features, N)
    covars    = covars_df,
    batch_col = 'batch',
    continuous_cols = [],
    categorical_cols = ['label']
)
X_harmonized = combat_out['data'].T   # back to (N, features)
print(f"ComBat done. Harmonized shape: {X_harmonized.shape}")

# ── 5-fold CV: before vs after ComBat ───────────────────────────────────────
with open(SPLITS_JSON) as f:
    splits = json.load(f)

def run_cv(X_feat, y_labels, splits, tag):
    results = []
    for fold_data in splits:
        fold    = fold_data['fold']
        tr_idx  = fold_data['train_idx']
        val_idx = fold_data['val_idx']
        X_tr, X_val = X_feat[tr_idx], X_feat[val_idx]
        y_tr, y_val = y_labels[tr_idx], y_labels[val_idx]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=N_PCA)),
            ('svm',    SVC(kernel='rbf', C=1.0, probability=True))
        ])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, proba)
        acc   = accuracy_score(y_val, (proba >= 0.5).astype(int))
        results.append({'fold': fold, 'auc': auc, 'accuracy': acc})
        print(f"  [{tag}] Fold {fold}: AUC={auc:.4f}, Acc={acc:.4f}")
    return results

print("\n--- WITHOUT ComBat (original features) ---")
res_orig = run_cv(X, y, splits, "Original")

print("\n--- WITH ComBat harmonization ---")
res_combat = run_cv(X_harmonized, y, splits, "ComBat  ")

# ── Summary ──────────────────────────────────────────────────────────────────
def summarize(results, tag):
    aucs = [r['auc'] for r in results]
    accs = [r['accuracy'] for r in results]
    return {
        'method'  : tag,
        'mean_auc': np.mean(aucs),
        'std_auc' : np.std(aucs, ddof=1),
        'mean_acc': np.mean(accs),
        'std_acc' : np.std(accs, ddof=1),
    }

s_orig   = summarize(res_orig,   'PCA+SVM (no harmonization)')
s_combat = summarize(res_combat, 'PCA+SVM (ComBat harmonized)')

print("\n=== SUMMARY ===")
for s in [s_orig, s_combat]:
    print(f"{s['method']:35s}  AUC={s['mean_auc']:.4f}±{s['std_auc']:.4f}  "
          f"Acc={s['mean_acc']:.4f}±{s['std_acc']:.4f}")

delta = s_combat['mean_auc'] - s_orig['mean_auc']
print(f"\nAUC change after ComBat: {delta:+.4f}")
if s_combat['mean_auc'] < 0.65:
    verdict = "CONFOUND FULLY EXPLAINS PERFORMANCE — AUC near chance after harmonization"
elif s_combat['mean_auc'] < 0.80:
    verdict = "PARTIAL CONFOUND — performance substantially reduced but some signal remains"
else:
    verdict = "GENUINE SIGNAL — high AUC survives harmonization"
print(f"Verdict: {verdict}")

# Save
fold_df = pd.DataFrame(
    [{'method': 'original', **r} for r in res_orig] +
    [{'method': 'combat',   **r} for r in res_combat]
)
fold_df.to_csv(f'{RESULTS_DIR}/fold_results.csv', index=False)

summary_df = pd.DataFrame([s_orig, s_combat])
summary_df['verdict'] = verdict
summary_df.to_csv(f'{RESULTS_DIR}/summary.csv', index=False)
print(f"\nSaved to {RESULTS_DIR}/")
