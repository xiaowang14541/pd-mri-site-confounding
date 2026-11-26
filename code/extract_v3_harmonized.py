"""Definitive extraction: removes the two remaining cross-dataset inequalities.

PROBLEM 1 -- heterogeneous upstream preprocessing. The public datasets arrive in
different states: fraction of exactly-zero voxels is 0.847 (ds001907, already heavily
skull-stripped by the providers), 0.505/0.501/0.428 (ds004392/ds003592/ds005892), and
0.247/0.240 (ds000245/ds004471, raw with skull). The pipeline applied the same crude
5th-percentile threshold to all of them, so "brain voxels" -- and therefore the z-score
used to normalise intensity -- meant something different in each dataset. This is why
ds001907's firstorder_Minimum sits at -2.99 while every other dataset is -0.40..-0.81,
and why it was the only cohort ever to reach significance.
FIX: derive the brain mask the SAME way for everyone, by warping the MNI152 brain mask
through each subject's own registration. Intensity is then z-scored inside that common
mask, so firstorder_Minimum is on one scale across datasets for the first time.

PROBLEM 2 -- ds004471 is 1.02x1.02x1.4mm anisotropic with 124 slices, while every other
dataset is 1.0mm isotropic. Thick slices blur through-plane boundaries; its warped
caudate mask came out 39144mm3 vs ~11000 elsewhere (CV 0.43).
FIX: measure in the common 2mm MNI grid as the primary output, so all subjects are on
an identical grid. Native-space values are also emitted for comparison.

Two feature sets per subject:
  tpl__*  image warped into MNI, AAL masks applied directly  (primary; comparable)
  nat__*  AAL masks warped back to native grid               (secondary; native texture)
"""
import os, sys, re
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, '/master/wang_zi_wei/cursor/pd_classification/src')
sys.path.insert(0, '/master/wang_zi_wei/cursor/pd_classification')

OUT_CSV = "/master/wang_zi_wei/cursor/pd_classification/data/radiomics_v3_harmonized.csv"
N_WORKERS = 22
RAW = "/master/wang_zi_wei/cursor/parkinson_mri_data"
ROI_GROUPS = {'caudate': ['Caudate_L', 'Caudate_R'], 'putamen': ['Putamen_L', 'Putamen_R'],
              'pallidum': ['Pallidum_L', 'Pallidum_R'], 'thalamus': ['Thal_']}
_C = {}


def get_ref():
    if 'r' in _C:
        return _C['r']
    from nilearn import datasets, image
    tpl_img = datasets.load_mni152_template(resolution=2)
    bm = datasets.load_mni152_brain_mask(resolution=2)
    bm = image.resample_to_img(bm, tpl_img, interpolation='nearest',
                               force_resample=True, copy_header=True)
    aal = datasets.fetch_atlas_aal()
    aal_r = image.resample_to_img(nib.load(aal.maps), tpl_img, interpolation='nearest',
                                  force_resample=True, copy_header=True)
    ad = np.asarray(aal_r.dataobj).astype(int)
    l2i = dict(zip(list(aal.labels), [int(i) for i in aal.indices]))
    masks = {}
    for roi, pres in ROI_GROUPS.items():
        vals = []
        for lab, idx in l2i.items():
            for p in pres:
                if p == 'Thal_':
                    if lab.startswith('Thal_'):
                        vals.append(idx)
                elif lab == p:
                    vals.append(idx)
        masks[roi] = np.isin(ad, vals)
    tpl = np.asarray(tpl_img.dataobj).astype(np.float32)
    tpl = tpl - tpl.min(); tpl = tpl / (tpl.max() + 1e-8)   # must stay non-negative
    _C['r'] = (tpl, tpl_img.affine, np.asarray(bm.dataobj).astype(bool), masks)
    return _C['r']


def build_extractor():
    from radiomics import featureextractor
    ex = featureextractor.RadiomicsFeatureExtractor(
        binCount=64, resampledPixelSpacing=None, interpolator='sitkBSpline',
        normalize=False, label=1)
    ex.disableAllFeatures()
    for c in ['firstorder', 'glcm', 'glrlm']:
        ex.enableFeatureClassByName(c)
    ex.disableAllImageTypes(); ex.enableImageTypeByName('Original')
    return ex


def zscore_in(img, mask):
    v = img[mask]
    v = v[np.isfinite(v)]
    if v.size < 10 or v.std() == 0:
        return None
    return (img - v.mean()) / v.std()


def feats_from(img, mask, spacing, prefix, out, extractor):
    if mask.sum() < 50:
        return
    im = sitk.GetImageFromArray(img.astype(np.float32)); im.SetSpacing(tuple(float(s) for s in spacing))
    mk = sitk.GetImageFromArray(mask.astype(np.uint8)); mk.SetSpacing(tuple(float(s) for s in spacing))
    for k, v in extractor.execute(im, mk).items():
        if not k.startswith('diagnostics_'):
            out[prefix + k] = float(v)


def process_one(args):
    fp, label, group, ds = args
    try:
        from dipy.align.imaffine import (MutualInformationMetric, AffineRegistration,
                                          transform_centers_of_mass)
        from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D,
                                            AffineTransform3D)
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
        from dipy.align.metrics import CCMetric

        tpl, tpl_a, brain_tpl, masks = get_ref()
        im = nib.load(fp)
        raw = np.asarray(im.dataobj).astype(np.float32)
        if raw.ndim == 4:
            raw = raw[..., 0]
        zooms = im.header.get_zooms()[:3]

        # registration driver: min-max only, NO thresholding -- so the upstream
        # stripping state does not change what the registration sees
        drive = raw - raw.min()
        if drive.max() > 0:
            drive = drive / drive.max()

        areg = AffineRegistration(metric=MutualInformationMetric(32, None),
                                   level_iters=[500, 100, 25], sigmas=[3.0, 1.0, 0.0],
                                   factors=[4, 2, 1], verbosity=0)
        com = transform_centers_of_mass(tpl, tpl_a, drive, im.affine)
        t = areg.optimize(tpl, drive, TranslationTransform3D(), None, tpl_a, im.affine,
                          starting_affine=com.affine)
        r = areg.optimize(tpl, drive, RigidTransform3D(), None, tpl_a, im.affine,
                          starting_affine=t.affine)
        a = areg.optimize(tpl, drive, AffineTransform3D(), None, tpl_a, im.affine,
                          starting_affine=r.affine)
        wa = a.transform(drive)
        mapping = SymmetricDiffeomorphicRegistration(CCMetric(3), [25, 15, 8],
                                                     inv_iter=20).optimize(tpl, wa)

        out = {'file_path': fp, 'label': label, 'group': group, 'dataset': ds,
               'zoom_z': float(zooms[2]), 'anisotropic': bool(abs(zooms[2] - zooms[0]) > 0.1),
               'frac_zero_raw': float((raw == 0).mean())}
        m1 = wa > np.percentile(wa[wa > 0], 5) if (wa > 0).any() else wa > 0
        out['reg_dice'] = float(2 * np.sum(m1 & (tpl > np.percentile(tpl, 60)))
                                / (m1.sum() + (tpl > np.percentile(tpl, 60)).sum() + 1e-9))
        ex = build_extractor()

        # ---- PRIMARY: common MNI grid, THREE normalisation variants ----
        # The variants exist to test, symmetrically, whether aggressive brain extraction
        # is what makes a PD effect appear. ds001907 arrives already stripped (85% exact
        # zeros) and is the only cohort that ever reached significance. If stripping is
        # the reason, variant C applied to every dataset should make the effect appear in
        # ds000245 and ds005892 too. If the effect stays ds001907-only under all three,
        # stripping was never the explanation. Justification for preferring a stripped
        # variant is a priori (skull/scalp carry scanner signature, no PD information),
        # NOT that any variant produced a significant result.
        img_tpl = mapping.transform(a.transform(raw))          # subject intensity in MNI
        tpl_sp = tuple(abs(tpl_a[i, i]) for i in range(3))
        nz = img_tpl > 0
        variants = {
            'A': img_tpl > np.percentile(img_tpl[nz], 5) if nz.any() else nz,  # legacy 5th pct
            'B': brain_tpl,                                                     # MNI mask
            'C': brain_tpl & nz,                                                # MNI mask AND non-zero
        }
        for vkey, vmask in variants.items():
            z = zscore_in(img_tpl, vmask)
            if z is None:
                continue
            out[f'normfrac_{vkey}'] = float(vmask.mean())
            for roi, m in masks.items():
                feats_from(z, m & vmask if vkey == 'C' else m, tpl_sp,
                           f'tpl{vkey}_{roi}__', out, ex)
        out['brain_mean_tpl'] = float(img_tpl[brain_tpl].mean())
        out['frac_zero_in_brainmask'] = float((~nz)[brain_tpl].mean())

        # ---- SECONDARY: masks warped back to the native grid ----
        bm_nat = a.transform_inverse(mapping.transform_inverse(
            brain_tpl.astype(np.float32), 'linear'), interpolation='linear') > 0.5
        out['brain_vol_native_mm3'] = float(bm_nat.sum() * np.prod(zooms))
        z_nat = zscore_in(raw, bm_nat)
        if z_nat is not None:
            for roi, m in masks.items():
                mn = a.transform_inverse(mapping.transform_inverse(
                    m.astype(np.float32), 'linear'), interpolation='linear') > 0.5
                out['vol_mm3_' + roi] = float(mn.sum() * np.prod(zooms))
                feats_from(z_nat, mn, zooms, f'nat_{roi}__', out, ex)
        return out
    except Exception as e:
        import traceback
        return {'file_path': fp, 'label': label, 'group': group, 'dataset': ds,
                'error': f'{type(e).__name__}: {e}', 'tb': traceback.format_exc()[-400:]}


def collect_all():
    from extract_radiomics_subcortical import collect_all as base
    tasks = [(f, l, '', d) for f, l, d in base()]
    p = pd.read_csv(os.path.join(RAW, 'ds005892', 'participants.tsv'), sep='\t')
    for _, row in p.iterrows():
        pid = row['participant_id']
        fp = os.path.join(RAW, 'ds005892', pid, 'anat', f'{pid}_T1w.nii.gz')
        if os.path.exists(fp):
            tasks.append((fp, 0 if row['group'] == 'Control' else 1, row['group'], 'ds005892'))
    return tasks


def main():
    tasks = collect_all()
    print(f'Total {len(tasks)}', flush=True)
    get_ref()
    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(process_one, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result(); rows.append(res)
            if 'error' in res:
                print(f'[{i}] ERR {res["error"]}', flush=True)
            elif i % 25 == 0 or i == len(tasks):
                print(f'[{i}/{len(tasks)}] dice={res.get("reg_dice",0):.3f} '
                      f'brain={res.get("brain_vol_native_mm3",0)/1000:.0f}mL '
                      f'caud={res.get("vol_mm3_caudate",0):.0f}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    n = df['error'].notna().sum() if 'error' in df.columns else 0
    print(f'\nDone. {len(df)-n}/{len(df)} ok -> {OUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
