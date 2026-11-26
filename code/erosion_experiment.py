"""P3: erode the caudate ROI away from the ventricular border and see if the effect dies.

H1 says firstorder_Minimum is reading CSF that leaks into the atlas ROI at the ventricular
boundary. Three direct tests, all in template space so every subject is on one grid:

  1. EROSION. Recompute the statistics on the caudate mask eroded by 0 / 1 / 2 voxels
     (0 / 2 / 4 mm). H1 predicts the PD-HC effect shrinks toward zero with erosion, because
     the contaminated rim is removed. A real tissue difference should survive.

  2. WHERE IS THE DARK VOXEL. For each subject, the distance from the ROI boundary to the
     10 darkest voxels (euclidean distance transform, in voxels). H1 predicts they sit on
     the rim (distance ~1). Tissue signal would sit anywhere.

  3. HOW MUCH CSF IS IN THE ROI. Fraction of ROI voxels below CSF-like intensity, and
     whether that fraction differs PD vs HC. H1 predicts PD > HC.

Only the three mixed cohorts are processed (145 subjects) -- they are the only ones where
PD and HC share a scanner, which is what the disease test requires.
"""
import os, sys, re
import numpy as np, pandas as pd, nibabel as nib
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0,'/master/wang_zi_wei/cursor/pd_classification/src')
sys.path.insert(0,'/master/wang_zi_wei/cursor/pd_classification')
OUT='/master/wang_zi_wei/cursor/pd_classification/data/erosion_features.csv'
RAW='/master/wang_zi_wei/cursor/parkinson_mri_data'
N_WORKERS=22
_C={}

def get_ref():
    if 'r' in _C: return _C['r']
    from nilearn import datasets, image
    tpl_img=datasets.load_mni152_template(resolution=2)
    bm=image.resample_to_img(datasets.load_mni152_brain_mask(resolution=2),tpl_img,
                             interpolation='nearest',force_resample=True,copy_header=True)
    aal=datasets.fetch_atlas_aal()
    aal_r=image.resample_to_img(nib.load(aal.maps),tpl_img,interpolation='nearest',
                                force_resample=True,copy_header=True)
    ad=np.asarray(aal_r.dataobj).astype(int)
    l2i=dict(zip(list(aal.labels),[int(i) for i in aal.indices]))
    masks={}
    for roi,names in {'caudate':['Caudate_L','Caudate_R'],
                      'putamen':['Putamen_L','Putamen_R']}.items():
        masks[roi]=np.isin(ad,[l2i[n] for n in names if n in l2i])
    tpl=np.asarray(tpl_img.dataobj).astype(np.float32)
    tpl=tpl-tpl.min(); tpl=tpl/(tpl.max()+1e-8)
    _C['r']=(tpl,tpl_img.affine,np.asarray(bm.dataobj).astype(bool),masks)
    return _C['r']

def stats_in(img, mask, prefix, out):
    v=img[mask]; v=v[np.isfinite(v)]
    if v.size<30: return
    out[prefix+'n']=int(v.size)
    out[prefix+'Minimum']=float(v.min())
    out[prefix+'p10']=float(np.percentile(v,10))
    out[prefix+'Median']=float(np.median(v))
    out[prefix+'Mean']=float(v.mean())
    for thr in (-1.0,-1.5):
        out[prefix+f'csffrac{abs(thr)}']=float((v<thr).mean())

def process_one(args):
    fp,label,ds=args
    try:
        from dipy.align.imaffine import (MutualInformationMetric, AffineRegistration,
                                          transform_centers_of_mass)
        from dipy.align.transforms import (TranslationTransform3D, RigidTransform3D,
                                            AffineTransform3D)
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
        from dipy.align.metrics import CCMetric
        tpl,tpl_a,brain,masks=get_ref()
        im=nib.load(fp); raw=np.asarray(im.dataobj).astype(np.float32)
        if raw.ndim==4: raw=raw[...,0]
        drive=raw-raw.min()
        if drive.max()>0: drive=drive/drive.max()
        areg=AffineRegistration(metric=MutualInformationMetric(32,None),
                                 level_iters=[500,100,25],sigmas=[3.0,1.0,0.0],
                                 factors=[4,2,1],verbosity=0)
        com=transform_centers_of_mass(tpl,tpl_a,drive,im.affine)
        t=areg.optimize(tpl,drive,TranslationTransform3D(),None,tpl_a,im.affine,starting_affine=com.affine)
        r=areg.optimize(tpl,drive,RigidTransform3D(),None,tpl_a,im.affine,starting_affine=t.affine)
        a=areg.optimize(tpl,drive,AffineTransform3D(),None,tpl_a,im.affine,starting_affine=r.affine)
        mapping=SymmetricDiffeomorphicRegistration(CCMetric(3),[25,15,8],inv_iter=20).optimize(tpl,a.transform(drive))
        img=mapping.transform(a.transform(raw))
        v=img[brain]; v=v[np.isfinite(v)]
        if v.size<100 or v.std()==0: raise RuntimeError('bad brain stats')
        z=(img-v.mean())/v.std()

        out={'file_path':fp,'label':label,'dataset':ds}
        for roi in ['caudate','putamen']:
            m0=masks[roi]
            for k in (0,1,2):
                m=ndimage.binary_erosion(m0,iterations=k) if k else m0
                if m.sum()<30: continue
                stats_in(z,m,f'{roi}_e{k}__',out)
            # where do the darkest voxels sit relative to the rim?
            dist=ndimage.distance_transform_edt(m0)      # voxels from outside boundary
            zz=z[m0]; dd=dist[m0]
            order=np.argsort(zz)
            out[f'{roi}__dist_darkest10']=float(dd[order[:10]].mean())
            out[f'{roi}__dist_all_mean']=float(dd.mean())
            out[f'{roi}__dist_brightest10']=float(dd[order[-10:]].mean())
        return out
    except Exception as e:
        import traceback
        return {'file_path':fp,'label':label,'dataset':ds,
                'error':f'{type(e).__name__}: {e}','tb':traceback.format_exc()[-300:]}

def collect():
    from extract_radiomics_subcortical import collect_all as base
    tasks=[(f,l,d) for f,l,d in base() if d in ('ds000245','ds001907')]
    p=pd.read_csv(os.path.join(RAW,'ds005892','participants.tsv'),sep='\t')
    for _,row in p.iterrows():
        pid=row['participant_id']
        fp=os.path.join(RAW,'ds005892',pid,'anat',f'{pid}_T1w.nii.gz')
        if os.path.exists(fp):
            tasks.append((fp,0 if row['group']=='Control' else 1,'ds005892'))
    return tasks

def main():
    tasks=collect(); print(f'{len(tasks)} subjects',flush=True)
    get_ref()
    rows=[]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs={pool.submit(process_one,t):t for t in tasks}
        for i,f in enumerate(as_completed(futs),1):
            res=f.result(); rows.append(res)
            if 'error' in res: print(f'[{i}] ERR {res["error"]}',flush=True)
            elif i%25==0 or i==len(tasks):
                print(f'[{i}/{len(tasks)}] caud e0 n={res.get("caudate_e0__n")} '
                      f'e1 n={res.get("caudate_e1__n")} e2 n={res.get("caudate_e2__n")}',flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT,index=False)
    n=df['error'].notna().sum() if 'error' in df.columns else 0
    print(f'\nDone {len(df)-n}/{len(df)} -> {OUT}',flush=True)

if __name__=='__main__': main()
