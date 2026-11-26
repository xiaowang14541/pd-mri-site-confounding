"""Locations of the darkest / brightest caudate voxels in MNI space, for the 3 mixed-diagnosis cohorts.
Re-uses the registration of extract_v3_harmonized.py unchanged (affine MI + SyN CC) and writes per-subject
voxel coordinates and distance-to-boundary values (Euclidean distance transform, boundary voxel = 1)."""
import sys, os, json, numpy as np, nibabel as nib
sys.path.insert(0,'/master/wang_zi_wei/cursor/pd_classification/src'); sys.path.insert(0,'/master/wang_zi_wei/cursor/pd_classification')
import extract_v3_harmonized as E
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor, as_completed
OUT='/master/wang_zi_wei/cursor/pd_classification/figdata/'
os.makedirs(OUT,exist_ok=True)
MIXED={'ds000245','ds001907','ds005892'}

def work(args):
    fp,label,group,ds,save_img=args
    try:
        from dipy.align.imaffine import MutualInformationMetric, AffineRegistration, transform_centers_of_mass
        from dipy.align.transforms import TranslationTransform3D, RigidTransform3D, AffineTransform3D
        from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
        from dipy.align.metrics import CCMetric
        tpl,tpl_a,brain_tpl,masks=E.get_ref()
        im=nib.load(fp); raw=np.asarray(im.dataobj).astype(np.float32)
        if raw.ndim==4: raw=raw[...,0]
        drive=raw-raw.min()
        if drive.max()>0: drive=drive/drive.max()
        areg=AffineRegistration(metric=MutualInformationMetric(32,None),level_iters=[500,100,25],sigmas=[3.0,1.0,0.0],factors=[4,2,1],verbosity=0)
        com=transform_centers_of_mass(tpl,tpl_a,drive,im.affine)
        t=areg.optimize(tpl,drive,TranslationTransform3D(),None,tpl_a,im.affine,starting_affine=com.affine)
        r=areg.optimize(tpl,drive,RigidTransform3D(),None,tpl_a,im.affine,starting_affine=t.affine)
        a=areg.optimize(tpl,drive,AffineTransform3D(),None,tpl_a,im.affine,starting_affine=r.affine)
        wa=a.transform(drive)
        mapping=SymmetricDiffeomorphicRegistration(CCMetric(3),[25,15,8],inv_iter=20).optimize(tpl,wa)
        img_tpl=mapping.transform(a.transform(raw))
        z=E.zscore_in(img_tpl,brain_tpl)
        m=masks['caudate']; dist=ndi.distance_transform_edt(m)
        idx=np.argwhere(m); vals=z[m]; order=np.argsort(vals)
        dark=idx[order[:10]]; bright=idx[order[-10:]]
        out=dict(fp=fp,label=int(label),ds=ds,dark=dark.tolist(),bright=bright.tolist(),
                 d_dark=float(dist[tuple(dark.T)].mean()),d_bright=float(dist[tuple(bright.T)].mean()),d_all=float(dist[m].mean()),
                 zmin=float(vals.min()),zmean=float(vals.mean()),zmedian=float(np.median(vals)))
        if save_img: np.savez_compressed(OUT+f'img_{ds}_{int(label)}.npz',z=z.astype(np.float16))
        return out
    except Exception as e:
        import traceback
        return dict(fp=fp,ds=ds,label=int(label),error=f'{type(e).__name__}: {e}',tb=traceback.format_exc()[-300:])

if __name__=='__main__':
    tasks=[t for t in E.collect_all() if t[3] in MIXED]
    seen=set(); jobs=[]
    for fp,l,g,d in tasks:
        key=(d,int(l)); s=key not in seen; seen.add(key); jobs.append((fp,l,g,d,s))
    print('subjects',len(jobs),flush=True)
    tpl,tpl_a,brain,masks=E.get_ref()
    np.savez_compressed(OUT+'ref.npz',tpl=tpl,brain=brain,affine=tpl_a,**{k:v for k,v in masks.items()})
    res=[]
    with ProcessPoolExecutor(max_workers=32) as pool:
        futs=[pool.submit(work,j) for j in jobs]
        for i,f in enumerate(as_completed(futs),1):
            r=f.result(); res.append(r)
            if 'error' in r: print('ERR',r['error'],flush=True)
            if i%10==0: print(i,'/',len(jobs),flush=True)
            json.dump(res,open(OUT+'anat_results.json','w'))
    print('DONE',len(res),flush=True)
