"""Recompute out-of-fold probabilities of the dataset-stratified 5-fold CV from the saved best_model.pth of each fold."""
import sys, os, json, numpy as np, pandas as pd, yaml, torch
B='/master/wang_zi_wei/cursor/pd_classification/'
sys.path.insert(0,B+'src')
from preprocessing.preprocess import MRIDataset
from models.classifier import get_model
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
cfg=yaml.safe_load(open(B+'configs/config_5fold.yaml'))
info=pd.read_csv(B+'data/splits/data_info.csv'); splits=json.load(open(B+'data/splits/cv_splits.json'))
rd=B+'results/densenet121_20260124_031248/'
Y=[];P=[];D=[];S=[];F=[]
for sp in splits:
    f=sp['fold']; vi=sp['val_idx']
    paths=info.file_path.values[vi].tolist(); labels=info.label.values[vi].tolist()
    ds=MRIDataset(paths,labels,slice_axis=1,augment=False)
    dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=4)
    m=get_model(model_name=cfg['model']['name'],num_classes=2,pretrained=False,dropout=cfg['model']['dropout'])
    m.load_state_dict(torch.load(rd+f'fold_{f}/best_model.pth',map_location='cpu')); m.eval()
    probs=[]
    with torch.no_grad():
        for x,y in dl: probs+=torch.softmax(m(x),1)[:,1].numpy().tolist()
    a=roc_auc_score(labels,probs); ref=json.load(open(rd+f'fold_{f}/metrics.json'))
    print('fold',f,'n',len(vi),'AUC recomputed %.4f'%a,'| saved',{k:round(v,4) for k,v in ref.items() if 'auc' in k.lower()},flush=True)
    Y+=labels;P+=probs;D+=info.dataset.values[vi].tolist();S+=info.subject_id.values[vi].tolist();F+=[f]*len(vi)
np.savez(B+'figdata/cv_oof.npz',y=np.array(Y),p=np.array(P,dtype=np.float32),dataset=np.array(D),subject=np.array(S),fold=np.array(F))
print('pooled AUC',roc_auc_score(Y,P),flush=True)
