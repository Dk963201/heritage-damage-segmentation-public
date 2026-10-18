"""Validate every draft mask; test three models on real images, NOT research accuracy."""
import csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
from model_check import build_model

def main():
    torch.manual_seed(42);torch.set_num_threads(2)
    rows=list(csv.DictReader(open('downloaded-800-manifest.csv')))
    audit={r['image_id']:r for r in csv.DictReader(open('pseudo-label-audit.csv'))}
    if len(audit)!=800:raise ValueError('800 draft masks required')
    candidates=[]
    for row in rows:
        path=Path('data/processed/clipseg-drafts-v2')/(row['image_id']+'.png')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=audit[row['image_id']]['mask_sha256']:raise ValueError('Mask hash mismatch')
        with Image.open(path) as mask:
            if mask.size!=(int(row['download_width']),int(row['download_height'])):raise ValueError('Dimension mismatch')
            values=set(np.unique(np.array(mask)).tolist())
            if not values.issubset({0,1,2,3,255}):raise ValueError('Unknown mask ID')
            # Pick up to two real examples with at least 1% assigned pixels.
            assigned=sum(int(audit[row['image_id']][f'pixels_{v}']) for v in range(4))
            if len(candidates)<2 and assigned/(mask.width*mask.height)>.01:candidates.append(row)
    if len(candidates)!=2:raise ValueError('Insufficient usable draft pixels for integration test')
    xs=[];ys=[]
    for row in candidates:
        with zipfile.ZipFile('data/raw/image-versions/'+row['archive']) as z:
            im=Image.open(io.BytesIO(z.read(row['archive_path']))).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
        mask=Image.open(Path('data/processed/clipseg-drafts-v2')/(row['image_id']+'.png')).resize((256,256),Image.Resampling.NEAREST)
        xs.append(torch.from_numpy(np.array(im).copy()).permute(2,0,1).float()/255)
        ys.append(torch.from_numpy(np.array(mask).copy()).long())
    x=torch.stack(xs);y=torch.stack(ys);records=[]
    for name in ['unet','deeplabv3plus','fpn']:
        model=build_model(name).train();opt=torch.optim.Adam(model.parameters(),lr=1e-5)
        logits=model(x)
        loss=smp.losses.FocalLoss(mode='multiclass',ignore_index=255)(logits,y)+smp.losses.DiceLoss(mode='multiclass',ignore_index=255)(logits,y)
        if not torch.isfinite(loss):raise ValueError('Non-finite loss')
        loss.backward();opt.step()
        records.append({'model':name,'output_shape':list(logits.shape),'finite_loss':True,'optimizer_step_passed':True})
        del model,opt,logits,loss
    Path('draft-pipeline-results.json').write_text(json.dumps({'validated_masks':800,'human_reviewed':0,
        'test':'ONE OPTIMIZER STEP PER MODEL ON TWO REAL IMAGES WITH UNREVIEWED DRAFT MASKS',
        'image_ids':[r['image_id'] for r in candidates],'models':records,'accuracy':None,'mean_iou':None,
        'weights_saved':False,'research_evaluation':False},indent=2)+'\n')
    print('PASS: 800 masks validated; 3 real-image integration checks passed',flush=True)
if __name__=='__main__':main()
