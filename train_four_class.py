"""Four-class assignment workflow. Requires reviewed masks; never manufactures labels."""
import argparse,csv,hashlib,json,random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageEnhance
import torch
from torch.utils.data import Dataset,DataLoader
import segmentation_models_pytorch as smp
from validate_training_manifest import validate
from metrics import segmentation_metrics
MODELS={'unet':smp.Unet,'deeplabv3plus':smp.DeepLabV3Plus,'fpn':smp.FPN}
CLASSES=['Healthy Masonry','Structural Crack','Surface Erosion','Vegetation Overgrowth']

def starts(n,p):
    return sorted(set(range(0,max(n-p,0)+1,p))|{max(n-p,0)})

def tensor(image,clahe=True):
    if clahe:
        lab=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(lab[:,:,0])
        image=Image.fromarray(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))
    x=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
    return (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

class Patches(Dataset):
    def __init__(self,rows,root,patch,clahe=True):
        self.root,self.patch,self.clahe=root,patch,clahe;self.items=[]
        for row in rows:
            with Image.open(root/row['image_path']) as image:w,h=image.size
            self.items.extend((row,x,y) for y in starts(h,patch) for x in starts(w,patch))
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        row,x,y=self.items[i];p=self.patch
        with Image.open(self.root/row['image_path']) as src:image=src.convert('RGB').crop((x,y,x+p,y+p))
        with Image.open(self.root/row['mask_path']) as src:
            valid=src.crop((x,y,min(x+p,src.width),min(y+p,src.height)))
            mask=Image.new('L',(p,p),255);mask.paste(valid,(0,0))
        for flip in [Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM]:
            if random.random()<.5:image,mask=image.transpose(flip),mask.transpose(flip)
        image=ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
        image=ImageEnhance.Color(image).enhance(random.uniform(.85,1.15))
        return tensor(image,self.clahe),torch.from_numpy(np.asarray(mask).copy()).long()

def loss_fn(logits,target):
    valid=target!=255
    if not valid.any():return logits.sum()*0
    safe=target.masked_fill(~valid,0);logp=torch.log_softmax(logits,dim=1)
    logpt=logp.gather(1,safe[:,None]).squeeze(1)
    focal=(-((1-logpt.exp())**2)*logpt)[valid].mean()
    p=logp.exp()*valid[:,None]
    truth=torch.nn.functional.one_hot(safe,4).permute(0,3,1,2)*valid[:,None]
    dims=(0,2,3);present=truth.sum(dims)>0
    dice=(2*(p*truth).sum(dims)+1e-6)/(p.sum(dims)+truth.sum(dims)+1e-6)
    return focal+(1-dice[present]).mean()

@torch.inference_mode()
def evaluate(model,rows,root,patch,clahe):
    model.eval();cm=np.zeros((4,4),dtype=np.int64);site_cm={};per_image=[]
    for row in rows:
        with Image.open(root/row['image_path']) as src:image=src.convert('RGB')
        w,h=image.size;scores=np.zeros((4,h,w),np.float32);counts=np.zeros((h,w),np.float32)
        for y in starts(h,patch):
            for x in starts(w,patch):
                out=model(tensor(image.crop((x,y,x+patch,y+patch)),clahe)[None])[0].numpy()
                rh,rw=min(patch,h-y),min(patch,w-x)
                scores[:,y:y+rh,x:x+rw]+=out[:,:rh,:rw];counts[y:y+rh,x:x+rw]+=1
        pred=(scores/counts[None]).argmax(0)
        with Image.open(root/row['mask_path']) as src:truth=np.asarray(src).astype(np.int64)
        valid=truth!=255
        image_cm=np.bincount((truth[valid]*4+pred[valid]).ravel(),minlength=16).reshape(4,4)
        cm+=image_cm
        site=row.get('site','').strip() or 'unspecified'
        site_cm.setdefault(site,np.zeros((4,4),dtype=np.int64))
        site_cm[site]+=image_cm
        per_image.append({'image_path':row['image_path'],'site':site,'building_group':row.get('building_group'),
                          'class_support':image_cm.sum(axis=1).tolist(),'confusion_matrix':image_cm.tolist(),
                          **segmentation_metrics(image_cm.tolist())})
    return {'confusion_matrix':cm.tolist(),'class_support':cm.sum(axis=1).tolist(),
            **segmentation_metrics(cm.tolist()),'per_image':per_image,
            'by_site':{site:{'confusion_matrix':matrix.tolist(),'class_support':matrix.sum(axis=1).tolist(),
                             **segmentation_metrics(matrix.tolist())} for site,matrix in site_cm.items()}}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--patch',type=int,default=256)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--no-clahe',action='store_true');p.add_argument('--random-init',action='store_true')
    p.add_argument('--evaluate-test',action='store_true',help='Only after freezing experimental choices')
    p.add_argument('--require-sites',action='store_true',help='Require both Erbil and Amedi in every building-group split')
    a=p.parse_args()
    if not(1<=a.epochs<=1000 and 32<=a.patch<=1024 and a.patch%32==0 and 2<=a.batch<=32 and 1<=a.warmup<=a.epochs and 0<a.lr<=.1):p.error('Invalid bounds: patch multiple of32, batch>=2, warmup<=epochs')
    root=a.manifest.resolve().parent
    try:
        with a.manifest.open(newline='') as stream:rows=list(csv.DictReader(stream))
        counts=validate(rows,root,require_sites=a.require_sites)
    except (ValueError,OSError,KeyError,csv.Error) as error:p.exit(2,f'Training data unavailable: {error}\n')
    if a.output.exists():p.error('Use a fresh output directory to preserve existing experiments')
    a.output.mkdir(parents=True);torch.set_num_threads(2)
    groups={s:[r for r in rows if r['split']==s] for s in counts}
    protocol={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    protocol.update(classes=CLASSES,encoder='efficientnet-b0',source_counts=counts,loss='Focal gamma2 + Dice',optimizer='Adam',torch=torch.__version__,smp=smp.__version__,device='cpu',manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),scope='SYNTHETIC SOFTWARE TEST ONLY' if all(r['annotator']=='SOFTWARE_FIXTURE' for r in rows) else 'REVIEWED FOUR-CLASS DEVELOPMENT',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    provenance=[{**r,'mask_sha256':hashlib.sha256((root/r['mask_path']).read_bytes()).hexdigest()} for r in rows]
    (a.output/'sources.json').write_text(json.dumps(provenance,indent=2));results=[]
    for name in a.models:
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        ds=Patches(groups['train'],root,a.patch,not a.no_clahe)
        if len(ds)<a.batch:raise ValueError('Insufficient patches for a full batch')
        loader=DataLoader(ds,batch_size=a.batch,shuffle=True,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(a.seed))
        model=MODELS[name](encoder_name='efficientnet-b0',encoder_weights=None if a.random_init else 'imagenet',classes=4)
        opt=torch.optim.Adam(model.parameters(),lr=a.lr);history=[];best=-1;best_scores=None;best_epoch=None
        for epoch in range(a.epochs):
            rate=a.lr*min(1.,(epoch+1)/a.warmup)
            for group in opt.param_groups:group['lr']=rate
            model.train();total=0.;steps=0
            for image,mask in loader:
                if not(mask!=255).any():continue
                opt.zero_grad(set_to_none=True);loss=loss_fn(model(image),mask)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();opt.step();total+=loss.item();steps+=1
            if not steps:raise ValueError('No annotated training pixels visited')
            scores=evaluate(model,groups['validation'],root,a.patch,not a.no_clahe)
            history.append(dict(epoch=epoch+1,lr=rate,training_loss=total/steps,validation=scores))
            if scores['mean_iou'] is not None and scores['mean_iou']>best:
                best=scores['mean_iou'];best_scores=scores;best_epoch=epoch+1
                torch.save(model.state_dict(),a.output/f'{name}-best.pt')
            (a.output/f'{name}-history.json').write_text(json.dumps(history,indent=2));print(name,epoch+1,scores['mean_iou'],flush=True)
        checkpoint=a.output/f'{name}-best.pt'
        result={'model':name,'best_epoch':best_epoch,'best_validation_miou':best,'validation':best_scores,'test':'NOT EVALUATED',
                'checkpoint':checkpoint.name,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'scope':protocol['scope']}
        (checkpoint.with_suffix('.metadata.json')).write_text(json.dumps({
            'checkpoint_sha256':result['checkpoint_sha256'],
            'protocol_sha256':hashlib.sha256((a.output/'protocol.json').read_bytes()).hexdigest(),
            'architecture':name,'num_classes':4,'scope':protocol['scope'],
            'manifest_sha256':protocol['manifest_sha256'],'best_epoch':best_epoch},indent=2))
        if a.evaluate_test:
            model.load_state_dict(torch.load(a.output/f'{name}-best.pt',map_location='cpu',weights_only=True))
            result['test']=evaluate(model,groups['test'],root,a.patch,not a.no_clahe)
        results.append(result);(a.output/'comparison.json').write_text(json.dumps(results,indent=2));del model,opt
if __name__=='__main__':main()
