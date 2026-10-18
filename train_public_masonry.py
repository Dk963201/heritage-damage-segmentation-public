"""Real supervised diagnostic on published UK crack masks; not Kurdish task results."""
import io,json,zipfile,hashlib,time,random,argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
import segmentation_models_pytorch as smp
from audit_pilot import EXPECTED

def scores(cm):
 tn,fp,fn,tp=cm.reshape(-1).tolist()
 return {'crack_dice':2*tp/max(2*tp+fp+fn,1),'crack_iou':tp/max(tp+fp+fn,1),'pixel_accuracy':(tp+tn)/max(cm.sum().item(),1),'all_background_accuracy':(tn+fp)/max(cm.sum().item(),1),'confusion_matrix':cm.tolist()}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--resume',default='local-results/expanded-pilot/unet-diagnostic.pt');parser.add_argument('--pretrained',action='store_true');parser.add_argument('--balanced',action='store_true');parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--unet-only',action='store_true');parser.add_argument('--output',default='local-results/binary-pilot');args=parser.parse_args()
 if not 1<=args.epochs<=100:raise ValueError('Epochs must be 1..100')
 torch.set_num_threads(2); torch.manual_seed(42); random.seed(42)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 audit=json.loads(Path('pilot-audit.json').read_text())
 for n,h in EXPECTED.items():
  assert hashlib.md5((Path('data/raw')/n).read_bytes()).hexdigest()==h
 xs=[];ys=[];names=[];regions=[]
 with zipfile.ZipFile('data/raw/masonry_crack_images.zip') as zi,zipfile.ZipFile('data/raw/masonry_crack_masks.zip') as zm:
  for r in audit['pairs']:
   name=r['image']; im=Image.open(io.BytesIO(zi.read(name))).convert('RGB'); ma=Image.open(io.BytesIO(zm.read(Path(name).stem+'_mask.png')))
   assert im.size==ma.size
   w,h=im.size;box=((w-256)//2,(h-256)//2,(w+256)//2,(h+256)//2)
   # Fixed center crop independent of labels; no full-image shrink that destroys thin cracks.
   x=np.asarray(im.crop(box)).copy(); y=np.asarray(ma.crop(box)).copy();assert set(np.unique(y))<= {0,255}
   xs.append(torch.from_numpy(x).permute(2,0,1).float()/255);ys.append(torch.from_numpy((y==255).astype('float32'))[None]);names.append(name);regions.append(box)
 perm=np.random.default_rng(42).permutation(len(names));tr=perm[:80];va=perm[80:]
 split={'train':[names[i] for i in tr],'validation':[names[i] for i in va],'crop_regions':dict(zip(names,regions)),'warning':'Image-level split only; building/scene groups unavailable. Validation previously inspected in external zero-shot benchmark. Development diagnostic, not independent research test.'}
 (out/'split.json').write_text(json.dumps(split,indent=2))
 # Expand TRAIN only with one native-resolution crack-centered crop per source.
 extra_indices=[];extra_regions=[];rng=np.random.default_rng(2026)
 with zipfile.ZipFile('data/raw/masonry_crack_images.zip') as zi,zipfile.ZipFile('data/raw/masonry_crack_masks.zip') as zm:
  for index in tr.copy():
   name=names[index];im=Image.open(io.BytesIO(zi.read(name))).convert('RGB');ma=Image.open(io.BytesIO(zm.read(Path(name).stem+'_mask.png')));a=np.asarray(ma);yy,xx=np.where(a==255)
   if not len(yy):continue
   j=int(rng.integers(len(yy)));w,h=im.size;l=max(0,min(w-256,int(xx[j])-128));t=max(0,min(h-256,int(yy[j])-128));box=(l,t,l+256,t+256)
   xs.append(torch.from_numpy(np.asarray(im.crop(box)).copy()).permute(2,0,1).float()/255);ys.append(torch.from_numpy((np.asarray(ma.crop(box))==255).astype('float32'))[None]);extra_indices.append(len(xs)-1);extra_regions.append({'source':name,'box':box})
 (out/'additional-training-crops.json').write_text(json.dumps(extra_regions,indent=2))
 tr=np.concatenate([tr,np.array(extra_indices,dtype=int)])
 public=json.loads(Path('public-masonry-audit.json').read_text());public_indices=[]
 for row in public['rows']:
  if row['status']!='eligible_training':continue
  im=Image.open(row['image']).convert('RGB');assert hashlib.sha256(im.tobytes()).hexdigest()==row['rgb_sha256']
  ma=Image.open(row['mask']).convert('L');im=im.resize((256,256),Image.Resampling.BILINEAR);ma=ma.resize((256,256),Image.Resampling.NEAREST)
  xs.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255);ys.append(torch.from_numpy((np.asarray(ma)==255).astype('float32'))[None]);public_indices.append(len(xs)-1)
 tr=np.concatenate([tr,np.array(public_indices,dtype=int)])
 X=torch.stack(xs);Y=torch.stack(ys);del xs,ys
 if args.pretrained:X=(X-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
 sampler=None
 if args.balanced:
  positive=Y[tr].flatten(1).any(1);n=int(positive.sum());assert 0<n<len(tr)
  weights=torch.where(positive,1/n,1/(len(tr)-n)).double()
  sampler=WeightedRandomSampler(weights,len(tr),replacement=True,generator=torch.Generator().manual_seed(42))
 train=DataLoader(TensorDataset(X[tr],Y[tr],torch.as_tensor(tr)),batch_size=2,shuffle=sampler is None,sampler=sampler,generator=torch.Generator().manual_seed(42))
 val=DataLoader(TensorDataset(X[va],Y[va]),batch_size=2)
 records=[]
 for name,ctor in [('unet',smp.Unet),('fpn',smp.FPN),('deeplabv3plus',smp.DeepLabV3Plus)]:
  if args.unet_only and name!='unet':continue
  torch.manual_seed(42);start=time.monotonic()
  model=ctor(encoder_name='efficientnet-b0',encoder_weights='imagenet' if args.pretrained else None,in_channels=3,classes=1)
  model.load_state_dict(torch.load(args.resume,map_location='cpu',weights_only=True))
  opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-5); dice=smp.losses.DiceLoss(mode='binary'); focal=smp.losses.FocalLoss(mode='binary')
  hist=[];seen_training_indices=set()
  for epoch in range(args.epochs):
   model.train();total=0
   # Preserve learned batch statistics with this small batch size.
   for module in model.modules():
    if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
   for x,y,indices in train:
    seen_training_indices.update(indices.tolist())
    # Paired, lossless geometric augmentation on TRAIN ONLY.
    k=random.randrange(4);x=torch.rot90(x,k,(-2,-1));y=torch.rot90(y,k,(-2,-1))
    if random.random()<.5:x=x.flip(-1);y=y.flip(-1)
    x=x.contiguous();y=y.contiguous()
    opt.zero_grad(set_to_none=True);p=model(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(p,y)+dice(p,y);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();total+=loss.item()
   model.eval();cm=torch.zeros((2,2),dtype=torch.int64)
   with torch.inference_mode():
    for x,y in val:
     p=model(x).sigmoid()>=.5;t=y.bool();cm+=torch.bincount((t.long()*2+p.long()).reshape(-1),minlength=4).reshape(2,2)
   scheduler.step()
   torch.save(model.state_dict(),out/f'{name}-latest.pt')
   rec={'epoch':epoch+1,'train_loss':total/len(train),**scores(cm)};hist.append(rec);(out/'progress.json').write_text(json.dumps(hist,indent=2));print(name,json.dumps(rec),flush=True)
  torch.save(model.state_dict(),out/f'{name}-diagnostic.pt')
  records.append({'model':name,'history':hist,'seconds':time.monotonic()-start})
  result={'scope':'UK BINARY CENTER-CROP DEVELOPMENT DIAGNOSTIC; NOT KURDISH FOUR-CLASS RESULTS','source':'https://zenodo.org/records/18458458','license':'CC-BY-4.0','device':'cpu','seed':42,'encoder':'efficientnet-b0','encoder_weights':'imagenet' if args.pretrained else None,'initialization':'ImageNet encoder; random decoder' if args.pretrained else 'random; no pretrained weights','balanced_training_sampler':args.balanced,'normalization':'ImageNet mean/std' if args.pretrained else 'RGB /255','patch_size':256,'batch_size':2,'epochs':args.epochs,'optimizer':'AdamW','learning_rate':.00005,'schedule':'cosine to 1e-5','augmentation':'paired rotations/flips on training only','batchnorm':'running statistics frozen','resume':args.resume,'loss':'binary cross entropy + dice','threshold':.5,'train_images_uk':80,'public_patches_added':len(public_indices),'public_source':'Dais et al.2021, pinned GPL-3.0 repository subset','public_resize':'224->256; bilinear RGB, nearest masks','public_unique_indices_seen':len(seen_training_indices.intersection(public_indices)),'train_patches':len(tr),'validation_images':20,'split_warning':split['warning'],'models':records}
  (out/'results.json').write_text(json.dumps(result,indent=2));del model,opt
if __name__=='__main__':main()
