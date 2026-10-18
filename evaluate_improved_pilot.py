"""Compare saved models on unchanged development images; no threshold search."""
import io,json,zipfile,hashlib,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
import segmentation_models_pytorch as smp
from train_binary_pilot import scores

def main():
 torch.set_num_threads(2)
 parser=argparse.ArgumentParser();parser.add_argument('--root',default='local-results/improved-pilot');parser.add_argument('--baseline',default='local-results/binary-balanced/unet-diagnostic.pt');args=parser.parse_args()
 root=Path(args.root);split=json.loads((root/'split.json').read_text())
 old=json.loads(Path('local-results/binary-balanced/split.json').read_text());assert split==old,'Split/crop changed'
 images=[];truth=[]
 with zipfile.ZipFile('data/raw/masonry_crack_images.zip') as zi,zipfile.ZipFile('data/raw/masonry_crack_masks.zip') as zm:
  for name in split['validation']:
   box=split['crop_regions'][name];images.append(np.asarray(Image.open(io.BytesIO(zi.read(name))).convert('RGB').crop(box)).copy());truth.append(np.asarray(Image.open(io.BytesIO(zm.read(Path(name).stem+'_mask.png'))).crop(box))==255)
 X=torch.from_numpy(np.stack(images)).permute(0,3,1,2).float()/255
 X=(X-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
 records={};allpred={}
 for label,weight in [('prior',Path(args.baseline)),('improved',root/'unet-diagnostic.pt')]:
  model=smp.Unet(encoder_name='efficientnet-b0',encoder_weights=None,in_channels=3,classes=1).eval();model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True));cm=torch.zeros((2,2),dtype=torch.int64);rows=[];predictions=[]
  with torch.inference_mode():
   for i,name in enumerate(split['validation']):
    p=model(X[i:i+1]).sigmoid()[0,0]>=.5;t=torch.from_numpy(truth[i]);c=torch.bincount((t.long()*2+p.long()).reshape(-1),minlength=4).reshape(2,2);cm+=c;rows.append({'image':name,'positive_pixels':int(t.sum()),**scores(c)});predictions.append(p.numpy())
  records[label]={'checkpoint_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'pooled':scores(cm),'per_image':rows};allpred[label]=predictions
 canvas=Image.new('RGB',(256*4,280*6),'white');draw=ImageDraw.Draw(canvas)
 for row,i in enumerate([i for i,t in enumerate(truth) if t.any()][:6]):
  panels=[images[i],np.repeat(truth[i][...,None]*255,3,axis=2).astype('uint8')]
  for label in ['prior','improved']:
   a=images[i].astype(float).copy();m=allpred[label][i];a[m]=a[m]*.4+np.array([255,0,0])*.6;panels.append(a.astype('uint8'))
  for col,a in enumerate(panels):canvas.paste(Image.fromarray(a),(col*256,row*280+24))
  draw.text((4,row*280+5),split['validation'][i]+' | truth | prior | improved',fill='black')
 canvas.save(root/'validation-comparison.jpg')
 (root/'comparison.json').write_text(json.dumps({'threshold':.5,'split_unchanged':True,'scope':'development validation only; no Kurdish ground truth','models':records},indent=2))
 print(json.dumps({k:v['pooled'] for k,v in records.items()},indent=2))
if __name__=='__main__':main()
