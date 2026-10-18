"""Exploratory threshold calibration on the reused UK validation set; expect optimistic bias."""
import io,json,zipfile,argparse
from pathlib import Path
import numpy as np,torch
from PIL import Image
import segmentation_models_pytorch as smp
from train_binary_pilot import scores
p=argparse.ArgumentParser();p.add_argument('--root',default='local-results/public-masonry');p.add_argument('--output',default='local-results/public-masonry/threshold-sweep.json');a=p.parse_args()
torch.set_num_threads(2);root=Path(a.root);split=json.loads((root/'split.json').read_text());names=split['validation'];xs=[];ys=[]
with zipfile.ZipFile('data/raw/masonry_crack_images.zip') as zi,zipfile.ZipFile('data/raw/masonry_crack_masks.zip') as zm:
 for name in names:
  box=split['crop_regions'][name];xs.append(np.asarray(Image.open(io.BytesIO(zi.read(name))).convert('RGB').crop(box)).copy());ys.append(np.asarray(Image.open(io.BytesIO(zm.read(Path(name).stem+'_mask.png'))).crop(box))==255)
X=torch.from_numpy(np.stack(xs)).permute(0,3,1,2).float()/255
X=(X-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
model=smp.Unet(encoder_name='efficientnet-b0',encoder_weights=None,in_channels=3,classes=1).eval();model.load_state_dict(torch.load(root/'unet-diagnostic.pt',map_location='cpu',weights_only=True));probs=[]
with torch.inference_mode():
 for x in X:probs.append(model(x[None]).sigmoid()[0,0].numpy())
records=[]
for threshold in np.arange(.1,.901,.01):
 cm=np.zeros((2,2),dtype=np.int64)
 for prob,truth in zip(probs,ys):
  pred=prob>=threshold;cm+=np.bincount((truth.astype(np.uint8)*2+pred.astype(np.uint8)).ravel(),minlength=4).reshape(2,2)
 records.append({'threshold':round(float(threshold),2),**scores(torch.from_numpy(cm))})
best=max(records,key=lambda x:x['crack_dice'])
Path(a.output).write_text(json.dumps({'scope':'EXPLORATORY THRESHOLD CALIBRATION ON REUSED UK DEVELOPMENT VALIDATION; OPTIMISTIC, NOT INDEPENDENT TEST','checkpoint':'local-results/public-masonry/unet-diagnostic.pt','n_validation_images':len(names),'thresholds':records,'best_on_reused_validation':best},indent=2));print(json.dumps(best,indent=2))
