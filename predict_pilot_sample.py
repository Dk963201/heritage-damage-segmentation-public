"""Visual transfer check on 0650; UNREVIEWED crack proposals, not ground truth."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp

def main():
 torch.set_num_threads(2)
 parser=argparse.ArgumentParser();parser.add_argument('--root',default='local-results/binary-balanced');args=parser.parse_args()
 root=Path(args.root);weight=root/'unet-diagnostic.pt'
 model=smp.Unet(encoder_name='efficientnet-b0',encoder_weights=None,in_channels=3,classes=1).eval()
 model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
 original=Image.open('data/processed/screening-sample/0650.jpg').convert('RGB');size=original.size
 im=original.copy();im.thumbnail((1024,1024));a=np.asarray(im);h,w=a.shape[:2]
 x=torch.from_numpy(a.copy()).permute(2,0,1).float()/255
 x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
 prob=np.zeros((h,w),dtype='float32');count=np.zeros((h,w),dtype='float32')
 yy=list(range(0,max(h-256,0)+1,192));xx=list(range(0,max(w-256,0)+1,192))
 if yy[-1]!=max(h-256,0):yy.append(max(h-256,0))
 if xx[-1]!=max(w-256,0):xx.append(max(w-256,0))
 with torch.inference_mode():
  for y in yy:
   for z in xx:
    tile=x[:,y:y+256,z:z+256];th,tw=tile.shape[1:];tile=torch.nn.functional.pad(tile,(0,256-tw,0,256-th))
    p=model(tile[None]).sigmoid()[0,0,:th,:tw].numpy();prob[y:y+th,z:z+tw]+=p;count[y:y+th,z:z+tw]+=1
 assert (count>0).all();prob/=count;mask=prob>=.5
 # Only predicted crack=1; all other pixels are UNKNOWN=255, never healthy.
 Image.fromarray(np.where(mask,1,255).astype('uint8')).save(root/'0650-UNREVIEWED-preview-mask.png')
 overlay=a.astype('float32');overlay[mask]=overlay[mask]*.4+np.array([255,0,0])*.6
 Image.fromarray(overlay.astype('uint8')).save(root/'0650-draft-overlay.jpg')
 im.save(root/'0650-preview.jpg')
 (root/'transfer-check.json').write_text(json.dumps({'image_id':'0650','source_dimensions':size,'preview_dimensions':im.size,'checkpoint_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'threshold':.5,'tile_size':256,'stride':192,'predicted_crack_fraction':float(mask.mean()),'labels':{'1':'unreviewed proposed crack','255':'unknown'},'warning':'Preview-resolution transfer demonstration only. No ground truth, no accuracy metric, no four-class mask. Do not train on this output.'},indent=2))
if __name__=='__main__':main()
