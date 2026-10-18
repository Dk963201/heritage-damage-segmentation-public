"""CLIPSeg draft annotations. Never ground truth; uncertain pixels remain 255."""
import argparse,csv,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import CLIPSegProcessor,CLIPSegForImageSegmentation
MODEL='CIDAS/clipseg-rd64-refined'
REVISION='999e0328d9e10b484360c477313983f9afdd7050'
PROMPTS=['healthy intact stone masonry wall','a crack in a masonry wall','eroded crumbling stone surface','vegetation growing on a masonry wall']

def select_labels(scores, threshold=0.5, margin=0.1):
    order=np.sort(scores,axis=0)
    labels=scores.argmax(axis=0).astype(np.uint8)
    labels[(order[-1]<threshold)|((order[-1]-order[-2])<margin)]=255
    return labels

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=800);args=ap.parse_args()
    out=Path('data/processed/clipseg-drafts-v2');out.mkdir(parents=True,exist_ok=True)
    processor=CLIPSegProcessor.from_pretrained(MODEL,revision=REVISION)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model=CLIPSegForImageSegmentation.from_pretrained(MODEL,revision=REVISION,use_safetensors=True).to(device).eval()
    with torch.inference_mode():
        text=processor.tokenizer(PROMPTS,padding=True,return_tensors='pt').to(device)
        conditions=model.get_conditional_embeddings(batch_size=4,input_ids=text['input_ids'],attention_mask=text['attention_mask'])
    rows=list(csv.DictReader(open('downloaded-800-manifest.csv')))[:args.limit]
    records=[];start=time.monotonic()
    for i,row in enumerate(rows,1):
        with zipfile.ZipFile('data/raw/image-versions/'+row['archive']) as archive:
            raw=archive.read(row['archive_path'])
        if hashlib.sha256(raw).hexdigest()!=row['download_sha256']:raise ValueError('Image hash mismatch')
        image=Image.open(io.BytesIO(raw)).convert('RGB')
        pixels=processor.image_processor(images=image,return_tensors='pt')['pixel_values'].to(device)
        with torch.inference_mode():
            vision=model.clip.get_image_features(pixel_values=pixels,interpolate_pos_encoding=True,output_hidden_states=True)
            activations=[vision.hidden_states[layer+1].repeat(4,1,1) for layer in model.extract_layers]
            scores=model.decoder(activations,conditions).logits.sigmoid().float().cpu().numpy()
            if i==1:
                baseline=model(pixel_values=pixels.repeat(4,1,1,1),conditional_embeddings=conditions).logits.sigmoid().float().cpu().numpy()
                error=float(np.max(np.abs(scores-baseline)))
                if error>0.001:raise ValueError(f'Cached encoder differs: {error}')
                print(f'Cached encoder equivalence max error: {error}',flush=True)
        mask=select_labels(scores)
        # Nearest-neighbor expansion preserves IDs, not lost image detail.
        full=Image.fromarray(mask).resize(image.size,Image.Resampling.NEAREST)
        path=out/(row['image_id']+'.png');full.save(path)
        vals,counts=np.unique(np.asarray(full),return_counts=True)
        record={'image_id':row['image_id'],'width':image.width,'height':image.height,'source_sha256':row['download_sha256'],
            'mask_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'UNREVIEWED PSEUDO-LABEL',
            **{f'pixels_{v}':int(dict(zip(vals,counts)).get(v,0)) for v in [0,1,2,3,255]}}
        records.append(record)
        # Resume/inspection record after every image, no user annotations overwritten.
        (out/'progress.json').write_text(json.dumps({'completed':i,'expected':len(rows)}))
        if i%20==0:print(f'Draft masks: {i}/{len(rows)}',flush=True)
    with open('pseudo-label-audit.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    total=sum(r['width']*r['height'] for r in records)
    summary={'model':MODEL,'revision':REVISION,'prompts':PROMPTS,'threshold':0.5,'margin':0.1,'device':device,
        'images':len(records),'seconds':round(time.monotonic()-start,2),'model_input':'processor default 352x352',
        'pixel_counts':{str(v):sum(r[f'pixels_{v}'] for r in records) for v in [0,1,2,3,255]},
        'human_verified_masks':0,'target_accuracy':None,'target_iou':None,'status':'DRAFT ONLY; NOT GROUND TRUTH'}
    summary['labelled_pixel_fraction']=1-summary['pixel_counts']['255']/total
    Path('pseudo-label-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with zipfile.ZipFile('pseudo-labels-unreviewed.zip','w',zipfile.ZIP_DEFLATED) as z:
        for r in records:z.write(out/(r['image_id']+'.png'),'masks/'+r['image_id']+'.png')
        z.write('pseudo-label-summary.json');z.write('pseudo-label-audit.csv')
        z.write('downloaded-800-manifest.csv');z.write('AUTO_LABELS.md')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
