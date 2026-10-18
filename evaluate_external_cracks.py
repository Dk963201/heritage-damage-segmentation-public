"""Zero-shot binary crack benchmark on an external HUMAN-ANNOTATED pilot."""
import csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import CLIPSegProcessor,CLIPSegForImageSegmentation
from auto_label import MODEL,REVISION
from metrics import segmentation_metrics
from audit_pilot import EXPECTED

def main():
    for name,expected in EXPECTED.items():
        if hashlib.md5((Path('data/raw')/name).read_bytes()).hexdigest()!=expected:raise ValueError('Pilot archive changed')
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    p=CLIPSegProcessor.from_pretrained(MODEL,revision=REVISION)
    m=CLIPSegForImageSegmentation.from_pretrained(MODEL,revision=REVISION,use_safetensors=True).to(device).eval()
    text=p.tokenizer(['a crack in a masonry wall'],return_tensors='pt').to(device)
    with torch.inference_mode():condition=m.get_conditional_embeddings(batch_size=1,input_ids=text['input_ids'],attention_mask=text['attention_mask'])
    matrix=np.zeros((2,2),dtype=np.int64);records=[]
    with zipfile.ZipFile('data/raw/masonry_crack_images.zip') as images,zipfile.ZipFile('data/raw/masonry_crack_masks.zip') as masks:
        names=sorted(n for n in images.namelist() if n.lower().endswith('.jpg'))
        if len(names)!=100:raise ValueError('Expected 100 external reference images')
        for i,name in enumerate(names,1):
            im=Image.open(io.BytesIO(images.read(name))).convert('RGB')
            gt_im=Image.open(io.BytesIO(masks.read(Path(name).stem+'_mask.png')))
            if gt_im.size!=im.size:raise ValueError('Pairing mismatch')
            gt=np.asarray(gt_im)
            if not set(np.unique(gt)).issubset({0,255}):raise ValueError('Unexpected reference labels')
            truth=gt==255
            pixels=p.image_processor(images=im,return_tensors='pt')['pixel_values'].to(device)
            with torch.inference_mode():score=m(pixel_values=pixels,conditional_embeddings=condition).logits.sigmoid().cpu().numpy().squeeze()
            pred=np.asarray(Image.fromarray((score>=.5).astype('uint8')).resize(im.size,Image.Resampling.NEAREST)).astype(bool)
            tp=int(np.count_nonzero(pred&truth));fp=int(np.count_nonzero(pred&~truth));fn=int(np.count_nonzero(~pred&truth));tn=int(truth.size-tp-fp-fn)
            matrix+=np.array([[tn,fp],[fn,tp]])
            records.append({'image':name,'tn':tn,'fp':fp,'fn':fn,'tp':tp})
            if i%20==0:print(f'External reference evaluation {i}/100',flush=True)
    scores=segmentation_metrics(matrix.tolist());total=int(matrix.sum());negative=int(matrix[0].sum())
    result={'scope':'EXTERNAL UK BINARY CRACK PILOT ONLY; NOT KURDISH FOUR-CLASS ACCURACY',
        'source':'https://zenodo.org/records/18458458','license':'CC-BY-4.0','model':MODEL,'revision':REVISION,
        'images':100,'training_or_threshold_tuning_on_reference':False,'threshold':.5,
        'evaluation_resolution':'original mask dimensions; 352px binary predictions expanded nearest-neighbor',
        'confusion_matrix_rows_truth':matrix.tolist(),'scores':scores,
        'all_background_pixel_accuracy':negative/total,'crack_prevalence':1-negative/total,
        'independence_note':'No images used for local fitting; overlap with foundation-model pretraining is unknown.'}
    Path('external-crack-results.json').write_text(json.dumps(result,indent=2)+'\n')
    with open('external-crack-per-image.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
