"""Pinned Moondream2 scene-screening pilot; outputs are not damage labels."""
import csv,json,time,os
from pathlib import Path
os.environ.setdefault('FIFTYONE_DO_NOT_TRACK','1')
import torch
from PIL import Image
from transformers import AutoModelForCausalLM
ROOT=Path(__file__).resolve().parent
REV='8ef2caddfdead307fe7f3a4ed596baca7eab8a2f'
PROMPT=('Classify this photo into exactly one category: close_up_masonry (a close view dominated by individual bricks or masonry stones), building_overview (whole buildings, streets or distant architecture), landscape_or_other (landscape or unrelated content), uncertain (cannot confidently determine). A carved natural rock relief is uncertain, not masonry. Return the category name only.')
if __name__=='__main__':
    import psutil, shutil
    if psutil.virtual_memory().total < 12 * 1024**3 or shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise SystemExit('Pilot blocked: use a machine with at least 12 GiB RAM and 8 GiB free disk after model download. These are conservative project safeguards, not vendor requirements.')
    torch.manual_seed(0)
    model=AutoModelForCausalLM.from_pretrained(str(ROOT/'models/moondream2'),trust_remote_code=True,local_files_only=True,torch_dtype=torch.float16,device_map={'':'mps'},low_cpu_mem_usage=True).eval()
    rows=list(csv.DictReader((ROOT/'screening-reference.csv').open()))
    out=[]
    for row in rows:
        start=time.monotonic()
        im=Image.open(ROOT/'data/processed/screening-sample'/f"{row['image_id']}.jpg").convert('RGB'); im.thumbnail((768,768))
        with torch.inference_mode():
            encoded=model.encode_image(im)
            caption=model.caption(encoded,length='short',settings={'max_tokens':80})['caption']
            answer=model.query(encoded,PROMPT,settings={'max_tokens':20})['answer'].strip()
        row.update(caption=caption,prediction=answer,agreement=answer==row['reference'],seconds=round(time.monotonic()-start,2))
        out.append(row)
        (ROOT/'screening-results.json').write_text(json.dumps({'model_revision':REV,'device':'mps','dtype':'float16','max_image_side':768,'prompt':PROMPT,'reference_type':'assistant visual review; not expert ground truth','results':out},indent=2))
        print(json.dumps(row),flush=True)
        del encoded
        torch.mps.empty_cache()
