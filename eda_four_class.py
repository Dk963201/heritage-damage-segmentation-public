"""Summarize approved masks and source color/edge distributions by split."""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from validate_training_manifest import validate
p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.manifest.resolve().parent;rows=list(csv.DictReader(a.manifest.open()));validate(rows,root)
if a.output.exists():p.error('Output exists; choose a new file')
records=[];splits={}
for row in rows:
    with Image.open(root/row['image_path']) as src:rgb=np.asarray(src.convert('RGB'))
    with Image.open(root/row['mask_path']) as src:mask=np.asarray(src)
    count=[int((mask==i).sum()) for i in [0,1,2,3,255]]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);edges=cv2.Canny(gray,100,200)
    hist=[np.histogram(rgb[:,:,i],bins=16,range=(0,256))[0].tolist() for i in range(3)]
    records.append({'image':row['image_path'],'building_group':row['building_group'],'split':row['split'],'width':rgb.shape[1],'height':rgb.shape[0],'pixels_0_1_2_3_255':count,'rgb_histograms_16_bins':hist,'edge_fraction':float((edges>0).mean())})
    splits.setdefault(row['split'],np.zeros(5,dtype=np.int64));splits[row['split']]+=count
result={'class_order':[0,1,2,3,255],'canny_thresholds':[100,200],'warning':'Edge frequency is descriptive, not a crack label. No site field exists; group summaries use supplied building IDs. Do not tune to test summaries.','split_pixel_counts':{k:v.tolist() for k,v in splits.items()},'sources':records}
a.output.write_text(json.dumps(result,indent=2))
