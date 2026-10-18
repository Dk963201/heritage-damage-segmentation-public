"""Reject unreviewed/invalid four-class masks and group leakage before training."""
import argparse,csv,hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def validate(rows,root,require_sites=False):
 if not rows:raise ValueError('Empty manifest')
 groups={};group_sites={};seen=set();counts={k:0 for k in ['train','validation','test']};classes=set()
 sites={k:set() for k in counts}
 for r in rows:
  required={'review_status','annotator','reviewer','split','building_group','source_sha256','image_path','mask_path'}
  if not required<=r.keys() or any(not isinstance(r[k],str) for k in required):raise ValueError('Missing required manifest fields')
  if r['review_status']!='approved' or not r['annotator'].strip() or not r['reviewer'].strip():raise ValueError('Unapproved mask or missing review identity')
  if r['split'] not in counts:raise ValueError('Invalid split')
  site=r.get('site','').strip()
  if site and site not in {'Erbil','Amedi'}:raise ValueError('Invalid site; expected Erbil or Amedi')
  if require_sites and not site:raise ValueError('Site is required for the two-city study')
  if site:sites[r['split']].add(site)
  g=r['building_group'].strip()
  if not g or g=='UNASSIGNED':raise ValueError('Missing building group')
  if site and g in group_sites and group_sites[g]!=site:raise ValueError('Conflicting site for the same building group')
  if site:group_sites[g]=site
  if g in groups and groups[g]!=r['split']:raise ValueError('Building leakage across splits')
  groups[g]=r['split']
  identity=r['source_sha256']
  if hashlib.sha256((root/r['image_path']).read_bytes()).hexdigest()!=identity:raise ValueError('Source hash mismatch')
  if identity in seen:raise ValueError('Duplicate source image; create patches after source split validation')
  seen.add(identity)
  with Image.open(root/r['image_path']) as im, Image.open(root/r['mask_path']) as ma:
   a=np.asarray(ma).copy();dimensions_match=ma.size==im.size
  if not dimensions_match or a.ndim!=2:raise ValueError('Mask dimensions or channels invalid')
  labels=set(np.unique(a).tolist())
  if not labels<={0,1,2,3,255}:raise ValueError('Invalid label ID')
  if labels=={255}:raise ValueError('Entire image ignored')
  if r['split']=='train':classes|=labels-{255}
  counts[r['split']]+=1
 if any(v==0 for v in counts.values()):raise ValueError('All three splits required')
 if classes!={0,1,2,3}:raise ValueError('Training split missing one or more classes')
 if require_sites and any(s!={'Erbil','Amedi'} for s in sites.values()):raise ValueError('Both Erbil and Amedi are required in each split for the two-city study')
 return counts
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--require-sites',action='store_true');a=p.parse_args()
 try:
  with a.manifest.open(newline='') as stream:rows=list(csv.DictReader(stream))
  counts=validate(rows,a.manifest.resolve().parent,require_sites=a.require_sites)
 except (ValueError,OSError,KeyError,csv.Error) as error:p.exit(2,f'Manifest unavailable: {error}\n')
 print(counts)
