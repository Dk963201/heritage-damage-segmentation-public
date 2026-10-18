"""Audit public image/mask pairs; conservative duplicate screening, not scene proof."""
import io,json,zipfile,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def dh(im):
 a=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS));bits=(a[:,1:]>a[:,:-1]).reshape(-1);return sum(int(v)<<i for i,v in enumerate(bits))
def rgbhash(im):return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
def main():
 root=Path('data/raw/dais-public/dataset');meta=json.loads(Path('public-masonry-download.json').read_text());assert len(meta['files'])==481
 ref=[]
 with zipfile.ZipFile('data/raw/masonry_crack_images.zip') as z:
  for name in sorted(n for n in z.namelist() if n.endswith('.jpg')):
   im=Image.open(io.BytesIO(z.read(name))).convert('RGB');w,h=im.size
   for label,a in [('full',im),('center',im.crop(((w-256)//2,(h-256)//2,(w+256)//2,(h+256)//2)))]:ref.append((name+':'+label,dh(a),rgbhash(a)))
 seen={};rows=[];tiles=[]
 for i,p in enumerate(sorted((root/'crack_detection_224_images').glob('*.png'))):
  im=Image.open(p).convert('RGB');mp=root/'crack_detection_224_masks'/p.name;ma=Image.open(mp).convert('L');assert im.size==ma.size==(224,224);a=np.asarray(ma);vals=np.unique(a).tolist();assert set(vals)<={0,255},(p,vals)
  h=rgbhash(im);d=dh(im);exact=seen.get(h);seen[h]=p.name;matches=[n for n,x,r in ref if h==r or (d^x).bit_count()<=4]
  status='excluded_duplicate' if exact else ('hold_possible_overlap' if matches else 'eligible_training')
  rows.append({'image':str(p),'mask':str(mp),'name':p.name,'width':224,'height':224,'mask_values':vals,'crack_fraction':float((a==255).mean()),'rgb_sha256':h,'dhash':hex(d),'status':status,'duplicate_of':exact,'possible_uk_overlap':matches})
  if i%20==0:
   tile=Image.new('RGB',(448,248),'white');tile.paste(im,(0,24));tile.paste(ma.convert('RGB'),(224,24));ImageDraw.Draw(tile).text((4,4),p.name,fill='black');tiles.append(tile)
 canvas=Image.new('RGB',(448*3,248*4),'white')
 for i,t in enumerate(tiles[:12]):canvas.paste(t,((i%3)*448,(i//3)*248))
 canvas.save('public-masonry-audit-contact.jpg')
 out={'source':meta['repository'],'commit':meta['commit'],'pairs':len(rows),'eligible':sum(r['status']=='eligible_training' for r in rows),'duplicate_screen':'decoded RGB exact within new set; exact RGB and64-bit dHash Hamming<=4 against all UK full images and center crops; cannot rule out all cropped or same-scene matches','rows':rows}
 Path('public-masonry-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
