import csv,io,json,zipfile,hashlib
from pathlib import Path
from PIL import Image,ImageDraw
p=Path(__file__).resolve().parent
ids=['0192','0250','0350','0450','0550','0650','0750']+[f'{i:04}' for i in range(200,800,20) if f'{i:04}' not in ['0250','0350','0450','0550','0650','0750']]
rows={r['image_id']:r for r in csv.DictReader((p/'downloaded-800-manifest.csv').open())}
out=p/'data/processed/annotation-queue';out.mkdir(parents=True,exist_ok=True)
queue=[]
for id in ids:
 r=rows[id]
 with zipfile.ZipFile(p/'data/raw/image-versions'/r['archive']) as z:
  b=z.read(r['archive_path']);assert hashlib.sha256(b).hexdigest()==r['download_sha256']
  im=Image.open(io.BytesIO(b)).convert('RGB');im.thumbnail((1200,1200));im.save(out/f'{id}-preview.jpg',quality=90)
 queue.append({'image_id':id,'source_url':r['source_url'],'source_sha256':r['download_sha256'],'status':'pending_visual_review','building_group':'UNASSIGNED','split':'UNASSIGNED','mask_status':'not_created','reviewer':'','notes':''})
for k in range(0,len(ids),12):
 grid=Image.new('RGB',(1200,1000),'white');d=ImageDraw.Draw(grid)
 for j,id in enumerate(ids[k:k+12]):
  im=Image.open(out/f'{id}-preview.jpg');im.thumbnail((395,220));x=(j%3)*400;y=(j//3)*250;grid.paste(im,(x,y+25));d.text((x+5,y+5),id,fill='black')
 grid.save(out/f'contact-{k//12+1}.jpg')
with (p/'annotation-queue.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=queue[0]);w.writeheader();w.writerows(queue)
print(len(queue),'previews prepared; no masks created')
