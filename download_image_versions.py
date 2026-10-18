"""Resumable collection of API-advertised Wikimedia image derivatives."""
import csv,hashlib,io,json,time,urllib.request,urllib.parse,urllib.error,zipfile,shutil
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/raw/image-versions';OUT.mkdir(parents=True,exist_ok=True)
ROWS=list(csv.DictReader(open(ROOT/'selected-800-candidates.csv')))
UA='HeritageDatasetResearch/1.0 (https://github.com/Dk963201/heritage-damage-segmentation-public)'
def request(url):
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':UA}),timeout=90) as r:return r.read()
 except urllib.error.HTTPError as e:
  if e.code in (403,429):
   (OUT/'STOP.json').write_text(json.dumps({'status':e.code,'retry_after':e.headers.get('Retry-After'),'time':time.time(),'url':url}));raise SystemExit(f'STOP HTTP {e.code}; Retry-After={e.headers.get("Retry-After")}')
  raise
meta_path=OUT/'image-version-urls.json'
meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
for start in range(0,len(ROWS),40):
 missing=[r['title'] for r in ROWS[start:start+40] if r['title'] not in meta]
 if not missing:continue
 params={'action':'query','format':'json','titles':'|'.join(missing),'prop':'imageinfo','iiprop':'url','iiurlwidth':1280}
 d=json.loads(request('https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params)))
 for p in d.get('query',{}).get('pages',{}).values():
  if p.get('imageinfo'):meta[p['title']]=p['imageinfo'][0]
 meta_path.write_text(json.dumps(meta));print('Metadata',len(meta),flush=True);time.sleep(1)
ledger_path=OUT/'download-ledger.json'
ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
from concurrent.futures import ThreadPoolExecutor
import threading
request_lock=threading.Lock()
last_start=[0.0]
stop=threading.Event()
def prepare(item):
 index,row=item
 if stop.is_set():raise RuntimeError('Transfer stopped')
 key=f'{index:04}'
 info=meta.get(row['title'])
 if not info:raise RuntimeError('Missing image metadata')
 url=info.get('responsiveUrls',{}).get('2',info.get('thumburl',info['url']))
 if urllib.parse.urlparse(url).hostname not in ('thumb.wikimedia.org','upload.wikimedia.org'):raise RuntimeError('Unexpected image host')
 with request_lock:
  time.sleep(max(0,2-(time.monotonic()-last_start[0])))
  if stop.is_set():raise RuntimeError('Transfer stopped')
  last_start[0]=time.monotonic()
 try:data=request(url)
 except BaseException:
  stop.set();raise
 with Image.open(io.BytesIO(data)) as image:
  image.load();width,height=image.size;fmt=image.format
 if fmt not in ('JPEG','PNG','MPO','WEBP'):raise RuntimeError('Unexpected image format: '+str(fmt))
 archive=f'heritage-versions-{(index-1)//80+1:02}.zip'
 name=f'images/{key}.'+{'JPEG':'jpg','MPO':'jpg','PNG':'png','WEBP':'webp'}[fmt]
 rec=dict(row,image_id=key,download_url=url,download_width=width,download_height=height,download_sha256=hashlib.sha256(data).hexdigest(),download_bytes=len(data),archive=archive,archive_path=name,version_note='API-advertised image version; original URL and source hash retained separately; visual review and annotation pending')
 return key,rec,data
pending=[(i,r) for i,r in enumerate(ROWS,1) if f'{i:04}' not in ledger]
with ThreadPoolExecutor(max_workers=2) as pool:
 for key,rec,data in pool.map(prepare,pending,buffersize=2):
  if shutil.disk_usage(OUT).free<2_000_000_000:
   stop.set();raise SystemExit('STOP low disk space')
  with zipfile.ZipFile(OUT/rec['archive'],'a',compression=zipfile.ZIP_STORED) as z:
   if rec['archive_path'] not in z.namelist():
    z.writestr(rec['archive_path'],data);z.writestr(f'attribution/{key}.json',json.dumps(rec,indent=2))
  ledger[key]=rec;ledger_path.write_text(json.dumps(ledger,indent=2))
  print(f'Downloaded {len(ledger)}/800; {key}: {rec["download_width"]}x{rec["download_height"]}',flush=True)
print('COMPLETE 800/800',flush=True)
