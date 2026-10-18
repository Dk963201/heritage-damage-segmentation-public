"""Fetch only pinned public dataset PNGs and license, not third-party code."""
from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures
REPO='dimitrisdais/crack_detection_CNN_masonry'
REV='83c371c70a27852f22ab3386269c6058a203ef38'
def main():
 root=Path('data/raw/dais-public');root.mkdir(parents=True,exist_ok=True)
 tree=json.loads(Path('public-dais-tree.json').read_text());assert tree['sha']==REV
 entries=[r for r in tree['tree'] if r['type']=='blob' and (r['path'].startswith('dataset/') and r['path'].endswith('.png') or r['path']=='LICENSE')]
 def fetch(r):
  name=r['path'];target=root/name;target.parent.mkdir(parents=True,exist_ok=True);url=f'https://raw.githubusercontent.com/{REPO}/{REV}/{name}'
  b=target.read_bytes() if target.exists() else urllib.request.urlopen(url,timeout=60).read()
  assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==r['sha'],name
  if not target.exists():target.write_bytes(b)
  return {'path':name,'source':url,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':r['sha']}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,entries))
 Path('public-masonry-download.json').write_text(json.dumps({'repository':REPO,'commit':REV,'files':rows},indent=2));print('Verified downloaded files:',len(rows))
if __name__=='__main__':main()
