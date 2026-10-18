import json, urllib.request, urllib.parse, time, csv, re
from pathlib import Path
OUT=Path(__file__).resolve().parent
BASE='https://commons.wikimedia.org/w/api.php'
def get(params):
 req=urllib.request.Request(BASE+'?'+urllib.parse.urlencode(params),headers={'User-Agent':'HeritageDatasetResearch/1.0 (educational source inventory)'})
 with urllib.request.urlopen(req,timeout=90) as r:return json.load(r)
rows=[]
for cat,site in [('Citadel of Erbil','Erbil'),('Amadiya','Amedi')]:
 params=dict(action='query',format='json',generator='categorymembers',gcmtitle='Category:'+cat,gcmtype='file',gcmlimit=500,prop='imageinfo',iiprop='url|size|sha1|extmetadata')
 while True:
  data=get(params)
  for p in data.get('query',{}).get('pages',{}).values():
   info=p.get('imageinfo',[{}])[0]; meta=info.get('extmetadata',{})
   def val(k):return re.sub('<[^>]+>','',meta.get(k,{}).get('value',''))
   rows.append(dict(title=p['title'],site=site,source_url=info.get('descriptionurl',''),original_url=info.get('url',''),width=info.get('width',0),height=info.get('height',0),bytes=info.get('size',0),sha1=info.get('sha1',''),license=val('LicenseShortName'),license_url=val('LicenseUrl'),author=val('Artist'),description=val('ImageDescription'),review_status='Unreviewed candidate; no masks; no damage-class verification'))
  if 'continue' not in data:break
  params.update(data['continue']); time.sleep(.3)
 print(site, sum(r['site']==site for r in rows),flush=True)
seen=set(); eligible=[]
for r in rows:
 if r['sha1'] in seen or not urllib.parse.urlparse(r['original_url']).path.lower().endswith(('.jpg','.jpeg','.png')):continue
 if min(r['width'],r['height'])<1500:continue
 if not (r['license'].startswith(('CC BY','CC0')) or r['license']=='Public domain'):continue
 seen.add(r['sha1']);eligible.append(r)
eligible.sort(key=lambda r:(0 if r['site']=='Amedi' else 1,0 if any(w in r['title'].lower() for w in ['wall','interior','gate','building']) else 1,r['title']))
for name,items in [('all-source-candidates.csv',rows),('selected-800-candidates.csv',eligible[:800])]:
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(items)
summary={'checked_date':'2026-10-18','all_count':len(rows),'eligible_unique_high_resolution_candidates':len(eligible),'selected_count':len(eligible[:800]),'selected_by_site':{s:sum(r['site']==s for r in eligible[:800]) for s in ['Erbil','Amedi']},'estimated_original_bytes':sum(r['bytes'] for r in eligible[:800]),'status':'Links and license metadata inventoried; photos not downloaded or visually validated; not an annotated dataset','resolution_rule':'Both dimensions >=1500px; provisional screening rule, not instructor confirmed'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
