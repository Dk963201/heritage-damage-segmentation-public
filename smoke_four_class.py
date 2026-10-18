"""Run tiny synthetic end-to-end checks; never research evidence."""
import csv,hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
with tempfile.TemporaryDirectory() as d:
    root=Path(d);rows=[]
    for n,split in enumerate(['train','train','validation','test']):
        rng=np.random.default_rng(n);im=root/f'{n}.png';mask=root/f'{n}-mask.png'
        Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(im)
        Image.fromarray(np.tile(np.arange(64,dtype=np.uint8)%4,(64,1))).save(mask)
        rows.append(dict(image_path=im.name,mask_path=mask.name,source_sha256=hashlib.sha256(im.read_bytes()).hexdigest(),building_group=f'SYNTHETIC-{n}',split=split,review_status='approved',annotator='SOFTWARE_FIXTURE',reviewer='SOFTWARE_FIXTURE'))
    with (root/'manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    subprocess.run([str(Path('.venv-ml/bin/python').absolute()),'train_four_class.py',str(root/'manifest.csv'),'--output',str(root/'run'),'--epochs','1','--warmup','1','--patch','64','--random-init','--evaluate-test'],check=True)
    subprocess.run([str(Path('.venv-ml/bin/python').absolute()),'eda_four_class.py',str(root/'manifest.csv'),'--output',str(root/'eda.json')],check=True)
    eda=json.loads((root/'eda.json').read_text());assert eda['split_pixel_counts']['train']==[2048,2048,2048,2048,0]
    results=json.loads((root/'run/comparison.json').read_text())
    assert len(results)==3 and all(isinstance(x['test'],dict) for x in results)
    for result in results:
        name=result['model'];destination=root/f'prediction-{name}'
        subprocess.run([str(Path('.venv-ml/bin/python').absolute()),'predict_four_class.py','--checkpoint',str(root/f'run/{name}-best.pt'),'--protocol',str(root/'run/protocol.json'),'--model',name,'--image',str(root/'3.png'),'--output',str(destination)],check=True)
        metadata=json.loads((destination/'metadata.json').read_text())
        assert metadata['checkpoint_sha256']==result['checkpoint_sha256']
        assert metadata['included_pixels']==64*64 and metadata['num_classes']==4
        assert sum(item['pixels'] for item in metadata['class_summary'])==64*64
    Path('four-class-software-check.json').write_text(json.dumps({'scope':'SYNTHETIC SOFTWARE TEST ONLY; NOT HERITAGE TRAINING','architectures':[r['model'] for r in results],'passed':['manifest validation','paired patch loading','CLAHE','focal plus Dice backward','Adam warmup','checkpoint save/reload','full-source evaluation','comparison export','pixel balance/color/edge EDA','all three saved checkpoint inference paths'],'research_metrics':None},indent=2))
