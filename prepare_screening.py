"""Download the pinned model and extract the pilot images from existing archives."""
import csv,io
from pathlib import Path
from zipfile import ZipFile
from PIL import Image
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent
REV='8ef2caddfdead307fe7f3a4ed596baca7eab8a2f'
if __name__=='__main__':
    import psutil, shutil
    if psutil.virtual_memory().total < 12 * 1024**3 or shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise SystemExit('Use a larger-memory machine with at least 12 GiB free disk before downloading this pilot model.')
    model=ROOT/'models/moondream2'
    snapshot_download('vikhyatk/moondream2',revision=REV,allow_patterns=['*.py','*.json','*.txt','model.safetensors'],local_dir=str(model))
    # Work around transformers 4.49 missing nested relative modules (same approach as zoo plugin).
    cache=Path.home()/'.cache/huggingface/modules/transformers_modules/moondream2'
    cache.mkdir(parents=True,exist_ok=True)
    for p in model.glob('*.py'):
        dest=cache/p.name
        if not dest.exists(): dest.symlink_to(p.resolve())
    ids={r['image_id'] for r in csv.DictReader((ROOT/'screening-reference.csv').open())}
    out=ROOT/'data/processed/screening-sample'; out.mkdir(parents=True,exist_ok=True)
    for r in csv.DictReader((ROOT/'downloaded-800-manifest.csv').open()):
        if r['image_id'] not in ids: continue
        with ZipFile(ROOT/'data/raw/image-versions'/Path(r['archive']).name) as z:
            im=Image.open(io.BytesIO(z.read(r['archive_path']))).convert('RGB')
            im.save(out/f"{r['image_id']}.jpg",quality=95)
