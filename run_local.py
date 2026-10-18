"""Reproducible full-image audit and CLAHE preprocessing; no fabricated labels."""
import argparse, csv, hashlib, io, json, time, zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def starts(length, patch=256, stride=256):
    if length < patch:
        raise ValueError('Image smaller than patch')
    return sorted(set(list(range(0, length-patch+1, stride)) + [length-patch]))


def clahe_rgb(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archives', type=Path, default=Path('data/raw/image-versions'))
    p.add_argument('--output', type=Path, default=Path('local-results'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open('downloaded-800-manifest.csv')))
    if len(rows) != 800 or len({r['image_id'] for r in rows}) != 800:
        raise ValueError('Expected 800 unique manifest IDs')
    results, failures = [], []
    begin = time.monotonic()
    for i, row in enumerate(rows, 1):
        try:
            with zipfile.ZipFile(args.archives / row['archive']) as z:
                raw = z.read(row['archive_path'])
            digest = hashlib.sha256(raw).hexdigest()
            if digest != row['download_sha256']:
                raise ValueError('SHA256 mismatch')
            with Image.open(io.BytesIO(raw)) as im:
                rgb = np.array(im.convert('RGB'))
            h, w = rgb.shape[:2]
            if (w, h) != (int(row['download_width']), int(row['download_height'])):
                raise ValueError('Manifest dimensions mismatch')
            processed = clahe_rgb(rgb)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            dhash = sum(int(v) << n for n, v in enumerate((small[:, 1:] > small[:, :-1]).flat))
            results.append(dict(image_id=row['image_id'], site=row['site'], width=w, height=h,
                sha256=digest, dhash=f'{dhash:016x}', mean_brightness=round(float(gray.mean()), 3),
                clahe_mean_brightness=round(float(cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY).mean()), 3),
                patch_count=len(starts(w))*len(starts(h)), status='PASS'))
        except Exception as exc:
            failures.append({'image_id':row['image_id'], 'error':str(exc)})
        if i % 50 == 0:
            print(f'Processed {i}/800; failures={len(failures)}', flush=True)
    if results:
        with open(args.output/'image-audit.csv', 'w') as f:
            writer=csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    # Similarity flags are review candidates, never automatic duplicate removals.
    pairs=[]
    for i, a in enumerate(results):
        for b in results[i+1:]:
            distance=(int(a['dhash'],16)^int(b['dhash'],16)).bit_count()
            if distance <= 4:
                pairs.append({'image_a':a['image_id'],'image_b':b['image_id'],'hamming_distance':distance})
    with open(args.output/'similarity-review.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=['image_a','image_b','hamming_distance']);writer.writeheader();writer.writerows(pairs)
    summary=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), input_images=len(rows),
        passed=len(results), failures=failures, sites=dict(Counter(r['site'] for r in results)),
        unique_sha256=len({r['sha256'] for r in results}),
        total_indexable_256px_patches=sum(r['patch_count'] for r in results),
        similarity_pairs_for_manual_review=len(pairs), elapsed_seconds=round(time.monotonic()-begin,2),
        annotated_images=0, training_run=False, accuracy=None, mean_iou=None,
        status='PREPROCESSING COMPLETE; ANNOTATION REQUIRED' if not failures else 'FAILED')
    (args.output/'run-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if failures: raise SystemExit(1)

if __name__=='__main__': main()
