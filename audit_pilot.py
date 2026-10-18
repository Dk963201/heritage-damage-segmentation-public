"""Audit the original Zenodo archives without extracting files."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
from PIL import Image

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "masonry_crack_images.zip": "434dd20461bc9ea5332beca8750e538e",
    "masonry_crack_masks.zip": "55e28405cd2d716f79da0b5bf1dcea7c",
}


def main():
    archives = {}
    for name, expected in EXPECTED.items():
        raw = (ROOT / "data/raw" / name).read_bytes()
        if hashlib.md5(raw).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
        archives[name] = {"md5": expected, "sha256": hashlib.sha256(raw).hexdigest()}
    pairs = []
    hashes = set()
    with ZipFile(ROOT / "data/raw/masonry_crack_images.zip") as images, ZipFile(ROOT / "data/raw/masonry_crack_masks.zip") as masks:
        names = sorted(n for n in images.namelist() if n.lower().endswith(".jpg"))
        expected_masks = {Path(n).stem + "_mask.png" for n in names}
        if expected_masks != set(masks.namelist()) or len(names) != 100:
            raise ValueError("Unexpected count or unmatched image/mask files")
        for name in names:
            raw = images.read(name)
            digest = hashlib.sha256(raw).hexdigest()
            if digest in hashes:
                raise ValueError(f"Duplicate image: {name}")
            hashes.add(digest)
            with Image.open(io.BytesIO(raw)) as image, Image.open(io.BytesIO(masks.read(Path(name).stem + "_mask.png"))) as mask:
                image.load()
                mask.load()
                if image.size != mask.size or mask.mode != "L":
                    raise ValueError(f"Dimension/mode mismatch: {name}")
                histogram = mask.histogram()
                if any(histogram[1:255]):
                    raise ValueError(f"Unexpected label values: {name}")
                pairs.append({"image": name, "width": image.width, "height": image.height,
                              "sha256": digest, "crack_fraction": histogram[255] / (image.width * image.height)})
    report = {"source": "https://zenodo.org/records/18458458", "license": "CC-BY-4.0",
              "scope": "UK binary crack pilot; not the required Kurdish four-class dataset",
              "checks": "Archive checksums, decoding, pairing, dimensions, labels and exact image duplicates; no visual annotation quality assessment",
              "pair_count": len(pairs), "archives": archives, "pairs": pairs}
    destination = ROOT / "pilot-audit.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(pairs)} image/mask pairs; wrote {destination.name}")


if __name__ == "__main__":
    main()
