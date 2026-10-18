"""Negative checks for source integrity and preserving pending review data."""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_two_site_review import (SELECTED, archive_paths, audit_reviewed, prepare,
                                    read_verified_image, sha, validate_selection)


def row(ident="0018", site="Amedi"):
    return {"image_id": ident, "site": site, "title": "File:Example wall.jpg",
            "source_url": "https://commons.wikimedia.org/wiki/File:Example_wall.jpg",
            "author": "Example", "license": "CC BY-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "archive": "example.zip", "archive_path": f"images/{ident}.png",
            "download_sha256": hashlib.sha256(ident.encode()).hexdigest(),
            "download_width": "2", "download_height": "2", "download_bytes": "1"}


class TwoSiteReviewTests(unittest.TestCase):
    def test_archive_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = row()
            stream = io.BytesIO()
            Image.new("RGB", (2, 2), "gray").save(stream, format="PNG")
            data = stream.getvalue()
            r["download_bytes"] = str(len(data))
            archive, member = archive_paths(root, r)
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(member, data)
            with zipfile.ZipFile(archive) as z:
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    read_verified_image(root, r, {str(archive): z})
                r["download_sha256"] = hashlib.sha256(data).hexdigest()
                actual, audit = read_verified_image(root, r, {str(archive): z})
            self.assertEqual(actual, data)
            self.assertEqual(audit["status"], "hash_bytes_dimensions_verified")

    def test_archive_path_escape_rejected(self):
        r = row()
        r["archive"] = "../../outside.zip"
        with self.assertRaises(ValueError):
            archive_paths(Path("/tmp/review"), r)
        r["archive"] = "example.zip"
        r["archive_path"] = "../outside.png"
        with self.assertRaises(ValueError):
            archive_paths(Path("/tmp/review"), r)

    def test_selection_balance_and_reviewed_exclusion(self):
        rows = [row(i, "Amedi" if n < 5 else "Erbil") for n, i in enumerate(SELECTED)]
        self.assertEqual(len(validate_selection(rows)), 9)
        rows[-1]["site"] = "Amedi"
        with self.assertRaisesRegex(ValueError, "five Amedi"):
            validate_selection(rows)
        with self.assertRaisesRegex(ValueError, "excluding reviewed"):
            validate_selection(rows + [row("0750", "Erbil")], SELECTED[:-1] + ("0750",))

    def test_existing_pending_review_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [row(i, "Amedi" if n < 5 else "Erbil") for n, i in enumerate(SELECTED)]
            with (root / "downloaded-800-manifest.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / "pack"
            output.mkdir()
            pending = output / "pending-four-class.csv"
            pending.write_text("USER REVIEW IN PROGRESS\n")
            before = pending.read_bytes()
            with self.assertRaises(FileExistsError):
                prepare(root, output)
            self.assertEqual(pending.read_bytes(), before)

    def test_pending_334_pixel_must_not_be_in_accepted_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versions = []
            for ident in ["0650", "0450", "0750"]:
                base = root / f"data/processed/review-{ident}"
                base.mkdir(parents=True)
                image = base / f"{ident}-source.jpg"
                Image.new("RGB", (2, 2), "gray").save(image)
                mask_path = base / "mask.png"
                if ident == "0750":
                    mask_path = base / "batch-27/partial-user-reviewed.png"
                    mask_path.parent.mkdir()
                Image.fromarray(np.array([[0, 1], [255, 255]], np.uint8)).save(mask_path)
                versions.append({"id": ident, "image": str(image.relative_to(root)),
                                 "image_sha256": sha(image), "mask": str(mask_path.relative_to(root)),
                                 "mask_sha256": sha(mask_path)})
            protocol = root / "local-results/reviewed-through-333/protocol.json"
            protocol.parent.mkdir(parents=True)
            protocol.write_text(json.dumps({"versions": versions}))
            pending = root / "data/processed/review-0750/batch-27"
            (pending / "review-manifest.json").write_text(json.dumps({
                "pending_region_ids": [334, 335, 336], "accepted_region_ids": list(range(328, 334))}))
            for ident in [334, 335, 336]:
                selection = np.array([[0, 0], [255, 0]], np.uint8)
                Image.fromarray(selection).save(pending / f"region-{ident}-selection.png")
            self.assertEqual(audit_reviewed(root)["pending_region_ids_preserved"], [334, 335, 336])
            # Pretend the pending selection intersects a valid reviewed pixel.
            Image.fromarray(np.array([[0, 255], [0, 0]], np.uint8)).save(pending / "region-334-selection.png")
            with self.assertRaisesRegex(ValueError, "Pending pixels"):
                audit_reviewed(root)


if __name__ == "__main__":
    unittest.main()
