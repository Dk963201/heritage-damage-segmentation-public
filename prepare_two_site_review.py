"""Audit local candidate photos and make a pending, human-reviewed two-site pack.

No inference, mask generation, approval or automatic split assignment occurs.
Full-resolution originals remain in their existing archives/files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import textwrap
import zipfile
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REVIEWED = {"0450", "0650", "0750"}
DEMO_IDS = {"0340", "0756"}
SELECTED = ("0018", "0021", "0022", "0023", "0024", "0725", "0736", "0751", "0777")
NOTES = {
    "0018": "Carved gateway and stone steps. Check joints versus fissures. Likely the same gateway as0021; confirm grouping.",
    "0021": "Closer gateway view with stepped masonry. Likely the same gateway as0018; confirm grouping.",
    "0022": "Qubehan school courtyard, masonry arches and vegetation. Compare structure identity with0023 and0024.",
    "0023": "Qubehan school wall, openings and vegetation. Compare structure identity with0022 and0024.",
    "0024": "Qubehan school wall and fallen stones. Decide annotation scope for rubble and vegetation; compare0022/0023.",
    "0725": "Broad brick wall with decorative openings. Distinguish openings and mortar from damage; review small details at native resolution.",
    "0736": "Decorated masonry facade and adjacent buildings. Decide which surfaces are eligible; exclude sky and unrelated background.",
    "0751": "Brick building facade with deep window shadows and branches. Check for overlap with previously reviewed0750 before any split.",
    "0777": "Tall brick walls with buttresses, shadows and support poles. Distinguish joints, boundaries and shadows from cracks.",
}
GENERATED = ("data-audit.json", "image-audit.csv", "source-list.csv", "pending-four-class.csv",
             "pending-severity.csv", "review-records.json", "contact-sheet.jpg", "review.html")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_id", "site", "title", "source_url", "author", "license", "license_url",
                "archive", "archive_path", "download_sha256", "download_width", "download_height", "download_bytes"}
    seen = set()
    for row in rows:
        if not required <= row.keys():
            raise ValueError("Manifest is missing required provenance fields")
        ident = row["image_id"]
        if not re.fullmatch(r"\d{4}", ident) or ident in seen:
            raise ValueError("Invalid or duplicate image ID")
        seen.add(ident)
        if not re.fullmatch(r"[0-9a-f]{64}", row["download_sha256"]):
            raise ValueError(f"Invalid SHA256 for{ident}")
        if row["site"] not in {"Erbil", "Amedi"}:
            raise ValueError(f"Unknown source-collection site for{ident}")
        for key in ("download_width", "download_height", "download_bytes"):
            if int(row[key]) <= 0:
                raise ValueError(f"Invalid positive value for{ident}:{key}")
        if not row["source_url"].startswith("https://commons.wikimedia.org/wiki/"):
            raise ValueError(f"Unexpected source URL for{ident}")
        if row["license_url"] and not row["license_url"].startswith(("https://", "http://")):
            raise ValueError(f"Invalid license URL for{ident}")
    if not rows:
        raise ValueError("Empty source manifest")
    return rows


def archive_paths(root: Path, row: dict) -> tuple[Path, str]:
    name = row["archive"]
    member = PurePosixPath(row["archive_path"])
    if Path(name).name != name or not name.endswith(".zip") or "\\" in name:
        raise ValueError("Archive must be a local filename")
    if member.is_absolute() or ".." in member.parts or "\\" in row["archive_path"]:
        raise ValueError("Unsafe archive member")
    return root / "data/raw/image-versions" / name, str(member)


def read_verified_image(root: Path, row: dict, archives: dict) -> tuple[bytes, dict]:
    archive, member = archive_paths(root, row)
    if str(archive) not in archives:
        raise ValueError("Archive not opened by caller")
    z = archives[str(archive)]
    info = z.getinfo(member)
    if info.file_size != int(row["download_bytes"]):
        raise ValueError(f"Archive byte count mismatch for{row['image_id']}")
    data = z.read(member)
    if hashlib.sha256(data).hexdigest() != row["download_sha256"]:
        raise ValueError(f"Source SHA256 mismatch for{row['image_id']}")
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        image.verify()
    if (width, height) != (int(row["download_width"]), int(row["download_height"])):
        raise ValueError(f"Image dimensions mismatch for{row['image_id']}")
    return data, {"image_id": row["image_id"], "site": row["site"], "archive": row["archive"],
                  "archive_member": member, "sha256": row["download_sha256"], "bytes": len(data),
                  "width": width, "height": height, "status": "hash_bytes_dimensions_verified"}


def validate_selection(rows: list[dict], ids=SELECTED) -> list[dict]:
    lookup = {r["image_id"]: r for r in rows}
    if len(ids) != 9 or len(set(ids)) != 9 or set(ids) & (REVIEWED | DEMO_IDS):
        raise ValueError("Select nine unique photos excluding reviewed photos and prior prediction demos")
    selected = [lookup[i] for i in ids]
    if Counter(r["site"] for r in selected) != {"Amedi": 5, "Erbil": 4}:
        raise ValueError("Review pack must contain five Amedi and four Erbil photos")
    reviewed_hashes = {lookup[i]["download_sha256"] for i in REVIEWED if i in lookup}
    hashes = [r["download_sha256"] for r in selected]
    if len(set(hashes)) != 9 or set(hashes) & reviewed_hashes:
        raise ValueError("Duplicate or previously reviewed source hash in pack")
    return selected


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_reviewed(root: Path) -> dict:
    protocol_path = root / "local-results/reviewed-through-333/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    records = []
    for version in protocol["versions"]:
        ident = version["id"]
        image_path, mask_path = root / version["image"], root / version["mask"]
        if sha(image_path) != version["image_sha256"] or sha(mask_path) != version["mask_sha256"]:
            raise ValueError("Previously reviewed data changed; inspect before preparing pack")
        mask = np.asarray(Image.open(mask_path))
        if set(np.unique(mask)) - {0, 1, 255}:
            raise ValueError("Unexpected labels in binary review snapshot")
        counts = {str(k): int((mask == k).sum()) for k in (0, 1, 255)}
        records.append({"image_id": ident, "site": "Erbil", "mask_path": version["mask"],
                        "mask_sha256": version["mask_sha256"], "counts": counts,
                        "status": "partial_user_reviewed_binary_only_not_approved_four_class"})
    pending = root / "data/processed/review-0750/batch-27"
    manifest_path = pending / "review-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["pending_region_ids"] != [334, 335, 336] or manifest["accepted_region_ids"] != list(range(328, 334)):
        raise ValueError("Review stop state changed; inspect before preparing pack")
    mask = np.asarray(Image.open(pending / "partial-user-reviewed.png"))
    for ident in manifest["pending_region_ids"]:
        selected = np.asarray(Image.open(pending / f"region-{ident}-selection.png")) > 0
        if not np.all(mask[selected] == 255):
            raise ValueError("Pending pixels must remain ignored")
    return {"source_protocol_sha256": sha(protocol_path), "photos": records,
            "photo_count": len(records), "site_counts": {"Erbil": len(records), "Amedi": 0},
            "reviewed_binary_pixels": sum(r["counts"]["0"] + r["counts"]["1"] for r in records),
            "binary_intact_pixels": sum(r["counts"]["0"] for r in records),
            "binary_crack_pixels": sum(r["counts"]["1"] for r in records),
            "pending_region_ids_preserved": [334, 335, 336],
            "pending_manifest_sha256": sha(manifest_path),
            "four_class_approval": False, "severity_labels_in_these_records": 0}


def audit_label_manifests(root: Path) -> dict:
    """Inspect actual task manifests, excluding synthetic tests and external datasets."""
    paths = sorted(root.glob("*.csv"))
    annotations = root / "data/annotations"
    if annotations.exists():
        paths += sorted(annotations.rglob("*.csv"))
    results = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            if not (({"mask_path", "review_status"} <= set(fields)) or any("severity" in f.lower() for f in fields)):
                continue
            rows = list(reader)
        approved = sum(r.get("review_status", "").strip().lower() == "approved" for r in rows)
        results.append({"path": str(path.relative_to(root)), "rows": len(rows),
                        "approved_row_count": approved, "sha256": sha(path), "fields": fields})
    return {"scope": "Root CSV manifests and data/annotations CSVs; excludes synthetic fixtures, public binary datasets and prediction outputs",
            "inspected": results, "approved_rows_found": sum(r["approved_row_count"] for r in results),
            "interpretation": "No approved four-class/severity labels are inferred from source titles, binary selections or predictions. A nonzero approval count requires a separate rubric/mask validation."}


def font(size: int):
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def existing_source(root: Path, row: dict) -> str:
    candidates = [root / f"local-results/wall-candidate-screen/{row['image_id']}-source.jpg",
                  root / f"data/processed/review-{row['image_id']}/{row['image_id']}-source.jpg"]
    for path in candidates:
        if path.is_file() and sha(path) == row["download_sha256"]:
            return str(path.relative_to(root))
    return ""


def prepare(root: Path, output: Path) -> dict:
    rows = read_manifest(root / "downloaded-800-manifest.csv")
    selected = validate_selection(rows)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in GENERATED) or (output / "thumbnails").exists():
        raise FileExistsError("Output contains a review pack; use a fresh directory to preserve human edits")
    reviewed_before = audit_reviewed(root)
    records, selected_data, archives_used, failures = [], {}, {}, []
    with ExitStack() as stack:
        archives = {}
        for row in rows:
            try:
                archive, member = archive_paths(root, row)
                if str(archive) not in archives:
                    archives[str(archive)] = stack.enter_context(zipfile.ZipFile(archive))
                    archives_used[row["archive"]] = {"path": str(archive.relative_to(root)), "present": True}
                data, rec = read_verified_image(root, row, archives)
                records.append(rec)
                if row["image_id"] in SELECTED:
                    selected_data[row["image_id"]] = data
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                failures.append({"image_id": row["image_id"], "error": str(exc)})
    audit = {"audited_at_utc": datetime.now(timezone.utc).isoformat(),
             "manifest_path": "downloaded-800-manifest.csv", "manifest_sha256": sha(root / "downloaded-800-manifest.csv"),
             "candidate_count": len(rows), "source_collection_site_counts": dict(Counter(r["site"] for r in rows)),
             "site_count_caveat": "These are source-collection labels, not verified counts of heritage buildings. Amedi collection includes regional landscapes, events and non-building imagery.",
             "unique_image_ids": len({r["image_id"] for r in rows}),
             "unique_download_hashes": len({r["download_sha256"] for r in rows}),
             "unique_original_sha1": len({r["sha1"] for r in rows}),
             "license_counts": dict(Counter(r["license"] for r in rows)),
             "missing_license_url_ids": [r["image_id"] for r in rows if not r["license_url"]],
             "archives": archives_used, "verified_image_count": len(records), "failures": failures,
             "verification": "Read every archive member; compare SHA256 and byte count, open/verify image header and check dimensions. Does not establish source relevance or pixel-label truth.",
             "existing_reviewed_data": reviewed_before, "label_manifest_audit": audit_label_manifests(root),
             "selected_ids": list(SELECTED), "selected_site_counts": dict(Counter(r["site"] for r in selected)),
             "selection": "Purposeful title/content screening before any pack predictions. No predictions generated. Not random or an independent test selection.",
             "screened_then_excluded": {"0001": "Rock relief; masonry-rubric applicability unclear", "0004": "Rock approach and steps; wall annotation scope less useful"},
             "unknown_building_groups": "UNASSIGNED for all nine; likely same-structure pairs must be confirmed and kept in the same split",
             "approved_four_class_masks_added": 0, "severity_labels_added": 0, "predictions_generated": 0}
    (output / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_csv(output / "image-audit.csv", records, ["image_id", "site", "archive", "archive_member", "sha256", "bytes", "width", "height", "status"])
    if failures:
        raise ValueError("Dataset audit found missing/corrupt files; see data-audit.json. No review pack generated.")
    (output / "thumbnails").mkdir()
    source_rows, pending_four, pending_severity, review_records = [], [], [], []
    sheet = Image.new("RGB", (1530, 1620), "#f6f7f9")
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 14), "Nine source photos | 5 Amedi + 4 Erbil | PENDING REVIEW", font=font(29), fill="#18212f")
    draw.text((24, 56), "No model overlays or approved labels. Confirm building groups before splitting data.", font=font(20), fill="#48566a")
    cards = []
    for n, row in enumerate(selected):
        ident = row["image_id"]
        im = Image.open(io.BytesIO(selected_data[ident])).convert("RGB")
        im.thumbnail((470, 290), Image.Resampling.LANCZOS)
        im.save(output / f"thumbnails/{ident}.jpg", quality=93)
        x, y = 15 + (n % 3) * 510, 105 + (n // 3) * 490
        draw.rounded_rectangle((x, y, x + 495, y + 475), radius=10, fill="white", outline="#d4dce7")
        draw.text((x + 12, y + 10), f"{n+1}. {ident} | {row['site']}", font=font(25), fill="#18212f")
        sheet.paste(im, (x + (495-im.width)//2, y + 49))
        title = row["title"].removeprefix("File:")
        lines = textwrap.wrap(title, width=53)[:2]
        draw.multiline_text((x + 12, y + 348), "\n".join(lines), font=font(16), fill="#253346", spacing=3)
        credit = textwrap.fill(f"{row['author']} | {row['license']}", width=54)
        draw.multiline_text((x + 12, y + 402), credit, font=font(15), fill="#526174", spacing=2)
        ref = existing_source(root, row)
        src = {**row, "repo_relative_existing_image_path": ref, "thumbnail_path": f"thumbnails/{ident}.jpg",
               "building_group": "UNASSIGNED", "split": "UNASSIGNED", "review_status": "pending",
               "visual_screening_note": NOTES[ident], "full_source_copied": False}
        source_rows.append(src)
        common = {"image_id": ident, "site": row["site"], "source_sha256": row["download_sha256"],
                  "archive": row["archive"], "archive_member": row["archive_path"],
                  "repo_relative_existing_image_path": ref, "building_group": "UNASSIGNED", "split": "UNASSIGNED",
                  "review_status": "pending", "annotator": "", "reviewer": "", "review_notes": ""}
        pending_four.append({**common, "mask_path": "", "mask_sha256": "", "rubric_version": "", "classes_present": ""})
        pending_severity.append({**common, "rubric_path": "", "rubric_version": "", "severity_label": "", "severity_evidence": ""})
        review_records.append({"image_id": ident, "site": row["site"], "source_sha256": row["download_sha256"],
                               "building_group": "UNASSIGNED", "split": "UNASSIGNED", "relevance_review": "pending",
                               "four_class": {"approved": False, "mask_path": None, "classes_present": None,
                                              "rubric_version": None, "annotator": None, "reviewer": None},
                               "severity": {"approved": False, "rubric_version": None, "label": None,
                                            "evidence": None, "annotator": None, "reviewer": None}})
        esc = html.escape
        cards.append(f'<article><h2>{n+1}. {ident} · {row["site"]}</h2><a href="{esc(row["source_url"],quote=True)}" target="_blank" rel="noopener noreferrer"><img src="thumbnails/{ident}.jpg" alt="{esc(title,quote=True)}"></a><p>{esc(NOTES[ident])}</p><p class="pending">Pending · group UNASSIGNED · split UNASSIGNED</p><p class="credit">{esc(row["author"])} · <a href="{esc(row["license_url"],quote=True)}">{esc(row["license"])}</a> · <a href="{esc(row["source_url"],quote=True)}">Source and original resolution</a></p></article>')
    draw.text((24, 1580), "Changes: resized thumbnails and layout only. Full source links and credits in review.html / source-list.csv.", font=font(17), fill="#48566a")
    sheet.save(output / "contact-sheet.jpg", quality=94)
    source_fields = ["image_id", "site", "title", "author", "license", "license_url", "source_url", "download_url",
                     "download_sha256", "download_width", "download_height", "archive", "archive_path",
                     "repo_relative_existing_image_path", "thumbnail_path", "building_group", "split", "review_status",
                     "visual_screening_note", "full_source_copied"]
    write_csv(output / "source-list.csv", source_rows, source_fields)
    write_csv(output / "pending-four-class.csv", pending_four, list(pending_four[0]))
    write_csv(output / "pending-severity.csv", pending_severity, list(pending_severity[0]))
    (output / "review-records.json").write_text(json.dumps(review_records, indent=2) + "\n")
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; object-src 'none'"><meta name="referrer" content="no-referrer"><title>Heritage Damage Study — Two-site source review</title><style>body{font:17px system-ui;background:#f6f7f9;color:#18212f;margin:0}main{max-width:1500px;padding:24px;margin:auto}h1{font-size:30px}p,li{line-height:1.5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}article{background:white;padding:18px;border:1px solid #d4dce7;border-radius:12px}img{width:100%;height:290px;object-fit:contain}h2{font-size:23px}.pending{color:#7c4a10}.credit{font-size:14px}a{color:#2357a0}</style></head><body><main><h1>Review nine source photos</h1><p>Five Amedi and four Erbil source photos, selected for visible structures. These are resized originals with no predictions or approved labels. Full originals remain in the existing local archives; source-list.csv records their locations. Click a photo for the source page and full resolution.</p><ol><li>Confirm each photo is relevant and readable; exclude unsuitable views.</li><li>Confirm building identity and near-duplicate groups.0018/0021 likely show one gateway;0022–0024 likely show Qubehan school. Compare0751 with the existing0750 photo. Keep every view/crop of a group in one split. Groups remain UNASSIGNED until reviewed.</li><li>Agree the four-class rubric, including plaster, rubble, shadows, joints, natural stone, background and ambiguous damage. Existing software IDs are0Healthy Masonry,1Structural Crack,2Surface Erosion,3Vegetation Overgrowth;255means ignore. The name Structural Crack does not establish structural safety from a photograph.</li><li>Annotate native-resolution masks and record reviewer identities. A photo-level keep decision does not approve any mask or prove that a class appears.</li><li>Obtain the supervisor-approved visual-severity rubric before assigning any severity label. Do not convert the colored area or crack percentage into a grade.</li></ol><p><strong>Human review is still required.</strong> These pending files are not training manifests. Resolve archive references into actual image paths only when preparing reviewed training data; do not feed empty masks, UNASSIGNED groups, or blank severity labels to training.</p><p><a href="contact-sheet.jpg">Contact sheet</a> · <a href="source-list.csv">Source list</a> · <a href="pending-four-class.csv">Pending segmentation records</a> · <a href="pending-severity.csv">Pending severity records</a> · <a href="data-audit.json">Data audit</a></p><section class="grid">''' + "".join(cards) + '''</section><p>Source credits and license links appear on every card and in source-list.csv. Changes: resized thumbnails and layout only; no image content or labels changed. Derived contact-sheet layout: CC BY-SA4.0. Site names are source-collection metadata; nine photos do not establish nine independent buildings or complete class coverage. No model score or Amedi performance is claimed.</p></main></body></html>'''
    (output / "review.html").write_text(page)
    if audit_reviewed(root) != reviewed_before:
        raise ValueError("Existing reviewed labels changed during preparation")
    return {"output": str(output), "verified_images": len(records), "selected_ids": list(SELECTED),
            "selected_site_counts": audit["selected_site_counts"], "status": "pending_human_review"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "local-results/two-site-review-20261018")
    args = parser.parse_args()
    try:
        result = prepare(ROOT, args.output)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Review preparation failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
