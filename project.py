"""Local entry point for the heritage study. Status separates code from evidence."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "check-segmentation-data": ("validate_training_manifest.py", [], ["--require-sites"]),
    "train-segmentation": ("train_four_class.py", [], ["--require-sites"]),
    "predict-segmentation": ("predict_four_class.py", [], []),
    "check-severity-data": ("severity_pipeline.py", ["validate"], []),
    "train-severity": ("severity_pipeline.py", ["train"], []),
    "predict-severity": ("severity_pipeline.py", ["predict"], []),
    "predict-cracks": ("predict_reviewed_cnn.py", [], []),
}


def read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def status(root: Path = ROOT) -> dict:
    """Report observable local evidence; file existence never proves accuracy."""
    manifest = root / "downloaded-800-manifest.csv"
    rows = list(csv.DictReader(manifest.open())) if manifest.exists() else []
    pilot = read_json(root / "local-results/reviewed-comparison.json")
    best = pilot.get("best_development_candidate", {}) if pilot else {}
    rubric = read_json(root / "severity-rubric-draft.json")
    software = {
        "four_class_training": (root / "train_four_class.py").is_file(),
        "four_class_inference": (root / "predict_four_class.py").is_file(),
        "severity_pipeline": (root / "severity_pipeline.py").is_file(),
        "two_site_review_preparation": (root / "prepare_two_site_review.py").is_file(),
    }
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local project evidence inventory; not a new model evaluation",
        "software_files_present": software,
        "candidate_image_rows": len(rows),
        "candidate_site_counts": dict(Counter(r.get("site", "unspecified") for r in rows)),
        "candidate_caution": "Collection labels require site/building verification. Downloads are not reviewed training masks.",
        "reviewed_binary_pilot": {
            "available": pilot is not None,
            "label_summary": pilot.get("labels") if pilot else None,
            "best_development_metrics": best.get("metrics"),
            "evidence": "Sparse labels from three Erbil photographs, repeatedly used for development; no independent test or Amedi accuracy claim.",
        },
        "severity_rubric_status": rubric.get("status", "not available") if rubric else "not available",
        "research_completion": "INCOMPLETE",
        "research_blockers": [
            "Reviewed four-class masks and verified building groups from both Erbil and Amedi.",
            "An approved visual-severity rubric and independently reviewed image-level severity labels.",
            "Target-domain training and a frozen, independent test for segmentation and severity, reported by site.",
            "A final research report based on those experiments and confirmation of the lecturer's dataset requirements.",
        ],
        "interpretation": "Runnable software and synthetic tests do not complete the research. No accuracy or severity grade is invented.",
    }


def markdown(report: dict) -> str:
    metrics = report["reviewed_binary_pilot"].get("best_development_metrics") or {}
    dice = metrics.get("dice")
    score = f"{100*dice:.2f}%" if dice is not None else "unavailable"
    lines = ["# Heritage study readiness", "", f"Checked: {report['checked_at_utc']}", "",
             "**Research status: incomplete.** This report inventories local evidence; it does not run evaluation.", "",
             "| Component | Software present | Research evidence still needed |", "|---|---|---|",
             f"| Four-class segmentation | {report['software_files_present']['four_class_training'] and report['software_files_present']['four_class_inference']} | Reviewed four-class masks, trained target models and held-out results |",
             f"| Severity classification | {report['software_files_present']['severity_pipeline']} | Approved rubric, reviewed severity labels and held-out results |",
             f"| Both cities | {report['software_files_present']['two_site_review_preparation']} | Verified buildings and reviewed labels from both cities |",
             "| Three CNN architectures | U-Net, DeepLabV3+, FPN training code | Fair target-domain comparison after annotations |", "",
             f"Candidate records: {report['candidate_image_rows']}; collection-site counts: {report['candidate_site_counts']}.", "",
             f"Existing binary development crack Dice: **{score}**. This is not four-class or severity accuracy.", "",
             f"Severity rubric: **{report['severity_rubric_status']}**.", "",
             "## What must happen next", ""]
    lines.extend(f"- {item}" for item in report["research_blockers"])
    lines.extend(["", "See ALL_FOUR_PARTS.md for commands and the current review packet.", ""])
    return "\n".join(lines)


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in COMMANDS:
        script, before, after = COMMANDS[raw[0]]
        if not (ROOT / script).is_file():
            raise SystemExit(f"Component unavailable: {script}")
        return subprocess.run([sys.executable, str(ROOT / script), *before, *raw[1:], *after], check=False).returncode
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("status", help="Report current evidence without inventing completion")
    show.add_argument("--output", type=Path, help="Optional fresh directory for JSON and Markdown")
    for command in COMMANDS:
        # Forward help and options to the actual tool, without shell evaluation.
        item = sub.add_parser(command, add_help=False)
        item.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(raw)
    if args.command == "status":
        report = status()
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            (args.output / "readiness.json").write_text(json.dumps(report, indent=2) + "\n")
            (args.output / "READINESS.md").write_text(markdown(report))
        print(markdown(report))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
