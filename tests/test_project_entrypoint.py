import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("heritage_project", ROOT / "project.py")
project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project)


class ProjectEntryTests(unittest.TestCase):
    def test_missing_evidence_never_claims_completed_research(self):
        with tempfile.TemporaryDirectory() as directory:
            report = project.status(Path(directory))
        self.assertEqual(report["research_completion"], "INCOMPLETE")
        self.assertEqual(report["candidate_image_rows"], 0)
        self.assertFalse(report["reviewed_binary_pilot"]["available"])
        self.assertIn("incomplete", project.markdown(report))

    def test_two_city_training_requires_site_checks(self):
        self.assertEqual(project.COMMANDS["train-segmentation"][2], ["--require-sites"])
        self.assertEqual(project.COMMANDS["check-segmentation-data"][2], ["--require-sites"])

    def test_option_first_forwarding_preserves_arguments_and_child_exit_code(self):
        with patch.object(project.subprocess, "run", return_value=SimpleNamespace(returncode=2)) as run:
            code = project.main(["train-segmentation", "--help"])
        self.assertEqual(code, 2)
        self.assertEqual(run.call_args.args[0][-2:], ["--help", "--require-sites"])
        self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
