import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
import pymupdf

spec = importlib.util.spec_from_file_location("package_submission", Path(__file__).resolve().parents[1] / "scripts/package_submission.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.pdf = self.folder / "fixture.pdf"
        from test_report import sample_state, sample_draft
        from ai_investment.agents.report import ReportAgent
        state = sample_state()
        agent = ReportAgent(output_dir=self.folder, draft_writer=lambda _: sample_draft())
        state.update(agent.run(state))
        state["eval_result"] = {"passed": True, "items": {}, "action": "pass", "target_views": [], "feedback": []}
        self.pdf = Path(agent.finalize(state)["report"])
        self.png = self.folder / "trace.png"
        Image.new("RGB", (8, 8)).save(self.png)
        (self.folder / ".env").write_text("SECRET=do-not-package")
        self.args = dict(repo_url="https://github.com/example/project/tree/report", report=self.pdf,
                         traces=[self.png], campus="TEST", class_name="1", names="tester", run_id="report-test-run",
                         output_dir=self.folder / "submission")

    def test_explicit_artifacts_only_and_no_overwrite(self):
        path = module.package_submission(**self.args)
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(set(archive.namelist()), {"report.pdf", "tracing-1.png", "GITHUB_LINK.txt", "manifest.json"})
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["run_id"], "report-test-run")
        with self.assertRaises(FileExistsError):
            module.package_submission(**self.args)

    def test_sample_pdf_cannot_be_submitted(self):
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((50, 60), "SUMMARY\nREFERENCE\nSUPERVISOR - EVIDENCE REVIEW / SAMPLE")
            pdf.save(self.folder / "sample.pdf")
        self.args["report"] = self.folder / "sample.pdf"
        with self.assertRaisesRegex(ValueError, "sample preview"):
            module.package_submission(**self.args)

    def test_mismatched_run_id_is_rejected(self):
        self.args["run_id"] = "different-run"
        with self.assertRaisesRegex(ValueError, "trace_id"):
            module.package_submission(**self.args)

    def test_pending_report_cannot_be_submitted(self):
        from test_report import sample_state, sample_draft
        from ai_investment.agents.report import ReportAgent
        agent = ReportAgent(output_dir=self.folder, draft_writer=lambda _: sample_draft())
        self.args["report"] = Path(agent.run(sample_state())["report"])
        with self.assertRaisesRegex(ValueError, "Finalize"):
            module.package_submission(**self.args)

    def test_missing_traces_and_bad_filename_fail(self):
        self.args["traces"] = []
        with self.assertRaisesRegex(ValueError, "real LangSmith"):
            module.package_submission(**self.args)
        self.args["names"] = "../../outside"
        with self.assertRaisesRegex(ValueError, "filename"):
            module.package_submission(**self.args)


if __name__ == "__main__":
    unittest.main()
