import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.ingest import (
    COLLECTIONS,
    CollectionSpec,
    TranscriptPage,
    has_text_layer,
    load_manifest,
    merge_page_text,
    parse_transcript,
)


class TranscriptTests(unittest.TestCase):
    def test_maps_one_based_headers_to_zero_based_pages(self):
        pages = parse_transcript(
            "<!-- ## p.9 in a comment is ignored -->\n"
            "## p.2\n첫 페이지 전사\n\n## p.8 replace\n로고맵 전사\n"
        )
        self.assertEqual(
            pages,
            {
                1: TranscriptPage("첫 페이지 전사"),
                7: TranscriptPage("로고맵 전사", replace=True),
            },
        )

    def test_rejects_duplicate_pages(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_transcript("## p.3\na\n## p.3\nb\n")


class MergeTests(unittest.TestCase):
    PDF_TEXT = "Global robot demand in factories doubles over 10 years +7%"

    def test_detects_missing_or_garbled_text_layer(self):
        self.assertFalse(has_text_layer(""))
        self.assertFalse(has_text_layer("\x01\x01\n\x01\x01"))
        self.assertFalse(has_text_layer("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Map!!"))
        self.assertTrue(has_text_layer(self.PDF_TEXT))

    def test_keeps_pdf_text_without_transcript(self):
        self.assertEqual(merge_page_text(f" {self.PDF_TEXT} ", None), (self.PDF_TEXT, "pdf"))

    def test_replaces_control_characters_used_as_spaces(self):
        text, _ = merge_page_text("SPRi\x01이슈리포트\n\uf0a7Key drivers", None)
        self.assertEqual(text, "SPRi 이슈리포트\n Key drivers")

    def test_appends_transcript_to_real_text_layer(self):
        text, extraction = merge_page_text(self.PDF_TEXT, TranscriptPage("2024년 542천 대"))
        self.assertEqual(extraction, "pdf+transcript")
        self.assertTrue(text.startswith(self.PDF_TEXT))
        self.assertTrue(text.endswith("2024년 542천 대"))

    def test_transcript_replaces_empty_or_flagged_page(self):
        self.assertEqual(
            merge_page_text("", TranscriptPage("전사")), ("전사", "transcript")
        )
        self.assertEqual(
            merge_page_text(self.PDF_TEXT, TranscriptPage("전사", replace=True)),
            ("전사", "transcript"),
        )


class SplitTests(unittest.TestCase):
    def test_sections_start_new_chunks_when_page_exceeds_chunk_size(self):
        from langchain_core.documents import Document

        from ai_investment.ingest import split_documents

        text = "### (4) AI·SW 플랫폼\n" + "플랫폼 기업 설명. " * 80 + "\n\n### 맺으며\n" + "로봇 밀도 1,012대. " * 20
        chunks = split_documents(
            CollectionSpec("test", Path("."), 1000, 150),
            [Document(page_content=text, metadata={"source": "a.pdf", "page": 6})],
        )
        self.assertTrue(any(c.page_content.startswith("### 맺으며") for c in chunks))
        self.assertFalse(any("AI·SW" in c.page_content and "맺으며" in c.page_content for c in chunks))


class ManifestTests(unittest.TestCase):
    def _spec(self, root: Path, manifest: dict) -> CollectionSpec:
        (root / "raw").mkdir()
        (root / "raw" / "a.pdf").write_bytes(b"%PDF")
        (root / "sources.json").write_text(json.dumps(manifest), encoding="utf-8")
        return CollectionSpec("test", root, 1000, 150)

    def test_rejects_pdf_missing_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(Path(tmp), {})
            with self.assertRaisesRegex(ValueError, "a.pdf"):
                load_manifest(spec)

    def test_rejects_entry_without_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._spec(Path(tmp), {"a.pdf": {"title": ""}})
            with self.assertRaisesRegex(ValueError, "title"):
                load_manifest(spec)

    def test_chunk_settings_follow_design_document(self):
        market, tech = COLLECTIONS["market"], COLLECTIONS["tech"]
        self.assertEqual((market.name, market.chunk_size, market.chunk_overlap), ("market_docs", 1000, 150))
        self.assertEqual((tech.name, tech.chunk_size, tech.chunk_overlap), ("tech_docs", 800, 100))


if __name__ == "__main__":
    unittest.main()
