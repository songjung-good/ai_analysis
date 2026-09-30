import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from ai_investment.retrieval_eval import chunk_language, recall_mrr, sample_chunks


def chunk(text):
    return SimpleNamespace(page_content=text, metadata={})


class RetrievalEvalTests(unittest.TestCase):
    def test_recall_and_mrr_at_5(self):
        ranked = [["a", "x"], ["x", "x", "b"], ["x"] * 5 + ["c"]]
        recall, mrr = recall_mrr(ranked, ["a", "b", "c"])
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertAlmostEqual(mrr, (1 + 1 / 3) / 3)

    def test_language_follows_dominant_script(self):
        self.assertEqual(chunk_language("협동로봇 설치 대수 Cobot"), "ko")
        self.assertEqual(chunk_language("Annual installations of industrial robots 542"), "en")

    def test_samples_requested_mix_and_skips_short_chunks(self):
        chunks = [chunk("가" * 250) for _ in range(3)] + [chunk("a" * 250) for _ in range(2)]
        chunks.append(chunk("나" * 10))
        sampled = sample_chunks(chunks, n_ko=2, n_en=1, seed=0)
        self.assertEqual([chunk_language(c.page_content) for c in sampled], ["ko", "ko", "en"])
        with self.assertRaisesRegex(ValueError, "en chunks"):
            sample_chunks(chunks, n_ko=1, n_en=3)


if __name__ == "__main__":
    unittest.main()
