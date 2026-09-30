import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_investment.tools import rag


class RagSetupTests(unittest.TestCase):
    def test_relative_storage_path_is_repository_relative(self):
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIRECTORY": "data/test-chroma"}):
            self.assertEqual(rag._persist_directory(), rag.PROJECT_ROOT / "data/test-chroma")

    def test_missing_database_fails_before_embedding_initialization(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"CHROMA_PERSIST_DIRECTORY": directory}
        ), patch.object(rag, "_store") as store:
            for search, command in (
                (rag.search_tech_docs, "scripts/ingest_tech_docs.py"),
                (rag.search_market_docs, "ai_investment.ingest market"),
            ):
                with self.subTest(command=command), self.assertRaisesRegex(RuntimeError, command):
                    search("robotics")
            store.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_empty_and_populated_collection(self):
        class NotFoundError(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "chroma.sqlite3").touch()
            client = Mock()
            factory = Mock(return_value=client)
            modules = {
                "chromadb": SimpleNamespace(PersistentClient=factory),
                "chromadb.errors": SimpleNamespace(NotFoundError=NotFoundError),
            }
            with patch.dict("os.environ", {"CHROMA_PERSIST_DIRECTORY": directory}), patch.dict(
                "sys.modules", modules
            ), patch.object(rag, "_store") as store:
                client.get_collection.side_effect = NotFoundError()
                with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                    rag._search_store(rag.MARKET_COLLECTION)
                client.get_collection.side_effect = None
                client.get_collection.return_value.count.return_value = 0
                with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                    rag._search_store(rag.MARKET_COLLECTION)
                store.assert_not_called()
                client.get_collection.return_value.count.return_value = 2
                self.assertIs(rag._search_store(rag.MARKET_COLLECTION), store.return_value)
                store.assert_called_once_with(rag.MARKET_COLLECTION)


if __name__ == "__main__":
    unittest.main()
