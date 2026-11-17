import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rag


class RAGTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_data_dir = rag.DATA_DIR
        self.original_database_path = rag.DATABASE_PATH
        rag.DATA_DIR = Path(self.temporary_directory.name)
        rag.DATABASE_PATH = rag.DATA_DIR / "test.sqlite3"

    def tearDown(self) -> None:
        rag.DATA_DIR = self.original_data_dir
        rag.DATABASE_PATH = self.original_database_path
        self.temporary_directory.cleanup()

    def test_local_api_base_rejects_non_loopback_hosts(self) -> None:
        with patch.dict(os.environ, {"RAG_TEST_URL": "http://127.0.0.1:8000/v1"}, clear=False):
            self.assertEqual(rag.local_api_base("RAG_TEST_URL"), "http://127.0.0.1:8000/v1")
        with patch.dict(os.environ, {"RAG_TEST_URL": "https://example.com/v1"}, clear=False):
            with self.assertRaises(rag.RAGError):
                rag.local_api_base("RAG_TEST_URL")

    def test_guide_parser_ignores_navigation_and_footer(self) -> None:
        parser = rag.GuideParser()
        parser.feed(
            "<header><p>Site nav</p></header><main><h1 id='install'>Install Edge Pack</h1>"
            "<p>TUI and CLI use the same package resolution.</p></main><footer><p>Legal footer</p></footer>"
        )
        parser.close()
        parser.flush_block()
        combined = " ".join(block[2] for block in parser.blocks)
        self.assertIn("Install Edge Pack", combined)
        self.assertIn("same package resolution", combined)
        self.assertNotIn("Site nav", combined)
        self.assertNotIn("Legal footer", combined)

    def test_guide_chunks_keep_heading_anchor_and_link(self) -> None:
        long_paragraph = "Prerequisite detail " * 180
        _, chunks = rag.parse_guide(
            "<main><h1 id='requirements'>System Requirements</h1>"
            f"<p>{long_paragraph}</p>"
            "<p>See <a href='https://docs.example.test/requirements.html'>system requirements</a>.</p>"
            "</main>",
            "2026-11-17",
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.locator == "#requirements" for chunk in chunks))
        self.assertTrue(any("https://docs.example.test/requirements.html" in chunk.content for chunk in chunks))

    def test_search_returns_cited_fts_match(self) -> None:
        connection = rag.open_database()
        with connection:
            source_cursor = connection.execute(
                "INSERT INTO sources(uri, kind, version, indexed_at) VALUES (?, ?, ?, ?)",
                ("https://github.com/open-edge-platform/edge-pack/blob/abc/README.md", "repository", "commit abc", "now"),
            )
            chunk_cursor = connection.execute(
                """INSERT INTO chunks(source_id, ordinal, title, locator, content, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (source_cursor.lastrowid, 0, "Install", "https://github.com/open-edge-platform/edge-pack/blob/abc/README.md#L1-L4", "The TUI installer supports Edge Pack profiles.", "hash"),
            )
            connection.execute(
                "INSERT INTO chunks_fts(rowid, title, content) VALUES (?, ?, ?)",
                (chunk_cursor.lastrowid, "Install", "The TUI installer supports Edge Pack profiles."),
            )
        connection.close()
        with patch.dict(os.environ, {"RAG_EMBEDDING_BASE_URL": ""}, clear=False):
            results, mode = rag.search_chunks("TUI installer profiles")
        self.assertEqual(mode, "lexical")
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["citation"].endswith("#L1-L4"))

    def test_fts_query_tokens_are_quoted(self) -> None:
        self.assertEqual(rag.tokenize_query("Ubuntu 24.04 apt-get"), ["Ubuntu", "24.04", "apt-get"])

    def test_generation_adds_source_markers_if_model_omits_them(self) -> None:
        sources = [{"title": "Install", "citation": "https://example.test/install", "version": "commit abc", "content": "TUI, CLI, APT"}]
        with patch.dict(os.environ, {"RAG_CHAT_BASE_URL": "http://127.0.0.1:11434/v1"}, clear=False):
            with patch.object(rag, "api_model_name", return_value="llama3.2:3b"):
                with patch.object(rag, "api_json", return_value={"choices": [{"message": {"content": "Edge Pack provides three installation methods."}}]}):
                    answer = rag.generate_answer("Which methods?", sources)
        self.assertIn("Sources: [S1]", answer)


if __name__ == "__main__":
    unittest.main()