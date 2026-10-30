"""Unit tests for the retriever."""

from __future__ import annotations

import unittest
from pathlib import Path

from orchestra.knowledge import Document, KnowledgeBase, tokenize

DATA = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"


class TokenizeTests(unittest.TestCase):
    def test_splits_latin_and_chinese(self) -> None:
        tokens = tokenize("CSV 导出")
        self.assertIn("csv", tokens)
        self.assertIn("导出", tokens)

    def test_emits_bigrams_for_chinese_runs(self) -> None:
        tokens = tokenize("发票抬头")
        self.assertEqual(tokens, ["发票", "票抬", "抬头"])


class KnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = KnowledgeBase.from_directory(DATA)

    def test_loads_every_markdown_note(self) -> None:
        self.assertEqual(len(self.kb.documents), 6)
        self.assertTrue(all(doc.doc_id.startswith("kb-") for doc in self.kb.documents))

    def test_routes_export_question_to_the_export_note(self) -> None:
        hits = self.kb.search("数据导出卡在90%", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, "kb-export")

    def test_routes_rate_limit_question_to_the_api_note(self) -> None:
        hits = self.kb.search("接口返回 429 Too Many Requests", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc_id, "kb-api-limit")

    def test_caps_results_and_returns_scores(self) -> None:
        hits = self.kb.search("发票 抬头 税号 退款", top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertGreater(hits[0].score, 0)

    def test_blank_query_returns_nothing(self) -> None:
        self.assertEqual(self.kb.search("   "), [])

    def test_excerpt_is_bounded(self) -> None:
        hits = self.kb.search("发票重开", top_k=1)
        self.assertLessEqual(len(hits[0].excerpt), 121)


class SmallCorpusTests(unittest.TestCase):
    def test_cosine_similarity_ranks_the_relevant_document_first(self) -> None:
        kb = KnowledgeBase(
            [
                Document("kb-a", "退款", "退款需要七个工作日到账。"),
                Document("kb-b", "登录", "密码连续输错五次会锁定账号。"),
            ]
        )
        hits = kb.search("账号锁定", top_k=1)
        self.assertEqual(hits[0].doc_id, "kb-b")


if __name__ == "__main__":
    unittest.main()
