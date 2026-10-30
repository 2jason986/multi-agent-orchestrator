"""A tiny TF-IDF retriever over a folder of Markdown notes.

Why hand-rolled instead of a vector database? Because the interesting part of an
agent project is the orchestration loop, and a dependency-free retriever keeps
``git clone && python -m orchestra.cli demo`` working on a bare Python install.

The tokenizer emits latin words plus character bigrams for CJK text, which is a
cheap way to get usable Chinese recall without a segmentation dependency.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LATIN = re.compile(r"[a-z0-9_]+")
CJK = re.compile(r"[\u4e00-\u9fff]+")
SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?\n])")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = LATIN.findall(lowered)
    for run in CJK.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    content: str

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.content}"


@dataclass(slots=True)
class SearchHit:
    doc_id: str
    title: str
    score: float
    excerpt: str

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "score": round(self.score, 4),
            "excerpt": self.excerpt,
        }


class KnowledgeBase:
    """In-memory TF-IDF index with cosine similarity scoring."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self._term_frequencies = [Counter(tokenize(doc.text)) for doc in documents]
        self._idf: dict[str, float] = {}
        self._build_idf()

    @classmethod
    def from_directory(cls, path: str | Path) -> KnowledgeBase:
        root = Path(path)
        documents: list[Document] = []
        for file in sorted(root.glob("*.md")):
            raw = file.read_text(encoding="utf-8").strip()
            first_line, _, rest = raw.partition("\n")
            title = first_line.lstrip("# ").strip() or file.stem
            documents.append(Document(doc_id=file.stem, title=title, content=rest.strip()))
        return cls(documents)

    @property
    def documents(self) -> list[Document]:
        return list(self._documents)

    def _build_idf(self) -> None:
        total = len(self._documents) or 1
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(frequencies.keys())
        self._idf = {
            term: math.log((total + 1) / (df + 1)) + 1.0 for term, df in document_frequency.items()
        }

    def _weight(self, frequencies: Counter[str]) -> dict[str, float]:
        total = sum(frequencies.values()) or 1
        return {
            term: (count / total) * self._idf.get(term, 1.0) for term, count in frequencies.items()
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        shared = set(left) & set(right)
        if not shared:
            return 0.0
        numerator = sum(left[term] * right[term] for term in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    def search(self, query: str, top_k: int = 3) -> list[SearchHit]:
        if not query.strip():
            return []
        query_weights = self._weight(Counter(tokenize(query)))
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self._term_frequencies):
            score = self._cosine(query_weights, self._weight(frequencies))
            if score > 0:
                scored.append((score, index))
        scored.sort(reverse=True)
        hits = []
        for score, index in scored[:top_k]:
            document = self._documents[index]
            hits.append(
                SearchHit(
                    doc_id=document.doc_id,
                    title=document.title,
                    score=score,
                    excerpt=self._excerpt(document, query_weights),
                )
            )
        return hits

    def _excerpt(
        self, document: Document, query_weights: dict[str, float], limit: int = 120
    ) -> str:
        """Return the sentence that best matches the query, trimmed to ``limit`` chars."""
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(document.content) if s.strip()]
        if not sentences:
            return document.content[:limit]
        best, best_score = sentences[0], -1.0
        for sentence in sentences:
            weights = self._weight(Counter(tokenize(sentence)))
            score = self._cosine(query_weights, weights)
            if score > best_score:
                best, best_score = sentence, score
        return best if len(best) <= limit else best[: limit - 1] + "…"
