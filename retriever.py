from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """Local TF-IDF retriever over document chunks and memory chunks."""

    def __init__(self, documents_dir: str = "documents", chunk_size: int = 900):
        self.documents_dir = Path(documents_dir)
        self.chunk_size = chunk_size
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunks: List[Dict[str, str]] = []
        self.matrix = None

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split a very long paragraph by words instead of cutting through words."""
        words = paragraph.split()
        chunks = []
        current_words = []
        current_length = 0

        for word in words:
            next_length = current_length + len(word) + (1 if current_words else 0)
            if current_words and next_length > self.chunk_size:
                chunks.append(" ".join(current_words))
                current_words = [word]
                current_length = len(word)
            else:
                current_words.append(word)
                current_length = next_length

        if current_words:
            chunks.append(" ".join(current_words))
        return chunks

    def _chunk_text(self, text: str, source: str) -> List[Dict[str, str]]:
        clean_text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
        if not clean_text:
            return []

        # Keep blank-line separated entries together when possible. This makes
        # search output easier to read and avoids chunks starting mid-word.
        paragraphs = [part.strip() for part in clean_text.split("\n\n") if part.strip()]
        chunks = []
        chunk_id = 1

        for paragraph in paragraphs:
            paragraph_parts = (
                self._split_long_paragraph(paragraph)
                if len(paragraph) > self.chunk_size
                else [paragraph]
            )

            for part in paragraph_parts:
                chunks.append(
                    {
                        "source": source,
                        "chunk_id": str(chunk_id),
                        "text": part,
                        "type": "document",
                    }
                )
                chunk_id += 1

        return chunks

    def load_document_chunks(self) -> List[Dict[str, str]]:
        chunks = []
        for path in sorted(self.documents_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8")
            chunks.extend(self._chunk_text(text, path.name))
        return chunks

    def build_index(self, memory_chunks: List[Dict[str, str]] | None = None) -> None:
        memory_chunks = memory_chunks or []
        self.chunks = self.load_document_chunks() + memory_chunks

        if not self.chunks:
            self.matrix = None
            return

        texts = [chunk["text"] for chunk in self.chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        if self.matrix is None or not self.chunks:
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self.chunks[index]
            results.append(
                {
                    "source": chunk["source"],
                    "chunk_id": chunk.get("chunk_id", ""),
                    "type": chunk.get("type", "document"),
                    "text": chunk["text"],
                    "score": score,
                }
            )
        return results


def context_strength(top_score: float) -> str:
    if top_score >= 0.25:
        return "strong"
    if top_score >= 0.10:
        return "weak"
    return "unreliable"
