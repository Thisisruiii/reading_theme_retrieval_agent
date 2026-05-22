import re
from typing import Dict, List

from memory import MemoryStore
from retriever import TfidfRetriever, context_strength


class ReadingThemeAgent:
    """Agent that retrieves local context before asking an LLM to answer."""

    QUERY_STOPWORDS = {
        "which",
        "what",
        "are",
        "is",
        "the",
        "a",
        "an",
        "for",
        "about",
        "discuss",
        "discussing",
        "book",
        "books",
        "useful",
        "recommend",
        "recommendation",
    }

    def __init__(self):
        self.memory = MemoryStore()
        self.retriever = TfidfRetriever()
        self.llm = None
        self.refresh_index()

    def refresh_index(self) -> None:
        self.retriever.build_index(self.memory.all_text_chunks())

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, object]]:
        self.refresh_index()
        return self.retriever.search(query, top_k=top_k)

    def remember(self, text: str) -> Dict[str, str]:
        item = self.memory.add(text)
        self.refresh_index()
        return item

    def ask(self, question: str, top_k: int = 5) -> Dict[str, object]:
        retrieval_query = self._retrieval_query_for_question(question)
        results = self.search(retrieval_query, top_k=top_k)
        top_score = results[0]["score"] if results else 0.0
        strength = context_strength(top_score)
        strength = self._adjust_strength_with_book_evidence(question, strength, results)
        sources = sorted({result["source"] for result in results})

        if strength == "unreliable":
            return {
                "answer": (
                    "I could not find reliable context in the local book notes, theme guide, "
                    "recommendation rules, or memory. I should not pretend that the notes contain "
                    "an answer to this question."
                ),
                "confidence": top_score,
                "strength": strength,
                "sources": sources,
            }

        context_text = self._format_context(results)
        warning = ""
        if strength == "weak":
            warning = (
                "The retrieved context is limited, so clearly say that the answer is tentative. "
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Reading Theme Retrieval Agent, a command-line AI agent for an "
                    "Information Retrieval course project. Answer using the retrieved context. "
                    "Do not invent book notes. Do not turn the answer into a political debate; "
                    "treat themes as a reading taxonomy. "
                    "Never use markdown tables. Terminal output must be plain text only. "
                    "Use this exact style: short bullet points, one book or idea per bullet. "
                    "Keep the answer concise and based only on the retrieved context. "
                    + warning
                    + "If recommending books, prefer books present in the retrieved context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved context:\n{context_text}\n\n"
                    "Write a clear, concise answer for a student demo. "
                    "Do not use a table. Use short bullet points."
                ),
            },
        ]

        if self.llm is None:
            from llm_client import LLMClient

            self.llm = LLMClient()
        answer = self.llm.chat(messages)
        if self._contains_markdown_table(answer):
            answer = self._rewrite_without_tables(answer)

        return {
            "answer": answer,
            "confidence": top_score,
            "strength": strength,
            "sources": sources,
        }

    @staticmethod
    def _format_context(results: List[Dict[str, object]]) -> str:
        lines = []
        for result in results:
            lines.append(
                f"[Source: {result['source']} | score: {result['score']:.3f}]\n{result['text']}"
            )
        return "\n\n---\n\n".join(lines)

    @staticmethod
    def _adjust_strength_with_book_evidence(
        question: str, strength: str, results: List[Dict[str, object]]
    ) -> str:
        """Downgrade book questions when retrieved evidence is not from book notes."""
        if strength == "unreliable":
            return strength

        if not ReadingThemeAgent._asks_about_books(question):
            return strength

        book_note_results = [
            result for result in results if result["source"] == "book_notes.txt"
        ]
        if not book_note_results:
            return "unreliable"

        important_terms = ReadingThemeAgent._important_query_terms(question)
        book_note_text = " ".join(str(result["text"]) for result in book_note_results)
        book_note_terms = set(re.findall(r"[a-z0-9]+", book_note_text.lower()))

        if not any(term in book_note_terms for term in important_terms):
            return "unreliable"
        return strength

    @staticmethod
    def _asks_about_books(question: str) -> bool:
        question_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        book_question_words = {"book", "books", "novel", "novels", "recommend"}
        return bool(question_terms & book_question_words)

    @staticmethod
    def _retrieval_query_for_question(question: str) -> str:
        """Use cleaner retrieval terms for book questions, but keep the original prompt."""
        if not ReadingThemeAgent._asks_about_books(question):
            return question
        important_terms = ReadingThemeAgent._important_query_terms(question)
        return " ".join(important_terms) if important_terms else question

    @staticmethod
    def _important_query_terms(question: str) -> List[str]:
        terms = re.findall(r"[a-z0-9]+", question.lower())
        return [
            term
            for term in terms
            if term not in ReadingThemeAgent.QUERY_STOPWORDS and len(term) > 1
        ]

    @staticmethod
    def _contains_markdown_table(text: str) -> bool:
        lines = text.splitlines()
        table_lines = [line for line in lines if line.strip().startswith("|")]
        return len(table_lines) >= 2

    def _rewrite_without_tables(self, answer: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the answer for a command-line terminal. Do not add new facts. "
                    "Do not use markdown tables. Use only short bullet points."
                ),
            },
            {
                "role": "user",
                "content": f"Rewrite this answer without tables:\n\n{answer}",
            },
        ]
        rewritten = self.llm.chat(messages)
        return self._remove_table_lines(rewritten)

    @staticmethod
    def _remove_table_lines(text: str) -> str:
        cleaned_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("|"):
                continue
            if stripped and set(stripped) <= {"|", "-", ":", " "}:
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()
