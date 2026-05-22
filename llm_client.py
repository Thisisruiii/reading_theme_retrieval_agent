import os

from dotenv import load_dotenv
from openai import OpenAI


class LLMClient:
    """Small wrapper around an OpenAI-compatible chat completion API."""

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = os.getenv("OPENAI_MODEL", "google/gemma-4-31b-it")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Create a .env file from .env.example first."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return self._clean_terminal_text(response.choices[0].message.content.strip())

    @staticmethod
    def _clean_terminal_text(text: str) -> str:
        """Replace common Unicode punctuation that can break some Windows terminals."""
        replacements = {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
        }
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        return text
