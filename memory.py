import json
from pathlib import Path
from typing import Dict, List


class MemoryStore:
    """A tiny JSON-backed memory store for user-provided notes."""

    def __init__(self, path: str = "memory_store/memory.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def load(self) -> List[Dict[str, str]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
        return data if isinstance(data, list) else []

    def add(self, text: str) -> Dict[str, str]:
        items = self.load()
        item = {
            "id": str(len(items) + 1),
            "text": text.strip(),
        }
        items.append(item)
        self.path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        return item

    def all_text_chunks(self) -> List[Dict[str, str]]:
        chunks = []
        for item in self.load():
            text = item.get("text", "").strip()
            if text:
                chunks.append(
                    {
                        "source": f"memory item {item.get('id', '?')}",
                        "text": text,
                        "type": "memory",
                    }
                )
        return chunks
