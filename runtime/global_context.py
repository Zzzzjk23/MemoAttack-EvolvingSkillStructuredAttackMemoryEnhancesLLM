from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Deque, Iterable


def _default_global_context_path() -> Path:
    return Path(__file__).resolve().parent.parent / "global_context.json"


@dataclass(frozen=True)
class GlobalContextEntry:
    score: int
    prompt: str

    @classmethod
    def from_item(cls, item) -> "GlobalContextEntry":
        if isinstance(item, cls):
            return item
        if isinstance(item, dict):
            return cls(
                score=int(item.get("score", 0) or 0),
                prompt=str(item.get("prompt", "") or ""),
            )
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            score, prompt = item[0], item[1]
            return cls(score=int(score or 0), prompt=str(prompt or ""))
        raise TypeError(f"Unsupported global context item: {item!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "prompt": self.prompt,
        }


class GlobalContextQueue:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.data: Deque[GlobalContextEntry] = deque()
        self.min_score = -1

    def _recalc_min_score(self) -> None:
        if not self.data:
            self.min_score = -1
            return
        self.min_score = min(entry.score for entry in self.data)

    def _remove_at_index(self, index: int) -> None:
        if index < 0 or index >= len(self.data):
            return
        self.data.rotate(-index)
        self.data.popleft()
        self.data.rotate(index)

    def _normalize_item(self, item) -> GlobalContextEntry:
        return GlobalContextEntry.from_item(item)

    def entries(self) -> list[GlobalContextEntry]:
        return list(self.data)

    def is_empty(self) -> bool:
        return len(self.data) == 0

    def enqueue(self, item) -> None:
        if self.capacity <= 0:
            return

        entry = self._normalize_item(item)
        score = entry.score

        if len(self.data) < self.capacity:
            self.data.append(entry)
            if self.min_score == -1 or score < self.min_score:
                self.min_score = score
            return

        current_min_score = None
        current_min_index = None
        for idx, existing in enumerate(self.data):
            if current_min_score is None or existing.score < current_min_score:
                current_min_score = existing.score
                current_min_index = idx

        if current_min_score is None or score < current_min_score:
            return

        if score == current_min_score:
            self._remove_at_index(current_min_index)
            self.data.append(entry)
            self._recalc_min_score()
            return

        self._remove_at_index(current_min_index)
        self.data.append(entry)
        self._recalc_min_score()

    def _payload(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "data": [entry.to_dict() for entry in self.data],
        }

    def convert_to_json(self) -> str:
        return json.dumps(self._payload()["data"], ensure_ascii=False)

    def save_to_file(self, filepath: str | None = None) -> None:
        path = Path(filepath) if filepath else _default_global_context_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self._payload(), handle, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str | None = None) -> None:
        path = Path(filepath) if filepath else _default_global_context_path()
        if not path.exists():
            self.data = deque()
            self._recalc_min_score()
            return

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        loaded_entries: Iterable[GlobalContextEntry] = (
            self._normalize_item(item) for item in payload.get("data", [])
        )
        self.data = deque()
        self.min_score = -1
        for entry in loaded_entries:
            self.enqueue(entry)


class FixedQueue(GlobalContextQueue):
    pass


__all__ = [
    "FixedQueue",
    "GlobalContextEntry",
    "GlobalContextQueue",
]
