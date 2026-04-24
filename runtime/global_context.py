from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable


PHASE_BOOTSTRAP = "bootstrap"
PHASE_POSTERIOR = "posterior"


def _default_global_context_path() -> Path:
    return Path(__file__).resolve().parent.parent / "global_context.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GlobalContextEntry:
    goal: str = ""
    goal_index: str = ""
    before_prompt: str = ""
    before_score: int = 0
    after_prompt: str = ""
    after_score: int = 0
    improvement: str = ""
    normalized_progress: float = 0.0
    target_response: str = ""
    depth: int = 0
    request_count: int = 0
    timestamp: str = field(default_factory=_utc_now)
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_item(cls, item) -> "GlobalContextEntry":
        if isinstance(item, cls):
            return item
        if isinstance(item, dict):
            if "after_prompt" in item or "before_prompt" in item:
                return cls(
                    goal=str(item.get("goal", "") or ""),
                    goal_index=str(item.get("goal_index", "") or item.get("index", "") or ""),
                    before_prompt=str(item.get("before_prompt", "") or ""),
                    before_score=int(item.get("before_score", 0) or 0),
                    after_prompt=str(item.get("after_prompt", "") or ""),
                    after_score=int(item.get("after_score", 0) or 0),
                    improvement=str(item.get("improvement", "") or ""),
                    normalized_progress=float(item.get("normalized_progress", 0.0) or 0.0),
                    target_response=str(item.get("target_response", "") or ""),
                    depth=int(item.get("depth", 0) or 0),
                    request_count=int(item.get("request_count", 0) or 0),
                    timestamp=str(item.get("timestamp", "") or _utc_now()),
                    metadata=dict(item.get("metadata", {}) or {}),
                )
            if "prompt" in item or "score" in item:
                return cls(
                    after_prompt=str(item.get("prompt", "") or ""),
                    after_score=int(item.get("score", 0) or 0),
                    timestamp=str(item.get("timestamp", "") or _utc_now()),
                )
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            score, prompt = item[0], item[1]
            return cls(after_score=int(score or 0), after_prompt=str(prompt or ""))
        raise TypeError(f"Unsupported global context item: {item!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "goal_index": self.goal_index,
            "before_prompt": self.before_prompt,
            "before_score": self.before_score,
            "after_prompt": self.after_prompt,
            "after_score": self.after_score,
            "improvement": self.improvement,
            "normalized_progress": self.normalized_progress,
            "target_response": self.target_response,
            "depth": self.depth,
            "request_count": self.request_count,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    def to_attacker_dict(self) -> dict[str, object]:
        return {
            "before_prompt": self.before_prompt,
            "before_score": self.before_score,
            "after_prompt": self.after_prompt,
            "after_score": self.after_score,
            "improvement": self.improvement,
        }


class GlobalContextQueue:
    def __init__(self, attacker_top_k: int):
        self.attacker_top_k = max(0, int(attacker_top_k))
        self.phase = PHASE_BOOTSTRAP
        self.bootstrap_success_count = 0
        self.posterior_built = False
        self.records: list[GlobalContextEntry] = []
        self.successful_goal_ids: set[str] = set()

    def entries(self) -> list[GlobalContextEntry]:
        return list(self.records)

    def is_empty(self) -> bool:
        return len(self.records) == 0

    def is_bootstrap_phase(self) -> bool:
        return self.phase != PHASE_POSTERIOR

    def is_posterior_phase(self) -> bool:
        return self.phase == PHASE_POSTERIOR

    def mark_posterior_built(self) -> None:
        self.posterior_built = True
        self.phase = PHASE_POSTERIOR

    def set_bootstrap_phase(self) -> None:
        self.phase = PHASE_BOOTSTRAP
        self.posterior_built = False

    def add_record(self, item) -> None:
        self.records.append(GlobalContextEntry.from_item(item))

    def add_records(self, items: Iterable[object]) -> None:
        for item in items:
            self.add_record(item)

    def commit_successful_goal(self, goal_id: str, items: Iterable[object]) -> bool:
        normalized_goal_id = str(goal_id or "").strip()
        if normalized_goal_id and normalized_goal_id in self.successful_goal_ids:
            return False
        self.add_records(items)
        if normalized_goal_id:
            self.successful_goal_ids.add(normalized_goal_id)
        self.bootstrap_success_count += 1
        return True

    def needs_posterior_build(self, target_success_count: int) -> bool:
        return (
            self.is_bootstrap_phase()
            and not self.posterior_built
            and self.bootstrap_success_count >= max(1, int(target_success_count))
        )

    def attacker_view(self, limit: int | None = None) -> list[dict[str, object]]:
        resolved_limit = self.attacker_top_k if limit is None else max(0, int(limit))
        ranked_records = sorted(
            self.records,
            key=lambda entry: (entry.after_score, entry.timestamp),
            reverse=True,
        )
        if resolved_limit:
            ranked_records = ranked_records[:resolved_limit]
        return [entry.to_attacker_dict() for entry in ranked_records]

    def convert_to_json(self, limit: int | None = None) -> str:
        return json.dumps(self.attacker_view(limit=limit), ensure_ascii=False)

    def _payload(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "bootstrap_success_count": self.bootstrap_success_count,
            "posterior_built": self.posterior_built,
            "attacker_top_k": self.attacker_top_k,
            "successful_goal_ids": sorted(self.successful_goal_ids),
            "records": [entry.to_dict() for entry in self.records],
        }

    def save_to_file(self, filepath: str | None = None) -> None:
        path = Path(filepath) if filepath else _default_global_context_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self._payload(), handle, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str | None = None) -> None:
        path = Path(filepath) if filepath else _default_global_context_path()
        if not path.exists():
            self.phase = PHASE_BOOTSTRAP
            self.bootstrap_success_count = 0
            self.posterior_built = False
            self.records = []
            self.successful_goal_ids = set()
            return

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            payload = {}

        legacy_records = payload.get("records")
        if legacy_records is None:
            legacy_records = payload.get("data", [])

        loaded_entries: Iterable[GlobalContextEntry] = (
            GlobalContextEntry.from_item(item) for item in legacy_records or []
        )
        self.records = list(loaded_entries)
        self.phase = str(payload.get("phase", PHASE_BOOTSTRAP) or PHASE_BOOTSTRAP)
        self.bootstrap_success_count = int(payload.get("bootstrap_success_count", 0) or 0)
        self.posterior_built = bool(payload.get("posterior_built", False))
        if "attacker_top_k" in payload:
            self.attacker_top_k = max(0, int(payload.get("attacker_top_k", self.attacker_top_k)))
        self.successful_goal_ids = {
            str(item).strip()
            for item in (payload.get("successful_goal_ids", []) or [])
            if str(item).strip()
        }
__all__ = [
    "GlobalContextEntry",
    "GlobalContextQueue",
    "PHASE_BOOTSTRAP",
    "PHASE_POSTERIOR",
]
