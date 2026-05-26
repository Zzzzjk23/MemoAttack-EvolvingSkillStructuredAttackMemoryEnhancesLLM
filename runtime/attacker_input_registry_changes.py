from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from config.default_config import AttackConfig
from methods.method_registry import MethodRegistry
from methods.method_schema import ACTIVE, ELIMINATED, RETIRED
from scoring.progress_metric import compute_normalized_gap_improvement


TRACKER_VERSION = 1
_FILENAME_RE = re.compile(r"openai_messages_(.+)_(\d+)$")
_SCORE_RE = re.compile(r"(?mi)^SCORE:\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                chunks.append(str(item["text"]))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(content)


def _extract_last_user_text(payload: list[Any]) -> str:
    for item in reversed(payload):
        if isinstance(item, dict) and item.get("role") == "user":
            return _message_content_to_text(item.get("content"))
    return ""


def _extract_parent_raw_score(user_text: str) -> float:
    match = _SCORE_RE.search(user_text or "")
    if match is None:
        return 0.0
    return float(match.group(1))


def _extract_previous_prompt(user_text: str) -> str:
    marker = "PREVIOUS ADVERSARIAL PROMPT:"
    text = user_text or ""
    start = text.find(marker)
    if start == -1:
        return ""
    remainder = text[start + len(marker) :]
    end_markers = [
        "\nSelected mode:",
        "\nYou should use",
        "\nCandidate attack methods:",
        "\nGLOBAL_CONTEXT_JSON:",
        "\nUse the previous",
        "\nNo candidate attack methods are available",
        "\nRelevant prior examples:",
        "\nRelevant examples:",
        "\nMethod selection rules:",
        "\nBegin.",
    ]
    end = len(remainder)
    for marker_text in end_markers:
        marker_index = remainder.find(marker_text)
        if marker_index != -1:
            end = min(end, marker_index)
    return remainder[:end].strip()


def _parse_log_stem(path: Path) -> tuple[str, int]:
    match = _FILENAME_RE.match(path.stem)
    if match is None:
        return path.stem, 0
    return match.group(1), int(match.group(2))


@dataclass
class LogEntry:
    path: Path
    payload: list[Any]
    metadata: dict[str, Any]
    goal_index: str
    request_count: int
    mtime_ns: int
    last_user_text: str


def load_log_entries(input_dir: str | Path) -> list[LogEntry]:
    resolved_dir = Path(input_dir)
    entries: list[LogEntry] = []
    for path in resolved_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list) or not payload:
            continue
        metadata = payload[0]
        if not isinstance(metadata, dict) or "role" in metadata:
            continue
        goal_index, request_count = _parse_log_stem(path)
        entries.append(
            LogEntry(
                path=path,
                payload=payload,
                metadata=metadata,
                goal_index=goal_index,
                request_count=request_count,
                mtime_ns=path.stat().st_mtime_ns,
                last_user_text=_extract_last_user_text(payload),
            )
        )
    return sorted(
        entries,
        key=lambda item: (item.mtime_ns, item.goal_index, item.request_count, item.path.name),
    )


@dataclass
class TrackedMethod:
    method_id: str
    method_name: str
    created_via: str = ""
    status: str = ACTIVE
    usage_count: int = 0
    progress_alpha: float = 1.0
    progress_beta: float = 1.0
    success_alpha: float = 1.0
    success_beta: float = 1.0
    recent_progress_history: list[bool] = field(default_factory=list)
    recent_success_history: list[bool] = field(default_factory=list)
    recent_progress_values: list[float] = field(default_factory=list)
    creation_order: int = 0
    inferred_existing: bool = False

    @property
    def progress_mean(self) -> float:
        total = self.progress_alpha + self.progress_beta
        return self.progress_alpha / total if total else 0.0

    @property
    def success_mean(self) -> float:
        total = self.success_alpha + self.success_beta
        return self.success_alpha / total if total else 0.0

    @property
    def recent_progress_rate(self) -> float:
        if not self.recent_progress_history:
            return 0.0
        return sum(1.0 for item in self.recent_progress_history if item) / len(
            self.recent_progress_history
        )

    @property
    def recent_success_rate(self) -> float:
        if not self.recent_success_history:
            return 0.0
        return sum(1.0 for item in self.recent_success_history if item) / len(
            self.recent_success_history
        )

    @property
    def recent_progress_value_mean(self) -> float:
        if not self.recent_progress_values:
            return 0.0
        return sum(self.recent_progress_values) / len(self.recent_progress_values)

    @property
    def utility_score(self) -> float:
        return 0.55 * self.progress_mean + 0.45 * self.success_mean

    def apply_attempt(
        self,
        *,
        made_progress: bool,
        final_success: bool,
        normalized_progress: float,
        window_size: int,
    ) -> None:
        self.usage_count += 1
        if made_progress:
            self.progress_alpha += 1.0
        else:
            self.progress_beta += 1.0
        if final_success:
            self.success_alpha += 1.0
        else:
            self.success_beta += 1.0
        self.recent_progress_history.append(made_progress)
        self.recent_success_history.append(final_success)
        self.recent_progress_values.append(normalized_progress)
        del self.recent_progress_history[:-window_size]
        del self.recent_success_history[:-window_size]
        del self.recent_progress_values[:-window_size]


class MethodRegistryChangeTracker:
    def __init__(self, config: Optional[AttackConfig] = None) -> None:
        self.config = config or AttackConfig()
        self.methods: dict[str, TrackedMethod] = {}
        self._next_creation_order = 0

    def seed_from_registry(self, registry: MethodRegistry) -> None:
        methods = list(registry.iter_methods())
        methods.sort(key=lambda item: getattr(item, "creation_time", "") or "")
        for method in methods:
            self._next_creation_order += 1
            self.methods[method.method_id] = TrackedMethod(
                method_id=method.method_id,
                method_name=method.method_name,
                created_via=method.created_via,
                status=method.status,
                usage_count=method.usage_count,
                progress_alpha=method.stats.progress_alpha,
                progress_beta=method.stats.progress_beta,
                success_alpha=method.stats.success_alpha,
                success_beta=method.stats.success_beta,
                recent_progress_history=list(method.recent_progress_history),
                recent_success_history=list(method.recent_success_history),
                recent_progress_values=list(method.recent_progress_values),
                creation_order=self._next_creation_order,
            )

    def ensure_method(
        self,
        method_id: str,
        method_name: str,
        *,
        created_via: str = "",
        inferred_existing: bool = False,
    ) -> tuple[TrackedMethod, bool]:
        existing = self.methods.get(method_id)
        if existing is not None:
            if method_name and not existing.method_name:
                existing.method_name = method_name
            if created_via and not existing.created_via:
                existing.created_via = created_via
            return existing, False
        self._next_creation_order += 1
        tracked = TrackedMethod(
            method_id=method_id,
            method_name=method_name,
            created_via=created_via,
            progress_alpha=self.config.newborn_progress_alpha,
            progress_beta=self.config.newborn_progress_beta,
            success_alpha=self.config.newborn_success_alpha,
            success_beta=self.config.newborn_success_beta,
            creation_order=self._next_creation_order,
            inferred_existing=inferred_existing,
        )
        self.methods[method_id] = tracked
        return tracked, True

    def counts(self) -> dict[str, int]:
        active = sum(1 for method in self.methods.values() if method.status == ACTIVE)
        retired = sum(1 for method in self.methods.values() if method.status == RETIRED)
        eliminated = sum(1 for method in self.methods.values() if method.status == ELIMINATED)
        return {
            "total": len(self.methods),
            "active": active,
            "retired": retired,
            "eliminated": eliminated,
        }

    def apply_attempt(
        self,
        method_ids: Iterable[str],
        *,
        current_raw_score: float,
        parent_raw_score: float,
    ) -> tuple[float, bool, bool, list[dict[str, Any]]]:
        current_score = max(
            0.0,
            min(self.config.max_score, current_raw_score / self.config.judge_max_score),
        )
        previous_score = max(
            0.0,
            min(self.config.max_score, parent_raw_score / self.config.judge_max_score),
        )
        normalized_progress = compute_normalized_gap_improvement(
            prev_score=previous_score,
            new_score=current_score,
            max_score=self.config.max_score,
            epsilon=self.config.epsilon,
        )
        made_progress = normalized_progress >= self.config.progress_threshold
        final_success = current_raw_score >= self.config.final_success_score_threshold
        updates: list[dict[str, Any]] = []
        for method_id in method_ids:
            tracked = self.methods[method_id]
            tracked.apply_attempt(
                made_progress=made_progress,
                final_success=final_success,
                normalized_progress=normalized_progress,
                window_size=self.config.recent_performance_window,
            )
            updates.append(
                {
                    "method_id": tracked.method_id,
                    "method_name": tracked.method_name,
                    "usage_count_after": tracked.usage_count,
                    "status_after": tracked.status,
                    "progress_mean_after": round(tracked.progress_mean, 6),
                    "success_mean_after": round(tracked.success_mean, 6),
                }
            )
        return normalized_progress, made_progress, final_success, updates

    def apply_lifecycle_rules(self) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        for method in self.methods.values():
            previous_status = method.status
            if method.usage_count >= self.config.retirement_min_support:
                if (
                    method.status == ACTIVE
                    and method.progress_mean <= self.config.retirement_progress_threshold
                    and method.success_mean <= self.config.retirement_success_threshold
                    and method.recent_progress_rate <= self.config.retirement_progress_threshold
                    and method.recent_success_rate <= self.config.retirement_success_threshold
                ):
                    method.status = RETIRED
            if method.usage_count >= self.config.elimination_min_support:
                if (
                    method.status == RETIRED
                    and method.progress_mean <= self.config.elimination_progress_threshold
                    and method.success_mean <= self.config.elimination_success_threshold
                    and method.recent_progress_rate <= self.config.elimination_progress_threshold
                    and method.recent_success_rate <= self.config.elimination_success_threshold
                ):
                    method.status = ELIMINATED
            if method.status != previous_status:
                changes.append(
                    {
                        "method_id": method.method_id,
                        "method_name": method.method_name,
                        "from_status": previous_status,
                        "to_status": method.status,
                    }
                )
        return changes

    def enforce_cap(self) -> list[dict[str, Any]]:
        cap = self.config.max_global_methods
        if cap <= 0:
            return []
        evicted: list[dict[str, Any]] = []
        while len(self.methods) > cap:
            victim = self._select_eviction_candidate()
            if victim is None:
                break
            evicted.append(
                {
                    "method_id": victim.method_id,
                    "method_name": victim.method_name,
                    "status": victim.status,
                    "usage_count": victim.usage_count,
                    "utility_score": round(victim.utility_score, 6),
                }
            )
            del self.methods[victim.method_id]
        return evicted

    def _select_eviction_candidate(self) -> Optional[TrackedMethod]:
        for status in (ELIMINATED, RETIRED):
            candidates = [method for method in self.methods.values() if method.status == status]
            if candidates:
                return min(candidates, key=self._eviction_key)
        active_methods = [method for method in self.methods.values() if method.status == ACTIVE]
        if not active_methods:
            return None
        return min(active_methods, key=self._eviction_key)

    @staticmethod
    def _eviction_key(method: TrackedMethod) -> tuple[float, int, float, int]:
        return (
            method.utility_score,
            method.usage_count,
            method.recent_progress_value_mean,
            -method.creation_order,
        )


def _build_change_record(
    entry: LogEntry,
    tracker: MethodRegistryChangeTracker,
    *,
    tracker_seeded: bool,
) -> dict[str, Any]:
    metadata = entry.metadata
    mode = str(metadata.get("mode", "") or "")
    phase = str(metadata.get("phase", "") or "")
    before_counts = tracker.counts()
    parent_raw_score = _extract_parent_raw_score(entry.last_user_text)
    previous_prompt = _extract_previous_prompt(entry.last_user_text)
    current_raw_score = _safe_float(metadata.get("outside_score"), 0.0)

    change_record: dict[str, Any] = {
        "tracker_version": TRACKER_VERSION,
        "mode": mode,
        "phase": phase,
        "goal_index": entry.goal_index,
        "request_count": entry.request_count,
        "library_counts_before": before_counts,
        "registered": [],
        "assumed_existing": [],
        "updated": [],
        "status_changes": [],
        "evicted": [],
        "selected_method_ids": list(metadata.get("selected_method_ids") or []),
        "selected_method_names": list(metadata.get("selected_method_names") or []),
        "parent_raw_score": parent_raw_score,
        "current_raw_score": current_raw_score,
        "previous_prompt_excerpt": previous_prompt[:200],
        "tracker_seeded_from_registry": tracker_seeded,
    }

    if phase != "posterior":
        change_record["note"] = "Bootstrap steps do not update the global method registry."
        change_record["library_counts_after"] = before_counts
        return change_record

    selected_ids = [str(item) for item in metadata.get("selected_method_ids") or [] if str(item)]
    selected_names = [str(item) for item in metadata.get("selected_method_names") or []]
    candidate_ids = [str(item) for item in metadata.get("candidate_method_ids") or [] if str(item)]
    candidate_names = [str(item) for item in metadata.get("candidate_method_names") or []]
    attack_method_id = str(metadata.get("attack_method_id", "") or "")
    attack_method_name = str(metadata.get("attack_method", "") or "")

    candidate_name_map = {
        method_id: candidate_names[index]
        for index, method_id in enumerate(candidate_ids)
        if index < len(candidate_names)
    }
    selected_name_map = {
        method_id: selected_names[index]
        for index, method_id in enumerate(selected_ids)
        if index < len(selected_names)
    }

    if attack_method_id:
        tracked, created = tracker.ensure_method(
            method_id=attack_method_id,
            method_name=attack_method_name,
            created_via=mode if mode in {"invent", "mutate"} else "",
            inferred_existing=False,
        )
        if created and mode in {"invent", "mutate"}:
            change_record["registered"].append(
                {
                    "method_id": tracked.method_id,
                    "method_name": tracked.method_name,
                    "created_via": mode,
                }
            )

    for method_id in candidate_ids:
        tracked, created = tracker.ensure_method(
            method_id=method_id,
            method_name=candidate_name_map.get(method_id, ""),
            created_via="",
            inferred_existing=(method_id not in tracker.methods and mode == "reuse"),
        )
        if created and mode == "reuse":
            change_record["assumed_existing"].append(
                {
                    "method_id": tracked.method_id,
                    "method_name": tracked.method_name,
                    "reason": "First seen in reuse mode without a seed registry snapshot.",
                }
            )

    for method_id in selected_ids:
        tracked, created = tracker.ensure_method(
            method_id=method_id,
            method_name=selected_name_map.get(method_id, candidate_name_map.get(method_id, "")),
            created_via="",
            inferred_existing=(method_id not in tracker.methods and mode == "reuse"),
        )
        if created and mode == "reuse":
            change_record["assumed_existing"].append(
                {
                    "method_id": tracked.method_id,
                    "method_name": tracked.method_name,
                    "reason": "Selected in reuse mode before appearing in the local replay state.",
                }
            )

    if selected_ids:
        normalized_progress, made_progress, final_success, updates = tracker.apply_attempt(
            selected_ids,
            current_raw_score=current_raw_score,
            parent_raw_score=parent_raw_score,
        )
        change_record["normalized_progress"] = round(normalized_progress, 6)
        change_record["made_progress"] = made_progress
        change_record["final_success"] = final_success
        change_record["updated"] = updates
        change_record["status_changes"] = tracker.apply_lifecycle_rules()
        change_record["evicted"] = tracker.enforce_cap()
    else:
        change_record["note"] = "No selected methods were present in this log entry."
        change_record["normalized_progress"] = 0.0
        change_record["made_progress"] = False
        change_record["final_success"] = current_raw_score >= tracker.config.final_success_score_threshold

    change_record["library_counts_after"] = tracker.counts()
    return change_record


def annotate_attacker_input_logs(
    input_dir: str | Path,
    *,
    config: Optional[AttackConfig] = None,
    seed_registry_path: Optional[str] = None,
) -> dict[str, Any]:
    resolved_config = config or AttackConfig()
    tracker = MethodRegistryChangeTracker(config=resolved_config)
    tracker_seeded = False
    if seed_registry_path:
        registry = MethodRegistry(config=resolved_config, load_path=seed_registry_path)
        if list(registry.iter_methods()):
            tracker.seed_from_registry(registry)
            tracker_seeded = True

    entries = load_log_entries(input_dir)
    updated_files = 0
    for entry in entries:
        change_record = _build_change_record(
            entry,
            tracker,
            tracker_seeded=tracker_seeded,
        )
        entry.metadata["method_registry_change"] = change_record
        entry.path.write_text(
            json.dumps(entry.payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        updated_files += 1

    return {
        "updated_files": updated_files,
        "seeded_from_registry": tracker_seeded,
        "final_library_counts": tracker.counts(),
    }
