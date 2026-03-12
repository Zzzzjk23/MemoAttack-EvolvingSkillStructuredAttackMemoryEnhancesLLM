from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

from config.default_config import AttackConfig
from methods.method_schema import (
    ACTIVE,
    ELIMINATED,
    RETIRED,
    AttackAttemptResult,
    AttackMethod,
    AttackMethodProposal,
    build_method_stats,
)


@dataclass
class MethodPool:
    config: AttackConfig
    methods: Dict[str, AttackMethod] = field(default_factory=dict)

    def get_method(self, method_id: str) -> Optional[AttackMethod]:
        return self.methods.get(method_id)

    def get_active_methods(self) -> List[AttackMethod]:
        return [method for method in self.methods.values() if method.status == ACTIVE]

    def get_all_methods(self) -> List[AttackMethod]:
        return list(self.methods.values())

    def has_active_methods(self) -> bool:
        return any(method.status == ACTIVE for method in self.methods.values())

    def find_duplicate(self, proposal: AttackMethodProposal) -> Optional[AttackMethod]:
        normalized_name = proposal.method_name.strip().lower()
        best_match: Optional[Tuple[AttackMethod, float]] = None
        for method in self.methods.values():
            if method.method_name.strip().lower() == normalized_name:
                return method
            similarity = method.similarity_to(proposal)
            if similarity >= self.config.duplicate_similarity_threshold and (
                best_match is None or similarity > best_match[1]
            ):
                best_match = (method, similarity)
        return best_match[0] if best_match else None

    def register_method(
        self,
        proposal: AttackMethodProposal,
        created_via: str,
        parent_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> AttackMethod:
        existing = self.find_duplicate(proposal)
        if existing is not None:
            if metadata:
                existing.metadata.update(metadata)
            return existing

        method = AttackMethod.from_proposal(
            proposal=proposal,
            created_via=created_via,
            parent_method_id=parent_method_id,
            stats=build_method_stats(
                progress_alpha=self.config.newborn_progress_alpha,
                progress_beta=self.config.newborn_progress_beta,
                success_alpha=self.config.newborn_success_alpha,
                success_beta=self.config.newborn_success_beta,
            ),
            metadata=metadata,
        )
        self.methods[method.method_id] = method
        return method

    def record_attempt(
        self,
        method_id: str,
        attempt_result: AttackAttemptResult,
        prompt_text: str,
        before_prompt: str,
        after_prompt: str,
        target_response: str,
    ) -> AttackMethod:
        method = self.methods[method_id]
        method.usage_count += 1
        method.stats.update_progress(attempt_result.made_progress)
        method.stats.update_success(attempt_result.final_success)
        method.update_recent_windows(
            made_progress=attempt_result.made_progress,
            final_success=attempt_result.final_success,
            normalized_progress=attempt_result.normalized_progress,
            window_size=self.config.recent_performance_window,
        )
        method.add_example(
            prompt_text=prompt_text,
            before_prompt=before_prompt,
            after_prompt=after_prompt,
            target_response=target_response,
            normalized_progress=attempt_result.normalized_progress,
            final_success=attempt_result.final_success,
        )
        return method

    def apply_lifecycle_rules(self) -> None:
        for method in self.methods.values():
            if method.usage_count < self.config.retirement_min_support:
                continue
            if (
                method.status == ACTIVE
                and method.stats.progress_mean <= self.config.retirement_progress_threshold
                and method.stats.success_mean <= self.config.retirement_success_threshold
                and method.recent_progress_rate <= self.config.retirement_progress_threshold
                and method.recent_success_rate <= self.config.retirement_success_threshold
            ):
                method.status = RETIRED
                continue
            if method.usage_count < self.config.elimination_min_support:
                continue
            if (
                method.status == RETIRED
                and method.stats.progress_mean <= self.config.elimination_progress_threshold
                and method.stats.success_mean <= self.config.elimination_success_threshold
                and method.recent_progress_rate <= self.config.elimination_progress_threshold
                and method.recent_success_rate <= self.config.elimination_success_threshold
            ):
                method.status = ELIMINATED

    def enforce_cap(self) -> None:
        cap = self.config.max_global_methods
        if cap <= 0:
            return
        while len(self.methods) > cap:
            victim = self._select_eviction_candidate()
            if victim is None:
                break
            del self.methods[victim.method_id]

    def _select_eviction_candidate(self) -> Optional[AttackMethod]:
        for status in (ELIMINATED, RETIRED):
            candidates = [method for method in self.methods.values() if method.status == status]
            if candidates:
                return min(candidates, key=self._status_eviction_key)
        active_methods = [method for method in self.methods.values() if method.status == ACTIVE]
        if not active_methods:
            return None
        return min(active_methods, key=self._active_eviction_key)

    def _status_eviction_key(self, method: AttackMethod) -> Tuple[float, int, float, float]:
        return (
            method.utility_score,
            method.usage_count,
            method.recent_progress_value_mean,
            self._creation_sort_value(method),
        )

    def _active_eviction_key(self, method: AttackMethod) -> Tuple[float, int, float, float]:
        return (
            method.utility_score,
            method.usage_count,
            method.recent_progress_value_mean,
            self._creation_sort_value(method),
        )

    @staticmethod
    def _creation_sort_value(method: AttackMethod) -> float:
        if not method.creation_time:
            return float("inf")
        try:
            return -datetime.fromisoformat(method.creation_time).timestamp()
        except ValueError:
            return float("inf")


class MethodRegistry:
    def __init__(
        self,
        config: Optional[AttackConfig] = None,
        load_path: Optional[str] = None,
    ):
        self.config = config or AttackConfig()
        self.path = load_path or self.config.persistence_path
        if load_path:
            self.path = load_path
        loaded = self._load_existing(self.path)
        if loaded is not None:
            self.pool = loaded.pool
            self.config = loaded.config
            self.path = loaded.path
            self.pool.config = self.config
        else:
            self.pool = MethodPool(config=self.config)

    @staticmethod
    def _load_existing(path: str) -> Optional["MethodRegistry"]:
        file_path = Path(path)
        if not file_path.exists():
            return None
        try:
            with file_path.open("rb") as handle:
                loaded = pickle.load(handle)
        except (pickle.PickleError, OSError, AttributeError, EOFError):
            return None
        if not isinstance(loaded, MethodRegistry):
            return None
        if not hasattr(loaded, "pool") or not isinstance(loaded.pool, MethodPool):
            return None
        return loaded

    def save(self, path: Optional[str] = None) -> str:
        save_path = Path(path or self.path)
        self.pool.config = self.config
        with save_path.open("wb") as handle:
            pickle.dump(self, handle)
        return str(save_path.resolve())

    def get_pool(self) -> MethodPool:
        return self.pool

    def register_method(
        self,
        proposal: AttackMethodProposal,
        created_via: str,
        parent_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> AttackMethod:
        return self.pool.register_method(
            proposal=proposal,
            created_via=created_via,
            parent_method_id=parent_method_id,
            metadata=metadata,
        )

    def record_attempt(
        self,
        method_id: str,
        attempt_result: AttackAttemptResult,
        prompt_text: str,
        before_prompt: str,
        after_prompt: str,
        target_response: str,
    ) -> AttackMethod:
        method = self.pool.record_attempt(
            method_id=method_id,
            attempt_result=attempt_result,
            prompt_text=prompt_text,
            before_prompt=before_prompt,
            after_prompt=after_prompt,
            target_response=target_response,
        )
        self.pool.apply_lifecycle_rules()
        self.pool.enforce_cap()
        return method

    def iter_methods(self) -> Iterable[AttackMethod]:
        return self.pool.get_all_methods()
