from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pickle
import random
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
        self.ensure_lifecycle_metadata()
        return [method for method in self.methods.values() if method.status == ACTIVE]

    def get_retired_methods(self) -> List[AttackMethod]:
        self.ensure_lifecycle_metadata()
        return [method for method in self.methods.values() if method.status == RETIRED]

    def get_all_methods(self) -> List[AttackMethod]:
        self.ensure_lifecycle_metadata()
        return list(self.methods.values())

    def has_active_methods(self) -> bool:
        self.ensure_lifecycle_metadata()
        return any(method.status == ACTIVE for method in self.methods.values())

    def has_retired_methods(self) -> bool:
        self.ensure_lifecycle_metadata()
        return any(method.status == RETIRED for method in self.methods.values())

    def has_selectable_methods(self) -> bool:
        return self.has_active_methods() or self.has_retired_methods()

    def get_methods_for_reuse(
        self,
        rng: Optional[random.Random] = None,
    ) -> List[AttackMethod]:
        active_methods = self.get_active_methods()
        retired_probe_methods = self.get_retired_probe_methods(
            rng=rng,
            force=not active_methods,
        )
        return active_methods + retired_probe_methods

    def get_retired_probe_methods(
        self,
        rng: Optional[random.Random] = None,
        force: bool = False,
    ) -> List[AttackMethod]:
        retired_methods = self.get_retired_methods()
        if not retired_methods:
            return []
        limit = max(0, self.config.retired_probe_candidate_count)
        if limit <= 0:
            return []
        rng = rng or random.Random()
        if not force and rng.random() >= self.config.retired_probe_probability:
            return []
        ranked = sorted(retired_methods, key=self._retired_probe_key, reverse=True)
        return ranked[:limit]

    def find_duplicate(self, proposal: AttackMethodProposal) -> Optional[AttackMethod]:
        normalized_name = proposal.method_name.strip().lower()
        matches: List[Tuple[AttackMethod, float, bool]] = []
        for method in self.methods.values():
            self._ensure_method_lifecycle_metadata(method)
            if method.method_name.strip().lower() == normalized_name:
                matches.append((method, 1.0, True))
                continue
            similarity = method.similarity_to(proposal)
            if similarity >= self.config.duplicate_similarity_threshold:
                matches.append((method, similarity, False))
        if not matches:
            return None
        return min(matches, key=self._duplicate_match_key)[0]

    def register_method(
        self,
        proposal: AttackMethodProposal,
        created_via: str,
        parent_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> AttackMethod:
        existing = self.find_duplicate(proposal)
        if existing is not None and existing.status == ACTIVE:
            if metadata:
                existing.metadata.update(metadata)
            return existing

        combined_metadata = dict(metadata or {})
        if existing is not None:
            combined_metadata.update(
                {
                    "forked_from_duplicate_method_id": existing.method_id,
                    "forked_from_duplicate_method_name": existing.method_name,
                    "forked_from_duplicate_status": existing.status,
                }
            )
            if parent_method_id is None:
                parent_method_id = existing.method_id

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
            metadata=combined_metadata,
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
        self._ensure_method_lifecycle_metadata(method)
        was_retired_probe = method.status == RETIRED
        method.usage_count += 1
        if was_retired_probe:
            method.retired_probe_count += 1
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
            self._ensure_method_lifecycle_metadata(method)
            if method.status == ELIMINATED:
                continue

            if method.status == RETIRED:
                if self._should_reactivate_retired(method):
                    self._mark_active(method)
                elif self._should_eliminate_retired(method):
                    method.status = ELIMINATED
                continue

            if method.status == ACTIVE and self._should_retire_active(method):
                self._mark_retired(method)

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
                return min(candidates, key=self._eviction_key)
        active_methods = [method for method in self.methods.values() if method.status == ACTIVE]
        if not active_methods:
            return None
        return min(active_methods, key=self._eviction_key)

    def _eviction_key(self, method: AttackMethod) -> Tuple[float, int, float, float]:
        return (
            method.utility_score,
            method.usage_count,
            method.recent_progress_value_mean,
            self._creation_sort_value(method),
        )

    def _retired_probe_key(self, method: AttackMethod) -> Tuple[int, float, float, float]:
        self._ensure_method_lifecycle_metadata(method)
        return (
            -method.retired_probe_count,
            method.utility_score,
            method.recent_progress_value_mean,
            self._creation_sort_value(method),
        )

    @staticmethod
    def _duplicate_match_key(
        match: Tuple[AttackMethod, float, bool]
    ) -> Tuple[int, int, float]:
        method, similarity, exact_name = match
        status_rank = {ACTIVE: 0, RETIRED: 1, ELIMINATED: 2}.get(method.status, 3)
        return (status_rank, 0 if exact_name else 1, -similarity)

    def _should_retire_active(self, method: AttackMethod) -> bool:
        return (
            method.usage_count >= self.config.retirement_min_support
            and method.stats.progress_mean <= self.config.retirement_progress_threshold
            and method.stats.success_mean <= self.config.retirement_success_threshold
            and method.recent_progress_rate <= self.config.retirement_progress_threshold
            and method.recent_success_rate <= self.config.retirement_success_threshold
        )

    def _should_reactivate_retired(self, method: AttackMethod) -> bool:
        last_progress = method.recent_progress_values[-1] if method.recent_progress_values else 0.0
        last_made_progress = (
            method.recent_progress_history[-1] if method.recent_progress_history else False
        )
        last_success = method.recent_success_history[-1] if method.recent_success_history else False
        return (
            last_success
            or last_made_progress
            or last_progress >= self.config.retired_reactivation_progress_threshold
            or method.recent_progress_rate >= self.config.retired_reactivation_recent_progress_rate
            or method.recent_success_rate >= self.config.retired_reactivation_recent_success_rate
        )

    def _should_eliminate_retired(self, method: AttackMethod) -> bool:
        return (
            method.usage_count >= self.config.elimination_min_support
            and method.retired_probe_count >= self.config.retired_probe_elimination_min_count
            and method.stats.progress_mean <= self.config.elimination_progress_threshold
            and method.stats.success_mean <= self.config.elimination_success_threshold
            and method.recent_progress_rate <= self.config.elimination_progress_threshold
            and method.recent_success_rate <= self.config.elimination_success_threshold
        )

    @staticmethod
    def _mark_active(method: AttackMethod) -> None:
        method.status = ACTIVE
        method.retired_probe_count = 0
        method.retired_since_usage_count = None

    @staticmethod
    def _mark_retired(method: AttackMethod) -> None:
        method.status = RETIRED
        method.retired_probe_count = 0
        method.retired_since_usage_count = method.usage_count

    def ensure_lifecycle_metadata(self) -> None:
        for method in self.methods.values():
            self._ensure_method_lifecycle_metadata(method)

    @staticmethod
    def _ensure_method_lifecycle_metadata(method: AttackMethod) -> None:
        if not hasattr(method, "retired_probe_count"):
            method.retired_probe_count = 0
        if not hasattr(method, "retired_since_usage_count"):
            method.retired_since_usage_count = None

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
        load_existing: bool = True,
    ):
        self.config = config or AttackConfig()
        self.path = load_path or self.config.persistence_path
        if load_path:
            self.path = load_path
        loaded = self._load_existing(self.path) if load_existing else None
        if loaded is not None:
            self.pool = loaded.pool
            self.config = loaded.config
            self.path = loaded.path
            self.pool.config = self.config
            self.pool.ensure_lifecycle_metadata()
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
