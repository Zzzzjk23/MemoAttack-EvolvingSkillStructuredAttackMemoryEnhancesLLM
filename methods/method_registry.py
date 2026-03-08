from __future__ import annotations

from dataclasses import dataclass, field
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
class CategoryMethodPool:
    category_id: str
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
            category_id=self.category_id,
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
            self.prompt_categories_dict = loaded.prompt_categories_dict
            self.config = loaded.config
            self.path = loaded.path
        else:
            self.prompt_categories_dict: Dict[str, CategoryMethodPool] = {}

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
        return loaded

    def save(self, path: Optional[str] = None) -> str:
        save_path = Path(path or self.path)
        with save_path.open("wb") as handle:
            pickle.dump(self, handle)
        return str(save_path.resolve())

    def create_category(self, category_id: str) -> CategoryMethodPool:
        if category_id not in self.prompt_categories_dict:
            self.prompt_categories_dict[category_id] = CategoryMethodPool(
                category_id=category_id,
                config=self.config,
            )
        return self.prompt_categories_dict[category_id]

    def get_or_create_pool(self, category_id: str) -> CategoryMethodPool:
        return self.create_category(category_id)

    def register_method(
        self,
        category_id: str,
        proposal: AttackMethodProposal,
        created_via: str,
        parent_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> AttackMethod:
        pool = self.get_or_create_pool(category_id)
        return pool.register_method(
            proposal=proposal,
            created_via=created_via,
            parent_method_id=parent_method_id,
            metadata=metadata,
        )

    def record_attempt(
        self,
        category_id: str,
        method_id: str,
        attempt_result: AttackAttemptResult,
        prompt_text: str,
        before_prompt: str,
        after_prompt: str,
        target_response: str,
    ) -> AttackMethod:
        pool = self.get_or_create_pool(category_id)
        method = pool.record_attempt(
            method_id=method_id,
            attempt_result=attempt_result,
            prompt_text=prompt_text,
            before_prompt=before_prompt,
            after_prompt=after_prompt,
            target_response=target_response,
        )
        pool.apply_lifecycle_rules()
        return method

    def iter_methods(self, category_id: str) -> Iterable[AttackMethod]:
        return self.get_or_create_pool(category_id).get_all_methods()
