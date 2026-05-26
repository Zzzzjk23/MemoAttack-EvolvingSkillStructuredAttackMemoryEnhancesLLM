from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np

from embeddings.prompt_embedding import cosine_similarity, embed_prompt


ACTIVE = "active"
RETIRED = "retired"
ELIMINATED = "eliminated"

MODE_REUSE = "reuse"
MODE_MUTATE = "mutate"
MODE_INVENT = "invent"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embedding_to_tuple(text: str) -> Optional[Tuple[float, ...]]:
    if not text:
        return None
    embedding = embed_prompt(text)
    if embedding is None:
        return None
    return tuple(np.asarray(embedding).reshape(-1).tolist())


def _tuple_to_array(value: Optional[Tuple[float, ...]]) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    return [str(item).strip() for item in items if str(item).strip()]


@dataclass
class MethodPosteriorStats:
    progress_alpha: float
    progress_beta: float
    success_alpha: float
    success_beta: float

    def update_progress(self, made_progress: bool) -> None:
        if made_progress:
            self.progress_alpha += 1.0
        else:
            self.progress_beta += 1.0

    def update_success(self, final_success: bool) -> None:
        if final_success:
            self.success_alpha += 1.0
        else:
            self.success_beta += 1.0

    @property
    def progress_mean(self) -> float:
        total = self.progress_alpha + self.progress_beta
        return self.progress_alpha / total if total else 0.0

    @property
    def success_mean(self) -> float:
        total = self.success_alpha + self.success_beta
        return self.success_alpha / total if total else 0.0


@dataclass
class MethodExampleRecord:
    prompt_text: str
    prompt_embedding: Optional[Tuple[float, ...]]
    before_prompt: str
    after_prompt: str
    target_response: str
    normalized_progress: float
    final_success: bool
    timestamp: str = field(default_factory=_utc_now)


@dataclass
class AttackMethodProposal:
    method_name: str
    method_description: str
    method_rationale: str
    mutation_of: Optional[str]
    prompt_template: str
    attack_plan: str
    applicability: str
    novelty_note: str
    expected_mechanism: str
    selected_parent_method_names: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackMethodProposal":
        proposal = cls(
            method_name=str(data.get("method_name", "")).strip(),
            method_description=str(data.get("method_description", "")).strip(),
            method_rationale=str(data.get("method_rationale", "")).strip(),
            mutation_of=(
                str(data["mutation_of"]).strip() if data.get("mutation_of") else None
            ),
            prompt_template=str(data.get("prompt_template", "")).strip(),
            attack_plan=str(data.get("attack_plan", "")).strip(),
            applicability=str(data.get("applicability", "")).strip(),
            novelty_note=str(data.get("novelty_note", "")).strip(),
            expected_mechanism=str(data.get("expected_mechanism", "")).strip(),
            selected_parent_method_names=_coerce_string_list(
                data.get("selected_parent_method_names")
            ),
            metadata=dict(data.get("metadata", {}) or {}),
        )
        proposal.validate()
        return proposal

    def validate(self) -> None:
        required_fields = [
            self.method_name,
            self.method_description,
            self.method_rationale,
            self.prompt_template,
            self.attack_plan,
            self.applicability,
            self.novelty_note,
            self.expected_mechanism,
        ]
        if any(not field for field in required_fields):
            raise ValueError("AttackMethodProposal contains empty required fields")

    def summary_text(self) -> str:
        return " ".join(
            [
                self.method_name,
                self.method_description,
                self.method_rationale,
                self.attack_plan,
                self.applicability,
                self.expected_mechanism,
            ]
        )


@dataclass
class AttackPromptDraft:
    improvement: str
    prompt: str
    selected_method_names: List[str] = field(default_factory=list)
    prompt_template: str = ""
    attack_plan: str = ""
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AttackPromptDraft":
        draft = cls(
            improvement=str(data.get("improvement", "")).strip(),
            prompt=str(data.get("prompt", "")).strip(),
            selected_method_names=_coerce_string_list(data.get("selected_method_names")),
            prompt_template=str(data.get("prompt_template", "")).strip(),
            attack_plan=str(data.get("attack_plan", "")).strip(),
            rationale=str(data.get("rationale", "")).strip(),
            metadata=dict(data.get("metadata", {}) or {}),
        )
        if not draft.improvement or not draft.prompt:
            raise ValueError("AttackPromptDraft requires non-empty improvement and prompt")
        return draft


@dataclass
class AttackAttemptResult:
    used_method_id: str
    mode: str
    prev_score: float
    new_score: float
    normalized_progress: float
    made_progress: bool
    final_success: bool
    response: Dict[str, Any]
    target_response: Optional[str]
    attack_prompt: str
    raw_prev_score: float = 0.0
    raw_new_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackState:
    goal: str
    target: str
    current_prompt: str
    current_target_response: Optional[str]
    current_raw_score: float
    current_score: float
    depth: int
    node_id: Optional[str]
    recent_method_id: Optional[str]
    recent_mode: Optional[str]
    history: List[AttackAttemptResult] = field(default_factory=list)
    goal_embedding: Optional[Tuple[float, ...]] = None
    prompt_embedding: Optional[Tuple[float, ...]] = None
    global_context_json: str = "[]"

    @property
    def score_trajectory(self) -> List[float]:
        values = [attempt.new_score for attempt in self.history]
        if self.current_score:
            values.append(self.current_score)
        return values

@dataclass
class AttackMethod:
    method_id: str
    method_name: str
    method_description: str
    method_rationale: str
    prompt_template: str
    attack_plan: str
    applicability: str
    novelty_note: str
    expected_mechanism: str
    parent_method_id: Optional[str]
    created_via: str
    creation_time: str
    stats: MethodPosteriorStats
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    recent_progress_history: List[bool] = field(default_factory=list)
    recent_success_history: List[bool] = field(default_factory=list)
    recent_progress_values: List[float] = field(default_factory=list)
    status: str = ACTIVE
    example_records: List[MethodExampleRecord] = field(default_factory=list)
    method_embedding: Optional[Tuple[float, ...]] = None

    @classmethod
    def from_proposal(
        cls,
        proposal: AttackMethodProposal,
        created_via: str,
        stats: MethodPosteriorStats,
        parent_method_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "AttackMethod":
        combined_metadata = dict(proposal.metadata)
        if metadata:
            combined_metadata.update(metadata)
        return cls(
            method_id=str(uuid.uuid4()),
            method_name=proposal.method_name,
            method_description=proposal.method_description,
            method_rationale=proposal.method_rationale,
            prompt_template=proposal.prompt_template,
            attack_plan=proposal.attack_plan,
            applicability=proposal.applicability,
            novelty_note=proposal.novelty_note,
            expected_mechanism=proposal.expected_mechanism,
            parent_method_id=parent_method_id,
            created_via=created_via,
            creation_time=_utc_now(),
            stats=stats,
            metadata=combined_metadata,
            method_embedding=_embedding_to_tuple(proposal.summary_text()),
        )

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
        return 0.55 * self.stats.progress_mean + 0.45 * self.stats.success_mean

    def update_recent_windows(
        self,
        made_progress: bool,
        final_success: bool,
        normalized_progress: float,
        window_size: int,
    ) -> None:
        self.recent_progress_history.append(made_progress)
        self.recent_success_history.append(final_success)
        self.recent_progress_values.append(normalized_progress)
        del self.recent_progress_history[:-window_size]
        del self.recent_success_history[:-window_size]
        del self.recent_progress_values[:-window_size]

    def add_example(
        self,
        prompt_text: str,
        before_prompt: str,
        after_prompt: str,
        target_response: str,
        normalized_progress: float,
        final_success: bool,
    ) -> None:
        self.example_records.append(
            MethodExampleRecord(
                prompt_text=prompt_text,
                prompt_embedding=_embedding_to_tuple(prompt_text),
                before_prompt=before_prompt,
                after_prompt=after_prompt,
                target_response=target_response,
                normalized_progress=normalized_progress,
                final_success=final_success,
            )
        )

    def get_ranked_examples(
        self,
        prompt_text: str,
        limit: int,
    ) -> List[MethodExampleRecord]:
        if not self.example_records:
            return []
        input_embedding = embed_prompt(prompt_text)
        if input_embedding is None:
            return self.example_records[-limit:]
        ranked = sorted(
            self.example_records,
            key=lambda record: cosine_similarity(
                input_embedding,
                _tuple_to_array(record.prompt_embedding),
            )
            if record.prompt_embedding is not None
            else -1.0,
            reverse=True,
        )
        return ranked[:limit]

    def similarity_to(self, proposal: AttackMethodProposal) -> float:
        if self.method_embedding is None:
            return 0.0
        proposal_embedding = _embedding_to_tuple(proposal.summary_text())
        if proposal_embedding is None:
            return 0.0
        return cosine_similarity(
            _tuple_to_array(self.method_embedding),
            _tuple_to_array(proposal_embedding),
        )


def tuple_to_array(value: Optional[Tuple[float, ...]]) -> Optional[np.ndarray]:
    return _tuple_to_array(value)


def build_method_stats(
    progress_alpha: float,
    progress_beta: float,
    success_alpha: float,
    success_beta: float,
) -> MethodPosteriorStats:
    return MethodPosteriorStats(
        progress_alpha=progress_alpha,
        progress_beta=progress_beta,
        success_alpha=success_alpha,
        success_beta=success_beta,
    )
