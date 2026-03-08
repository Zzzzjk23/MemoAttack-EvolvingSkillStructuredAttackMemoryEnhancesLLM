from __future__ import annotations

from typing import Dict

from config.default_config import AttackConfig
from methods.method_registry import CategoryMethodPool
from methods.method_schema import AttackMethod, AttackState, tuple_to_array
from embeddings.prompt_embedding import cosine_similarity


def build_context_features(
    state: AttackState,
    method: AttackMethod,
    pool: CategoryMethodPool,
) -> Dict[str, float]:
    goal_similarity = 0.0
    if state.goal_embedding is not None and method.method_embedding is not None:
        goal_similarity = cosine_similarity(
            tuple_to_array(state.goal_embedding),
            tuple_to_array(method.method_embedding),
        )
    score_gap = max(0.0, 1.0 - state.current_score)
    repeat_penalty = 1.0 if state.recent_method_id == method.method_id else 0.0
    total_usage = sum(item.usage_count for item in pool.get_all_methods())
    pool_share = method.usage_count / max(total_usage, 1)
    return {
        "goal_similarity": goal_similarity,
        "score_gap": score_gap,
        "recent_progress_rate": method.recent_progress_rate,
        "recent_success_rate": method.recent_success_rate,
        "repeat_penalty": repeat_penalty,
        "pool_usage_share": pool_share,
        "is_newborn": 1.0 if method.usage_count == 0 else 0.0,
    }


def compute_context_bonus(
    context_features: Dict[str, float],
    config: AttackConfig,
) -> float:
    bonus = 0.0
    bonus += config.context_similarity_weight * context_features["goal_similarity"]
    bonus += (
        config.context_recent_progress_weight
        * context_features["recent_progress_rate"]
    )
    bonus += (
        config.context_recent_success_weight
        * context_features["recent_success_rate"]
    )
    bonus += config.context_gap_weight * context_features["score_gap"]
    bonus += config.newborn_bonus * context_features["is_newborn"]
    bonus -= config.overuse_penalty_weight * context_features["pool_usage_share"]
    bonus -= config.mode_repeat_penalty * context_features["repeat_penalty"]
    return bonus
