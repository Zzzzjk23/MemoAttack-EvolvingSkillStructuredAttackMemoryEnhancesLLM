from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Optional

from bandit.contextual_bandit import build_context_features, compute_context_bonus
from bandit.posterior import sample_progress, sample_success
from config.default_config import AttackConfig
from methods.method_registry import MethodPool
from methods.method_schema import AttackMethod, AttackState


@dataclass
class ThompsonSample:
    method: AttackMethod
    utility: float
    sample_progress_value: float
    sample_success_value: float
    context_bonus: float


def sample_method_utility(
    method: AttackMethod,
    state: AttackState,
    pool: MethodPool,
    config: AttackConfig,
    rng: random.Random,
) -> ThompsonSample:
    sampled_progress = sample_progress(method.stats, rng)
    sampled_success = sample_success(method.stats, rng)
    context_features = build_context_features(state, method, pool)
    context_bonus = compute_context_bonus(context_features, config)
    utility = (
        config.thompson_progress_weight * sampled_progress
        + config.thompson_success_weight * sampled_success
        + context_bonus
    )
    return ThompsonSample(
        method=method,
        utility=utility,
        sample_progress_value=sampled_progress,
        sample_success_value=sampled_success,
        context_bonus=context_bonus,
    )


def select_method_via_thompson(
    state: AttackState,
    pool: MethodPool,
    config: AttackConfig,
    rng: Optional[random.Random] = None,
) -> Optional[ThompsonSample]:
    rng = rng or random.Random()
    methods: List[AttackMethod] = pool.get_active_methods()
    if not methods:
        return None
    samples = [
        sample_method_utility(
            method=method,
            state=state,
            pool=pool,
            config=config,
            rng=rng,
        )
        for method in methods
    ]
    return max(samples, key=lambda item: item.utility)
