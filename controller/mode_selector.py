from __future__ import annotations

import math
import random
from typing import Dict

from config.default_config import AttackConfig
from methods.method_registry import MethodPool
from methods.method_schema import MODE_INVENT, MODE_MUTATE, MODE_REUSE, AttackState


def _softmax_sample(scores: Dict[str, float], rng: random.Random) -> str:
    keys = list(scores.keys())
    values = list(scores.values())
    max_score = max(values)
    weights = [math.exp(value - max_score) for value in values]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    threshold = rng.random()
    cumulative = 0.0
    for key, probability in zip(keys, probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return key
    return keys[-1]


def select_mode(
    context: AttackState,
    method_pool: MethodPool,
    config: AttackConfig,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random()
    if not method_pool.has_selectable_methods():
        return MODE_INVENT

    scores = {
        MODE_REUSE: config.mode_reuse_bias,
        MODE_MUTATE: config.mode_mutate_bias,
        MODE_INVENT: config.mode_invent_bias,
    }
    active_methods = method_pool.get_active_methods()
    retired_methods = method_pool.get_retired_methods()
    if len(active_methods) <= config.sparse_pool_threshold:
        cold_start_bonus = config.mode_cold_start_bonus
        if retired_methods:
            cold_start_bonus *= 0.5
            scores[MODE_REUSE] += config.mode_cold_start_bonus
        scores[MODE_INVENT] += cold_start_bonus
    if context.current_score <= config.low_score_threshold:
        scores[MODE_INVENT] += config.mode_low_score_invent_bonus
    if context.recent_mode:
        scores[context.recent_mode] -= config.mode_repeat_penalty
    recent_history = context.history[-config.recent_performance_window :]
    if recent_history and not any(item.made_progress for item in recent_history):
        scores[MODE_MUTATE] += config.mode_stagnation_mutate_bonus
    if recent_history and any(item.made_progress for item in recent_history[-2:]):
        scores[MODE_REUSE] += config.mode_recent_progress_reuse_bonus
    return _softmax_sample(scores, rng)
