from __future__ import annotations

import random

from methods.method_schema import MethodPosteriorStats


def sample_progress(stats: MethodPosteriorStats, rng: random.Random) -> float:
    return rng.betavariate(stats.progress_alpha, stats.progress_beta)


def sample_success(stats: MethodPosteriorStats, rng: random.Random) -> float:
    return rng.betavariate(stats.success_alpha, stats.success_beta)


def posterior_mean(alpha: float, beta: float) -> float:
    total = alpha + beta
    return alpha / total if total else 0.0
