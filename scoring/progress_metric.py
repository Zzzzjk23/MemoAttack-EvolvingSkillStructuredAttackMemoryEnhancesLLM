from __future__ import annotations


def compute_normalized_gap_improvement(
    prev_score: float,
    new_score: float,
    max_score: float = 1.0,
    epsilon: float = 1e-6,
) -> float:
    remaining_gap_prev = max_score - prev_score
    remaining_gap_new = max_score - new_score
    improvement = remaining_gap_prev - remaining_gap_new
    denominator = max(remaining_gap_prev, epsilon)
    return improvement / denominator
