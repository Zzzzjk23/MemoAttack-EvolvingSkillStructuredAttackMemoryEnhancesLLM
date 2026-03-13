from __future__ import annotations

from typing import Iterable

from config.default_config import AttackConfig
from methods.method_schema import AttackMethod, AttackMethodProposal, AttackState


def mutate_method(
    attacker_llm,
    candidate_parent_methods: Iterable[AttackMethod],
    state: AttackState,
    existing_methods: Iterable[AttackMethod],
    config: AttackConfig,
) -> AttackMethodProposal:
    candidate_parent_methods = list(candidate_parent_methods)
    last_error = None
    for _ in range(max(1, config.proposal_retry_limit)):
        try:
            return attacker_llm.generate_method_proposal(
                goal=state.goal,
                target_str=state.target,
                attack_state=state,
                mode="mutate",
                existing_methods=list(existing_methods),
                candidate_parent_methods=candidate_parent_methods,
            )
        except Exception as exc:  # pragma: no cover - network/runtime failure path
            last_error = exc
    raise last_error
