from __future__ import annotations

import random
from typing import List, Optional

from bandit.thompson_sampling import (
    ThompsonSample,
    select_methods_via_thompson,
)
from config.default_config import AttackConfig
from methods.method_registry import MethodPool
from methods.method_schema import AttackState


def select_existing_methods(
    context: AttackState,
    method_pool: MethodPool,
    config: AttackConfig,
    rng: Optional[random.Random] = None,
    limit: Optional[int] = None,
) -> List[ThompsonSample]:
    resolved_limit = max(1, limit or config.thompson_candidate_method_count)
    return select_methods_via_thompson(
        state=context,
        pool=method_pool,
        config=config,
        rng=rng,
        limit=resolved_limit,
    )


def select_parent_methods_for_mutation(
    context: AttackState,
    method_pool: MethodPool,
    config: AttackConfig,
    rng: Optional[random.Random] = None,
    limit: Optional[int] = None,
) -> List[ThompsonSample]:
    return select_existing_methods(
        context=context,
        method_pool=method_pool,
        config=config,
        rng=rng,
        limit=limit,
    )
