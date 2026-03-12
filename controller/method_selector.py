from __future__ import annotations

import random
from typing import Optional

from bandit.thompson_sampling import ThompsonSample, select_method_via_thompson
from config.default_config import AttackConfig
from methods.method_registry import MethodPool
from methods.method_schema import AttackState


def select_existing_method(
    context: AttackState,
    method_pool: MethodPool,
    config: AttackConfig,
    rng: Optional[random.Random] = None,
) -> Optional[ThompsonSample]:
    return select_method_via_thompson(
        state=context,
        pool=method_pool,
        config=config,
        rng=rng,
    )


def select_parent_method_for_mutation(
    context: AttackState,
    method_pool: MethodPool,
    config: AttackConfig,
    rng: Optional[random.Random] = None,
) -> Optional[ThompsonSample]:
    return select_existing_method(
        context=context,
        method_pool=method_pool,
        config=config,
        rng=rng,
    )
