from __future__ import annotations

import random
from typing import Optional

from bandit.thompson_sampling import ThompsonSample, select_method_via_thompson
from config.default_config import AttackConfig
from methods.method_registry import CategoryMethodPool
from methods.method_schema import AttackState


def select_existing_method(
    context: AttackState,
    category_state: CategoryMethodPool,
    config: AttackConfig,
    rng: Optional[random.Random] = None,
) -> Optional[ThompsonSample]:
    return select_method_via_thompson(
        state=context,
        pool=category_state,
        config=config,
        rng=rng,
    )


def select_parent_method_for_mutation(
    context: AttackState,
    category_state: CategoryMethodPool,
    config: AttackConfig,
    rng: Optional[random.Random] = None,
) -> Optional[ThompsonSample]:
    return select_existing_method(
        context=context,
        category_state=category_state,
        config=config,
        rng=rng,
    )
