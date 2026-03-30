from __future__ import annotations

from config.default_config import AttackConfig
from methods.method_registry import MethodRegistry
from methods.method_schema import AttackAttemptResult, AttackMethod
from runtime.global_context import GlobalContextQueue


MODE_BOOTSTRAP_DISTILLED = "bootstrap_distilled"


def _seed_method_for_bootstrap(method: AttackMethod, config: AttackConfig) -> None:
    floor = max(float(config.posterior_beta_floor), 1e-6)
    method.stats.progress_alpha = floor
    method.stats.progress_beta = floor
    method.stats.success_alpha = floor
    method.stats.success_beta = floor
    method.usage_count = 0
    method.recent_progress_history = []
    method.recent_success_history = []
    method.recent_progress_values = []
    method.example_records = []
    method.metadata["bootstrap_seeded"] = True


def build_registry_from_global_context(
    *,
    global_context: GlobalContextQueue,
    evaluator_llm,
    config: AttackConfig,
) -> MethodRegistry:
    registry = MethodRegistry(config=config, load_existing=False)
    sorted_records = sorted(
        global_context.entries(),
        key=lambda entry: (entry.after_score, entry.timestamp),
        reverse=True,
    )
    for entry in sorted_records:
        existing_methods = registry.get_pool().get_all_methods()
        proposal = evaluator_llm.distill_method_proposal(
            record=entry.to_dict(),
            existing_methods=existing_methods,
        )
        method = registry.register_method(
            proposal=proposal,
            created_via=MODE_BOOTSTRAP_DISTILLED,
            parent_method_id=None,
            metadata={
                "bootstrap_distilled": True,
                "source_goal_index": entry.goal_index,
            },
        )
        if not method.metadata.get("bootstrap_seeded", False):
            _seed_method_for_bootstrap(method, config)

        attempt_result = AttackAttemptResult(
            used_method_id=method.method_id,
            mode=MODE_BOOTSTRAP_DISTILLED,
            prev_score=max(0.0, min(config.max_score, entry.before_score / config.judge_max_score)),
            new_score=max(0.0, min(config.max_score, entry.after_score / config.judge_max_score)),
            normalized_progress=entry.normalized_progress,
            made_progress=True,
            final_success=True,
            response={
                "improvement": entry.improvement,
                "prompt": entry.after_prompt,
                "selected_method_names": [method.method_name],
            },
            target_response=entry.target_response,
            attack_prompt=entry.after_prompt,
            raw_prev_score=entry.before_score,
            raw_new_score=entry.after_score,
            metadata={
                "bootstrap_seed": True,
                "goal_index": entry.goal_index,
                "goal": entry.goal,
            },
        )
        registry.record_attempt(
            method_id=method.method_id,
            attempt_result=attempt_result,
            prompt_text=entry.goal,
            before_prompt=entry.before_prompt,
            after_prompt=entry.after_prompt,
            target_response=entry.target_response,
        )
    return registry
