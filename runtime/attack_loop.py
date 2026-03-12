from __future__ import annotations

from controller.method_selector import (
    select_existing_method,
    select_parent_method_for_mutation,
)
from controller.mode_selector import select_mode
from methods.invention import invent_method
from methods.method_schema import (
    MODE_INVENT,
    MODE_MUTATE,
    MODE_REUSE,
    AttackAttemptResult,
    AttackMethod,
)
from methods.mutation import mutate_method
from runtime.state_tracker import append_attempt_to_history, build_attack_state
from scoring.progress_metric import compute_normalized_gap_improvement


def _materialize_method(tree, state, pool, mode: str):
    debug = {}
    if mode == MODE_REUSE:
        selection = select_existing_method(state, pool, tree.config, rng=tree.random)
        if selection is None:
            mode = MODE_INVENT
        else:
            debug["thompson"] = {
                "utility": selection.utility,
                "sample_progress": selection.sample_progress_value,
                "sample_success": selection.sample_success_value,
                "context_bonus": selection.context_bonus,
            }
            return mode, selection.method, debug
    if mode == MODE_MUTATE:
        selection = select_parent_method_for_mutation(
            state,
            pool,
            tree.config,
            rng=tree.random,
        )
        if selection is None:
            mode = MODE_INVENT
        else:
            parent_method = selection.method
            proposal = mutate_method(
                attacker_llm=tree.attacker_llm,
                parent_method=parent_method,
                state=state,
                existing_methods=pool.get_all_methods(),
                config=tree.config,
            )
            method = tree.method_registry.register_method(
                proposal=proposal,
                created_via=MODE_MUTATE,
                parent_method_id=parent_method.method_id,
                metadata={"mutation_parent_name": parent_method.method_name},
            )
            debug["mutation_parent"] = parent_method.method_id
            return mode, method, debug
    proposal = invent_method(
        attacker_llm=tree.attacker_llm,
        state=state,
        existing_methods=pool.get_all_methods(),
        config=tree.config,
    )
    method = tree.method_registry.register_method(
        proposal=proposal,
        created_via=MODE_INVENT,
        parent_method_id=None,
        metadata={},
    )
    return MODE_INVENT, method, debug


def execute_attack_step(tree, parent_node):
    state = build_attack_state(tree, parent_node)
    pool = tree.method_registry.get_pool()
    mode = select_mode(state, pool, tree.config, rng=tree.random)
    mode, method, debug = _materialize_method(tree, state, pool, mode)

    examples = method.get_ranked_examples(
        prompt_text=tree.goal,
        limit=tree.config.method_example_limit,
    )
    conversation = tree.build_attack_conversation(
        parent_node=parent_node,
        attack_method=method,
        mode=mode,
        examples=examples,
    )
    draft = tree.attacker_llm.generate_attack_prompt(
        conversation=conversation,
        goal=tree.goal,
        target_str=tree.target,
        attack_state=state,
        attack_method=method,
        mode=mode,
        examples=examples,
    )
    conversation.append_message(
        conversation.roles[1],
        {
            "improvement": draft.improvement,
            "prompt": draft.prompt,
            "prompt_template": draft.prompt_template,
            "attack_plan": draft.attack_plan,
            "rationale": draft.rationale,
        },
    )

    on_topic = tree.evaluate_on_topic(draft.prompt)
    target_response = tree.get_target_response(draft.prompt) if on_topic else None
    raw_score = tree.evaluate_outside_score(target_response) if on_topic else 0
    normalized_score = tree.normalize_score(raw_score)
    prev_score = state.current_score
    normalized_progress = compute_normalized_gap_improvement(
        prev_score=prev_score,
        new_score=normalized_score,
        max_score=tree.config.max_score,
        epsilon=tree.config.epsilon,
    )
    made_progress = normalized_progress >= tree.config.progress_threshold
    final_success = raw_score >= tree.config.final_success_score_threshold

    attempt_result = AttackAttemptResult(
        used_method_id=method.method_id,
        mode=mode,
        prev_score=prev_score,
        new_score=normalized_score,
        normalized_progress=normalized_progress,
        made_progress=made_progress,
        final_success=final_success,
        response={
            "improvement": draft.improvement,
            "prompt": draft.prompt,
            "prompt_template": draft.prompt_template,
            "attack_plan": draft.attack_plan,
            "rationale": draft.rationale,
        },
        target_response=target_response,
        attack_prompt=draft.prompt,
        raw_prev_score=state.current_raw_score,
        raw_new_score=raw_score,
        metadata={
            "mode_debug": debug,
            "method_name": method.method_name,
            "method_status": method.status,
        },
    )

    tree.method_registry.record_attempt(
        method_id=method.method_id,
        attempt_result=attempt_result,
        prompt_text=tree.goal,
        before_prompt=getattr(parent_node, "prompt", tree.goal),
        after_prompt=draft.prompt,
        target_response=target_response or "",
    )

    child_node = tree.create_child_node(parent_node=parent_node)
    child_node.populate_from_attempt(
        prompt=draft.prompt,
        improvement=draft.improvement,
        conv=conversation,
        attack_method=method,
        mode=mode,
        attempt_result=attempt_result,
        on_topic=on_topic,
        target_response=target_response,
        outside_score=raw_score,
        history=append_attempt_to_history(parent_node, attempt_result),
    )
    tree.dump_attacker_input(child_node)
    return child_node
