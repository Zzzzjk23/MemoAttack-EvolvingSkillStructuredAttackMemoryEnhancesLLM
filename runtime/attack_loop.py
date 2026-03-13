from __future__ import annotations

from controller.method_selector import (
    select_existing_methods,
    select_parent_methods_for_mutation,
)
from controller.mode_selector import select_mode
from methods.invention import invent_method
from methods.method_schema import (
    MODE_INVENT,
    MODE_MUTATE,
    MODE_REUSE,
    AttackAttemptResult,
)
from methods.mutation import mutate_method
from runtime.state_tracker import append_attempt_to_history, build_attack_state
from scoring.progress_metric import compute_normalized_gap_improvement


def _serialize_thompson_samples(samples):
    return [
        {
            "rank": index,
            "method_id": sample.method.method_id,
            "method_name": sample.method.method_name,
            "utility": sample.utility,
            "sample_progress": sample.sample_progress_value,
            "sample_success": sample.sample_success_value,
            "context_bonus": sample.context_bonus,
        }
        for index, sample in enumerate(samples, start=1)
    ]


def _resolve_selected_methods(candidate_methods, selected_names, fallback_text: str | None = None):
    if not candidate_methods:
        return []
    resolved = []
    seen_ids = set()
    normalized_name_map = {
        method.method_name.strip().lower(): method for method in candidate_methods
    }
    for name in selected_names or []:
        method = normalized_name_map.get(str(name).strip().lower())
        if method is not None and method.method_id not in seen_ids:
            resolved.append(method)
            seen_ids.add(method.method_id)
    if resolved:
        return resolved

    if fallback_text:
        fallback_text_lower = fallback_text.lower()
        for method in candidate_methods:
            if method.method_name.lower() in fallback_text_lower and method.method_id not in seen_ids:
                resolved.append(method)
                seen_ids.add(method.method_id)
    if resolved:
        return resolved
    return [candidate_methods[0]]


def _collect_examples(candidate_methods, prompt_text: str, per_method_limit: int):
    examples = []
    seen = set()
    for method in candidate_methods:
        for example in method.get_ranked_examples(prompt_text=prompt_text, limit=per_method_limit):
            key = (
                example.prompt_text,
                example.before_prompt,
                example.after_prompt,
                example.target_response,
            )
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
    return examples


def _materialize_method(tree, state, pool, mode: str):
    debug = {}
    if mode == MODE_REUSE:
        selections = select_existing_methods(state, pool, tree.config, rng=tree.random)
        if not selections:
            mode = MODE_INVENT
        else:
            candidate_methods = [sample.method for sample in selections]
            debug["thompson_candidates"] = _serialize_thompson_samples(selections)
            return mode, candidate_methods[0], candidate_methods, debug
    if mode == MODE_MUTATE:
        selections = select_parent_methods_for_mutation(
            state,
            pool,
            tree.config,
            rng=tree.random,
        )
        if not selections:
            mode = MODE_INVENT
        else:
            candidate_parent_methods = [sample.method for sample in selections]
            proposal = mutate_method(
                attacker_llm=tree.attacker_llm,
                candidate_parent_methods=candidate_parent_methods,
                state=state,
                existing_methods=pool.get_all_methods(),
                config=tree.config,
            )
            selected_parent_methods = _resolve_selected_methods(
                candidate_parent_methods,
                proposal.selected_parent_method_names,
                fallback_text=proposal.mutation_of,
            )
            primary_parent_method = selected_parent_methods[0]
            method = tree.method_registry.register_method(
                proposal=proposal,
                created_via=MODE_MUTATE,
                parent_method_id=primary_parent_method.method_id,
                metadata={
                    "mutation_parent_name": primary_parent_method.method_name,
                    "candidate_parent_method_ids": [
                        item.method_id for item in candidate_parent_methods
                    ],
                    "candidate_parent_method_names": [
                        item.method_name for item in candidate_parent_methods
                    ],
                    "selected_parent_method_ids": [
                        item.method_id for item in selected_parent_methods
                    ],
                    "selected_parent_method_names": [
                        item.method_name for item in selected_parent_methods
                    ],
                },
            )
            debug["thompson_candidates"] = _serialize_thompson_samples(selections)
            debug["selected_parent_method_ids"] = [
                item.method_id for item in selected_parent_methods
            ]
            debug["selected_parent_method_names"] = [
                item.method_name for item in selected_parent_methods
            ]
            return mode, method, [method], debug
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
    return MODE_INVENT, method, [method], debug


def execute_attack_step(tree, parent_node):
    state = build_attack_state(tree, parent_node)
    pool = tree.method_registry.get_pool()
    mode = select_mode(state, pool, tree.config, rng=tree.random)
    mode, method, candidate_methods, debug = _materialize_method(tree, state, pool, mode)

    examples = _collect_examples(
        candidate_methods=candidate_methods,
        prompt_text=tree.goal,
        per_method_limit=tree.config.method_example_limit,
    )
    conversation = tree.build_attack_conversation(
        parent_node=parent_node,
        candidate_methods=candidate_methods,
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
        candidate_methods=candidate_methods,
    )
    conversation.append_message(
        conversation.roles[1],
        {
            "improvement": draft.improvement,
            "prompt": draft.prompt,
            "selected_method_names": draft.selected_method_names,
            "prompt_template": draft.prompt_template,
            "attack_plan": draft.attack_plan,
            "rationale": draft.rationale,
        },
    )

    selected_methods = _resolve_selected_methods(
        candidate_methods,
        draft.selected_method_names,
    )
    primary_selected_method = selected_methods[0]

    on_topic = tree.evaluate_on_topic(draft.prompt)
    target_response = tree.get_target_response(draft.prompt) if on_topic else None
    raw_score = tree.evaluate_outside_score(target_response) if on_topic else 0
    tree.global_context.enqueue((raw_score, draft.prompt))
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
        used_method_id=primary_selected_method.method_id,
        mode=mode,
        prev_score=prev_score,
        new_score=normalized_score,
        normalized_progress=normalized_progress,
        made_progress=made_progress,
        final_success=final_success,
        response={
            "improvement": draft.improvement,
            "prompt": draft.prompt,
            "selected_method_names": draft.selected_method_names,
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
            "method_name": primary_selected_method.method_name,
            "method_status": primary_selected_method.status,
            "candidate_method_ids": [item.method_id for item in candidate_methods],
            "candidate_method_names": [item.method_name for item in candidate_methods],
            "selected_method_ids": [item.method_id for item in selected_methods],
            "selected_method_names": [item.method_name for item in selected_methods],
            "materialized_method_id": method.method_id,
            "materialized_method_name": method.method_name,
        },
    )

    for selected_method in selected_methods:
        tree.method_registry.record_attempt(
            method_id=selected_method.method_id,
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
        attack_method=primary_selected_method,
        selected_methods=selected_methods,
        candidate_methods=candidate_methods,
        mode=mode,
        attempt_result=attempt_result,
        on_topic=on_topic,
        target_response=target_response,
        outside_score=raw_score,
        history=append_attempt_to_history(parent_node, attempt_result),
    )
    tree.dump_attacker_input(child_node)
    return child_node
