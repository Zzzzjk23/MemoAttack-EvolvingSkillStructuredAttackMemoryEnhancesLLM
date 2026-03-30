from __future__ import annotations

from embeddings.prompt_embedding import embed_prompt
from methods.method_schema import AttackState


def build_attack_state(tree, node) -> AttackState:
    history = list(getattr(node, "history", []) or [])
    current_prompt = getattr(node, "prompt", tree.goal)
    goal_embedding = tuple(tree.goal_ebd.tolist()) if tree.goal_ebd is not None else None
    global_context = getattr(tree, "global_context", None)
    prompt_embedding = (
        tuple(embed_prompt(current_prompt).tolist()) if current_prompt else None
    )
    return AttackState(
        goal=tree.goal,
        target=tree.target,
        current_prompt=current_prompt,
        current_target_response=getattr(node, "target_response", None),
        current_raw_score=float(getattr(node, "outside_score", 0) or 0),
        current_score=float(getattr(node, "normalized_score", 0.0) or 0.0),
        depth=getattr(node, "depth", 0),
        node_id=getattr(node, "id", None),
        recent_method_id=getattr(node, "attack_method_id", None),
        recent_mode=getattr(node, "mode", None),
        history=history,
        goal_embedding=goal_embedding,
        prompt_embedding=prompt_embedding,
        global_context_json=(
            tree.get_attacker_global_context_json() if global_context is not None else "[]"
        ),
    )


def append_attempt_to_history(node, attempt_result):
    history = list(getattr(node, "history", []) or [])
    history.append(attempt_result)
    return history
