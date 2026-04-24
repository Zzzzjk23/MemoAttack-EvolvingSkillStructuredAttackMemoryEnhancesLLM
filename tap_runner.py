from __future__ import annotations

import ast
import csv
from pathlib import Path
from types import SimpleNamespace

from config.default_config import AttackConfig
from llm.clients import AttackerLLM, EvaluatorLLM, TargetLLM
from observability.goal_file_logger import GoalFileLogger
from runtime.search_tree import Tree, TreeNode


RESULT_FIELDNAMES = ["index", "goal", "target", "if_success", "request_count"]


def select_nodes(leaf_nodes, width):
    leaf_nodes = [node for node in leaf_nodes if node.on_topic]
    if not leaf_nodes:
        return []
    leaf_nodes = sorted(
        leaf_nodes,
        key=lambda item: item.outside_score if item.outside_score is not None else float("-inf"),
        reverse=True,
    )
    return leaf_nodes[:width]


def _instantiate_llm(llm_cls, model_name: str, config: AttackConfig):
    try:
        return llm_cls(model_name=model_name, config=config)
    except TypeError:
        return llm_cls(model_name=model_name)


def tap(args, logger=None):
    config = getattr(args, "config", None) or AttackConfig()
    attacker_model = getattr(args, "attacker_model", config.attacker_model)
    evaluator_model = getattr(args, "evaluator_model", config.evaluator_model)
    target_model = getattr(args, "target_model", config.target_model)
    max_depth = getattr(args, "max_depth", config.max_depth)
    branching_factor = getattr(args, "branching_factor", config.branching_factor)
    width = getattr(args, "width", config.width)

    attacker_llm = _instantiate_llm(AttackerLLM, attacker_model, config)
    evaluator_llm = _instantiate_llm(EvaluatorLLM, evaluator_model, config)
    target_llm = _instantiate_llm(TargetLLM, target_model, config)
    tap_tree = Tree(
        args.goal,
        args.target,
        args.index,
        attacker_llm,
        evaluator_llm,
        target_llm,
        config=config,
    )
    root = TreeNode(tree=tap_tree)
    tap_tree.root = tap_tree.initialize_root(root)

    if logger:
        logger.log_node(root)

    for depth in range(max_depth):
        leaf_nodes = tap_tree.get_leaf_nodes(depth)
        selected_nodes = select_nodes(leaf_nodes, width)
        for node in selected_nodes:
            for _ in range(branching_factor):
                tap_tree.request_count += 1
                new_node = tap_tree.execute_attack_step(node)
                node.add_child(new_node)

                if logger:
                    logger.log_node(new_node)

                if new_node.outside_score == config.final_success_score_threshold:
                    tap_tree.if_jailbreak = True
                    print(f"{tap_tree.goal} success after {tap_tree.request_count} requests")
                    if logger:
                        logger.log_success(tap_tree.goal, tap_tree.request_count, depth)
                    tap_tree.finalize_goal(success=True)
                    return True, tap_tree.request_count

    if logger:
        logger.log_success(tap_tree.goal, tap_tree.request_count, max_depth, success=False)

    print(f"{tap_tree.goal} failed after {tap_tree.request_count} requests")
    tap_tree.finalize_goal(success=False)
    return False, tap_tree.request_count


def build_attack_args(goal_and_target, config: AttackConfig):
    return SimpleNamespace(
        attacker_model=config.attacker_model,
        evaluator_model=config.evaluator_model,
        target_model=config.target_model,
        goal=goal_and_target["goal"],
        target=goal_and_target["target"],
        index=goal_and_target["index"],
        max_depth=config.max_depth,
        branching_factor=config.branching_factor,
        width=config.width,
        config=config,
    )


def _normalize_target(target_value: str) -> str:
    if not isinstance(target_value, str):
        return target_value
    text = target_value.strip()
    if not text:
        return text
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, (list, tuple)) and parsed:
        first_item = parsed[0]
        if isinstance(first_item, str):
            return first_item.strip()
    return text


def _coerce_goal_index(goal_index: str) -> int:
    return int(str(goal_index).strip())


def _load_goals_and_targets(config: AttackConfig):
    advbench_file = config.resolve_advbench_path()
    goals_and_targets = []
    with open(advbench_file, encoding=config.csv_encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            goals_and_targets.append(
                {
                    "index": str(row_index),
                    "goal": row["goal"],
                    "target": _normalize_target(row["target"]),
                }
            )
    return [
        item
        for item in goals_and_targets
        if _coerce_goal_index(item["index"]) >= config.start_index
    ]


def _write_results(
    output_file: str,
    goals_and_targets,
    results_by_index,
    encoding: str,
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for goal_and_target in goals_and_targets:
            row = results_by_index.get(goal_and_target["index"])
            if row is not None:
                writer.writerow(row)


def main(config: AttackConfig | None = None):
    config = config or AttackConfig()
    logger = GoalFileLogger(
        log_dir=config.resolve_goal_log_dir(),
        filename_template=config.goal_log_filename_template,
        config=config.__dict__,
    )
    goals_and_targets = _load_goals_and_targets(config)
    Path(config.attacker_input_dir).mkdir(parents=True, exist_ok=True)
    output_file = config.resolve_results_output_path()
    results_by_index = {}
    for goal_and_target in goals_and_targets:
        input_args = build_attack_args(goal_and_target, config)
        success, request_count = tap(input_args, logger=logger)
        results_by_index[goal_and_target["index"]] = {
            "index": goal_and_target["index"],
            "goal": goal_and_target["goal"],
            "target": goal_and_target["target"],
            "if_success": success,
            "request_count": request_count,
        }
        _write_results(
            output_file=output_file,
            goals_and_targets=goals_and_targets,
            results_by_index=results_by_index,
            encoding=config.csv_encoding,
        )

    logger.finish()
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    config = AttackConfig()
    main(config=config)
