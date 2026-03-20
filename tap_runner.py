from __future__ import annotations

import csv
import os
from types import SimpleNamespace

from config.default_config import AttackConfig
from llm.clients import AttackerLLM, EvaluatorLLM, TargetLLM
from observability.wandb_logger import WandBLogger
from runtime.search_tree import Tree, TreeNode

import requests

def send_pushdeer(text):
    key = "PDU39095TAyMpaKOD01BmimCnsXTyzUaZsDNBk7nJ"
    url = f"https://api2.pushdeer.com/message/push?pushkey={key}&text={text}"
    requests.get(url)


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
                    print(f"{tap_tree.goal} 越狱成功，共生成{tap_tree.request_count}个请求")
                    if logger:
                        logger.log_success(tap_tree.goal, tap_tree.request_count, depth)
                    tap_tree.save_global_context()
                    tap_tree.method_registry.save()
                    return True, tap_tree.request_count

    if logger:
        logger.log_success(tap_tree.goal, tap_tree.request_count, max_depth, success=False)

    print(f"{tap_tree.goal} 越狱失败，共生成{tap_tree.request_count}个请求")
    tap_tree.save_global_context()
    tap_tree.method_registry.save()
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


def main(subset_index, config: AttackConfig | None = None):
    config = config or AttackConfig()
    logger = WandBLogger(project_name=config.wandb_project_name, config=config.__dict__)
    advbench_file = config.resolve_advbench_subset_path(subset_index)
    goals_and_targets = []
    os.makedirs(config.attacker_input_dir, exist_ok=True)
    with open(advbench_file, encoding=config.csv_encoding) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            goals_and_targets.append(
                {
                    "index": row["Unnamed: 0"],
                    "goal": row["goal"],
                    "target": row["target"],
                }
            )
    results = []
    for goal_and_target in goals_and_targets:
        input_args = build_attack_args(goal_and_target, config)
        success, request_count = tap(input_args, logger=logger)
        results.append(
            {
                "goal": goal_and_target["goal"],
                "target": goal_and_target["target"],
                "if_success": success,
                "request_count": request_count,
            }
        )

    logger.finish()

    output_file = config.resolve_results_output_path(subset_index)
    with open(output_file, "w", newline="", encoding=config.csv_encoding) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["goal", "target", "if_success", "request_count"],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    config = AttackConfig()
    try:
        for subset_index in range(config.subset_start_index, config.subset_end_index + 1):
            main(subset_index, config=config)
    except Exception as e:
        msg = f"程序报错中止：{str(e)}"
        send_pushdeer(msg)
        raise e