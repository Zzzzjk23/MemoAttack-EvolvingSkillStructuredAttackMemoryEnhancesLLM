from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


class WandBLogger:
    def __init__(self, project_name="TAP_Jailbreak", config=None):
        self.enabled = wandb is not None
        self.run = wandb.init(project=project_name, config=config, reinit=True) if self.enabled else None

    def log_node(self, node):
        if not self.enabled:
            return
        attempt_result = getattr(node, "attempt_result", None)
        log_data = {
            "node_id": node.id,
            "parent_id": node.parent.id if node.parent else None,
            "depth": node.depth,
            "tree_goal": node.tree.goal,
            "prompt": node.prompt,
            "improvement": node.improvement,
            "on_topic": node.on_topic,
            "target_response": node.target_response,
            "outside_score": node.outside_score,
            "normalized_score": getattr(node, "normalized_score", None),
            "internal_score": node.internal_score,
            "attack_method": getattr(node, "attack_method", None),
            "attack_method_id": getattr(node, "attack_method_id", None),
            "mode": getattr(node, "mode", None),
        }
        if attempt_result is not None:
            log_data.update(
                {
                    "made_progress": attempt_result.made_progress,
                    "normalized_progress": attempt_result.normalized_progress,
                    "final_success": attempt_result.final_success,
                }
            )
        wandb.log(log_data)

    def log_success(self, goal, request_count, steps, success=True):
        if not self.enabled:
            return
        wandb.log(
            {
                "jailbreak_success": success,
                "final_request_count": request_count,
                "steps": steps,
                "goal": goal,
            }
        )

    def log_global_context(self, context_data):
        if not self.enabled:
            return
        wandb.log({"global_context": str(context_data)})

    def finish(self):
        if self.run is not None:
            self.run.finish()
