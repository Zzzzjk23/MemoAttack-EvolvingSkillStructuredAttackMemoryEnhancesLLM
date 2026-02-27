import wandb
import os
import json

class WandBLogger:
    def __init__(self, project_name="TAP_Jailbreak", config=None):
        """
        Initialize the WandB logger.
        :param project_name: Name of the WandB project.
        :param config: Dictionary containing run configuration (args, hyperparameters).
        """
        # Ensure we don't fail if wandb is not configured, but user specifically asked for it, 
        # so we assume they want it to run. 
        self.run = wandb.init(project=project_name, config=config, reinit=True)
        
        # We can create a Table for nodes to log them row by row if we want tabular view
        # Or just log metrics. The user asked for "log necessary info", a Table is often good for structural data.
        self.nodes_table = wandb.Table(columns=[
            "id", "parent_id", "depth", "goal", "target", "prompt", 
            "improvement", "on_topic", "target_response", 
            "outside_score", "internal_score", "timestamp"
        ])

    def log_node(self, node):
        """
        Log a TreeNode's information.
        """
        # Construct the data dictionary
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
            "internal_score": node.internal_score,
            # Flattening tree info if helpful, though redundant if logged in config
        }
        
        # Log as metrics for charts
        wandb.log(log_data)
        
        # Log to the table for detailed inspection
        # 'timestamp' isn't explicitly in node, we can add current wandb step or time
        # self.nodes_table.add_data(...) - Doing this row by row might be heavy if we re-log the whole table.
        # Better to log a separate artifact or just rely on the step-wise logging.
        # But for 'reading existing project', maybe they want to see the text.
        # Let's log specific important text fields clearly.

    def log_success(self, goal, request_count, steps, success=True):
        """
        Log success/failure event.
        """
        wandb.log({
            "jailbreak_success": success,
            "final_request_count": request_count,
            "goal": goal
        })

    def log_global_context(self, context_data):
        """
        Log global context queue.
        context_data: output of global_context.convert_to_json() or similar list
        """
        wandb.log({"global_context": str(context_data)})

    def finish(self):
        self.run.finish()
