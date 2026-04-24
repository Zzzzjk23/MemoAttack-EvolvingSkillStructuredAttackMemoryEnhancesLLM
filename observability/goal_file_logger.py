from __future__ import annotations

import json
from pathlib import Path


class GoalFileLogger:
    def __init__(
        self,
        log_dir: str = "goal_logs",
        filename_template: str = "goal_{goal_index}.json",
        config=None,
    ):
        self.log_dir = Path(log_dir)
        self.filename_template = filename_template
        self.config = dict(config or {})
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._goal_payloads: dict[str, dict] = {}
        self._latest_goal_key: str | None = None

    def _get_goal_key(self, node) -> str:
        return str(node.tree.index)

    def _get_goal_path(self, goal_key: str) -> Path:
        return self.log_dir / self.filename_template.format(goal_index=goal_key)

    def _serialize_attempt_result(self, attempt_result):
        if attempt_result is None:
            return None
        return {
            "made_progress": attempt_result.made_progress,
            "normalized_progress": attempt_result.normalized_progress,
            "final_success": attempt_result.final_success,
            "raw_prev_score": getattr(attempt_result, "raw_prev_score", None),
            "raw_new_score": getattr(attempt_result, "raw_new_score", None),
            "mode": getattr(attempt_result, "mode", None),
            "used_method_id": getattr(attempt_result, "used_method_id", None),
        }

    def _serialize_node(self, node) -> dict:
        return {
            "node_id": node.id,
            "parent_id": node.parent.id if node.parent else None,
            "depth": node.depth,
            "request_count": node.tree.request_count,
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
            "selected_method_names": list(getattr(node, "selected_method_names", []) or []),
            "candidate_method_names": list(getattr(node, "candidate_method_names", []) or []),
            "attempt_result": self._serialize_attempt_result(
                getattr(node, "attempt_result", None)
            ),
        }

    def _build_goal_payload(self, node) -> dict:
        return {
            "goal_index": str(node.tree.index),
            "goal": node.tree.goal,
            "target": node.tree.target,
            "status": "running",
            "request_count": node.tree.request_count,
            "node_count": 0,
            "steps": None,
            "success": None,
            "config": self.config,
            "nodes": [],
        }

    def _write_payload(self, goal_key: str) -> None:
        path = self._get_goal_path(goal_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self._goal_payloads[goal_key], handle, ensure_ascii=False, indent=2)

    def log_node(self, node):
        goal_key = self._get_goal_key(node)
        if node.parent is None or goal_key not in self._goal_payloads:
            self._goal_payloads[goal_key] = self._build_goal_payload(node)
        payload = self._goal_payloads[goal_key]
        payload["status"] = "running"
        payload["request_count"] = node.tree.request_count
        payload["node_count"] = len(payload["nodes"]) + 1
        payload["last_node_id"] = node.id
        payload["nodes"].append(self._serialize_node(node))
        self._latest_goal_key = goal_key
        self._write_payload(goal_key)

    def log_success(self, goal, request_count, steps, success=True):
        if self._latest_goal_key is None:
            return
        payload = self._goal_payloads[self._latest_goal_key]
        payload["goal"] = goal
        payload["request_count"] = request_count
        payload["steps"] = steps
        payload["success"] = success
        payload["status"] = "success" if success else "failed"
        self._write_payload(self._latest_goal_key)

    def finish(self):
        return None
