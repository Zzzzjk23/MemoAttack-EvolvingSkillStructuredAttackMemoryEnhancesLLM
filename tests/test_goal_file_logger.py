from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from observability.goal_file_logger import GoalFileLogger


class GoalFileLoggerTests(unittest.TestCase):
    def _build_node(
        self,
        *,
        tree,
        node_id: str,
        parent=None,
        depth: int = 0,
        outside_score: int = 0,
        attempt_result=None,
    ):
        return SimpleNamespace(
            id=node_id,
            parent=parent,
            depth=depth,
            tree=tree,
            prompt=f"prompt-{node_id}",
            improvement=f"improvement-{node_id}",
            on_topic=True,
            target_response=f"target-{node_id}",
            outside_score=outside_score,
            normalized_score=outside_score / 10.0,
            internal_score=0,
            attack_method=None,
            attack_method_id=None,
            mode="bootstrap",
            selected_method_names=[],
            candidate_method_names=[],
            attempt_result=attempt_result,
        )

    def test_logger_updates_goal_file_on_every_logged_node(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = GoalFileLogger(log_dir=temp_dir, config={"example": True})
            tree = SimpleNamespace(index=7, goal="Goal text", target="Target prefix", request_count=0)
            root = self._build_node(tree=tree, node_id="root")

            logger.log_node(root)

            log_path = os.path.join(temp_dir, "goal_7.json")
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["goal_index"], "7")
            self.assertEqual(payload["node_count"], 1)
            self.assertEqual(len(payload["nodes"]), 1)
            self.assertEqual(payload["nodes"][0]["node_id"], "root")

            tree.request_count = 1
            attempt_result = SimpleNamespace(
                made_progress=True,
                normalized_progress=0.4,
                final_success=False,
                raw_prev_score=0,
                raw_new_score=4,
                mode="bootstrap",
                used_method_id="",
            )
            child = self._build_node(
                tree=tree,
                node_id="child",
                parent=root,
                depth=1,
                outside_score=4,
                attempt_result=attempt_result,
            )

            logger.log_node(child)

            with open(log_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["request_count"], 1)
            self.assertEqual(payload["node_count"], 2)
            self.assertEqual(len(payload["nodes"]), 2)
            self.assertEqual(payload["nodes"][1]["node_id"], "child")
            self.assertTrue(payload["nodes"][1]["attempt_result"]["made_progress"])

            logger.log_success("Goal text", request_count=1, steps=1, success=True)

            with open(log_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["status"], "success")
            self.assertTrue(payload["success"])
            self.assertEqual(payload["steps"], 1)


if __name__ == "__main__":
    unittest.main()
