from __future__ import annotations

import unittest

from config.default_config import AttackConfig
from runtime.search_tree import Tree


class _MockAttackerLLM:
    def __init__(self):
        self.goal = ""
        self.target_str = ""


class _MockEvaluatorLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def out_of_target(self, off_target_system_prompt, off_target_prompt):
        self.calls.append((off_target_system_prompt, off_target_prompt))
        if self.error is not None:
            raise self.error
        return self.response

    def judge(self, judge_system_prompt, target_response):
        return "Rating: [[1]]"


class _MockTargetLLM:
    def target_response(self, prompt):
        return prompt


class OnTopicEvaluationTests(unittest.TestCase):
    def _build_tree(self, evaluator):
        return Tree(
            goal="Goal text",
            target_str="Target prefix",
            index=0,
            attacker_llm=_MockAttackerLLM(),
            evaluator_llm=evaluator,
            target_llm=_MockTargetLLM(),
            config=AttackConfig(bootstrap_resume_from_disk=False),
        )

    def test_evaluate_on_topic_returns_true_for_yes_response(self):
        evaluator = _MockEvaluatorLLM(response="Response: [[YES]]")
        tree = self._build_tree(evaluator)

        self.assertTrue(tree.evaluate_on_topic("Prompt text"))
        self.assertEqual(len(evaluator.calls), 1)
        self.assertEqual(evaluator.calls[0][1], "Prompt text")

    def test_evaluate_on_topic_returns_false_for_no_response(self):
        evaluator = _MockEvaluatorLLM(response="Response: [[NO]]")
        tree = self._build_tree(evaluator)

        self.assertFalse(tree.evaluate_on_topic("Prompt text"))

    def test_evaluate_on_topic_fails_open_on_unexpected_format(self):
        evaluator = _MockEvaluatorLLM(response="unexpected output")
        tree = self._build_tree(evaluator)

        self.assertTrue(tree.evaluate_on_topic("Prompt text"))

    def test_evaluate_on_topic_fails_open_on_evaluator_error(self):
        evaluator = _MockEvaluatorLLM(error=RuntimeError("temporary evaluator failure"))
        tree = self._build_tree(evaluator)

        self.assertTrue(tree.evaluate_on_topic("Prompt text"))


if __name__ == "__main__":
    unittest.main()
