from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm.clients import AttackerLLM, TargetLLM


def _build_text_response(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _build_tool_response(arguments: str):
    tool_call = SimpleNamespace(function=SimpleNamespace(arguments=arguments))
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        response = self._responses[self.calls]
        self.calls += 1
        return response


class _FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_FakeCompletions(responses))


class LLMClientRetryTests(unittest.TestCase):
    def test_target_response_retries_until_third_attempt(self):
        fake_client = _FakeClient([None, None, _build_text_response("final answer")])
        with patch("llm.clients._create_client", return_value=fake_client):
            target = TargetLLM(model_name="mock-target")

        result = target.target_response("prompt")

        self.assertEqual(result, "final answer")
        self.assertEqual(fake_client.chat.completions.calls, 3)

    def test_target_response_raises_after_three_invalid_attempts(self):
        fake_client = _FakeClient([None, None, None])
        with patch("llm.clients._create_client", return_value=fake_client):
            target = TargetLLM(model_name="mock-target")

        with self.assertRaises(TypeError) as context:
            target.target_response("prompt")

        self.assertIn("after 3 attempts", str(context.exception))
        self.assertEqual(fake_client.chat.completions.calls, 3)

    def test_structured_tool_call_retries_for_invalid_response(self):
        fake_client = _FakeClient(
            [
                None,
                _build_tool_response(
                    '{"improvement":"retry worked","prompt":"prompt text","selected_method_names":["method"]}'
                ),
            ]
        )
        with patch("llm.clients._create_client", return_value=fake_client):
            attacker = AttackerLLM(model_name="mock-attacker")

        payload = attacker._call_structured_tool(
            messages=[{"role": "user", "content": "hello"}],
            tool_schema={"type": "function", "function": {"name": "attack_prompt_draft"}},
            tool_name="attack_prompt_draft",
        )

        self.assertEqual(payload["prompt"], "prompt text")
        self.assertEqual(fake_client.chat.completions.calls, 2)


if __name__ == "__main__":
    unittest.main()
