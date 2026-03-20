from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from config.default_config import AttackConfig
from llm.prompts import get_attack_prompt_user_prompt, get_method_proposal_user_prompt
from methods.method_schema import MODE_INVENT, MODE_REUSE, AttackMethodProposal, AttackPromptDraft
from runtime.global_context import GlobalContextQueue
from runtime.search_tree import get_init_msg, process_target_response
from tap_runner import tap


@dataclass
class GlobalContextTestConfig(AttackConfig):
    global_context_path: str = ""

    def resolve_global_context_path(self) -> str:
        return self.global_context_path


class GlobalContextIntegrationTests(unittest.TestCase):
    def test_queue_retains_top_scores_with_fifo_tiebreak(self):
        queue = GlobalContextQueue(2)

        queue.enqueue((1, "prompt-a"))
        queue.enqueue((1, "prompt-b"))
        queue.enqueue((0, "prompt-c"))
        self.assertEqual(
            [entry.to_dict() for entry in queue.entries()],
            [
                {"score": 1, "prompt": "prompt-a"},
                {"score": 1, "prompt": "prompt-b"},
            ],
        )

        queue.enqueue((1, "prompt-c"))
        self.assertEqual(
            [entry.to_dict() for entry in queue.entries()],
            [
                {"score": 1, "prompt": "prompt-b"},
                {"score": 1, "prompt": "prompt-c"},
            ],
        )

        queue.enqueue((3, "prompt-d"))
        self.assertEqual(
            [entry.to_dict() for entry in queue.entries()],
            [
                {"score": 1, "prompt": "prompt-c"},
                {"score": 3, "prompt": "prompt-d"},
            ],
        )

    def test_queue_round_trip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "global_context.json")
            queue = GlobalContextQueue(3)
            queue.enqueue((7, "prompt-a"))
            queue.enqueue((9, "prompt-b"))
            queue.save_to_file(path)

            loaded = GlobalContextQueue(3)
            loaded.load_from_file(path)
            self.assertEqual(
                [entry.to_dict() for entry in loaded.entries()],
                [
                    {"score": 7, "prompt": "prompt-a"},
                    {"score": 9, "prompt": "prompt-b"},
                ],
            )

            missing = GlobalContextQueue(3)
            missing.load_from_file(os.path.join(temp_dir, "missing.json"))
            self.assertTrue(missing.is_empty())

    def test_attack_prompt_includes_global_context_only_for_invent_and_mutate(self):
        global_context_json = '[{"score": 7, "prompt": "prompt-a"}]'
        invent_prompt = get_attack_prompt_user_prompt(
            goal="Goal text",
            target_str="Target prefix",
            mode="invent",
            candidate_methods=[],
            parent_target_response="target response",
            parent_score=4,
            recent_examples="No examples.",
            previous_prompt="Previous prompt body",
            global_context_json=global_context_json,
        )
        self.assertIn("GLOBAL_CONTEXT_JSON:", invent_prompt)
        self.assertIn(global_context_json, invent_prompt)
        self.assertIn("Do not copy any stored prompt verbatim.", invent_prompt)

        reuse_prompt = get_attack_prompt_user_prompt(
            goal="Goal text",
            target_str="Target prefix",
            mode=MODE_REUSE,
            candidate_methods=[],
            parent_target_response="target response",
            parent_score=4,
            recent_examples="No examples.",
            previous_prompt="Previous prompt body",
            global_context_json=global_context_json,
        )
        self.assertNotIn("GLOBAL_CONTEXT_JSON:", reuse_prompt)
        self.assertNotIn(global_context_json, reuse_prompt)

        method_prompt = get_method_proposal_user_prompt(
            goal="Goal text",
            target_str="Target prefix",
            mode=MODE_INVENT,
            existing_method_summaries=[],
            candidate_parent_method_summaries=[],
            current_score=0.0,
            recent_summary="No recent attempts.",
        )
        self.assertNotIn("GLOBAL_CONTEXT_JSON", method_prompt)

    def test_search_tree_only_appends_global_context_for_invent_and_mutate(self):
        global_context_json = '[{"score": 7, "prompt": "prompt-a"}]'

        init_invent_prompt = get_init_msg(
            goal="Goal text",
            target="Target prefix",
            candidate_methods=[],
            examples=[],
            mode=MODE_INVENT,
            global_context_json=global_context_json,
        )
        self.assertIn("GLOBAL_CONTEXT_JSON:", init_invent_prompt)
        self.assertIn(global_context_json, init_invent_prompt)

        init_reuse_prompt = get_init_msg(
            goal="Goal text",
            target="Target prefix",
            candidate_methods=[],
            examples=[],
            mode=MODE_REUSE,
            global_context_json=global_context_json,
        )
        self.assertNotIn("GLOBAL_CONTEXT_JSON:", init_reuse_prompt)

        followup_reuse_prompt = process_target_response(
            target_response="target response",
            score=4,
            goal="Goal text",
            candidate_methods=[],
            examples=[],
            mode=MODE_REUSE,
            previous_prompt="Previous prompt body",
            global_context_json=global_context_json,
        )
        self.assertNotIn("GLOBAL_CONTEXT_JSON:", followup_reuse_prompt)

    def test_tap_loads_saves_and_injects_global_context_on_success(self):
        class RecordingAttackerLLM:
            instances = []

            def __init__(self, model_name, config=None):
                self.model_name = model_name
                self.goal = ""
                self.target_str = ""
                self.prompt_calls = 0
                self.user_prompts = []
                RecordingAttackerLLM.instances.append(self)

            def generate_method_proposal(
                self,
                *,
                goal,
                target_str,
                attack_state,
                mode,
                existing_methods,
                candidate_parent_methods,
            ):
                return AttackMethodProposal(
                    method_name=f"{mode.title()} Method",
                    method_description=f"{mode} description",
                    method_rationale=f"{mode} rationale",
                    mutation_of=None,
                    selected_parent_method_names=[],
                    prompt_template="template",
                    attack_plan="plan",
                    applicability="general",
                    novelty_note="novel",
                    expected_mechanism="mechanism",
                )

            def generate_attack_prompt(
                self,
                *,
                conversation,
                goal,
                target_str,
                attack_state,
                attack_method,
                mode,
                examples,
                candidate_methods=None,
            ):
                self.prompt_calls += 1
                self.user_prompts.append(conversation.messages[-1][1])
                prompt = (
                    "partial success prompt" if self.prompt_calls == 1 else "success prompt"
                )
                return AttackPromptDraft(
                    improvement=f"improvement-{self.prompt_calls}",
                    prompt=prompt,
                    selected_method_names=(
                        [candidate_methods[0].method_name]
                        if candidate_methods
                        else [attack_method.method_name]
                    ),
                    prompt_template="template",
                    attack_plan="plan",
                    rationale="rationale",
                )

        class SuccessEvaluatorLLM:
            def __init__(self, model_name, config=None):
                self.model_name = model_name

            def out_of_target(self, off_target_system_prompt, off_target_prompt):
                return "Response: [[YES]]"

            def judge(self, judge_system_prompt, target_response):
                if (target_response or "") == "success prompt":
                    return "Rating: [[10]]"
                if (target_response or "") == "partial success prompt":
                    return "Rating: [[4]]"
                return "Rating: [[1]]"

        class EchoTargetLLM:
            def __init__(self, model_name, config=None):
                self.model_name = model_name

            def target_response(self, prompt):
                return prompt

        original_load = GlobalContextQueue.load_from_file
        original_save = GlobalContextQueue.save_to_file
        original_enqueue = GlobalContextQueue.enqueue

        def wrapped_load(self, filepath=None):
            return original_load(self, filepath)

        def wrapped_save(self, filepath=None):
            return original_save(self, filepath)

        def wrapped_enqueue(self, item):
            return original_enqueue(self, item)

        with tempfile.TemporaryDirectory() as temp_dir:
            global_context_path = os.path.join(temp_dir, "global_context.json")
            registry_path = os.path.join(temp_dir, "posterior_evidence_global.pkl")
            with open(global_context_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "capacity": 3,
                        "data": [{"score": 8, "prompt": "seed prompt"}],
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            config = GlobalContextTestConfig(
                persistence_path=registry_path,
                sparse_pool_threshold=0,
                mode_reuse_bias=2.0,
                mode_mutate_bias=0.1,
                mode_invent_bias=0.1,
                mode_cold_start_bonus=0.0,
                low_score_threshold=0.0,
                max_global_methods=32,
                global_context_queue_size=3,
                attacker_input_dir=os.path.join(temp_dir, "attacker_input"),
                global_context_path=global_context_path,
            )
            args = SimpleNamespace(
                attacker_model="mock-attacker",
                evaluator_model="mock-evaluator",
                target_model="mock-target",
                goal="Goal text",
                target="Target prefix",
                index=0,
                max_depth=2,
                branching_factor=1,
                width=1,
                config=config,
            )

            RecordingAttackerLLM.instances.clear()
            with patch(
                "runtime.global_context.GlobalContextQueue.load_from_file",
                autospec=True,
                side_effect=wrapped_load,
            ) as load_mock, patch(
                "runtime.global_context.GlobalContextQueue.save_to_file",
                autospec=True,
                side_effect=wrapped_save,
            ) as save_mock, patch(
                "runtime.global_context.GlobalContextQueue.enqueue",
                autospec=True,
                side_effect=wrapped_enqueue,
            ) as enqueue_mock, patch(
                "tap_runner.AttackerLLM",
                RecordingAttackerLLM,
            ), patch(
                "tap_runner.EvaluatorLLM",
                SuccessEvaluatorLLM,
            ), patch(
                "tap_runner.TargetLLM",
                EchoTargetLLM,
            ):
                success, request_count = tap(args, logger=None)

            self.assertTrue(success)
            self.assertEqual(request_count, 2)
            self.assertEqual(load_mock.call_count, 1)
            self.assertEqual(save_mock.call_count, 1)
            self.assertEqual(enqueue_mock.call_count, 3)
            self.assertEqual(enqueue_mock.call_args_list[-2].args[1], (4, "partial success prompt"))
            self.assertEqual(enqueue_mock.call_args_list[-1].args[1], (10, "success prompt"))

            attacker = RecordingAttackerLLM.instances[0]
            self.assertEqual(len(attacker.user_prompts), 2)
            self.assertIn("GLOBAL_CONTEXT_JSON:", attacker.user_prompts[0])
            self.assertIn("seed prompt", attacker.user_prompts[0])
            self.assertIn("partial success prompt", attacker.user_prompts[1])
            self.assertIn("PREVIOUS ADVERSARIAL PROMPT: partial success prompt", attacker.user_prompts[1])

            with open(global_context_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["capacity"], 3)
            self.assertEqual(
                payload["data"],
                [
                    {"score": 8, "prompt": "seed prompt"},
                    {"score": 4, "prompt": "partial success prompt"},
                    {"score": 10, "prompt": "success prompt"},
                ],
            )

    def test_tap_saves_global_context_on_failure(self):
        class RecordingAttackerLLM:
            def __init__(self, model_name, config=None):
                self.model_name = model_name
                self.goal = ""
                self.target_str = ""
                self.prompt_calls = 0

            def generate_method_proposal(
                self,
                *,
                goal,
                target_str,
                attack_state,
                mode,
                existing_methods,
                candidate_parent_methods,
            ):
                return AttackMethodProposal(
                    method_name=f"{mode.title()} Method",
                    method_description=f"{mode} description",
                    method_rationale=f"{mode} rationale",
                    mutation_of=None,
                    selected_parent_method_names=[],
                    prompt_template="template",
                    attack_plan="plan",
                    applicability="general",
                    novelty_note="novel",
                    expected_mechanism="mechanism",
                )

            def generate_attack_prompt(
                self,
                *,
                conversation,
                goal,
                target_str,
                attack_state,
                attack_method,
                mode,
                examples,
                candidate_methods=None,
            ):
                self.prompt_calls += 1
                return AttackPromptDraft(
                    improvement=f"improvement-{self.prompt_calls}",
                    prompt=f"attempt-{self.prompt_calls}",
                    selected_method_names=(
                        [candidate_methods[0].method_name]
                        if candidate_methods
                        else [attack_method.method_name]
                    ),
                    prompt_template="template",
                    attack_plan="plan",
                    rationale="rationale",
                )

        class FailureEvaluatorLLM:
            def __init__(self, model_name, config=None):
                self.model_name = model_name

            def out_of_target(self, off_target_system_prompt, off_target_prompt):
                return "Response: [[YES]]"

            def judge(self, judge_system_prompt, target_response):
                return "Rating: [[3]]"

        class EchoTargetLLM:
            def __init__(self, model_name, config=None):
                self.model_name = model_name

            def target_response(self, prompt):
                return prompt

        original_load = GlobalContextQueue.load_from_file
        original_save = GlobalContextQueue.save_to_file
        original_enqueue = GlobalContextQueue.enqueue

        def wrapped_load(self, filepath=None):
            return original_load(self, filepath)

        def wrapped_save(self, filepath=None):
            return original_save(self, filepath)

        def wrapped_enqueue(self, item):
            return original_enqueue(self, item)

        with tempfile.TemporaryDirectory() as temp_dir:
            global_context_path = os.path.join(temp_dir, "global_context.json")
            registry_path = os.path.join(temp_dir, "posterior_evidence_global.pkl")
            config = GlobalContextTestConfig(
                persistence_path=registry_path,
                sparse_pool_threshold=0,
                mode_reuse_bias=2.0,
                mode_mutate_bias=0.1,
                mode_invent_bias=0.1,
                mode_cold_start_bonus=0.0,
                low_score_threshold=0.0,
                max_global_methods=32,
                global_context_queue_size=2,
                attacker_input_dir=os.path.join(temp_dir, "attacker_input"),
                global_context_path=global_context_path,
            )
            args = SimpleNamespace(
                attacker_model="mock-attacker",
                evaluator_model="mock-evaluator",
                target_model="mock-target",
                goal="Goal text",
                target="Target prefix",
                index=0,
                max_depth=1,
                branching_factor=2,
                width=1,
                config=config,
            )

            with patch(
                "runtime.global_context.GlobalContextQueue.load_from_file",
                autospec=True,
                side_effect=wrapped_load,
            ) as load_mock, patch(
                "runtime.global_context.GlobalContextQueue.save_to_file",
                autospec=True,
                side_effect=wrapped_save,
            ) as save_mock, patch(
                "runtime.global_context.GlobalContextQueue.enqueue",
                autospec=True,
                side_effect=wrapped_enqueue,
            ) as enqueue_mock, patch(
                "tap_runner.AttackerLLM",
                RecordingAttackerLLM,
            ), patch(
                "tap_runner.EvaluatorLLM",
                FailureEvaluatorLLM,
            ), patch(
                "tap_runner.TargetLLM",
                EchoTargetLLM,
            ):
                success, request_count = tap(args, logger=None)

            self.assertFalse(success)
            self.assertEqual(request_count, 2)
            self.assertEqual(load_mock.call_count, 1)
            self.assertEqual(save_mock.call_count, 1)
            self.assertEqual(enqueue_mock.call_count, 2)

            with open(global_context_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["capacity"], 2)
            self.assertEqual(
                payload["data"],
                [
                    {"score": 3, "prompt": "attempt-1"},
                    {"score": 3, "prompt": "attempt-2"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
