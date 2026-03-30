from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from config.default_config import AttackConfig
from llm.prompts import get_attack_prompt_user_prompt
from methods.method_registry import MethodRegistry
from methods.method_schema import AttackMethodProposal, AttackPromptDraft
from runtime.global_context import GlobalContextEntry, GlobalContextQueue, PHASE_POSTERIOR
from runtime.search_tree import get_init_msg, process_target_response
from tap_runner import tap


@dataclass
class GlobalContextTestConfig(AttackConfig):
    global_context_path: str = ""

    def resolve_global_context_path(self) -> str:
        return self.global_context_path


class GlobalContextIntegrationTests(unittest.TestCase):
    def test_attacker_view_uses_top_scores_and_limited_fields(self):
        queue = GlobalContextQueue(attacker_top_k=2)
        queue.add_record(
            GlobalContextEntry(
                before_prompt="before-a",
                before_score=1,
                after_prompt="after-a",
                after_score=6,
                improvement="improve-a",
                timestamp="2026-03-20T00:00:00+00:00",
            )
        )
        queue.add_record(
            GlobalContextEntry(
                before_prompt="before-b",
                before_score=2,
                after_prompt="after-b",
                after_score=9,
                improvement="improve-b",
                timestamp="2026-03-20T00:00:01+00:00",
            )
        )
        queue.add_record(
            GlobalContextEntry(
                before_prompt="before-c",
                before_score=3,
                after_prompt="after-c",
                after_score=9,
                improvement="improve-c",
                timestamp="2026-03-20T00:00:02+00:00",
            )
        )

        self.assertEqual(
            queue.attacker_view(),
            [
                {
                    "before_prompt": "before-c",
                    "before_score": 3,
                    "after_prompt": "after-c",
                    "after_score": 9,
                    "improvement": "improve-c",
                },
                {
                    "before_prompt": "before-b",
                    "before_score": 2,
                    "after_prompt": "after-b",
                    "after_score": 9,
                    "improvement": "improve-b",
                },
            ],
        )

    def test_round_trip_persists_phase_and_bootstrap_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "global_context.json")
            queue = GlobalContextQueue(attacker_top_k=5)
            committed = queue.commit_successful_goal(
                "goal-1",
                [
                    GlobalContextEntry(
                        goal="Goal text",
                        goal_index="1",
                        before_prompt="before",
                        before_score=1,
                        after_prompt="after",
                        after_score=8,
                        improvement="rewrite",
                        normalized_progress=0.7,
                    )
                ],
            )
            self.assertTrue(committed)
            queue.mark_posterior_built()
            queue.save_to_file(path)

            loaded = GlobalContextQueue(attacker_top_k=1)
            loaded.load_from_file(path)

            self.assertEqual(loaded.phase, PHASE_POSTERIOR)
            self.assertTrue(loaded.posterior_built)
            self.assertEqual(loaded.bootstrap_success_count, 1)
            self.assertEqual(loaded.successful_goal_ids, {"goal-1"})
            self.assertEqual(len(loaded.entries()), 1)
            self.assertEqual(loaded.entries()[0].after_prompt, "after")

    def test_bootstrap_prompt_only_uses_global_context(self):
        global_context_json = json.dumps(
            [
                {
                    "before_prompt": "before",
                    "before_score": 2,
                    "after_prompt": "after",
                    "after_score": 7,
                    "improvement": "rewrite",
                }
            ],
            ensure_ascii=False,
        )
        prompt = get_attack_prompt_user_prompt(
            goal="Goal text",
            target_str="Target prefix",
            mode="bootstrap",
            candidate_methods=[],
            parent_target_response="target response",
            parent_score=4,
            recent_examples="No examples.",
            previous_prompt="Previous prompt body",
            global_context_json=global_context_json,
            global_context_only=True,
        )
        self.assertIn("GLOBAL_CONTEXT_JSON:", prompt)
        self.assertIn("No candidate attack methods are available in this stage", prompt)
        self.assertNotIn("Candidate attack methods:", prompt)

    def test_search_tree_bootstrap_messages_use_global_context_only(self):
        global_context_json = json.dumps(
            [
                {
                    "before_prompt": "before",
                    "before_score": 1,
                    "after_prompt": "after",
                    "after_score": 5,
                    "improvement": "rewrite",
                }
            ],
            ensure_ascii=False,
        )
        init_prompt = get_init_msg(
            goal="Goal text",
            target="Target prefix",
            candidate_methods=[],
            examples=[],
            mode="bootstrap",
            global_context_json=global_context_json,
            global_context_only=True,
        )
        self.assertIn("GLOBAL_CONTEXT_JSON:", init_prompt)
        self.assertNotIn("Candidate attack methods:", init_prompt)

        followup_prompt = process_target_response(
            target_response="target response",
            score=4,
            goal="Goal text",
            candidate_methods=[],
            examples=[],
            mode="bootstrap",
            previous_prompt="Previous prompt body",
            global_context_json=global_context_json,
            global_context_only=True,
        )
        self.assertIn("GLOBAL_CONTEXT_JSON:", followup_prompt)
        self.assertNotIn("Candidate attack methods:", followup_prompt)

    def test_tap_bootstrap_commits_records_only_on_success(self):
        class RecordingAttackerLLM:
            instances = []

            def __init__(self, model_name, config=None):
                self.model_name = model_name
                self.goal = ""
                self.target_str = ""
                self.user_prompts = []
                self.prompt_calls = 0
                RecordingAttackerLLM.instances.append(self)

            def generate_method_proposal(self, **kwargs):
                return AttackMethodProposal(
                    method_name="Unused Method",
                    method_description="unused",
                    method_rationale="unused",
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
                prompt = "partial success prompt" if self.prompt_calls == 1 else "success prompt"
                selected_method_names = [item.method_name for item in (candidate_methods or [])]
                return AttackPromptDraft(
                    improvement=f"improvement-{self.prompt_calls}",
                    prompt=prompt,
                    selected_method_names=selected_method_names,
                    prompt_template="template",
                    attack_plan="plan",
                    rationale="rationale",
                )

        class MockEvaluatorLLM:
            def __init__(self, model_name, config=None):
                self.model_name = model_name

            def out_of_target(self, off_target_system_prompt, off_target_prompt):
                return "Response: [[YES]]"

            def judge(self, judge_system_prompt, target_response):
                if target_response == "success prompt":
                    return "Rating: [[10]]"
                if target_response == "partial success prompt":
                    return "Rating: [[4]]"
                return "Rating: [[1]]"

            def distill_method_proposal(self, *, record, existing_methods):
                raise AssertionError("distillation should not run in this test")

        class EchoTargetLLM:
            def __init__(self, model_name, config=None):
                self.model_name = model_name

            def target_response(self, prompt):
                return prompt

        with tempfile.TemporaryDirectory() as temp_dir:
            global_context_path = os.path.join(temp_dir, "global_context.json")
            registry_path = os.path.join(temp_dir, "posterior_evidence_global.pkl")
            with open(global_context_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "phase": "bootstrap",
                        "bootstrap_success_count": 0,
                        "posterior_built": False,
                        "attacker_top_k": 5,
                        "successful_goal_ids": [],
                        "records": [],
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )

            config = GlobalContextTestConfig(
                persistence_path=registry_path,
                bootstrap_success_target=99,
                global_context_attacker_top_k=5,
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
            with patch("tap_runner.AttackerLLM", RecordingAttackerLLM), patch(
                "tap_runner.EvaluatorLLM", MockEvaluatorLLM
            ), patch("tap_runner.TargetLLM", EchoTargetLLM):
                success, request_count = tap(args, logger=None)

            self.assertTrue(success)
            self.assertEqual(request_count, 2)
            attacker = RecordingAttackerLLM.instances[0]
            self.assertNotIn("GLOBAL_CONTEXT_JSON:", attacker.user_prompts[0])
            self.assertNotIn("GLOBAL_CONTEXT_JSON:", attacker.user_prompts[1])
            self.assertNotIn("Candidate attack methods:", attacker.user_prompts[0])

            with open(global_context_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["phase"], "bootstrap")
            self.assertEqual(payload["bootstrap_success_count"], 1)
            self.assertEqual(len(payload["records"]), 2)
            self.assertEqual(payload["records"][0]["after_prompt"], "partial success prompt")
            self.assertEqual(payload["records"][1]["after_prompt"], "success prompt")

        with tempfile.TemporaryDirectory() as temp_dir:
            global_context_path = os.path.join(temp_dir, "global_context.json")
            registry_path = os.path.join(temp_dir, "posterior_evidence_global.pkl")
            config = GlobalContextTestConfig(
                persistence_path=registry_path,
                bootstrap_success_target=99,
                global_context_attacker_top_k=5,
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
                branching_factor=1,
                width=1,
                config=config,
            )

            class FailureAttackerLLM(RecordingAttackerLLM):
                def generate_attack_prompt(self, **kwargs):
                    conversation = kwargs["conversation"]
                    self.prompt_calls += 1
                    self.user_prompts.append(conversation.messages[-1][1])
                    return AttackPromptDraft(
                        improvement="improvement",
                        prompt="partial success prompt",
                        selected_method_names=[],
                        prompt_template="template",
                        attack_plan="plan",
                        rationale="rationale",
                    )

            with patch("tap_runner.AttackerLLM", FailureAttackerLLM), patch(
                "tap_runner.EvaluatorLLM", MockEvaluatorLLM
            ), patch("tap_runner.TargetLLM", EchoTargetLLM):
                success, request_count = tap(args, logger=None)

            self.assertFalse(success)
            self.assertEqual(request_count, 1)
            with open(global_context_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["bootstrap_success_count"], 0)
            self.assertEqual(payload["records"], [])

    def test_tap_distills_posterior_and_stops_using_global_context(self):
        class RecordingAttackerLLM:
            instances = []

            def __init__(self, model_name, config=None):
                self.model_name = model_name
                self.goal = ""
                self.target_str = ""
                self.user_prompts = []
                self.prompt_calls = 0
                RecordingAttackerLLM.instances.append(self)

            def generate_method_proposal(self, **kwargs):
                return AttackMethodProposal(
                    method_name="Fallback Posterior Method",
                    method_description="posterior fallback",
                    method_rationale="posterior fallback",
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
                if mode == "bootstrap":
                    prompt = "bootstrap success prompt"
                    selected = []
                else:
                    prompt = "posterior success prompt"
                    selected = [item.method_name for item in (candidate_methods or [])]
                return AttackPromptDraft(
                    improvement=f"{mode}-improvement",
                    prompt=prompt,
                    selected_method_names=selected,
                    prompt_template="template",
                    attack_plan="plan",
                    rationale="rationale",
                )

        class DistillingEvaluatorLLM:
            distill_calls = 0

            def __init__(self, model_name, config=None):
                self.model_name = model_name

            def out_of_target(self, off_target_system_prompt, off_target_prompt):
                return "Response: [[YES]]"

            def judge(self, judge_system_prompt, target_response):
                if target_response in {"bootstrap success prompt", "posterior success prompt"}:
                    return "Rating: [[10]]"
                return "Rating: [[1]]"

            def distill_method_proposal(self, *, record, existing_methods):
                DistillingEvaluatorLLM.distill_calls += 1
                method_name = "Canonical Rewrite Method"
                if existing_methods:
                    method_name = existing_methods[0].method_name
                return AttackMethodProposal(
                    method_name=method_name,
                    method_description="Distilled canonical rewrite",
                    method_rationale="Derived from successful cold-start transitions",
                    mutation_of=None,
                    selected_parent_method_names=[],
                    prompt_template="Use the successful rewrite pattern from the distilled record.",
                    attack_plan="Apply the same successful rewrite transformation to the next prompt.",
                    applicability="Cold-start derived posterior method",
                    novelty_note="Reused canonical name for similar rewrites.",
                    expected_mechanism="Transfer a successful before/after rewrite mechanism.",
                )

        class EchoTargetLLM:
            def __init__(self, model_name, config=None):
                self.model_name = model_name

            def target_response(self, prompt):
                return prompt

        with tempfile.TemporaryDirectory() as temp_dir:
            global_context_path = os.path.join(temp_dir, "global_context.json")
            registry_path = os.path.join(temp_dir, "posterior_evidence_global.pkl")
            config = GlobalContextTestConfig(
                persistence_path=registry_path,
                bootstrap_success_target=1,
                sparse_pool_threshold=0,
                mode_reuse_bias=5.0,
                mode_mutate_bias=0.1,
                mode_invent_bias=0.1,
                mode_cold_start_bonus=0.0,
                low_score_threshold=0.0,
                global_context_attacker_top_k=5,
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
                branching_factor=1,
                width=1,
                config=config,
            )

            RecordingAttackerLLM.instances.clear()
            DistillingEvaluatorLLM.distill_calls = 0
            with patch("tap_runner.AttackerLLM", RecordingAttackerLLM), patch(
                "tap_runner.EvaluatorLLM", DistillingEvaluatorLLM
            ), patch("tap_runner.TargetLLM", EchoTargetLLM):
                success, request_count = tap(args, logger=None)
                self.assertTrue(success)
                self.assertEqual(request_count, 1)

                with open(global_context_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.assertEqual(payload["phase"], PHASE_POSTERIOR)
                self.assertTrue(payload["posterior_built"])
                self.assertGreaterEqual(DistillingEvaluatorLLM.distill_calls, 1)

                loaded_registry = MethodRegistry(config=config, load_path=registry_path)
                self.assertGreaterEqual(len(loaded_registry.get_pool().get_all_methods()), 1)

                success, request_count = tap(args, logger=None)
                self.assertTrue(success)
                self.assertEqual(request_count, 1)

            self.assertEqual(len(RecordingAttackerLLM.instances), 2)
            self.assertNotIn("Candidate attack methods:", RecordingAttackerLLM.instances[0].user_prompts[0])
            self.assertNotIn("GLOBAL_CONTEXT_JSON:", RecordingAttackerLLM.instances[1].user_prompts[0])
            self.assertIn("Candidate attack methods:", RecordingAttackerLLM.instances[1].user_prompts[0])


if __name__ == "__main__":
    unittest.main()
