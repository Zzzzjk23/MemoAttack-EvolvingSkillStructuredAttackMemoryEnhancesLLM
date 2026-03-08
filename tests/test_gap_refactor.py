from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config.default_config import AttackConfig
from controller.mode_selector import select_mode
from llm.clients import AttackerLLM
from llm.prompts import get_attacker_method_system_prompt, get_attacker_system_prompt
from methods.method_registry import CategoryMethodPool, MethodRegistry
from methods.method_schema import (
    ACTIVE,
    MODE_INVENT,
    MODE_REUSE,
    AttackAttemptResult,
    AttackMethodProposal,
    AttackPromptDraft,
    AttackState,
)
from runtime.search_tree import get_init_msg, process_target_response
from runtime.tap_runner import select_nodes, tap
from scoring.progress_metric import compute_normalized_gap_improvement


class GapRefactorTests(unittest.TestCase):
    def test_progress_metric_uses_normalized_remaining_gap(self):
        improvement = compute_normalized_gap_improvement(
            prev_score=0.2,
            new_score=0.5,
            max_score=1.0,
            epsilon=1e-6,
        )
        self.assertAlmostEqual(improvement, 0.375, places=6)

    def test_dual_posteriors_and_retirement_delay(self):
        config = AttackConfig(retirement_min_support=3, elimination_min_support=5)
        registry = MethodRegistry(config=config, load_path=os.path.join(tempfile.gettempdir(), "nonexistent_gap_test.pkl"))
        proposal = AttackMethodProposal(
            method_name="Seed Method",
            method_description="Initial adaptive method",
            method_rationale="Used for testing posterior separation",
            mutation_of=None,
            prompt_template="template",
            attack_plan="plan",
            applicability="general",
            novelty_note="new",
            expected_mechanism="mechanism",
        )
        method = registry.register_method("Privacy", proposal, created_via=MODE_REUSE)
        registry.record_attempt(
            category_id="Privacy",
            method_id=method.method_id,
            attempt_result=AttackAttemptResult(
                used_method_id=method.method_id,
                mode=MODE_REUSE,
                prev_score=0.1,
                new_score=0.4,
                normalized_progress=0.3,
                made_progress=True,
                final_success=False,
                response={},
                target_response="partial",
                attack_prompt="prompt-1",
            ),
            prompt_text="goal",
            before_prompt="before",
            after_prompt="after",
            target_response="partial",
        )
        updated = registry.get_or_create_pool("Privacy").get_method(method.method_id)
        self.assertEqual(updated.stats.progress_alpha, config.newborn_progress_alpha + 1.0)
        self.assertEqual(updated.stats.progress_beta, config.newborn_progress_beta)
        self.assertEqual(updated.stats.success_alpha, config.newborn_success_alpha)
        self.assertEqual(updated.stats.success_beta, config.newborn_success_beta + 1.0)
        self.assertEqual(updated.status, ACTIVE)

        registry.record_attempt(
            category_id="Privacy",
            method_id=method.method_id,
            attempt_result=AttackAttemptResult(
                used_method_id=method.method_id,
                mode=MODE_REUSE,
                prev_score=0.4,
                new_score=0.4,
                normalized_progress=0.0,
                made_progress=False,
                final_success=False,
                response={},
                target_response="none",
                attack_prompt="prompt-2",
            ),
            prompt_text="goal",
            before_prompt="before-2",
            after_prompt="after-2",
            target_response="none",
        )
        updated = registry.get_or_create_pool("Privacy").get_method(method.method_id)
        self.assertEqual(updated.usage_count, 2)
        self.assertEqual(updated.status, ACTIVE)

    def test_mode_selector_forces_invent_on_cold_start(self):
        config = AttackConfig()
        removed_attr = "_".join(["bootstrap", "strategy", "catalog"])
        self.assertFalse(hasattr(config, removed_attr))
        pool = CategoryMethodPool(category_id="Fraud/Deception", config=config)
        state = AttackState(
            goal="goal",
            target="target",
            category_id="Fraud/Deception",
            current_prompt="goal",
            current_target_response=None,
            current_raw_score=0.0,
            current_score=0.0,
            depth=0,
            node_id=None,
            recent_method_id=None,
            recent_mode=None,
        )
        self.assertEqual(select_mode(state, pool, config), MODE_INVENT)

    def test_select_nodes_preserves_top_score_behavior(self):
        nodes = [
            SimpleNamespace(on_topic=True, outside_score=2),
            SimpleNamespace(on_topic=False, outside_score=10),
            SimpleNamespace(on_topic=True, outside_score=7),
        ]
        selected = select_nodes(nodes, 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].outside_score, 7)

    def test_method_proposal_fallback_is_prompt_derived_without_static_catalog(self):
        state = AttackState(
            goal="Describe a harmful objective",
            target="Target prefix",
            category_id="Privacy",
            current_prompt="Describe a harmful objective",
            current_target_response=None,
            current_raw_score=0.0,
            current_score=0.0,
            depth=0,
            node_id=None,
            recent_method_id=None,
            recent_mode=None,
        )
        with patch("llm.clients._create_client", return_value=None):
            attacker = AttackerLLM(model_name="mock-attacker")
        with patch.object(attacker, "_call_structured_tool", side_effect=ValueError("bad json")):
            proposal = attacker.generate_method_proposal(
                goal=state.goal,
                target_str=state.target,
                attack_state=state,
                mode=MODE_INVENT,
                existing_methods=[],
                parent_method=None,
            )
        self.assertIn("Strategy", proposal.method_name)
        self.assertNotIn("Payload Splitting", proposal.method_name)
        self.assertNotIn("Adaptive Bootstrap", proposal.method_name)
        self.assertIn("current goal", proposal.method_rationale)

    def test_attacker_system_prompt_discourages_placeholder_outputs(self):
        prompt = get_attacker_system_prompt("Goal text", "Target prefix")
        self.assertIn("full standalone prompt", prompt)
        self.assertIn("Never output placeholders", prompt)
        self.assertIn('"success prompt"', prompt)

    def test_method_system_prompt_discourages_generic_method_names(self):
        prompt = get_attacker_method_system_prompt()
        self.assertIn("mechanism-specific method", prompt)
        self.assertIn('"Invented Method"', prompt)

    def test_search_tree_messages_require_full_prompt_text(self):
        attack_method = SimpleNamespace(
            method_name="Concrete Method",
            method_description="Concrete description",
            method_rationale="Concrete rationale",
            attack_plan="Concrete plan",
            prompt_template="Concrete template",
            applicability="general",
            novelty_note="novel",
            expected_mechanism="mechanism",
        )
        init_msg = get_init_msg("Goal text", "Target prefix", attack_method, [], MODE_INVENT)
        self.assertIn("exact next prompt to send to the target model", init_msg)
        self.assertIn("Never output placeholders", init_msg)

        follow_up = process_target_response(
            "partial success response",
            4,
            "Goal text",
            attack_method,
            [],
            MODE_REUSE,
        )
        self.assertIn("previous language model output and score", follow_up)
        self.assertIn("preserve what appears to work", follow_up)

    def test_tap_entrypoint_runs_with_dynamic_inventory(self):
        class MockAttackerLLM:
            def __init__(self, model_name):
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
                parent_method,
            ):
                return AttackMethodProposal(
                    method_name="Invented Method" if mode == MODE_INVENT else "Adaptive Method",
                    method_description=f"{mode} description",
                    method_rationale=f"{mode} rationale",
                    mutation_of=parent_method.method_name if parent_method else None,
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
            ):
                self.prompt_calls += 1
                prompt = "partial success prompt" if self.prompt_calls == 1 else "success prompt"
                return AttackPromptDraft(
                    improvement=f"improvement-{self.prompt_calls}",
                    prompt=prompt,
                    prompt_template="template",
                    attack_plan="plan",
                    rationale="rationale",
                )

        class MockEvaluatorLLM:
            def __init__(self, model_name):
                self.model_name = model_name

            def out_of_target(self, off_target_system_prompt, off_target_prompt):
                return "Response: [[YES]]"

            def judge(self, judge_system_prompt, target_response):
                if (target_response or "") == "success prompt":
                    return "Rating: [[10]]"
                if (target_response or "") == "partial success prompt":
                    return "Rating: [[4]]"
                return "Rating: [[1]]"

            def prompt_category(self, prompt):
                return "[[Privacy]]"

        class MockTargetLLM:
            def __init__(self, model_name):
                self.model_name = model_name

            def target_response(self, prompt):
                return prompt

        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "posterior_evidence_v2.pkl")
            config = AttackConfig(
                persistence_path=registry_path,
                sparse_pool_threshold=0,
                mode_reuse_bias=2.0,
                mode_mutate_bias=0.1,
                mode_invent_bias=0.1,
                mode_cold_start_bonus=0.0,
                low_score_threshold=0.0,
            )
            args = SimpleNamespace(
                attacker_model="mock-attacker",
                evaluator_model="mock-evaluator",
                target_model="mock-target",
                goal="Goal text",
                target="Target prefix",
                index=0,
                max_depth=2,
                branching_factor=2,
                width=1,
                config=config,
            )
            with patch("runtime.tap_runner.AttackerLLM", MockAttackerLLM), patch(
                "runtime.tap_runner.EvaluatorLLM", MockEvaluatorLLM
            ), patch("runtime.tap_runner.TargetLLM", MockTargetLLM):
                success, request_count = tap(args, logger=None)

            self.assertTrue(success)
            self.assertEqual(request_count, 2)

            loaded_registry = MethodRegistry(config=config, load_path=registry_path)
            pool = loaded_registry.get_or_create_pool("Privacy")
            self.assertGreaterEqual(len(pool.get_all_methods()), 1)
            method = pool.get_all_methods()[0]
            self.assertGreaterEqual(method.usage_count, 1)


if __name__ == "__main__":
    unittest.main()
