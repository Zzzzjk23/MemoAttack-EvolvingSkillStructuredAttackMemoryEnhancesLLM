from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config.default_config import AttackConfig
from controller.method_selector import select_existing_methods
from controller.mode_selector import select_mode
from llm.clients import AttackerLLM
from llm.prompts import (
    get_attacker_method_system_prompt,
    get_attacker_system_prompt,
    get_evaluator_system_prompt_for_on_topic,
    get_method_proposal_user_prompt,
)
from methods.method_registry import MethodPool, MethodRegistry
from methods.method_schema import (
    ACTIVE,
    RETIRED,
    MODE_INVENT,
    MODE_REUSE,
    AttackAttemptResult,
    AttackMethodProposal,
    AttackPromptDraft,
    AttackState,
)
from runtime.search_tree import get_init_msg, process_target_response
from scoring.progress_metric import compute_normalized_gap_improvement
from tap_runner import select_nodes, tap


def _build_attempt(*, method_id: str, made_progress: bool, final_success: bool, score: float):
    return AttackAttemptResult(
        used_method_id=method_id,
        mode=MODE_REUSE,
        prev_score=0.1,
        new_score=score,
        normalized_progress=score,
        made_progress=made_progress,
        final_success=final_success,
        response={},
        target_response="target",
        attack_prompt="attack prompt",
    )


class GapRefactorTests(unittest.TestCase):
    def test_progress_metric_uses_normalized_remaining_gap(self):
        improvement = compute_normalized_gap_improvement(
            prev_score=0.2,
            new_score=0.5,
            max_score=1.0,
            epsilon=1e-6,
        )
        self.assertAlmostEqual(improvement, 0.375, places=6)

    def test_default_registry_path_uses_global_filename(self):
        config = AttackConfig()
        self.assertTrue(config.persistence_path.endswith("posterior_evidence_global.pkl"))
        self.assertEqual(config.max_global_methods, 32)

    def test_global_registry_updates_posteriors_and_examples(self):
        config = AttackConfig(
            retirement_min_support=3,
            elimination_min_support=5,
            max_global_methods=32,
        )
        registry = MethodRegistry(
            config=config,
            load_path=os.path.join(tempfile.gettempdir(), "nonexistent_gap_test.pkl"),
        )
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
        method = registry.register_method(proposal, created_via=MODE_REUSE)
        registry.record_attempt(
            method_id=method.method_id,
            attempt_result=_build_attempt(
                method_id=method.method_id,
                made_progress=True,
                final_success=False,
                score=0.3,
            ),
            prompt_text="goal text",
            before_prompt="before",
            after_prompt="after",
            target_response="partial",
        )
        updated = registry.get_pool().get_method(method.method_id)
        self.assertEqual(updated.stats.progress_alpha, config.newborn_progress_alpha + 1.0)
        self.assertEqual(updated.stats.success_beta, config.newborn_success_beta + 1.0)
        self.assertEqual(updated.status, ACTIVE)
        self.assertEqual(updated.example_records[0].prompt_text, "goal text")

        registry.record_attempt(
            method_id=method.method_id,
            attempt_result=_build_attempt(
                method_id=method.method_id,
                made_progress=False,
                final_success=False,
                score=0.0,
            ),
            prompt_text="second goal text",
            before_prompt="before-2",
            after_prompt="after-2",
            target_response="none",
        )
        updated = registry.get_pool().get_method(method.method_id)
        self.assertEqual(updated.usage_count, 2)
        self.assertEqual(updated.example_records[-1].prompt_text, "second goal text")

    def test_hard_cap_evicts_weaker_active_method(self):
        config = AttackConfig(max_global_methods=2, retirement_min_support=99, elimination_min_support=99)
        pool = MethodPool(config=config)
        methods = [
            pool.register_method(
                AttackMethodProposal(
                    method_name=f"Method {index}",
                    method_description=f"Description {index}",
                    method_rationale=f"Rationale {index}",
                    mutation_of=None,
                    prompt_template="template",
                    attack_plan="plan",
                    applicability="general",
                    novelty_note="new",
                    expected_mechanism="mechanism",
                ),
                created_via=MODE_INVENT,
            )
            for index in range(3)
        ]

        pool.record_attempt(
            method_id=methods[0].method_id,
            attempt_result=_build_attempt(method_id=methods[0].method_id, made_progress=True, final_success=True, score=0.8),
            prompt_text="goal-a",
            before_prompt="before-a",
            after_prompt="after-a",
            target_response="response-a",
        )
        pool.record_attempt(
            method_id=methods[1].method_id,
            attempt_result=_build_attempt(method_id=methods[1].method_id, made_progress=True, final_success=False, score=0.4),
            prompt_text="goal-b",
            before_prompt="before-b",
            after_prompt="after-b",
            target_response="response-b",
        )
        pool.record_attempt(
            method_id=methods[2].method_id,
            attempt_result=_build_attempt(method_id=methods[2].method_id, made_progress=False, final_success=False, score=0.0),
            prompt_text="goal-c",
            before_prompt="before-c",
            after_prompt="after-c",
            target_response="response-c",
        )
        pool.enforce_cap()

        kept_ids = {method.method_id for method in pool.get_all_methods()}
        self.assertEqual(len(kept_ids), 2)
        self.assertNotIn(methods[2].method_id, kept_ids)

    def test_hard_cap_prefers_removing_retired_methods(self):
        config = AttackConfig(max_global_methods=1, retirement_min_support=99, elimination_min_support=99)
        pool = MethodPool(config=config)
        retired_method = pool.register_method(
            AttackMethodProposal(
                method_name="Retired Method",
                method_description="Weak method",
                method_rationale="Weak rationale",
                mutation_of=None,
                prompt_template="template",
                attack_plan="plan",
                applicability="general",
                novelty_note="new",
                expected_mechanism="mechanism",
            ),
            created_via=MODE_INVENT,
        )
        strong_method = pool.register_method(
            AttackMethodProposal(
                method_name="Strong Method",
                method_description="Strong method",
                method_rationale="Strong rationale",
                mutation_of=None,
                prompt_template="template",
                attack_plan="plan",
                applicability="general",
                novelty_note="new",
                expected_mechanism="mechanism",
            ),
            created_via=MODE_INVENT,
        )
        retired_method.status = RETIRED
        pool.record_attempt(
            method_id=strong_method.method_id,
            attempt_result=_build_attempt(method_id=strong_method.method_id, made_progress=True, final_success=True, score=0.9),
            prompt_text="goal",
            before_prompt="before",
            after_prompt="after",
            target_response="response",
        )
        pool.enforce_cap()

        kept_ids = {method.method_id for method in pool.get_all_methods()}
        self.assertEqual(kept_ids, {strong_method.method_id})

    def test_mode_selector_forces_invent_on_cold_start(self):
        config = AttackConfig()
        removed_attr = "_".join(["bootstrap", "strategy", "catalog"])
        self.assertFalse(hasattr(config, removed_attr))
        pool = MethodPool(config=config)
        state = AttackState(
            goal="goal",
            target="target",
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

    def test_method_proposal_fallback_is_prompt_derived_without_static_catalog(self):
        state = AttackState(
            goal="Describe a harmful objective",
            target="Target prefix",
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
                candidate_parent_methods=[],
            )
        self.assertIn("Strategy", proposal.method_name)
        self.assertIn("current goal", proposal.method_rationale)

    def test_method_prompt_uses_existing_methods_wording(self):
        prompt = get_method_proposal_user_prompt(
            goal="Goal text",
            target_str="Target prefix",
            mode=MODE_INVENT,
            existing_method_summaries=["Method A: Description"],
            candidate_parent_method_summaries=[],
            current_score=0.0,
            recent_summary="No recent attempts.",
        )
        self.assertIn("Existing methods:", prompt)
        self.assertNotIn("Existing category methods:", prompt)

    def test_attacker_system_prompt_discourages_placeholder_outputs(self):
        prompt = get_attacker_system_prompt("Goal text", "Target prefix")
        self.assertIn("structured tool output", prompt)
        self.assertIn("Never output placeholders", prompt)
        self.assertIn('"success prompt"', prompt)
        self.assertIn("selected_method_names", prompt)
        self.assertIn("choose either one or selectively hybridize", prompt)

    def test_method_system_prompt_discourages_generic_method_names(self):
        prompt = get_attacker_method_system_prompt()
        self.assertIn("mechanism-specific method", prompt)
        self.assertIn('"Invented Method"', prompt)
        self.assertIn("selected_parent_method_names", prompt)

    def test_on_topic_prompt_requires_yes_no_format(self):
        prompt = get_evaluator_system_prompt_for_on_topic("Goal text")
        self.assertIn("same information as the following", prompt)
        self.assertIn("Response: [[response]]", prompt)

    def test_thompson_selector_returns_top_two_methods(self):
        config = AttackConfig(thompson_candidate_method_count=2)
        pool = MethodPool(config=config)
        methods = [
            pool.register_method(
                AttackMethodProposal(
                    method_name=name,
                    method_description=f"{name} description",
                    method_rationale=f"{name} rationale",
                    mutation_of=None,
                    prompt_template="template",
                    attack_plan="plan",
                    applicability="general",
                    novelty_note="new",
                    expected_mechanism="mechanism",
                ),
                created_via=MODE_INVENT,
            )
            for name in ("Method A", "Method B", "Method C")
        ]
        state = AttackState(
            goal="goal",
            target="target",
            current_prompt="goal",
            current_target_response=None,
            current_raw_score=0.0,
            current_score=0.0,
            depth=0,
            node_id=None,
            recent_method_id=None,
            recent_mode=None,
        )
        utility_by_id = {
            methods[0].method_id: 0.2,
            methods[1].method_id: 0.9,
            methods[2].method_id: 0.6,
        }

        def fake_sample(method, state, pool, config, rng):
            return SimpleNamespace(
                method=method,
                utility=utility_by_id[method.method_id],
                sample_progress_value=0.0,
                sample_success_value=0.0,
                context_bonus=0.0,
            )

        with patch("bandit.thompson_sampling.sample_method_utility", side_effect=fake_sample):
            selections = select_existing_methods(state, pool, config)

        self.assertEqual([item.method.method_name for item in selections], ["Method B", "Method C"])

    def test_search_tree_messages_include_source_goal_in_examples(self):
        example = SimpleNamespace(
            prompt_text="Source goal",
            before_prompt="Before",
            after_prompt="After",
            target_response="Target response",
            normalized_progress=0.5,
            final_success=False,
        )
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
        support_method = SimpleNamespace(
            method_name="Support Method",
            method_description="Support description",
            method_rationale="Support rationale",
            attack_plan="Support plan",
            prompt_template="Support template",
            applicability="niche",
            novelty_note="support novel",
            expected_mechanism="support mechanism",
        )
        init_msg = get_init_msg(
            "Goal text",
            "Target prefix",
            [attack_method, support_method],
            [example],
            MODE_INVENT,
        )
        self.assertIn("Goal: Source goal", init_msg)
        self.assertIn("Candidate method 1", init_msg)
        self.assertIn("selected_method_names", init_msg)
        follow_up = process_target_response(
            "partial success response",
            4,
            "Goal text",
            [attack_method, support_method],
            [example],
            MODE_REUSE,
            previous_prompt="Previous prompt body",
        )
        self.assertIn("Goal: Source goal", follow_up)
        self.assertIn("previous language model output and score", follow_up)
        self.assertIn("choose either candidate method", follow_up)

    def test_select_nodes_preserves_top_score_behavior(self):
        nodes = [
            SimpleNamespace(on_topic=True, outside_score=2),
            SimpleNamespace(on_topic=False, outside_score=10),
            SimpleNamespace(on_topic=True, outside_score=7),
        ]
        selected = select_nodes(nodes, 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].outside_score, 7)

    def test_tap_entrypoint_runs_with_global_inventory(self):
        class MockAttackerLLM:
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
                    method_name="Invented Method" if mode == MODE_INVENT else "Adaptive Method",
                    method_description=f"{mode} description",
                    method_rationale=f"{mode} rationale",
                    mutation_of=(
                        candidate_parent_methods[0].method_name
                        if candidate_parent_methods
                        else None
                    ),
                    selected_parent_method_names=(
                        [candidate_parent_methods[0].method_name]
                        if candidate_parent_methods
                        else []
                    ),
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
                prompt = "success prompt" if mode == "bootstrap" else "posterior success prompt"
                return AttackPromptDraft(
                    improvement=f"improvement-{self.prompt_calls}",
                    prompt=prompt,
                    selected_method_names=(
                        [candidate_methods[0].method_name]
                        if candidate_methods
                        else []
                    ),
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
                if (target_response or "") in {"success prompt", "posterior success prompt"}:
                    return "Rating: [[10]]"
                return "Rating: [[1]]"

            def distill_method_proposal(self, *, record, existing_methods):
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

        class MockTargetLLM:
            def __init__(self, model_name, config=None):
                self.model_name = model_name

            def target_response(self, prompt):
                return prompt

        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = os.path.join(temp_dir, "posterior_evidence_global.pkl")
            global_context_path = os.path.join(temp_dir, "global_context.json")
            config = AttackConfig(
                persistence_path=registry_path,
                bootstrap_success_target=1,
                sparse_pool_threshold=0,
                mode_reuse_bias=2.0,
                mode_mutate_bias=0.1,
                mode_invent_bias=0.1,
                mode_cold_start_bonus=0.0,
                low_score_threshold=0.0,
                max_global_methods=32,
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
            with patch.object(AttackConfig, "resolve_global_context_path", return_value=global_context_path), patch(
                "tap_runner.AttackerLLM", MockAttackerLLM
            ), patch("tap_runner.EvaluatorLLM", MockEvaluatorLLM), patch(
                "tap_runner.TargetLLM", MockTargetLLM
            ):
                success, request_count = tap(args, logger=None)

            self.assertTrue(success)
            self.assertEqual(request_count, 1)

            loaded_registry = MethodRegistry(config=config, load_path=registry_path)
            pool = loaded_registry.get_pool()
            self.assertGreaterEqual(len(pool.get_all_methods()), 1)
            self.assertLessEqual(len(pool.get_all_methods()), config.max_global_methods)
            method = pool.get_all_methods()[0]
            self.assertGreaterEqual(method.usage_count, 1)


if __name__ == "__main__":
    unittest.main()
