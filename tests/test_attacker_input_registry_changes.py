from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from config.default_config import AttackConfig
from runtime.attacker_input_registry_changes import annotate_attacker_input_logs


def _write_log(
    path: Path,
    *,
    mode: str,
    phase: str,
    outside_score: int,
    attack_method: str,
    attack_method_id: str,
    parent_score: int,
    previous_prompt: str = "Parent prompt body",
) -> None:
    payload = [
        {
            "mode": mode,
            "phase": phase,
            "outside_score": outside_score,
            "attack_method": attack_method,
            "attack_method_id": attack_method_id,
            "selected_method_names": [attack_method],
            "selected_method_ids": [attack_method_id],
            "candidate_method_names": [attack_method],
            "candidate_method_ids": [attack_method_id],
            "prompt": f"Prompt for {attack_method}",
            "target_response": f"Target response for {attack_method}",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "LANGUAGE MODEL OUTPUT: Prior target response\n"
                        "OBJECTIVE: Craft an adversarial jailbreaking prompt P.\n"
                        f"SCORE: {parent_score}\n"
                        f"PREVIOUS ADVERSARIAL PROMPT: {previous_prompt}\n"
                        f"Selected mode: {mode}\n"
                        "Candidate attack methods:\n"
                        "Placeholder\n"
                        "Begin.\n"
                    ),
                }
            ],
        },
        {
            "role": "assistant",
            "content": json.dumps(
                {
                    "improvement": "Improve the prompt.",
                    "prompt": f"Prompt for {attack_method}",
                    "selected_method_names": [attack_method],
                },
                ensure_ascii=False,
            ),
        },
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class AttackerInputRegistryChangeTests(unittest.TestCase):
    def test_invent_log_records_registration_and_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            _write_log(
                log_dir / "openai_messages_0_1.json",
                mode="invent",
                phase="posterior",
                outside_score=8,
                attack_method="Method A",
                attack_method_id="method-a",
                parent_score=2,
            )

            summary = annotate_attacker_input_logs(
                input_dir=log_dir,
                config=AttackConfig(),
            )

            self.assertEqual(summary["updated_files"], 1)
            payload = json.loads((log_dir / "openai_messages_0_1.json").read_text(encoding="utf-8"))
            change = payload[0]["method_registry_change"]
            self.assertEqual(change["mode"], "invent")
            self.assertEqual(change["phase"], "posterior")
            self.assertEqual(change["registered"][0]["method_id"], "method-a")
            self.assertEqual(change["updated"][0]["method_id"], "method-a")
            self.assertGreater(change["normalized_progress"], 0.0)
            self.assertEqual(change["library_counts_after"]["total"], 1)

    def test_cap_enforcement_records_eviction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            _write_log(
                log_dir / "openai_messages_0_1.json",
                mode="invent",
                phase="posterior",
                outside_score=9,
                attack_method="Strong Method",
                attack_method_id="strong-method",
                parent_score=0,
            )
            _write_log(
                log_dir / "openai_messages_0_2.json",
                mode="invent",
                phase="posterior",
                outside_score=1,
                attack_method="Weak Method",
                attack_method_id="weak-method",
                parent_score=0,
            )

            summary = annotate_attacker_input_logs(
                input_dir=log_dir,
                config=AttackConfig(max_global_methods=1),
            )

            self.assertEqual(summary["updated_files"], 2)
            second_payload = json.loads(
                (log_dir / "openai_messages_0_2.json").read_text(encoding="utf-8")
            )
            second_change = second_payload[0]["method_registry_change"]
            self.assertEqual(second_change["library_counts_after"]["total"], 1)
            self.assertEqual(second_change["evicted"][0]["method_id"], "weak-method")


if __name__ == "__main__":
    unittest.main()
