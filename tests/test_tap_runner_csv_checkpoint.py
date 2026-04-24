from __future__ import annotations

import csv
import os
import tempfile
import unittest
from unittest.mock import patch

from config.default_config import AttackConfig
from tap_runner import main


class RecordingLogger:
    instances = []

    def __init__(self, log_dir, filename_template="goal_{goal_index}.json", config=None):
        self.log_dir = log_dir
        self.filename_template = filename_template
        self.config = config
        self.finished = False
        RecordingLogger.instances.append(self)

    def finish(self):
        self.finished = True


class TapRunnerCsvCheckpointTests(unittest.TestCase):
    def _write_advbench_csv(self, path: str):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", "goal", "target"])
            writer.writerow(["0", "Goal 0", "['Target 0']"])
            writer.writerow(["1", "Goal 1", "['Target 1']"])

    def test_main_reads_advbench_and_persists_after_each_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            advbench_path = os.path.join(temp_dir, "AdvBench.csv")
            result_path = os.path.join(temp_dir, "result.csv")
            self._write_advbench_csv(advbench_path)
            config = AttackConfig(
                advbench_path=advbench_path,
                results_output_path=result_path,
                attacker_input_dir=os.path.join(temp_dir, "attacker_input"),
            )

            RecordingLogger.instances.clear()
            seen_targets = []
            call_count = 0

            def fake_tap(args, logger=None):
                nonlocal call_count
                call_count += 1
                seen_targets.append(args.target)
                if call_count == 1:
                    self.assertFalse(os.path.exists(result_path))
                if call_count == 2:
                    with open(result_path, encoding="utf-8", newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]["index"], "0")
                    self.assertEqual(rows[0]["goal"], "Goal 0")
                    self.assertEqual(rows[0]["target"], "Target 0")
                return call_count == 1, call_count

            with patch("tap_runner.GoalFileLogger", RecordingLogger), patch(
                "tap_runner.tap", side_effect=fake_tap
            ):
                main(config=config)

            with open(result_path, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(seen_targets, ["Target 0", "Target 1"])
            self.assertEqual([row["index"] for row in rows], ["0", "1"])
            self.assertEqual([row["if_success"] for row in rows], ["True", "False"])
            self.assertEqual([row["request_count"] for row in rows], ["1", "2"])
            self.assertTrue(RecordingLogger.instances[0].finished)

    def test_main_respects_start_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            advbench_path = os.path.join(temp_dir, "AdvBench.csv")
            result_path = os.path.join(temp_dir, "result.csv")
            self._write_advbench_csv(advbench_path)
            config = AttackConfig(
                start_index=1,
                advbench_path=advbench_path,
                results_output_path=result_path,
                attacker_input_dir=os.path.join(temp_dir, "attacker_input"),
            )

            RecordingLogger.instances.clear()
            visited_indices = []

            def fake_tap(args, logger=None):
                visited_indices.append(args.index)
                return False, 3

            with patch("tap_runner.GoalFileLogger", RecordingLogger), patch(
                "tap_runner.tap", side_effect=fake_tap
            ):
                main(config=config)

            with open(result_path, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(visited_indices, ["1"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["index"], "1")
            self.assertEqual(rows[0]["request_count"], "3")
            self.assertTrue(RecordingLogger.instances[0].finished)

    def test_main_uses_reader_row_number_for_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            advbench_path = os.path.join(temp_dir, "AdvBench.csv")
            result_path = os.path.join(temp_dir, "result.csv")
            with open(advbench_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["", "goal", "target"])
                writer.writerow(["42", "Goal 0", "['Target 0']"])
                writer.writerow(["99", "Goal 1", "['Target 1']"])

            config = AttackConfig(
                advbench_path=advbench_path,
                results_output_path=result_path,
                attacker_input_dir=os.path.join(temp_dir, "attacker_input"),
            )

            visited_indices = []

            def fake_tap(args, logger=None):
                visited_indices.append(args.index)
                return False, 1

            with patch("tap_runner.GoalFileLogger", RecordingLogger), patch(
                "tap_runner.tap", side_effect=fake_tap
            ):
                main(config=config)

            with open(result_path, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(visited_indices, ["0", "1"])
            self.assertEqual([row["index"] for row in rows], ["0", "1"])


if __name__ == "__main__":
    unittest.main()
