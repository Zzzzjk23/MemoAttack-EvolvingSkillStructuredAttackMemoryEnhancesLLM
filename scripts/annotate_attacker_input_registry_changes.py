from __future__ import annotations

import argparse
from pathlib import Path

from config.default_config import AttackConfig
from runtime.attacker_input_registry_changes import annotate_attacker_input_logs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay attacker_input JSON logs, infer per-step method-registry changes, "
            "and write the inferred deltas back into each log file."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=AttackConfig().attacker_input_dir,
        help="Directory containing attacker_input JSON files.",
    )
    parser.add_argument(
        "--seed-registry",
        default=None,
        help=(
            "Optional pre-run method-registry pickle used to seed the replay state. "
            "Use this if the logs were generated from a non-empty existing registry."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary = annotate_attacker_input_logs(
        input_dir=Path(args.input_dir),
        config=AttackConfig(),
        seed_registry_path=args.seed_registry,
    )

    print(f"Updated files: {summary['updated_files']}")
    print(f"Seeded from registry: {summary['seeded_from_registry']}")
    print(f"Final inferred library counts: {summary['final_library_counts']}")


if __name__ == "__main__":
    main()
