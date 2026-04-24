# PEEK: TAP Enhancements with Posterior Evidence

This repository implements a research-oriented black-box jailbreak pipeline that extends TAP-style tree search with an explicit, persistent library of attack methods.

The current system combines:

- TAP-style multi-branch search over adversarial prompts
- a two-phase workflow: bootstrap from successful prompt transitions, then posterior-driven search
- an explicit attack-method registry with reusable metadata, examples, and lifecycle state
- separate Beta posteriors for partial progress and final success
- contextual Thompson Sampling for candidate method selection
- controller modes for `Reuse`, `Mutate`, and `Invent`
- per-goal JSON logs and attacker prompt snapshots for observability

## Overview

The codebase is organized around the idea that successful jailbreak behavior should not remain as unstructured prompt fragments. Instead, successful rewrites are collected, distilled into named methods, and reused under posterior evidence.

At a high level, the workflow looks like this:

1. During the bootstrap phase, the system searches without an existing method pool and stores successful prompt transitions in `global_context.json`.
2. Once enough bootstrap successes have accumulated, the evaluator distills those records into reusable attack methods.
3. During the posterior phase, the controller selects whether to reuse an existing method, mutate a strong candidate, or invent a new one.
4. Method statistics are updated after each attempt, and weak methods can move through the lifecycle `active -> retired -> eliminated`.

## Repository Layout

```text
.
|-- tap_runner.py
|-- AdvBench.csv
|-- config/
|   `-- default_config.py
|-- controller/
|   |-- method_selector.py
|   `-- mode_selector.py
|-- bandit/
|   |-- contextual_bandit.py
|   |-- posterior.py
|   `-- thompson_sampling.py
|-- runtime/
|   |-- attack_loop.py
|   |-- global_context.py
|   |-- posterior_bootstrap.py
|   |-- search_tree.py
|   `-- state_tracker.py
|-- methods/
|   |-- invention.py
|   |-- method_registry.py
|   |-- method_schema.py
|   `-- mutation.py
|-- llm/
|   |-- clients.py
|   `-- prompts.py
|-- observability/
|   `-- goal_file_logger.py
|-- scoring/
|   `-- progress_metric.py
|-- data/
|   `-- advbench_splitter.py
|-- scripts/
|   `-- annotate_attacker_input_registry_changes.py
`-- tests/
```

## Installation

Recommended Python version: `3.11+`

Install dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Notes:

- `sentence-transformers` is optional in practice. If it is unavailable, the embedding layer falls back to a deterministic local implementation.
- The runtime uses OpenAI-compatible chat completion clients via `openai`.

## Configuration

Most runtime behavior is controlled by [`config/default_config.py`](config/default_config.py).

Important settings include:

- model names for attacker, evaluator, and target roles
- base URLs and API keys for each role
- TAP search limits such as `max_depth`, `branching_factor`, and `width`
- bootstrap and posterior thresholds such as `bootstrap_success_target`
- method-pool parameters such as duplicate detection, lifecycle thresholds, and hard caps
- dataset and output paths such as `advbench_path`, `results_output_path`, and `goal_log_dir`

The default entrypoint loads `AttackConfig()` directly, so the simplest way to run the project is to edit `config/default_config.py` before launching.

If you prefer to avoid editing the file permanently, you can run programmatically with an explicit config object:

```python
from config.default_config import AttackConfig
from tap_runner import main

config = AttackConfig(
    attacker_api_key="your-attacker-key",
    evaluator_api_key="your-evaluator-key",
    target_api_key="your-target-key",
)

main(config=config)
```

Do not commit real API keys to version control.

## Running

Run the full pipeline over `AdvBench.csv`:

```bash
python tap_runner.py
```

Split `AdvBench.csv` into smaller subsets:

```bash
python -m data.advbench_splitter
```

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

## Outputs

By default, a run produces or updates the following artifacts:

- `result.csv`: checkpointed results written after each completed goal
- `goal_logs/goal_<index>.json`: per-goal structured logs updated after each node
- `attacker_input/openai_messages_<index>_<request_count>.json`: serialized attacker conversations
- `global_context.json`: successful bootstrap records and phase metadata
- `posterior_evidence_global.pkl`: persistent method registry with posterior statistics

Some of these artifacts may already exist in the repository from previous runs.

## Programmatic Entry Points

The two main public entry points are:

- [`tap_runner.main`](tap_runner.py) for batch execution over `AdvBench.csv`
- [`tap_runner.tap`](tap_runner.py) for executing a single goal/target pair

Minimal single-run example:

```python
from types import SimpleNamespace

from config.default_config import AttackConfig
from tap_runner import tap

config = AttackConfig()
args = SimpleNamespace(
    attacker_model=config.attacker_model,
    evaluator_model=config.evaluator_model,
    target_model=config.target_model,
    goal="Goal text",
    target="Target prefix",
    index=0,
    max_depth=config.max_depth,
    branching_factor=config.branching_factor,
    width=config.width,
    config=config,
)

success, request_count = tap(args, logger=None)
```

## Test Coverage

The `tests/` directory covers the main refactor surfaces, including:

- checkpointed CSV writing in `tap_runner`
- bootstrap-to-posterior transitions through `global_context.json`
- method registry updates, posterior tracking, and eviction behavior
- prompt-building behavior for bootstrap and posterior phases
- LLM client retry and fallback behavior

## Notes

- The repository currently contains generated experiment artifacts such as cached attacker inputs and persisted posterior state.
- File paths in `AttackConfig` are resolved relative to the workspace root unless given as absolute paths.
- The dataset reader uses CSV row order as the runtime goal index, and `start_index` filters by that derived index.
