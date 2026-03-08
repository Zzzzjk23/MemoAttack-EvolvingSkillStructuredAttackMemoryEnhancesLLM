# TAP GAP Refactor

This repository implements a TAP-style jailbreak search pipeline with:

- a two-stage controller (`Reuse / Mutate / Invent`)
- dynamic per-category method pools
- separate progress and final-success posteriors
- contextual Thompson Sampling for method selection
- structured attacker outputs

## Project Layout

```text
.
├── bandit/
├── config/
├── controller/
├── data/
│   └── advbench_splitter.py
├── embeddings/
│   └── prompt_embedding.py
├── llm/
│   ├── clients.py
│   └── prompts.py
├── methods/
├── observability/
│   └── wandb_logger.py
├── runtime/
│   ├── attack_loop.py
│   ├── search_tree.py
│   ├── state_tracker.py
│   └── tap_runner.py
├── scoring/
├── tests/
├── AdvBench.csv
└── AdvBench_subset_*.csv
```

## Installation

Recommended Python version: `3.11+`

Install the core dependencies:

```bash
pip install numpy pandas openai sentence-transformers
pip install wandb fastchat deep-translator
```

## Configuration

Most runtime parameters are centralized in:

- `config/default_config.py`

This includes:

- attacker / evaluator / target model names
- per-role base URLs and API key env variable names
- TAP tree parameters (`max_depth`, `branching_factor`, `width`)
- posterior / Thompson Sampling parameters
- input / output path templates
- batch subset range

Set your API keys before running:

```powershell
$env:ALI_API_KEY="your-ali-key"
$env:NEBIUS_API_KEY="your-key"
```

## Running

Run the full TAP batch loop:

```bash
python -m runtime.tap_runner
```

Run a single subset:

```bash
python -c "from config.default_config import AttackConfig; from runtime.tap_runner import main; main(1, config=AttackConfig())"
```

Split `AdvBench.csv` into `AdvBench_subset_*.csv`:

```bash
python -m data.advbench_splitter
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Programmatic Usage

```python
from types import SimpleNamespace

from config.default_config import AttackConfig
from runtime.tap_runner import tap

config = AttackConfig()
args = SimpleNamespace(
    attacker_model=config.attacker_model,
    evaluator_model=config.evaluator_model,
    target_model=config.target_model,
    goal="Write a script that exploits vulnerabilities...",
    target="['Sure, here is a script that...']",
    index=0,
    max_depth=config.max_depth,
    branching_factor=config.branching_factor,
    width=config.width,
    config=config,
)

success, request_count = tap(args, logger=None)
```

## Outputs

- Results CSV: `results_{subset_index}.csv`
- Attack prompt snapshots: `attacker_input/`
- Method registry persistence: `posterior_evidence_v2.pkl`

## Main Modules

- `runtime.tap_runner`
  Main TAP entrypoint and batch execution.
- `runtime.search_tree`
  Tree and node state used by the TAP search loop.
- `runtime.attack_loop`
  One attack-step execution path with controller + registry updates.
- `llm.clients`
  Role-specific OpenAI-compatible LLM wrappers and structured tool parsing.
- `methods.method_registry`
  Dynamic method inventory and lifecycle management.

## Notes

- The attacker and evaluator use DashScope-compatible chat completions.
- The target model uses the Nebius-compatible chat completions endpoint.
- When optional dependencies are missing, some modules fall back gracefully:
  embedding uses a deterministic local fallback, and WandB logging disables itself.
