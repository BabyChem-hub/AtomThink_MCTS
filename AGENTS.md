# Repository Guidelines

## Project Structure & Module Organization
Core training and evaluation code lives under `src/llamafactory`, which is split into `train/`, `evaluation/`, and `model/` subpackages. Entry script `src/train.py` consumes YAML presets stored in `configs/` (for example, `configs/train_full/`). Place curated datasets in `data/`, keep shared visuals in `figures/`, and stash reproducible metrics or logs in `evaluation_results/` so they remain separate from raw inputs.

## Build, Test, and Development Commands
Set up the environment with `pip install -r requirements.txt`. Launch fine-tuning via `python src/train.py configs/<profile>.yaml`; wrap the same command with `torchrun --nproc_per_node <n>` when scaling to multiple GPUs. Evaluate checkpoints with `python src/llamafactory/evaluation/run_evaluation_parallel.py --config <yaml> --task <benchmark> --method <mode> --temperature 0.0` to mirror paper metrics. Use helper scripts such as `download_llama.py` and `update_config.py` whenever checkpoints or model configs need to be synchronized.

## Coding Style & Naming Conventions
Favor PEP 8 defaults with 4-space indentation and descriptive, lowercase module names. Classes stay in PascalCase, functions and variables in snake_case, and configuration YAML files in lowercase with hyphen separators. Add type hints to public functions where practical. Reformat Python files with `yapf -i <path>` before committing, and keep docstrings in Google style to match existing modules.

## Testing Guidelines
Base regression checks on the evaluation harness: provide at least one deterministic run per change and capture the resulting metrics in `evaluation_results/`. Unit tests should sit beside the code under `src/llamafactory/<area>/tests/`, follow the `test_<feature>.py` pattern, and execute with `python -m pytest`. Target at least 70% statement coverage on code you touch; explain any intentional gaps directly in the pull request.

## Commit & Pull Request Guidelines
Commit subjects should be short, imperative, and under 72 characters, mirroring the current history. Pull requests need a clear motivation, the commands or scripts you executed, and any metric tables or screenshots that support new results. Reference the relevant config or dataset paths and request reviewers who maintain the affected subsystem.

## Security & Configuration Tips
Keep secrets such as `OPENAI_API_KEY` in your shell profile or a `.env` file excluded by `.gitignore`. Validate hashes before trusting downloaded checkpoints, and avoid committing large binaries. When authoring YAML, template machine-specific paths through environment variables to keep shared configs portable.
