# Compare: Package Hallucination Benchmarks (HuggingFace backend)

This directory contains a HuggingFace-based reimplementation of the package
hallucination benchmarks and mitigation strategies.

The core pieces are:

- `data/` – instruction datasets and package lists
  - `data/instruction/packages_<Language>_instruct.json` – prompts per package
  - `data/package_list/*.txt` – ground-truth package name lists
  - `data/rag/*` – RAG corpus (`RAG_data.jsonl`, `pypi_descriptions.jsonl`)
- `mitigation/` – decoding and mitigation strategies implemented on top of
  HuggingFace models:
  - `baseline.py` – temperature / top-p sampling
  - `greedy.py` – greedy decoding (argmax)
  - `dola.py` – DoLa (Decoding by Contrasting Layers)
  - `nudging.py` – inference-time nudging with base + nudging models
  - `rag.py` – retrieval-augmented generation wrapper
  - `self_refine.py` – iterative self-refinement with package validation
  - `contrastive_decoding.py` – expert vs amateur contrastive decoding
  - `interface.py` – common `Generator` / `ChatMessage` abstractions
- `output/` – generated answers and metrics for each model/strategy
- `evaluation/` – evaluation helpers (e.g. `phr.py` for Package Hallucination Rate)
- `test/` – notebooks for quick experiments, e.g.
  `test/generation_strategies.ipynb`.

## Environment and dependencies

Install the Python dependencies for the module:

```bash
pip install -r requirements.txt
```

You will need a working CUDA setup if you want to run larger models or use
`device_map: auto` in the config.

## Configuration (`config.yml`)

Global configuration is centralised in `config.yml`:

- `data.instruction_dir` – directory containing the
  `packages_<Language>_instruct.json` files.
- `data.languages` – list of language suffixes to process
  (e.g. `["Python", "JavaScript", "Rust", "Ruby"]`).
- `data.batch_size` – default batch size for batched generation (per forward pass).
- `model.name` – HuggingFace model id to evaluate.
- `model.short_name` – short identifier for the output subdirectory name.
- `model.torch_dtype` / `model.device_map` – optional HF loading hints.
- `output.root_dir` – root directory for evaluation outputs (default: `output/`).
- `strategies.*` – per-strategy configuration:
  - `enabled` – whether to run this strategy.
  - `subdir` – subdirectory under `output/<short_name>/`.
  - decoding / mitigation-specific parameters
    (e.g. `max_new_tokens`, `temperature`, `max_rounds`, `k`, `top_prob_thres`).
  - optional per-strategy `batch_size` to override the global one.

## Running evaluations

Use `run_mitigation_eval.py` to generate answers for all enabled strategies and
languages defined in `config.yml`:

```bash
python run_mitigation_eval.py
```

You can override the config path and languages at the CLI:

```bash
python run_mitigation_eval.py --config custom_config.yml --languages Python JavaScript
```

For each strategy `S` and language `L`, outputs are written to:

- `output/<short_name>/baseline/packages_<L>_instruct_vllm.json`
- `output/<short_name>/greedy/packages_<L>_instruct_vllm.json`
- `output/<short_name>/self_refine/packages_<L>_instruct_vllm_sr.json`
- `output/<short_name>/dola/packages_<L>_instruct_vllm.json`
- `output/<short_name>/rag/packages_<L>_instruct_vllm.json`
- `output/<short_name>/nudging/packages_<L>_instruct_vllm.json`
- `output/<short_name>/contrastive_decoding/packages_<L>_instruct_vllm.json`

The exact set depends on which strategies are marked `enabled: true`.

## Measuring hallucinations

The `evaluation/phr.py` module provides a simple Package Hallucination Rate
(PHR) calculation given:

- a language (`\"Python\"`, `\"JavaScript\"`, …),
- a list of predicted package names,
- and the corresponding ground-truth package list file under `data/package_list/`.

You can import `calculate_phr` in a notebook or script and pass it the model
outputs for coarse-grained hallucination statistics.

## Quick manual testing

For an interactive overview of the generation strategies, open:

- `test/generation_strategies.ipynb`

This notebook:

- Instantiates the different generators (`baseline`, `greedy`, `dola`,
  `nudging`, `rag`, `self_refine`, `contrastive_decoding`).
- Runs them on a shared test instruction.
- Explains their configuration and qualitative behaviour side by side.

