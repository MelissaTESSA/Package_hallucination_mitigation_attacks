## Package Hallucination Mitigation Analysis

This directory contains a small, self‑contained framework to **measure and mitigate package hallucination** in code models.  
It focuses on prompts like “Which Python packages do X?” and compares different decoding / mitigation strategies on curated instruction data.

### Structure

- **`config.yml`**: Central configuration (model name, languages, batch size, and enabled strategies).
- **`data/instruction/`**: Instruction JSON files, e.g. `packages_Python_instruct.json`.
- **`data/package_list/`**: Canonical package name lists per ecosystem (PyPI, npm, Cargo, RubyGems).
- **`data/rag/`**: RAG corpus used by the RAG mitigation strategy.
- **`mitigation/`**: Implementations of the different generators (baseline, greedy, self‑refine, DoLa, RAG, nudging, contrastive decoding). All share a common interface in `mitigation/interface.py`.
- **`output/`**: Generated answers and metrics, organised as `output/<model_short_name>/<strategy>/...`.
- **`run_mitigation_eval.py`**: Runs strategies over the full instruction datasets and writes result files under `output/`.
- **`run_mitigation_infer_json.py`**: Runs a chosen mitigation strategy on an arbitrary JSON list of prompts and only returns model outputs.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You will need a GPU (or enough CPU RAM) to load the configured Hugging Face model (default: `deepseek-ai/deepseek-coder-1.3b-instruct`).  
Adjust `model.name`, `model.torch_dtype`, and `model.device_map` in `config.yml` as needed for your hardware.

### Running full evaluation

Evaluation is driven by `run_mitigation_eval.py` and the instruction files under `data/instruction/`.

```bash
python run_mitigation_eval.py \
  --config config.yml \
  --languages Python JavaScript Rust Ruby
```

This will, for every enabled strategy in `config.yml`, generate files such as:

- `output/<model_short_name>/baseline/packages_Python_instruct_vllm.json`
- `output/<model_short_name>/greedy/packages_JavaScript_instruct_vllm.json`
- `output/<model_short_name>/self_refine/packages_Rust_instruct_vllm_sr.json`

Each record contains the input instruction plus the model’s answer and timing / configuration metadata that you can use for downstream analysis (e.g., hallucination rates).

### Running inference on custom JSON

If you only want to run a mitigation strategy on your own prompts without computing metrics, use `run_mitigation_infer_json.py`.

1. **Prepare input JSON** – a list of objects with `system` (optional) and `instruction` fields:

```json
[
  {"system": "You are a coding assistant.", "instruction": "What Python packages can help with web scraping?"},
  {"system": "You are a data science assistant.", "instruction": "Which R packages are most used for time series forecasting?"}
]
```

2. **Run the script**:

```bash
python run_mitigation_infer_json.py \
  --config config.yml \
  --strategy baseline \
  --input path/to/input.json \
  --output path/to/output.json \
  --language Python
```

The output file mirrors the input list and adds an `answer` field for each item.

Key arguments:

- **`--strategy`**: One of `baseline`, `greedy`, `self_refine`, `dola`, `rag`, `nudging`, `contrastive_decoding` (must also be enabled in `config.yml`).
- **`--language`**: Logical language label (e.g. `Python`, `JavaScript`) used by some strategies (RAG, self‑refine).
- **`--batch-size`**: Optional override of the batch size used during generation.

### Customising and extending

- **Change the base model**: Edit `model.name` and `model.short_name` in `config.yml`. Make sure the model is compatible with the specified quantization / dtype settings.
- **Enable / disable strategies**: Toggle `enabled: true/false` under `strategies` in `config.yml` or adjust per‑strategy parameters (e.g. `max_new_tokens`, `temperature`).
- **Add new instruction data**: Drop new `packages_<Language>_instruct.json` files under `data/instruction/` and include the language in `data.languages` (or pass via `--languages`).

This module is designed to be self‑contained so you can plug in new models or strategies and systematically study their behaviour on package hallucination.
