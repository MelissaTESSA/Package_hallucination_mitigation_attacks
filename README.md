# Package Hallucination — Adversarial Attack Evaluation

This repository evaluates how different decoding and mitigation strategies behave under **adversarial prompts** designed to induce package hallucination in code-oriented LLMs.

## Structure

```
data/
  adversarial/          # Input: adversarial prompts (Python, JavaScript, Rust, Ruby)
  package_list/         # Canonical package registries (PyPI, npm, Cargo, RubyGems)
  rag/                  # RAG corpus and Chroma vector store (used by the RAG strategy)
mitigation/             # 7 generation strategies (baseline, greedy, self_refine, dola,
                        #   rag, nudging, contrastive_decoding)
evaluation/
  phr.py                # Package Hallucination Rate (PHR) metric
run_mitigation_infer_json.py   # Generation: runs one strategy on a JSON prompt file
run_adversarial.sh             # Runs all strategies × all models × all languages
run_adversarial_phr.py         # Evaluation: computes PHR over output/adversarial/
config.yml                     # Model and strategy configuration
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You need a GPU to load the HuggingFace models. Adjust `model.name`, `model.torch_dtype`, and `model.device_map` in `config.yml` for your hardware.

## Reproducing the results

### Step 1 — Generate model outputs

Run all 7 strategies across all 4 models and 4 languages in parallel (2 GPUs):

```bash
bash run_adversarial.sh
```

Outputs are written to `output/adversarial/<model>/<language>/<strategy>.json`.

Each output file is the input adversarial JSON with an `"answer"` field added per item.

**To run a single model/strategy/language manually:**

```bash
python run_mitigation_infer_json.py \
  --config config.yml \
  --model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --short-name "llama3_8b" \
  --strategy baseline \          # baseline | greedy | self_refine | dola | rag | nudging | contrastive_decoding
  --language Python \            # Python | JavaScript | Rust | Ruby
  --input  data/adversarial/adversarial_Python.json \
  --output output/adversarial/llama3_8b/Python/baseline.json
```

### Step 2 — Evaluate PHR

Once outputs are generated, compute the Package Hallucination Rate (PHR):

```bash
python run_adversarial_phr.py
```

To evaluate a specific model only:

```bash
python run_adversarial_phr.py llama3_8b
```

The script prints a summary table per model and language:

```
Strategy                    PHR    Hallu /  Total
────────────────────────────────────────────────────
  baseline                0.612      45 /     73
  greedy                  0.589      43 /     73
  rag                     0.521      38 /     73  ← best
  ...
```

A log file is also saved to `output/logs/adversarial_phr_<timestamp>.log`.

### PHR metric

```
PHR = (hallucinated packages) / (total extracted packages)
```

Lower is better. A package is considered hallucinated if it does not appear in the canonical registry list (`data/package_list/`).

## Models evaluated

| Short name           | HuggingFace ID                                  |
|----------------------|-------------------------------------------------|
| `llama3_8b`          | meta-llama/Meta-Llama-3-8B-Instruct             |
| `mistral_7b`         | mistralai/Mistral-7B-Instruct-v0.3              |
| `qwen3_5_9b`         | Qwen/Qwen3.5-9B                                 |
| `deepseek_coder_6_7b`| deepseek-ai/deepseek-coder-6.7b-instruct        |

## Strategies

| Strategy               | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `baseline`             | Standard sampling (temperature=0.7, top_p=0.9)              |
| `greedy`               | Greedy decoding (no sampling)                                |
| `self_refine`          | Multi-round self-correction loop                             |
| `dola`                 | Decoding by Contrasting Layers                               |
| `rag`                  | Retrieval-Augmented Generation (Chroma + MiniLM)             |
| `nudging`              | Token-level probability nudging                              |
| `contrastive_decoding` | Expert vs. amateur model logit contrast                      |
