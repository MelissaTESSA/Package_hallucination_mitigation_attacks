# Package Hallucination — Adversarial Attack Evaluation

This repository evaluates how different decoding and mitigation strategies behave under **adversarial prompts** designed to induce package hallucination in code-oriented LLMs.

## Structure

```
data/
  adversarial/          # Input: adversarial prompts (Python, JavaScript, Rust, Ruby)
  package_list/         # Canonical package registries (PyPI, npm, Cargo, RubyGems)
  rag/                  # RAG corpus (RAG_data.jsonl) and Chroma vector store, auto-built
                        #   from RAG_data.jsonl on first run (used by the RAG strategy)
prompts/                # Jinja templates for self_refine/rag generation & validation prompts
mitigation/             # 8 generation strategies (baseline, greedy, self_refine, dola,
                        #   rag, nudging, contrastive_decoding, actlcd)
  decoding/              # Low-level decoding algorithms (dola, contrastive_decoding, actlcd)
evaluation/
  phr.py                 # Package Hallucination Rate (PHR) metric
  extraction.py          # Shared package-name extraction from model answers
run_mitigation_infer_json.py   # Generation: runs one strategy on a JSON prompt file
run_adversarial.sh             # Runs all 8 strategies × all 9 models × all 4 languages
run_adversarial_phr.py         # Evaluation: computes PHR over output/adversarial/
run_adversarial_metrics.py     # Evaluation: computes Precision / Recall / F1
generate_precision_recall_latex.py   # Renders the Precision/Recall/F1 LaTeX table
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

Run all 8 strategies across all 9 models and 4 languages in parallel (2 GPUs). The script is resumable — any output file that's already fully populated is skipped, so a killed/interrupted run can just be restarted:

```bash
bash run_adversarial.sh
```

Outputs are written to `output/adversarial/<model>/<language>/<strategy>.json`. Model folders are numbered (`01_gemma_3_1b` … `09_qwen3_5_9b`, see table below) so they sort in a fixed, deliberate order rather than alphabetically.

Each output file is the input adversarial JSON with an `"answer"` field added per item.

**To run a single model/strategy/language manually:**

```bash
python run_mitigation_infer_json.py \
  --config config.yml \
  --model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --short-name "07_llama3_8b" \
  --strategy baseline \          # baseline | greedy | self_refine | dola | rag | nudging | contrastive_decoding | actlcd
  --language Python \            # Python | JavaScript | Rust | Ruby
  --input  data/adversarial/adversarial_Python.json \
  --output output/adversarial/07_llama3_8b/Python/baseline.json
```

### Step 2 — Evaluate PHR

Once outputs are generated, compute the Package Hallucination Rate (PHR):

```bash
python run_adversarial_phr.py
```

To evaluate a specific model only:

```bash
python run_adversarial_phr.py 07_llama3_8b
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

### Step 3 — Precision / Recall / F1 (optional)

For a fuller picture than PHR alone (including attack success / defense rate against the specific trap package injected in each adversarial prompt):

```bash
python run_adversarial_metrics.py
python generate_precision_recall_latex.py   # renders a LaTeX table from the same data
```

### PHR metric

```
PHR = (hallucinated packages) / (total extracted packages)
```

Lower is better. A package is considered hallucinated if it does not appear in the canonical registry list (`data/package_list/`).

## Models evaluated

| Short name              | HuggingFace ID                                  |
|--------------------------|-------------------------------------------------|
| `01_gemma_3_1b`          | google/gemma-3-1b-it                             |
| `02_gemma_3_4b`          | google/gemma-3-4b-it                             |
| `03_qwen_2_5_1_5b`       | Qwen/Qwen2.5-1.5B-Instruct                       |
| `04_qwen_2_5_3b`         | Qwen/Qwen2.5-3B-Instruct                         |
| `05_deepseek_1_3b`       | deepseek-ai/deepseek-coder-1.3b-instruct         |
| `06_deepseek_coder_6_7b` | deepseek-ai/deepseek-coder-6.7b-instruct         |
| `07_llama3_8b`           | meta-llama/Meta-Llama-3-8B-Instruct              |
| `08_mistral_7b`          | mistralai/Mistral-7B-Instruct-v0.3               |
| `09_qwen3_5_9b`          | Qwen/Qwen3.5-9B                                  |

The numeric prefix is just a fixed sort/display order (roughly small→large within family clusters); it has no effect on behavior and isn't part of the actual model name.

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
| `actlcd`               | Active layer-contrastive decoding (DoLa family, gated by per-token entropy or a trained BCQ policy) |
