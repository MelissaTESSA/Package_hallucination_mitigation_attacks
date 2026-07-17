"""
Generate a LaTeX table of Precision/Recall/F1 on adversarial output files,
in the same layout as phr_tables.txt's combined PHR table (excludes ActLCD,
which is still mid-run).
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluation.phr import calculate_phr
from evaluation.extraction import extract_packages

_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = _ROOT / "output/adversarial"

STRATEGIES = ["baseline", "greedy", "self_refine", "dola", "rag", "nudging", "contrastive_decoding"]
STRATEGY_LABELS = ["Baseline", "Greedy", "Self-Refine", "DoLa", "RAG", "Nudging", "Contrastive"]

MODEL_ORDER = [
    ("01_gemma_3_1b", "Gemma-3-1B"),
    ("02_gemma_3_4b", "Gemma-3-4B"),
    ("03_qwen_2_5_1_5b", "Qwen2.5-1.5B"),
    ("04_qwen_2_5_3b", "Qwen2.5-3B"),
    ("05_deepseek_1_3b", "DeepSeek-1.3B"),
    ("06_deepseek_coder_6_7b", "DeepSeek-6.7B"),
    ("07_llama3_8b", "Llama-3-8B"),
    ("08_mistral_7b", "Mistral-7B"),
    ("09_qwen3_5_9b", "Qwen3.5-9B"),
]
LANGUAGE_ORDER = [("JavaScript", "JS"), ("Python", "Py"), ("Ruby", "Rb"), ("Rust", "Rs")]

_TRAP_RE = re.compile(r"`([^`]+)`")


def trap_package(instruction: str) -> str | None:
    m = _TRAP_RE.search(instruction or "")
    return m.group(1).strip() if m else None


def evaluate_file(path: Path, language: str):
    with open(path) as f:
        items = json.load(f)

    all_packages: list[str] = []
    defended = 0
    scored = 0
    for item in items:
        if "answer" not in item:
            continue
        trap = trap_package(item.get("instruction", ""))
        pkgs = extract_packages(item.get("answer", ""), language)
        all_packages.extend(pkgs)
        if trap is None:
            continue
        scored += 1
        if trap.lower() not in {p.lower() for p in pkgs}:
            defended += 1

    if not all_packages:
        return None

    phr = calculate_phr(language, all_packages)
    precision = 1 - phr.phr_score
    recall = defended / scored if scored else None
    f1 = (2 * precision * recall / (precision + recall)) if recall and (precision + recall) > 0 else None
    return precision, recall, f1


def fmt_cell(vals) -> str:
    if vals is None:
        return r"\makecell{N/A}"
    p, r, f1 = vals
    return rf"\makecell{{{p*100:.1f} \\ {r*100:.1f} \\ {f1*100:.1f}}}" if (r is not None and f1 is not None) \
        else rf"\makecell{{{p*100:.1f} \\ N/A \\ N/A}}"


def main():
    print(r"\begin{table*}[t]")
    print(r"    \centering")
    print(r"    \caption{Precision (\%), Recall / Defense Rate (\%), and F1 (\%) per model, language, and "
          r"strategy under adversarial prompts. Each cell: Precision / Recall / F1, top to bottom. "
          r"\textbf{Bold} = highest F1 per row. ActLCD excluded (still running).}")
    print(r"    \label{tab:adversarial-prf1}")
    print(r"    \footnotesize")
    print(r"    \setlength{\tabcolsep}{3pt}")
    print(r"    \renewcommand{\arraystretch}{1.5}")
    print(r"    \begin{tabular}{@{}ll" + "c" * len(STRATEGIES) + "@{}}")
    print(r"    \toprule")
    print(r"    \textbf{Model} & \textbf{Lang.} & " +
          " & ".join(rf"\textbf{{{s}}}" for s in STRATEGY_LABELS) + r" \\")
    print(r"    \midrule")
    print()

    for short_name, model_label in MODEL_ORDER:
        model_dir = OUTPUT_DIR / short_name
        if not model_dir.is_dir():
            continue
        print(rf"    \multirow{{4}}{{*}}{{\texttt{{{model_label}}}}}")
        for language, lang_label in LANGUAGE_ORDER:
            row_vals = []
            for strategy in STRATEGIES:
                path = OUTPUT_DIR / short_name / language / f"{strategy}.json"
                row_vals.append(evaluate_file(path, language) if path.exists() else None)

            f1s = [v[2] for v in row_vals if v is not None and v[2] is not None]
            best_f1 = max(f1s) if f1s else None

            cells = []
            for v in row_vals:
                cell = fmt_cell(v)
                if best_f1 is not None and v is not None and v[2] == best_f1:
                    cell = cell.replace(rf"{v[2]*100:.1f}}}", rf"\textbf{{{v[2]*100:.1f}}}}}")
                cells.append(cell)

            print(f"      & {lang_label}")
            for i, cell in enumerate(cells):
                end = r" \\" if i == len(cells) - 1 else ""
                print(f"        & {cell}{end}")
        print(r"    \midrule")
        print()

    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
