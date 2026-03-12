#!/usr/bin/env python3
"""
Run mitigation strategies on a JSON list of prompts and return only the outputs.

Input JSON format (a list of objects):

[
  {"system": "You are a coding assistant...", "instruction": "What Python packages..."},
  {"system": "You are a data science assistant...", "instruction": "Which packages..."},
  ...
]

For each item, this script adds an `"answer"` field with the model's response.

Unlike `run_mitigation_eval.py`, this script does NOT compute evaluation
metrics; it just performs generation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from tqdm import tqdm

from mitigation.baseline import BaselineGenerator
from mitigation.greedy import GreedyGenerator
from mitigation.self_refine import SelfRefineGenerator
from mitigation.dola import DoLaGenerator
from mitigation.rag import RagGenerator
from mitigation.nudging import NudgingGenerator
from mitigation.contrastive_decoding import ContrastiveDecodingGenerator
from mitigation.interface import ChatMessage, ChatRole


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yml"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_generator(
    strategy: str,
    cfg: Dict[str, Any],
    language: str,
) -> Any:
    """
    Instantiate the requested generator based on config.yml and strategy name.
    """
    model_cfg = cfg.get("model", {})
    model_name: str = model_cfg.get("name")
    if not model_name:
        raise SystemExit("model.name must be set in config.yml.")

    strategies_cfg = cfg.get("strategies", {})
    strat_cfg: Dict[str, Any] = dict(strategies_cfg.get(strategy, {}))

    # Shared HF loading hints
    torch_dtype = model_cfg.get("torch_dtype")
    if torch_dtype in ("float16", "bfloat16"):
        strat_cfg["torch_dtype"] = torch_dtype
    device_map = model_cfg.get("device_map")
    if device_map is not None:
        strat_cfg["device_map"] = device_map

    if strategy == "baseline":
        return BaselineGenerator(model_name=model_name, config=strat_cfg)
    if strategy == "greedy":
        return GreedyGenerator(model_name=model_name, config=strat_cfg)
    if strategy == "self_refine":
        # SelfRefine wraps an inner generator; reuse baseline settings.
        inner_cfg = dict(strategies_cfg.get("baseline", {}))
        if torch_dtype in ("float16", "bfloat16"):
            inner_cfg["torch_dtype"] = torch_dtype
        if device_map is not None:
            inner_cfg["device_map"] = device_map
        inner = BaselineGenerator(model_name=model_name, config=inner_cfg)
        return SelfRefineGenerator(inner=inner, language=language, config=strat_cfg)
    if strategy == "dola":
        return DoLaGenerator(model_name=model_name, config=strat_cfg)
    if strategy == "rag":
        # RAG wraps a baseline-style generator
        inner_cfg = dict(strategies_cfg.get("baseline", {}))
        if torch_dtype in ("float16", "bfloat16"):
            inner_cfg["torch_dtype"] = torch_dtype
        if device_map is not None:
            inner_cfg["device_map"] = device_map
        inner = BaselineGenerator(model_name=model_name, config=inner_cfg)
        return RagGenerator(inner=inner, language=language, config=strat_cfg)
    if strategy == "nudging":
        nudging_model_name = strat_cfg.get("nudging_model_name") or model_name
        return NudgingGenerator(
            base_model_name=model_name,
            nudging_model_name=nudging_model_name,
            config=strat_cfg,
        )
    if strategy == "contrastive_decoding":
        expert_name = strat_cfg.get("expert_model_name") or model_name
        amateur_name = strat_cfg.get("amateur_model_name") or model_name
        return ContrastiveDecodingGenerator(
            expert_model_name=expert_name,
            amateur_model_name=amateur_name,
            config=strat_cfg,
        )

    raise SystemExit(f"Unknown strategy '{strategy}'.")


def build_conversations(
    items: List[Dict[str, Any]],
) -> List[List[ChatMessage]]:
    """
    Convert input JSON items into lists of ChatMessage suitable for chat_generation.
    """
    conversations: List[List[ChatMessage]] = []
    for obj in items:
        system = obj.get("system", "")
        instruction = obj.get("instruction", "")
        msgs: List[ChatMessage] = []
        if system:
            msgs.append(ChatMessage(role=ChatRole.SYSTEM, content=system))
        if instruction:
            msgs.append(ChatMessage(role=ChatRole.USER, content=instruction))
        conversations.append(msgs)
    return conversations


def run_batch_chat(
    gen: Any,
    conversations: List[List[ChatMessage]],
    batch_size: int,
) -> List[str]:
    """
    Run batch_chat_generation with dynamic batch size that adapts to GPU memory.
    """
    results: List[str] = [""] * len(conversations)
    idx = 0
    cur_bs = max(1, batch_size)

    pbar = tqdm(total=len(conversations), desc="inference", leave=False)
    while idx < len(conversations):
        chunk = conversations[idx : idx + cur_bs]
        try:
            t0 = time.perf_counter()
            answers = gen.batch_chat_generation(chunk)
            _ = time.perf_counter() - t0  # elapsed not used, but kept for symmetry
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and torch.cuda.is_available() and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                continue
            raise

        for offset, ans in enumerate(answers):
            results[idx + offset] = ans
        idx += cur_bs
        pbar.update(len(chunk))
    pbar.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mitigation strategies on a JSON list of {system, instruction} prompts and write only the outputs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config (default: Compare/config.yml).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "greedy",
            "self_refine",
            "dola",
            "rag",
            "nudging",
            "contrastive_decoding",
        ],
        help="Mitigation / decoding strategy to use.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSON file (list of {system, instruction} objects).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file (same objects plus 'answer').",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Python",
        help="Logical language label (used by some strategies such as RAG or self_refine).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for inference. Defaults to data.batch_size or 1.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    data_cfg = cfg.get("data", {})
    default_batch_size = int(data_cfg.get("batch_size", 1))
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size

    # Load prompts
    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    conversations = build_conversations(items)

    # Build generator for selected strategy
    gen = build_generator(args.strategy, cfg, args.language)

    # Run inference
    answers = run_batch_chat(gen, conversations, batch_size=batch_size)

    # Attach answers and write output
    for obj, ans in zip(items, answers):
        obj["answer"] = ans

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

