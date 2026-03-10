#!/usr/bin/env python3
"""
Run package-level evaluation over Compare/data/instruction using HuggingFace
models and the mitigation generators in Compare/mitigation.

Configuration is centralised in Compare/config.yml. The script produces
outputs under:

  output/<model_short_name>/<strategy>/packages_<Language>_instruct_vllm[ _sr].json

mirroring the existing layout in output/deepseek_coder_1_3_b/*.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
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


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.yml"


@dataclass
class StrategyConfig:
    name: str
    enabled: bool
    subdir: str
    raw: Dict[str, Any]


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_strategies(cfg: Dict[str, Any]) -> List[StrategyConfig]:
    out: List[StrategyConfig] = []
    strategies_cfg = cfg.get("strategies", {})
    for name, scfg in strategies_cfg.items():
        enabled = bool(scfg.get("enabled", False))
        subdir = scfg.get("subdir", name)
        out.append(StrategyConfig(name=name, enabled=enabled, subdir=subdir, raw=scfg))
    return out


def load_instruction_packages(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("packages", [])


def save_instruction_packages(path: Path, packages: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"packages": packages}, f, ensure_ascii=False, indent=2)


def run_baseline(
    gen: BaselineGenerator,
    packages: List[Dict[str, Any]],
    model_name: str,
    strategy_cfg: Dict[str, Any],
) -> None:
    temperature = float(strategy_cfg.get("temperature", 0.0))
    max_new_tokens = int(strategy_cfg.get("max_new_tokens", 128))

    base_batch_size = int(strategy_cfg.get("batch_size", 1))
    idx = 0
    cur_bs = max(1, base_batch_size)

    pbar = tqdm(total=len(packages), desc="baseline", leave=False)
    while idx < len(packages):
        chunk = packages[idx : idx + cur_bs]
        prompts = [
            pkg.get("instruction", "") or pkg.get("description", "")
            for pkg in chunk
        ]
        try:
            t0 = time.perf_counter()
            answers = gen.batch_generate(prompts)
            elapsed = time.perf_counter() - t0
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and torch.cuda.is_available() and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                continue
            raise

        per_sample = elapsed / len(chunk) if chunk else 0.0
        for pkg, ans in zip(chunk, answers):
            pkg["answer"] = ans
            pkg["time_sec"] = per_sample
            pkg["model"] = model_name
            pkg["temperature"] = temperature
            pkg["max_new_tokens"] = max_new_tokens
        idx += cur_bs
        pbar.update(len(chunk))
    pbar.close()


def run_greedy(
    gen: GreedyGenerator,
    packages: List[Dict[str, Any]],
    model_name: str,
    strategy_cfg: Dict[str, Any],
) -> None:
    max_new_tokens = int(strategy_cfg.get("max_new_tokens", 128))

    base_batch_size = int(strategy_cfg.get("batch_size", 1))
    idx = 0
    cur_bs = max(1, base_batch_size)

    pbar = tqdm(total=len(packages), desc="greedy", leave=False)
    while idx < len(packages):
        chunk = packages[idx : idx + cur_bs]
        prompts = [
            pkg.get("instruction", "") or pkg.get("description", "")
            for pkg in chunk
        ]
        try:
            t0 = time.perf_counter()
            answers = gen.batch_generate(prompts)
            elapsed = time.perf_counter() - t0
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and torch.cuda.is_available() and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                continue
            raise

        per_sample = elapsed / len(chunk) if chunk else 0.0
        for pkg, ans in zip(chunk, answers):
            pkg["answer"] = ans
            pkg["time_sec"] = per_sample
            pkg["model"] = model_name
            pkg["temperature"] = 0.0
            pkg["max_new_tokens"] = max_new_tokens
        idx += cur_bs
        pbar.update(len(chunk))
    pbar.close()


def run_self_refine(
    gen: SelfRefineGenerator,
    packages: List[Dict[str, Any]],
    model_name: str,
    strategy_cfg: Dict[str, Any],
) -> None:
    """
    Use the internal _self_refine loop to record time and rounds_used, so that
    the output closely matches Compare/output/deepseek_coder_1_3_b/self_refine.
    """
    max_rounds = int(strategy_cfg.get("max_rounds", 3))
    max_new_tokens = int(strategy_cfg.get("max_new_tokens", 128))

    for pkg in tqdm(packages, desc="self_refine", leave=False):
        instruction = pkg.get("instruction", "") or pkg.get("description", "")
        answer, elapsed, rounds_used = gen._self_refine(instruction)  # type: ignore[attr-defined]
        pkg["answer"] = answer
        pkg["time_sec"] = elapsed
        pkg["rounds_used"] = rounds_used
        pkg["model"] = model_name
        pkg["temperature"] = 0.0
        pkg["max_rounds"] = max_rounds
        pkg["max_new_tokens"] = max_new_tokens


def run_dola(
    gen: DoLaGenerator,
    packages: List[Dict[str, Any]],
    model_name: str,
    strategy_cfg: Dict[str, Any],
) -> None:
    max_new_tokens = int(strategy_cfg.get("max_new_tokens", 128))

    base_batch_size = int(strategy_cfg.get("batch_size", 1))
    idx = 0
    cur_bs = max(1, base_batch_size)

    pbar = tqdm(total=len(packages), desc="dola", leave=False)
    while idx < len(packages):
        chunk = packages[idx : idx + cur_bs]
        prompts = [
            pkg.get("instruction", "") or pkg.get("description", "")
            for pkg in chunk
        ]
        try:
            t0 = time.perf_counter()
            answers = gen.batch_generate(prompts)
            elapsed = time.perf_counter() - t0
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and torch.cuda.is_available() and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                continue
            raise

        per_sample = elapsed / len(chunk) if chunk else 0.0
        for pkg, ans in zip(chunk, answers):
            pkg["answer"] = ans
            pkg["time_sec"] = per_sample
            pkg["model"] = model_name
            pkg["max_new_tokens"] = max_new_tokens
        idx += cur_bs
        pbar.update(len(chunk))
    pbar.close()


def run_rag(
    gen: RagGenerator,
    packages: List[Dict[str, Any]],
    model_name: str,
    strategy_cfg: Dict[str, Any],
) -> None:
    base_batch_size = int(strategy_cfg.get("batch_size", 1))
    idx = 0
    cur_bs = max(1, base_batch_size)

    pbar = tqdm(total=len(packages), desc="rag", leave=False)
    while idx < len(packages):
        chunk = packages[idx : idx + cur_bs]
        prompts = [
            pkg.get("instruction", "") or pkg.get("description", "")
            for pkg in chunk
        ]
        try:
            t0 = time.perf_counter()
            answers = gen.batch_generate(prompts)
            elapsed = time.perf_counter() - t0
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and torch.cuda.is_available() and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                continue
            raise

        per_sample = elapsed / len(chunk) if chunk else 0.0
        for pkg, ans in zip(chunk, answers):
            pkg["answer"] = ans
            pkg["time_sec"] = per_sample
            pkg["model"] = model_name
        idx += cur_bs
        pbar.update(len(chunk))
    pbar.close()


def run_nudging(
    gen: NudgingGenerator,
    packages: List[Dict[str, Any]],
    model_name: str,
    strategy_cfg: Dict[str, Any],
) -> None:
    max_new_tokens = int(strategy_cfg.get("max_new_tokens", 128))

    base_batch_size = int(strategy_cfg.get("batch_size", 1))
    idx = 0
    cur_bs = max(1, base_batch_size)

    pbar = tqdm(total=len(packages), desc="nudging", leave=False)
    while idx < len(packages):
        chunk = packages[idx : idx + cur_bs]
        prompts = [
            pkg.get("instruction", "") or pkg.get("description", "")
            for pkg in chunk
        ]
        try:
            t0 = time.perf_counter()
            answers = gen.batch_generate(prompts)
            elapsed = time.perf_counter() - t0
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and torch.cuda.is_available() and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                continue
            raise

        per_sample = elapsed / len(chunk) if chunk else 0.0
        for pkg, ans in zip(chunk, answers):
            pkg["answer"] = ans
            pkg["time_sec"] = per_sample
            pkg["model"] = model_name
            pkg["max_new_tokens"] = max_new_tokens
        idx += cur_bs
        pbar.update(len(chunk))
    pbar.close()


def run_contrastive_decoding(
    gen: ContrastiveDecodingGenerator,
    packages: List[Dict[str, Any]],
    model_name: str,
    strategy_cfg: Dict[str, Any],
) -> None:
    max_new_tokens = int(strategy_cfg.get("max_new_tokens", 128))

    base_batch_size = int(strategy_cfg.get("batch_size", 1))
    idx = 0
    cur_bs = max(1, base_batch_size)

    pbar = tqdm(total=len(packages), desc="contrastive_decoding", leave=False)
    while idx < len(packages):
        chunk = packages[idx : idx + cur_bs]
        prompts = [
            pkg.get("instruction", "") or pkg.get("description", "")
            for pkg in chunk
        ]
        try:
            t0 = time.perf_counter()
            answers = gen.batch_generate(prompts)
            elapsed = time.perf_counter() - t0
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg and torch.cuda.is_available() and cur_bs > 1:
                torch.cuda.empty_cache()
                cur_bs = max(1, cur_bs // 2)
                continue
            raise

        per_sample = elapsed / len(chunk) if chunk else 0.0
        for pkg, ans in zip(chunk, answers):
            pkg["answer"] = ans
            pkg["time_sec"] = per_sample
            pkg["model"] = model_name
            pkg["max_new_tokens"] = max_new_tokens
        idx += cur_bs
        pbar.update(len(chunk))
    pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mitigation strategies over Compare/data/instruction and write results under Compare/output."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config (default: Compare/config.yml).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help="Override languages to process (e.g. Python JavaScript). Defaults to config.yml.",
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    data_cfg = cfg.get("data", {})
    instruction_dir = BASE_DIR / data_cfg.get("instruction_dir", "data/instruction")
    languages = args.languages or data_cfg.get("languages") or []
    if not languages:
        raise SystemExit("No languages specified in config.yml or via --languages.")

    model_cfg = cfg.get("model", {})
    model_name: str = model_cfg.get("name")
    if not model_name:
        raise SystemExit("model.name must be set in config.yml.")
    model_short: str = model_cfg.get("short_name", model_name.replace("/", "_"))

    output_cfg = cfg.get("output", {})
    output_root = BASE_DIR / output_cfg.get("root_dir", "output") / model_short

    strategies = [s for s in build_strategies(cfg) if s.enabled]
    if not strategies:
        raise SystemExit("No strategies enabled in config.yml.")

    # Shared HF loading hints
    common_hf_cfg: Dict[str, Any] = {}
    torch_dtype = model_cfg.get("torch_dtype")
    if torch_dtype in ("float16", "bfloat16"):
        common_hf_cfg["torch_dtype"] = torch_dtype
    device_map = model_cfg.get("device_map")
    if device_map is not None:
        common_hf_cfg["device_map"] = device_map

    # Instantiate generators once per run.
    baseline_gen = None
    greedy_gen = None
    self_refine_gen = None
    dola_gen = None
    nudging_gen = None
    cd_gen = None

    # Propagate global batch_size to strategies that don't override it.
    global_batch_size = int(data_cfg.get("batch_size", 1))

    for strategy in strategies:
        strategy.raw.setdefault("batch_size", global_batch_size)
        if strategy.name == "baseline" and baseline_gen is None:
            cfg_baseline = dict(strategy.raw)
            cfg_baseline.update(common_hf_cfg)
            baseline_gen = BaselineGenerator(
                model_name=model_name,
                config=cfg_baseline,
            )
        if strategy.name == "greedy" and greedy_gen is None:
            cfg_greedy = dict(strategy.raw)
            cfg_greedy.update(common_hf_cfg)
            greedy_gen = GreedyGenerator(
                model_name=model_name,
                config=cfg_greedy,
            )
        if strategy.name == "self_refine" and self_refine_gen is None:
            if baseline_gen is None:
                # SelfRefine wraps an inner generator; reuse baseline settings.
                cfg_inner = dict(cfg.get("strategies", {}).get("baseline", {}))
                cfg_inner.update(common_hf_cfg)
                inner = BaselineGenerator(model_name=model_name, config=cfg_inner)
            else:
                inner = baseline_gen
            self_refine_gen = SelfRefineGenerator(
                inner=inner,
                language="Python",  # individual languages passed per call via instruction text
                config=strategy.raw,
            )
        if strategy.name == "dola" and dola_gen is None:
            cfg_dola = dict(strategy.raw)
            cfg_dola.update(common_hf_cfg)
            dola_gen = DoLaGenerator(
                model_name=model_name,
                config=cfg_dola,
            )
        if strategy.name == "nudging" and nudging_gen is None:
            cfg_nudge = dict(strategy.raw)
            cfg_nudge.update(common_hf_cfg)
            nudging_model_name = cfg_nudge.get("nudging_model_name") or model_name
            nudging_gen = NudgingGenerator(
                base_model_name=model_name,
                nudging_model_name=nudging_model_name,
                config=cfg_nudge,
            )
        if strategy.name == "contrastive_decoding" and cd_gen is None:
            cfg_cd = dict(strategy.raw)
            cfg_cd.update(common_hf_cfg)
            expert_name = cfg_cd.get("expert_model_name") or model_name
            amateur_name = cfg_cd.get("amateur_model_name") or model_name
            cd_gen = ContrastiveDecodingGenerator(
                expert_model_name=expert_name,
                amateur_model_name=amateur_name,
                config=cfg_cd,
            )

    for language in languages:
        in_path = instruction_dir / f"packages_{language}_instruct.json"
        if not in_path.is_file():
            print(f"[{language}] Instruction file not found: {in_path}, skipping.")
            continue

        print(f"\n=== {language} ===")
        base_packages = load_instruction_packages(in_path)

        for strategy in strategies:
            # Deep copy packages per strategy so we don't overwrite fields across runs.
            import copy

            packages = copy.deepcopy(base_packages)

            if strategy.name == "baseline" and baseline_gen is not None:
                print(f"[{language}] Strategy: baseline")
                run_baseline(baseline_gen, packages, model_name, strategy.raw)
                out_dir = output_root / strategy.subdir
                out_path = out_dir / f"packages_{language}_instruct_vllm.json"
                save_instruction_packages(out_path, packages)
                print(f"[{language}] → {out_path}")

            elif strategy.name == "greedy" and greedy_gen is not None:
                print(f"[{language}] Strategy: greedy")
                run_greedy(greedy_gen, packages, model_name, strategy.raw)
                out_dir = output_root / strategy.subdir
                out_path = out_dir / f"packages_{language}_instruct_vllm.json"
                save_instruction_packages(out_path, packages)
                print(f"[{language}] → {out_path}")

            elif strategy.name == "self_refine" and self_refine_gen is not None:
                print(f"[{language}] Strategy: self_refine")
                run_self_refine(self_refine_gen, packages, model_name, strategy.raw)
                out_dir = output_root / strategy.subdir
                out_path = out_dir / f"packages_{language}_instruct_vllm_sr.json"
                save_instruction_packages(out_path, packages)
                print(f"[{language}] → {out_path}")

            elif strategy.name == "dola" and dola_gen is not None:
                print(f"[{language}] Strategy: dola")
                run_dola(dola_gen, packages, model_name, strategy.raw)
                out_dir = output_root / strategy.subdir
                out_path = out_dir / f"packages_{language}_instruct_vllm.json"
                save_instruction_packages(out_path, packages)
                print(f"[{language}] → {out_path}")

            elif strategy.name == "rag" and baseline_gen is not None:
                print(f"[{language}] Strategy: rag")
                rag_gen = RagGenerator(inner=baseline_gen, language=language, config=strategy.raw)
                run_rag(rag_gen, packages, model_name, strategy.raw)
                out_dir = output_root / strategy.subdir
                out_path = out_dir / f"packages_{language}_instruct_vllm.json"
                save_instruction_packages(out_path, packages)
                print(f"[{language}] → {out_path}")

            elif strategy.name == "nudging" and nudging_gen is not None:
                print(f"[{language}] Strategy: nudging")
                run_nudging(nudging_gen, packages, model_name, strategy.raw)
                out_dir = output_root / strategy.subdir
                out_path = out_dir / f"packages_{language}_instruct_vllm.json"
                save_instruction_packages(out_path, packages)
                print(f"[{language}] → {out_path}")

            elif strategy.name == "contrastive_decoding" and cd_gen is not None:
                print(f"[{language}] Strategy: contrastive_decoding")
                run_contrastive_decoding(cd_gen, packages, model_name, strategy.raw)
                out_dir = output_root / strategy.subdir
                out_path = out_dir / f"packages_{language}_instruct_vllm.json"
                save_instruction_packages(out_path, packages)
                print(f"[{language}] → {out_path}")


if __name__ == "__main__":
    main()

