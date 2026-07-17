"""
ActLCD mitigation strategy — Active Layer-Contrastive Decoding.
Zhang et al., EMNLP 2025.  https://arxiv.org/abs/2505.23657

The BCQAgent class is a lightweight re-implementation of the offline
reinforcement learning policy from the paper. It is trained separately and
loaded at inference time through the ``actlcd_policy_path`` config key.

When ``actlcd_policy_path`` is null (or the file is absent) ActLCD degrades
gracefully to standard DoLa — the contrastive adjustment is applied at
every token step, exactly as in the original DoLa paper.

Config keys follow this repo's ``dola_*`` naming convention (see
mitigation/dola.py):
    actlcd_mature_layer          (int)    Final transformer layer index.
    actlcd_early_exit_layers     (str)    Comma-separated candidate premature
                                           layer indices, e.g. "4,8,12,16".
    actlcd_relative_top          (float)  Relative-top filter threshold.
    actlcd_repetition_penalty    (float)  Repetition penalty.
    actlcd_policy_type           (str)    "bcq" (default) or "entropy".
    actlcd_entropy_threshold     (float)  Entropy threshold (policy_type=entropy).
    actlcd_policy_path           (str)    Path to a .pth BCQAgent checkpoint, or null.
    actlcd_bc_threshold          (float)  BC probability threshold for BCQ.
    max_new_tokens                (int)   Maximum generated tokens (default 128).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .decoding.actlcd import actlcd, TOP_K_STATE
from .interface import Generator, ChatMessage


# ---------------------------------------------------------------------------
# Entropy policy  (heuristic, no training required)
# ---------------------------------------------------------------------------

class EntropyPolicy:
    """
    Simple entropy-threshold gating policy for ActLCD.

    At each token step the Shannon entropy of the mature layer's probability
    distribution is computed.  If the entropy exceeds ``threshold`` (model is
    uncertain) the DoLa contrastive adjustment is applied; otherwise the mature
    layer logits are used directly (model is already confident).

    Threshold selection (in nats): ~1.0 is a heuristic midpoint. Calibrate on
    a held-out set by logging per-token entropies and setting the threshold
    around the 75th-80th percentile.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def select_action(
        self,
        state: np.ndarray,
        mature_logits: Optional[torch.Tensor] = None,
    ) -> int:
        if mature_logits is not None:
            probs   = torch.softmax(mature_logits.float(), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
        else:
            top5_probs = state[-(TOP_K_STATE * 2)::2]
            top5_probs = np.clip(top5_probs, 1e-10, 1.0)
            entropy    = float(-np.sum(top5_probs * np.log(top5_probs)))
        return 1 if entropy > self.threshold else 0


# ---------------------------------------------------------------------------
# BCQ policy
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCQAgent:
    """
    Batch Constrained Q-learning agent used by ActLCD.

    State dimension:  (num_premature_layers + 1) x TOP_K_STATE x 2
    Action space:      {0=skip contrast, 1=apply contrast}
    """

    ACTION_DIM = 2

    def __init__(
        self,
        state_dim: int,
        hidden: List[int] = (1024, 512, 256),
        bc_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.state_dim    = state_dim
        self.hidden       = list(hidden)
        self.bc_threshold = bc_threshold
        self.device       = torch.device(device)

        self.q_net  = _MLP(state_dim, self.ACTION_DIM, self.hidden).to(self.device)
        self.bc_net = _MLP(state_dim, self.ACTION_DIM, self.hidden).to(self.device)

    def select_action(
        self,
        state: np.ndarray,
        mature_logits: Optional[torch.Tensor] = None,  # unused; kept for API symmetry
    ) -> int:
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            bc_probs = F.softmax(self.bc_net(t), dim=1).squeeze(0)
            q_vals   = self.q_net(t).squeeze(0)

        allowed = (bc_probs > self.bc_threshold).nonzero(as_tuple=True)[0]
        if len(allowed) == 0:
            return int(bc_probs.argmax().item())
        return int(allowed[q_vals[allowed].argmax()].item())

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dim":    self.state_dim,
                "hidden":       self.hidden,
                "bc_threshold": self.bc_threshold,
                "q_net":        self.q_net.state_dict(),
                "bc_net":       self.bc_net.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "BCQAgent":
        ckpt  = torch.load(path, map_location=device)
        agent = cls(
            state_dim    = ckpt["state_dim"],
            hidden       = ckpt["hidden"],
            bc_threshold = ckpt.get("bc_threshold", 0.3),
            device       = device,
        )
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.bc_net.load_state_dict(ckpt["bc_net"])
        agent.q_net.eval()
        agent.bc_net.eval()
        return agent

    @staticmethod
    def state_dim_for(num_premature_layers: int) -> int:
        return (num_premature_layers + 1) * TOP_K_STATE * 2


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@dataclass
class ActLCDGenerator(Generator):
    """ActLCD mitigation strategy — wraps the token-by-token ActLCD generation loop."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: Dict[str, Any]
    _policy: Optional[Any]
    _mature_layer: int
    _premature_layers: List[int]

    def __init__(
        self,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.config = dict(config or {})

        tokenizer_id = tokenizer_name or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        elif self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
        if self.tokenizer.eos_token_id is None and self.tokenizer.pad_token_id is not None:
            self.tokenizer.eos_token_id = self.tokenizer.pad_token_id

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        torch_dtype = self.config.get("torch_dtype")
        if torch_dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif torch_dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        quant_cfg = self.config.get("quantization") or {}
        if quant_cfg.get("load_in_4bit") or quant_cfg.get("load_in_8bit"):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_cfg)

        device_map = self.config.get("device_map")
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

        # Layer configuration (dola_*-style: comma-separated string for early_exit_layers).
        # Falls back to text_config.num_hidden_layers for multimodal configs (e.g. Qwen3.5).
        cfg = self.model.config
        default_layers = (
            getattr(cfg, "num_hidden_layers", None)
            or getattr(getattr(cfg, "text_config", None), "num_hidden_layers", 32)
        )
        self._mature_layer = int(self.config.get("actlcd_mature_layer", default_layers))
        early_exit_layers_str = self.config.get("actlcd_early_exit_layers", "")
        self._premature_layers = (
            [int(x) for x in early_exit_layers_str.split(",") if x.strip()]
            if early_exit_layers_str
            else []
        )

        # Policy (optional)
        self._policy = self._load_policy()

    # ------------------------------------------------------------------ #
    # Policy loading
    # ------------------------------------------------------------------ #

    def _load_policy(self) -> Optional[object]:
        policy_type = self.config.get("actlcd_policy_type", "bcq").lower()

        if policy_type == "entropy":
            threshold = float(self.config.get("actlcd_entropy_threshold", 1.0))
            print(f"[ActLCD] Using entropy policy (threshold={threshold} nats).")
            return EntropyPolicy(threshold=threshold)

        # Default: BCQ (trained RL policy)
        policy_path = self.config.get("actlcd_policy_path") or None
        if not policy_path:
            return None
        path = Path(policy_path)
        if not path.is_file():
            print(
                f"[ActLCD] Warning: actlcd_policy_path '{policy_path}' not found. "
                "Running without BCQ policy (= standard DoLa)."
            )
            return None
        device_str = str(self.device)
        agent = BCQAgent.load(path, device=device_str)
        expected_dim = BCQAgent.state_dim_for(len(self._premature_layers))
        if agent.state_dim != expected_dim:
            raise ValueError(
                f"[ActLCD] Loaded policy state_dim={agent.state_dim} does not match "
                f"expected {expected_dim} for {len(self._premature_layers)} premature layers. "
                "Re-train the policy with the correct layer configuration."
            )
        print(f"[ActLCD] Loaded BCQ policy from {path} (state_dim={agent.state_dim}).")
        return agent

    # ------------------------------------------------------------------ #
    # Core generation
    # ------------------------------------------------------------------ #

    def _run_actlcd(self, prompt: str) -> str:
        max_len_raw = getattr(self.tokenizer, "model_max_length", 4096) or 4096
        max_len = min(int(max_len_raw), 4096)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(self.model.device)

        input_ids      = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_len     = int(attention_mask.sum().item())

        output_ids = actlcd(
            model                       = self.model,
            tokenizer                   = self.tokenizer,
            input_ids                   = input_ids,
            attention_mask              = attention_mask,
            max_new_tokens              = int(self.config.get("max_new_tokens", 128)),
            mature_layer                = self._mature_layer,
            candidate_premature_layers  = self._premature_layers,
            relative_top                = float(self.config.get("actlcd_relative_top", 0.1)),
            repetition_penalty          = float(self.config.get("actlcd_repetition_penalty", 1.2)),
            eos_token_id                = self.tokenizer.eos_token_id,
            early_stop                  = bool(self.config.get("early_stop", True)),
            policy                      = self._policy,
        )

        gen_ids = output_ids[0, prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True).strip()

    # ------------------------------------------------------------------ #
    # Generator interface
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str) -> str:
        return self._run_actlcd(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        # ActLCD processes one sequence at a time (BCQ state is per-token, per-sequence).
        # Each step recomputes hidden states for the full growing sequence (no KV-cache,
        # since candidate premature layers need fresh logits every token). Clearing the
        # cache between sequences avoids allocator fragmentation across ~1000 calls.
        results = []
        for p in prompts:
            results.append(self._run_actlcd(p))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        lines = [f"{m.role.value}: {m.content}" for m in messages if m.content]
        return self.generate("\n".join(lines) + "\nassistant: ")

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        return [self.chat_generation(conv) for conv in messages]
