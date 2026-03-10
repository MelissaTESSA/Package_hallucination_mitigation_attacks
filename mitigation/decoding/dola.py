import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList
from typing import Dict, List


def relative_top_filter(
    scores: torch.FloatTensor,
    relative_top: float = 0.1,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filters logits below a relative threshold based on the top token probability.
    Used in DoLa to mask out unlikely tokens before contrasting layer distributions.
    """
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, _ = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh).unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized


def dola(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    max_new_tokens: int = 128,
    mature_layer: int = None,
    base_layer: int = None,
    candidate_premature_layers: List[int] = None,
    relative_top: float = 0.1,
    repetition_penalty: float = 1.2,
    eos_token_id: int = None,
    early_stop: bool = True,
    stopping_criteria: StoppingCriteriaList = None,
) -> torch.LongTensor:
    """
    Decoding by Contrasting Layers (DoLa) — Chuang et al. 2023.
    https://arxiv.org/abs/2309.03883

    Args:
        model:                      A HuggingFace CausalLM (must support output_hidden_states).
        tokenizer:                  Corresponding tokenizer.
        input_ids:                  Prompt token ids, shape (batch, seq_len).
        attention_mask:             Attention mask, shape (batch, seq_len).
        max_new_tokens:             Maximum number of tokens to generate.
        mature_layer:               Index of the final (mature) layer to use as the signal.
                                    Typically set to model.config.num_hidden_layers.
                                    NOTE: hidden_states[0] = embedding output,
                                          hidden_states[i] = output of transformer block i.
        base_layer:                 If set, contrast mature_layer against this single fixed
                                    layer instead of dynamically selecting via JS divergence.
        candidate_premature_layers: List of early layer indices to contrast against.
                                    Ignored when base_layer is set.
                                    Example for a 6-layer model: [1, 2, 3, 4, 5]
        relative_top:               Filters tokens below this fraction of the top probability
                                    before contrasting. Set to 0.0 to disable.
        repetition_penalty:         Repetition penalty applied to logits (>1 discourages repeats).
        eos_token_id:               Override the tokenizer's EOS token id.
        early_stop:                 If True, allow generation to stop at EOS.
                                    If False, EOS is suppressed (generates exactly max_new_tokens).
        stopping_criteria:          Optional custom StoppingCriteriaList.

    Returns:
        Full token sequence including the original prompt, shape (batch, seq_len + new_tokens).
    """
    if mature_layer is None:
        raise ValueError(
            "mature_layer must be specified. "
            "Set it to model.config.num_hidden_layers for the final layer."
        )

    candidate_premature_layers = candidate_premature_layers or []
    early_exit_layers = candidate_premature_layers + [mature_layer]

    batch_size = input_ids.shape[0]
    device = input_ids.device

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_tensor = (
        torch.tensor([eos_token_id], device=device) if eos_token_id is not None else None
    )
    unfinished = input_ids.new_ones(batch_size)  # 1 = still generating, 0 = done

    processors = LogitsProcessorList()
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    stopping_criteria = stopping_criteria or StoppingCriteriaList()

    # Track which premature layer was chosen each step (for analysis)
    premature_layer_dist: Dict[int, int] = {l: 0 for l in candidate_premature_layers}

    for step in range(max_new_tokens):
        # ── Position IDs ──────────────────────────────────────────────────────
        # Compute explicitly from attention_mask to avoid RoPE size mismatches.
        # input_ids and attention_mask are always in sync at this point.
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        position_ids = position_ids.masked_fill(attention_mask == 0, 1)

        # ── Forward pass ──────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
                output_hidden_states=True,
            )

        # ── Extract logits from each layer of interest ────────────────────────
        # hidden_states is a tuple of length (num_layers + 1):
        #   index 0   → embedding output
        #   index i   → output of transformer block i
        layer_logits: Dict[int, torch.Tensor] = {
            layer: model.lm_head(outputs.hidden_states[layer])
            for layer in early_exit_layers
        }

        # ── Contrastive logit computation ─────────────────────────────────────
        if base_layer is not None:
            # Static contrast: mature vs a fixed base layer
            final_logits = layer_logits[mature_layer][:, -1, :]
            base_logits  = layer_logits[base_layer][:, -1, :]

            if relative_top > 0.0:
                final_logits = relative_top_filter(final_logits, relative_top)
                base_logits  = base_logits.log_softmax(dim=-1)
                base_logits[final_logits < -1e3] = -1e3

            next_token_logits = final_logits - base_logits

        elif len(candidate_premature_layers) == 0:
            # No contrasting — plain greedy from mature layer
            next_token_logits = layer_logits[mature_layer][:, -1, :]

        else:
            # Dynamic contrast: choose premature layer with highest JS divergence
            # from the mature layer (i.e., the layer that differs most → most "uncertain")
            mature_last  = layer_logits[mature_layer][:, -1, :]       # (B, V)
            stacked_pre  = torch.stack(
                [layer_logits[l][:, -1, :] for l in candidate_premature_layers], dim=0
            )                                                           # (C, B, V)

            sm_mature = F.softmax(mature_last, dim=-1)                 # (B, V)
            sm_pre    = F.softmax(stacked_pre, dim=-1)                 # (C, B, V)

            # Jensen-Shannon divergence between mature and each premature layer
            M    = 0.5 * (sm_mature[None] + sm_pre)                   # (C, B, V)
            kl1  = F.kl_div(F.log_softmax(mature_last, dim=-1)[None], M, reduction="none").mean(-1)
            kl2  = F.kl_div(F.log_softmax(stacked_pre, dim=-1),       M, reduction="none").mean(-1)
            js   = (0.5 * (kl1 + kl2)).mean(-1)                       # (C,)

            best_pre_idx   = int(js.argmax().cpu().item())
            best_pre_layer = candidate_premature_layers[best_pre_idx]
            premature_layer_dist[best_pre_layer] += 1

            final_logits = mature_last
            base_logits  = layer_logits[best_pre_layer][:, -1, :]

            if relative_top > 0.0:
                final_logits = relative_top_filter(final_logits, relative_top)
                base_logits  = base_logits.log_softmax(dim=-1)
                base_logits[final_logits < -1e3] = -1e3

            next_token_logits = final_logits - base_logits

        # ── Post-processing ───────────────────────────────────────────────────
        next_token_logits = next_token_logits.to(device)
        scores = processors(input_ids, next_token_logits)

        if not early_stop and eos_tensor is not None:
            scores[:, eos_token_id] = -float("inf")

        next_tokens = torch.argmax(scores, dim=-1)

        # Pad finished sequences instead of appending real tokens
        next_tokens = next_tokens * unfinished + tokenizer.pad_token_id * (1 - unfinished)

        # ── Append token and extend mask (keep them in sync) ─────────────────
        input_ids      = torch.cat([input_ids,      next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)

        # ── Check stopping conditions ─────────────────────────────────────────
        if eos_tensor is not None:
            unfinished = unfinished.mul(
                next_tokens.tile(eos_tensor.shape[0], 1)
                .ne(eos_tensor.unsqueeze(1))
                .prod(dim=0)
            )
        unfinished = unfinished & ~stopping_criteria(input_ids, None)
        if unfinished.max() == 0:
            break

    return input_ids