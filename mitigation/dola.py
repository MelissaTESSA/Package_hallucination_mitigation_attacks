from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .decoding.dola import dola
from .interface import Generator, ChatMessage


@dataclass
class DoLaGenerator(Generator):
    """
    Generator that uses DoLa decoding on HuggingFace models.
    """

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    config: Dict[str, Any]

    def __init__(self, model_name: str, tokenizer_name: str | None = None, config: Dict[str, Any] | None = None):
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

        # Optional bitsandbytes quantization
        quant_cfg = self.config.get("quantization") or {}
        use_quant = bool(quant_cfg.get("load_in_4bit") or quant_cfg.get("load_in_8bit"))
        if use_quant:
            bnb_config = BitsAndBytesConfig(**quant_cfg)
            model_kwargs["quantization_config"] = bnb_config

        device_map = self.config.get("device_map")
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device_map is None:
            self.model.to(self.device)
        self.model.eval()

    def _encode(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        # Some tokenizers set an extremely large model_max_length which can
        # overflow internal truncation logic. Cap to a reasonable window.
        max_len_raw = getattr(self.tokenizer, "model_max_length", 4096) or 4096
        max_len = min(int(max_len_raw), 4096)
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(self.model.device)

    def _decode_from_ids(self, input_ids: torch.LongTensor, prompt_lens: List[int]) -> List[str]:
        decoded: List[str] = []
        for i in range(input_ids.size(0)):
            start = prompt_lens[i]
            seq = input_ids[i, start:]
            text = self.tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded.append(text.strip())
        return decoded

    def _run_dola(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []

        inputs = self._encode(prompts)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_lens = attention_mask.sum(dim=1).tolist()

        mature_layer = int(self.config.get("dola_mature_layer", self.model.config.num_hidden_layers))
        base_layer = self.config.get("dola_base_layer")
        early_exit_layers_str = self.config.get("dola_early_exit_layers", "")
        early_exit_layers = (
            [int(x) for x in early_exit_layers_str.split(",") if x.strip()]
            if early_exit_layers_str
            else None
        )

        generated = dola(
            self.model,
            self.tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(self.config.get("max_new_tokens", 128)),
            mature_layer=mature_layer,
            base_layer=base_layer,
            candidate_premature_layers=early_exit_layers,
            relative_top=float(self.config.get("dola_relative_top", 0.1)),
            repetition_penalty=float(self.config.get("dola_repetition_penalty", 1.2)),
            eos_token_id=self.tokenizer.eos_token_id,
            early_stop=bool(self.config.get("early_stop", True)),
        )
        return self._decode_from_ids(generated, prompt_lens)

    def generate(self, prompt: str) -> str:
        results = self._run_dola([prompt])
        return results[0] if results else ""

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return self._run_dola(prompts)

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        if not messages:
            return ""
        lines = [f"{m.role.value}: {m.content}" for m in messages if m.content]
        prompt = "\n".join(lines) + "\nassistant: "
        return self.generate(prompt)

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        prompts: List[str] = []
        for conv in messages:
            if not conv:
                prompts.append("")
                continue
            lines = [f"{m.role.value}: {m.content}" for m in conv if m.content]
            prompts.append("\n".join(lines) + "\nassistant: ")
        return self.batch_generate(prompts)