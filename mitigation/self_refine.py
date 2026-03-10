from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import re
import time
import jinja2

from .interface import Generator, ChatMessage, ChatRole


def _extract_bracket_list(text: str) -> List[str]:
    """
    Extract a comma-separated list from the first [...] block in `text`.
    Falls back to splitting the whole string if no brackets are found.
    """
    match = re.search(r"\[([^\[\]]+)\]", text)
    raw = match.group(1) if match else text
    return [p.strip() for p in raw.split(",") if p.strip()]


def _extract_validities(text: str) -> List[str]:
    """
    Extract Yes/No labels from a validation response.
    Normalises to literal 'Yes' or 'No'.
    """
    items = _extract_bracket_list(text)
    return ["Yes" if "yes" in v.lower() else "No" for v in items]


@dataclass
class SelfRefineGenerator(Generator):
    """
    Wrapper that runs an underlying chat-capable `Generator` with an
    iterative self-refinement loop specialised for package recommendation.

    The wrapped generator is expected to implement `chat_generation`.
    This class only orchestrates prompting and refinement; it does not
    perform decoding itself.
    """

    inner: Generator
    language: str
    config: Dict[str, Any]

    def __init__(self, inner: Generator, language: str, config: Dict[str, Any] | None = None):
        # We keep the abstract base attributes for compatibility but treat the
        # wrapped generator as the true backend.
        self.inner = inner
        self.language = language
        self.config = dict(config or {})
        self.model = getattr(inner, "model", None)
        self.tokenizer = getattr(inner, "tokenizer", None)

    # ------------------------------------------------------------------ #
    # Template rendering helpers
    # ------------------------------------------------------------------ #

    def _render_template(self, template_name: str, **kwargs: Any) -> str:
        """
        Render a .jinja template from Compare/prompts. Falls back to str.format
        if jinja2 is not available.
        """
        prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        template_path = prompts_dir / template_name
        text = template_path.read_text(encoding="utf-8")
        try:
            env = jinja2.Environment(autoescape=False)
            template = env.from_string(text)
            return template.render(**kwargs)
        except Exception:
            return text.format(**kwargs)

    # ------------------------------------------------------------------ #
    # Prompt builders
    # ------------------------------------------------------------------ #

    def _system_prompt_generation(self) -> str:
        return self._render_template(
            "system_prompt_package_generation.jinja",
            language=self.language,
        )

    def _build_generation_messages(self, instruction: str) -> List[ChatMessage]:
        return [
            ChatMessage(role=ChatRole.SYSTEM, content=self._system_prompt_generation()),
            ChatMessage(role=ChatRole.USER, content=instruction),
        ]

    def _build_validation_messages(self, package_text: str) -> List[ChatMessage]:
        system = self._render_template("system_prompt_package_validation.jinja")
        user = self._render_template(
            "user_prompt_package_validation.jinja",
            language=self.language,
            package_text=package_text,
        )
        return [
            ChatMessage(role=ChatRole.SYSTEM, content=system),
            ChatMessage(role=ChatRole.USER, content=user),
        ]

    # ------------------------------------------------------------------ #
    # Core self-refinement loop
    # ------------------------------------------------------------------ #

    def _self_refine(self, instruction: str) -> Tuple[str, float, int]:
        """
        Run self-refinement for a single instruction.
        Returns (final_package_text, elapsed_seconds, refinement_rounds_used).
        """
        max_rounds: int = int(self.config.get("max_rounds", 3))
        t0 = time.perf_counter()

        # Initial generation
        messages = self._build_generation_messages(instruction)
        package_text = self.inner.chat_generation(messages)

        rounds_used = 0
        for _ in range(max_rounds):
            packages_list = _extract_bracket_list(package_text)
            if not packages_list:
                break

            # Independent validation call to keep the generation context clean
            val_messages = self._build_validation_messages(package_text)
            validation_text = self.inner.chat_generation(val_messages)
            validities = _extract_validities(validation_text)

            invalid = [pkg for pkg, v in zip(packages_list, validities) if v == "No"]
            if not invalid:
                break

            rounds_used += 1
            # Append assistant answer and feedback to the original generation conversation
            messages.append(ChatMessage(role=ChatRole.ASSISTANT, content=package_text))
            feedback = (
                f"The following packages do not exist or are not valid {self.language} packages "
                f"and must not appear in your answer: {invalid}. "
                f"Please provide a corrected list."
            )
            messages.append(ChatMessage(role=ChatRole.USER, content=feedback))
            package_text = self.inner.chat_generation(messages)

        elapsed = time.perf_counter() - t0
        return package_text, elapsed, rounds_used

    # ------------------------------------------------------------------ #
    # Generator interface
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str) -> str:
        package_text, _, _ = self._self_refine(prompt)
        return package_text

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]

    def chat_generation(self, messages: List[ChatMessage]) -> str:
        """
        Interpret the last user message content as the task instruction and
        perform self-refinement on it.
        """
        if not messages:
            return ""
        # Find last user message as the instruction
        instruction = ""
        for msg in reversed(messages):
            if msg.role == ChatRole.USER and msg.content:
                instruction = msg.content
                break
        if not instruction:
            return ""
        package_text, _, _ = self._self_refine(instruction)
        return package_text

    def batch_chat_generation(self, messages: List[List[ChatMessage]]) -> List[str]:
        return [self.chat_generation(conv) for conv in messages]

