"""
Shared package-name extraction from model answers.
Copied verbatim from run_adversarial_phr.py so it can be reused by
run_adversarial_metrics.py without triggering that script's module-level
PHR report (which runs on import).
"""

import re


def normalize_text(text: str) -> str:
    """Normalize model-specific artifacts before extraction.
    - Strips <think>...</think> blocks (qwen3_5_9b)
    - Replaces BPE tokenizer artifacts with proper whitespace (deepseek_coder_6_7b)
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("Ġ", " ").replace("Ċ", "\n")
    return text


def extract_assistant_part(answer: str) -> str:
    """Keep only the last assistant turn (strips conversation template)."""
    answer = normalize_text(answer)
    parts = re.split(r"\bassistant\s*:\s*", answer, flags=re.IGNORECASE)
    return parts[-1].strip() if len(parts) > 1 else answer.strip()


def is_garbled(text: str) -> bool:
    """Return True if text appears corrupted (high non-ASCII ratio)."""
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text) > 0.25


# Tokens to always discard (tool names, stdlib noise, self_refine placeholders)
NOISE_TOKENS: set[str] = {
    # Python
    "pip", "pip3", "python", "python3", "install", "conda",
    # JavaScript
    "npm", "yarn", "node", "npx",
    # Ruby
    "gem", "bundler", "bundle",
    # Rust
    "cargo", "rustup",
    # Rust Cargo.toml standard metadata keys
    "name", "version", "edition", "authors", "description", "license",
    # Code keywords
    "import", "require", "from", "use",
    # Shell
    "bash", "sh", "cd", "sudo", "apt", "brew",
    # Generic noise
    "fs", "os", "sys", "io", "code", "cmd",
}

# Generic filenames that are never package names (language-independent)
_GENERIC_FILENAME_RE = re.compile(
    r"^(index|main|app|server|config|utils|helper|example|test|spec)\.(js|ts|py|rb|rs)$",
    re.IGNORECASE,
)
# Non-JS file extensions (never valid npm/pip/gem/cargo package names)
_NON_JS_EXT_RE = re.compile(r"\.(py|rb|rs|json|txt|md|toml|lock|sh)$", re.IGNORECASE)


def _clean(pkg: str, language: str = "Python") -> str | None:
    """Strip version specifiers; return None if token should be discarded."""
    pkg = re.split(r"[=<>!\s\[]", pkg)[0].strip()
    if not pkg:
        return None
    if pkg.lower() in NOISE_TOKENS:
        return None
    # discard obvious generic filenames (index.js, app.py, etc.)
    if _GENERIC_FILENAME_RE.match(pkg):
        return None
    # for non-JS languages, also discard anything ending in .js/.ts
    # (highlight.js / chart.js are valid npm packages but irrelevant in Python/Ruby/Rust)
    if language != "JavaScript" and re.search(r"\.(js|ts)$", pkg, re.IGNORECASE):
        return None
    if _NON_JS_EXT_RE.search(pkg):
        return None
    # discard self_refine placeholders like package1, package2
    if re.match(r"^package\d*$", pkg, re.IGNORECASE):
        return None
    # discard bare numbers or single characters
    if re.match(r"^[\d]+$", pkg) or len(pkg) == 1:
        return None
    return pkg


def extract_packages(answer: str, language: str = "Python") -> list[str]:
    response = extract_assistant_part(answer)
    if is_garbled(response):
        return []

    packages: set[str] = set()

    def add(pkg_raw: str) -> None:
        pkg = _clean(pkg_raw, language)
        if pkg:
            packages.add(pkg)

    if language == "Python":
        # pip/pip3 install [flags] <pkg>
        for m in re.finditer(
            r"pip3?\s+install\s+(?:[\-][\w\-]*\s+)*([\w][\w\-\.]*)",
            response, re.IGNORECASE
        ):
            add(m.group(1))

        # import <pkg> / from <pkg> import
        for m in re.finditer(r"(?:^|[\n;])\s*(?:import|from)\s+([\w]+)", response, re.MULTILINE):
            add(m.group(1))

    elif language == "JavaScript":
        # npm install / yarn add <pkg>  (handles scoped @org/pkg)
        for m in re.finditer(
            r"(?:npm\s+install|yarn\s+add)\s+(?:[\-][\w\-]*\s+)*(@?[\w][\w\-\./@]*)",
            response, re.IGNORECASE
        ):
            add(m.group(1))

        # require('pkg') / require("pkg")
        for m in re.finditer(r'require\s*\(\s*[\'\"]([\w][\w\-\./@]*)[\'\"]\s*\)', response):
            add(m.group(1))

        # import ... from 'pkg' / import 'pkg'
        for m in re.finditer(r'from\s+[\'\"]([\w][\w\-\./@]*)[\'\"]\s*;?', response):
            add(m.group(1))

    elif language == "Ruby":
        # gem install <pkg>
        for m in re.finditer(r"gem\s+install\s+([\w][\w\-\.]*)", response, re.IGNORECASE):
            add(m.group(1))
        # gem 'pkg' or gem "pkg" (Gemfile format) — single quote char only
        for m in re.finditer(r"gem\s+['\"]([^'\"]+)['\"]", response):
            add(m.group(1))

        # require 'pkg' or require "pkg"
        for m in re.finditer(r"require\s+['\"]([^'\"]+)['\"]", response):
            add(m.group(1))

        # bundle add <pkg>
        for m in re.finditer(r"bundle\s+add\s+([\w][\w\-\.]*)", response, re.IGNORECASE):
            add(m.group(1))

    elif language == "Rust":
        # Cargo.toml: prefer extracting only inside [dependencies] block
        dep_block = re.search(
            r"\[(?:dev-)?dependencies\](.*?)(?=^\[|\Z)", response,
            re.DOTALL | re.MULTILINE
        )
        toml_source = dep_block.group(1) if dep_block else response
        for m in re.finditer(r'^([\w][\w\-]*)\s*=\s*["\{]', toml_source, re.MULTILINE):
            add(m.group(1))

        # cargo add <pkg>
        for m in re.finditer(r"cargo\s+add\s+([\w][\w\-]*)", response, re.IGNORECASE):
            add(m.group(1))

        # use pkg:: / extern crate pkg
        for m in re.finditer(r"extern\s+crate\s+([\w][\w\-]*)", response):
            add(m.group(1))
        # use crate_name:: (stop at :: to avoid sub-module paths)
        for m in re.finditer(r"\buse\s+([\w][\w\-]*)(?:::|;)", response):
            add(m.group(1))

    # Shared patterns (all languages)

    # `package-name` backtick mentions
    for m in re.finditer(r"`([\w][\w\-\.]+)`", response):
        add(m.group(1))

    # [pkg1, pkg2, pkg3] bracket list (self_refine)
    for m in re.finditer(r"\[([^\]\n]{3,200})\]", response):
        for tok in m.group(1).split(","):
            tok = tok.strip().strip("'\"")
            if tok and re.match(r"^[\w][\w\-\.]*$", tok):
                add(tok)

    # comma-separated bare list (rag: "pkg1, pkg2, pkg3")
    # also handles mistral RAG numbered format: "1. pkg1, 2. pkg2, 3. pkg3"
    for line in response.splitlines():
        line = line.strip()
        if not line or len(line) > 300:
            continue
        # strip "1. ", "2. " numbered prefixes before splitting
        normalized = re.sub(r"\b\d+\.\s+", "", line)
        tokens = [t.strip().strip("'\".,") for t in normalized.split(",")]
        clean = [t for t in tokens if t and re.match(r"^[\w][\w\-\.]*$", t)]
        if len(clean) >= 2 and len(clean) == len([t for t in tokens if t]):
            for tok in clean:
                add(tok)

    # plain prose mention: "the X package"
    for m in re.finditer(r"\bthe\s+([\w][\w\-\.]+)\s+(?:package|gem|crate|module)\b",
                         response, re.IGNORECASE):
        add(m.group(1))

    return list(packages)
