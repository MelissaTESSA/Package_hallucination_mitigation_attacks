"""
Evaluate PHR (Package Hallucination Rate) on adversarial output files.
Extracts package names from model answers, then validates against the PyPI list.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluation.phr import calculate_phr

_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = _ROOT / "output/adversarial"
LOG_DIR = _ROOT / "output/logs"


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.__stdout__

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = LOG_DIR / f"adversarial_phr_{_timestamp}.log"
sys.stdout = _Tee(_log_path)
print(f"Log: {_log_path}\nRun started: {datetime.now()}\n")
STRATEGIES = ["baseline", "greedy", "self_refine", "dola", "rag",
              "nudging", "contrastive_decoding"]

# Optional model filter (e.g. "mistral_7b").
# Pass as first CLI arg: python run_adversarial_phr.py mistral_7b
# If omitted, all model subdirectories under OUTPUT_DIR are evaluated.
MODEL = sys.argv[1] if len(sys.argv) > 1 else None


def get_models() -> list[str]:
    """Return models to evaluate: CLI arg if given, else all model subdirectories."""
    if MODEL:
        return [MODEL]
    return sorted(d.name for d in OUTPUT_DIR.iterdir() if d.is_dir())


def resolve_path(model: str, strategy: str, language: str) -> Path:
    """Return the JSON path for a given model/strategy/language."""
    return OUTPUT_DIR / model / language / f"{strategy}.json"


def get_languages(model: str) -> list[str]:
    """Return all language subdirectories for a given model."""
    model_dir = OUTPUT_DIR / model
    return sorted(d.name for d in model_dir.iterdir() if d.is_dir())


def normalize_text(text: str) -> str:
    """Normalize model-specific artifacts before extraction.
    - Strips <think>...</think> blocks (qwen3_5_9b)
    - Replaces BPE tokenizer artifacts with proper whitespace (deepseek_coder_6_7b)
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("\u0120", " ").replace("\u010a", "\n")
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


def evaluate_file(path: Path, language: str) -> dict:
    with open(path) as f:
        items = json.load(f)

    print(f"\n{'═'*70}")
    print(f"  {path.stem}")
    print(f"{'═'*70}")

    all_packages: list[str] = []
    for i, item in enumerate(items, 1):
        answer = item.get("answer", "")
        pkgs = extract_packages(answer, language)
        all_packages.extend(pkgs)

        print(f"\n[{i}] {item.get('instruction', '')}")
        print(f"  ANSWER   : {extract_assistant_part(answer)[:200].strip()!r}")
        print(f"  EXTRACTED: {pkgs}")

    strategy = path.stem.replace(f"{language}_", "").replace(f"{language}", "")
    if not all_packages:
        return {"strategy": strategy, "phr_score": None, "total_generated": 0, "total_valid": 0, "hallucinations": []}

    phr = calculate_phr(language, all_packages)
    hallucinations = sorted(set(
        pkg for pkg, valid in zip(phr.response, phr.validation) if not valid
    ))
    return {
        "strategy": strategy,
        "phr_score": phr.phr_score,
        "total_generated": phr.total_generated_packages,
        "total_valid": phr.total_valid_packages,
        "hallucinations": hallucinations,
    }


for model in get_models():
    print(f"\n{'█'*70}")
    print(f"  MODEL: {model}")
    print(f"{'█'*70}")

    for language in get_languages(model):
        results = []
        for strategy in STRATEGIES:
            path = resolve_path(model, strategy, language)
            if not path.exists():
                print(f"[skip] {path} not found")
                continue
            r = evaluate_file(path, language)
            results.append(r)

        if not results:
            continue

        results.sort(key=lambda x: x.get("phr_score") if x.get("phr_score") is not None else 2, reverse=False)
        best = results[0]["strategy"] if results else None

        print(f"\n{'═'*52}")
        print(f"  {model} / {language}")
        print(f"{'═'*52}")
        print(f"{'Strategy':<25} {'PHR':>7}  {'Hallu':>6} / {'Total':>6}")
        print("─" * 52)
        for r in results:
            if r["phr_score"] is None:
                print(f"  {r['strategy']:<23} {'N/A':>7}  (garbled output)")
            else:
                flag = " ← best" if r["strategy"] == best else ""
                n_hallu = r['total_generated'] - r['total_valid']
                print(f"  {r['strategy']:<23} {r['phr_score']:>7.3f}  {n_hallu:>6} / {r['total_generated']:>6}{flag}")

        print(f"\n  Hallucinations per strategy:")
        print("─" * 52)
        for r in results:
            hallus = r.get("hallucinations", [])
            if not hallus:
                print(f"  {r['strategy']:<23}  (none)")
            else:
                # display up to 10 per line, then wrap
                chunks = [hallus[i:i+10] for i in range(0, len(hallus), 10)]
                prefix = f"  {r['strategy']:<23}  "
                indent = " " * len(prefix)
                for j, chunk in enumerate(chunks):
                    line_prefix = prefix if j == 0 else indent
                    print(f"{line_prefix}{', '.join(chunk)}")

print("\nPHR = fraction of extracted packages NOT in registry  |  Lower → fewer hallucinations")
