"""
Two independent tasks:
  1. Collect top packages per language from Libraries.io  (collect_packages)
  2. Generate coding instructions per package via DeepSeek  (generate_instructions)

Run from the data/ directory:
  python package_collection.py collect   # fetch packages → package_description/
  python package_collection.py instruct  # add instructions → package_description/*_instruct.json
  python package_collection.py all       # both in order
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import List, Optional

import requests
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration – read secrets from environment, never hard-code
# ---------------------------------------------------------------------------
LIBRARIES_IO_KEY: str = os.environ.get("LIBRARIES_IO_KEY", "1d51844239c782ec6c7494dffa67b21f")
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "sk-c80a05a86cdf40b59f6ff56bf9f9644b")
LIBRARIES_IO_BASE = "https://libraries.io/api"

PACKAGES_PROVIDERS = {
    "Python":     "Pypi",
    "JavaScript": "NPM",
    "Java":       "Maven",
    "CSharp":     "NuGet",
    "Ruby":       "Rubygems",   # API expects lowercase 'g'
    "Go":         "Go",
    "PHP":        "Packagist",
    "Swift":      "SwiftPM",
    "R":          "CRAN",
    "Rust":       "Cargo",
}

MAX_PAGES    = 350    # 50 pages × 100 per page = 5 000 packages max
PER_PAGE     = 100   # Libraries.io max is 100
SLEEP_OK     = 0.5   # seconds between successful calls
SLEEP_ERR    = 5    # seconds after a failed call
MAX_RETRIES  = 3
MAX_WORKERS  = 16    # concurrent DeepSeek threads

OUTPUT_DIR   = "package_description"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Package:
    name: str
    description: str


@dataclass
class PackageProvider:
    name: str
    language: str
    packages: List[Package]


# ---------------------------------------------------------------------------
# Libraries.io helpers
# ---------------------------------------------------------------------------
def get_top_packages(platform: str, page: int = 1, per_page: int = PER_PAGE) -> Optional[list]:
    """Fetch one page of top-ranked packages for *platform* from Libraries.io."""
    if not LIBRARIES_IO_KEY:
        raise EnvironmentError("LIBRARIES_IO_KEY environment variable is not set.")
    url = (
        f"{LIBRARIES_IO_BASE}/search"
        f"?platforms={platform}&sort=rank"
        f"&api_key={LIBRARIES_IO_KEY}&page={page}&per_page={per_page}"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            print(f"  Error on attempt {attempt}: {e}")
            time.sleep(SLEEP_ERR)
    return None


def parse_packages(raw: list) -> List[Package]:
    return [
        Package(name=p.get("name", ""), description=p.get("description") or "")
        for p in raw
        if p.get("name")
    ]


def collect_packages(languages: Optional[List[str]] = None) -> None:
    """Fetch top packages for each language and save to OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    targets = languages or list(PACKAGES_PROVIDERS.keys())

    for language in targets:
        platform = PACKAGES_PROVIDERS[language]
        out_file = f"{OUTPUT_DIR}/packages_{language}.json"
        all_packages: List[Package] = []

        print(f"\n[{language}] Fetching from platform '{platform}' …")
        for page in tqdm(range(1, MAX_PAGES + 1), desc=language):
            raw = get_top_packages(platform, page, PER_PAGE)
            if raw is None:
                print(f"  Page {page} failed after retries, skipping.")
            elif not raw:
                break   # no more results
            else:
                all_packages.extend(parse_packages(raw))
            time.sleep(SLEEP_OK)

        # Deduplicate by name, preserving order
        seen: set = set()
        unique: List[Package] = []
        for pkg in all_packages:
            if pkg.name not in seen:
                seen.add(pkg.name)
                unique.append(pkg)

        provider = PackageProvider(name=language, language=platform, packages=unique)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(asdict(provider), f, ensure_ascii=False, indent=2)
        print(f"[{language}] Saved {len(unique)} unique packages → {out_file}")


# ---------------------------------------------------------------------------
# Instruction generation (DeepSeek)
# ---------------------------------------------------------------------------
def _make_client() -> OpenAI:
    if not DEEPSEEK_API_KEY:
        raise EnvironmentError("DEEPSEEK_API_KEY environment variable is not set.")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def _system_prompt(language: str) -> str:
    return (
        f"You are a coding assistant that assists users in creating simple prompts "
        f"that will be used to generate {language} code. No code should be used in the response."
    )


def _user_prompt(language: str, description: str) -> str:
    return (
        f"Your answer must begin with 'Generate {language} code that' and must not be longer than one sentence. "
        f"Do not include extra text or formatting (i.e. do not start with 'Sure! Here's a prompt...'). "
        f"Write a prompt that would generate {language} code to accomplish the same tasks as the following "
        f"package description: {description}."
    )


def generate_instruction(client: OpenAI, language: str, description: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": _system_prompt(language)},
            {"role": "user",   "content": _user_prompt(language, description)},
        ],
    )
    return response.choices[0].message.content


def generate_instructions(languages: Optional[List[str]] = None) -> None:
    """Read collected packages, add an 'instruction' field, and save *_instruct.json."""
    client = _make_client()
    if languages is None:
        languages = [
            f.replace("packages_", "").replace(".json", "")
            for f in os.listdir(OUTPUT_DIR)
            if f.startswith("packages_") and not f.endswith("_instruct.json") and f.endswith(".json")
        ]

    for language in languages:
        in_path  = f"{OUTPUT_DIR}/packages_{language}.json"
        out_path = f"{OUTPUT_DIR}/packages_{language}_instruct.json"

        if not os.path.exists(in_path):
            print(f"[{language}] Source file not found: {in_path}, skipping.")
            continue

        with open(in_path, "r", encoding="utf-8") as f:
            packages = json.load(f)["packages"]

        print(f"\n[{language}] Generating instructions for {len(packages)} packages …")

        # Capture language in the default arg to avoid closure-over-loop-var issues
        def _task(desc: str, lang: str = language) -> str:
            return generate_instruction(client, lang, desc)

        instructions: List[Optional[str]] = [None] * len(packages)
        errors = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {
                executor.submit(_task, pkg["description"]): i
                for i, pkg in enumerate(packages)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(packages)):
                idx = future_to_idx[future]
                try:
                    instructions[idx] = future.result()
                except Exception as exc:
                    instructions[idx] = ""
                    errors += 1
                    print(f"  Error on sample {idx}: {exc}")

        for pkg, instr in zip(packages, instructions):
            pkg["instruction"] = instr or ""

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"packages": packages}, f, ensure_ascii=False, indent=2)

        print(f"[{language}] Saved {len(packages)} packages ({errors} errors) → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    extra_langs = sys.argv[2:] or None   # optional: python package_collection.py collect Python Java

    if cmd in ("collect", "all"):
        collect_packages(extra_langs)
    if cmd in ("instruct", "all"):
        generate_instructions(extra_langs)
    if cmd not in ("collect", "instruct", "all"):
        print(f"Unknown command '{cmd}'. Use: collect | instruct | all")
        sys.exit(1)
