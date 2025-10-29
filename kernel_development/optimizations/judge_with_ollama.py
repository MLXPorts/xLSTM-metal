
"""
Judge optimizer outputs with a local Ollama model (e.g., qwen3-coder:30b).

For each output file in an optimizer run's outputs directory, sends a
structured evaluation prompt to the Ollama HTTP API and records the model's
JSON rating. Results are stored as ratings_ollama.jsonl and ratings_ollama.csv.

Requirements:
- Ollama running locally (default: http://localhost:11434)
- Model pulled (e.g.,: `ollama pull qwen3-coder:30b`)

Usage:
  python scripts/judge_with_ollama.py \
    --model qwen3-coder:30b \
    --outputs runs/mps_opt/<run_dir>/outputs \
    --prompt-file /path/to/long_prompt.txt \
    --api http://localhost:11434 \
    --max-prompt-chars 20000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict

import requests


SYSTEM_INSTRUCTIONS = (
    "You are an expert evaluator for generated text from long-context RNNs. "
    "Given a prompt excerpt and a continuation, strictly return a compact JSON object with numeric ratings and a short rationale."
)

SCHEMA_HINT = {
    "coherence": "integer 1-10 (global consistency)",
    "relevance": "integer 1-10 (on-topic with prompt)",
    "fluency": "integer 1-10 (grammar/style)",
    "overall": "integer 1-10 (holistic score)",
    "rationale": "one sentence"
}


def build_prompt(prompt_text: str, cont_text: str) -> str:
    schema = json.dumps(SCHEMA_HINT, indent=2)
    return (
        f"SYSTEM:\n{SYSTEM_INSTRUCTIONS}\n\n"
        f"PROMPT_EXCERPT:\n{prompt_text}\n\n"
        f"CONTINUATION:\n{cont_text}\n\n"
        f"Output strictly JSON with fields:\n{schema}\n"
        f"No prose outside JSON."
    )


def call_ollama(api: str, model: str, prompt: str, stream: bool = False) -> str:
    url = f"{api.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": stream}
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def maybe_truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n...\n" + text[-tail:]


def parse_params_from_name(name: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for part in name.split("__"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[k] = v
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Optional JSON config to override CLI")
    ap.add_argument("--model", type=str, default="qwen3-coder:30b")
    ap.add_argument("--outputs", type=str, required=True, help="Directory with *.txt outputs")
    ap.add_argument("--prompt-file", type=str, required=True)
    ap.add_argument("--api", type=str, default="http://localhost:11434")
    ap.add_argument("--max-prompt-chars", type=int, default=20000)
    ap.add_argument("--max-cont-chars", type=int, default=8000)
    args = ap.parse_args()

    # Load JSON config (if provided) to override args
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
        for sect in ("runner", "optimizer"):
            if isinstance(cfg.get(sect), dict):
                for k, v in cfg[sect].items():
                    if hasattr(args, k):
                        setattr(args, k, v)

    out_dir = Path(args.outputs)
    files = sorted(out_dir.glob("*.txt"))
    assert files, f"No outputs found in {out_dir}"

    prompt_text_full = Path(args.prompt_file).read_text()
    prompt_text = maybe_truncate(prompt_text_full, args.max_prompt_chars)

    ratings_jsonl = out_dir / "ratings_ollama.jsonl"
    ratings_csv = out_dir / "ratings_ollama.csv"

    csvf = open(ratings_csv, "w", newline="")
    writer = csv.DictWriter(csvf, fieldnames=[
        "file", "backend", "heads_per_band", "chunk_size", "workers",
        "coherence", "relevance", "fluency", "overall", "rationale"
    ])
    writer.writeheader()

    with open(ratings_jsonl, "w") as jf:
        for fp in files:
            name = fp.stem
            params = parse_params_from_name(name)
            cont_text_full = fp.read_text()
            cont_text = maybe_truncate(cont_text_full, args.max_cont_chars)
            prompt = build_prompt(prompt_text, cont_text)
            try:
                resp = call_ollama(args.api, args.model, prompt, stream=False)
                text = resp.strip()
                start = text.find("{")
                end = text.rfind("}")
                if start >= 0 and end > start:
                    text = text[start:end+1]
                data = json.loads(text)
            except Exception as e:
                data = {"coherence": None, "relevance": None, "fluency": None, "overall": None, "rationale": f"parse_error: {e}"}

            rec = {
                "file": fp.name,
                "backend": params.get("b"),
                "heads_per_band": params.get("h"),
                "chunk_size": params.get("ck"),
                "workers": params.get("w"),
                "coherence": data.get("coherence"),
                "relevance": data.get("relevance"),
                "fluency": data.get("fluency"),
                "overall": data.get("overall"),
                "rationale": data.get("rationale"),
            }
            jf.write(json.dumps(rec) + "\n")
            writer.writerow(rec)
            csvf.flush()
            print(f"Judged {fp.name}: overall={rec['overall']} coherence={rec['coherence']}")

    csvf.close()


if __name__ == "__main__":
    main()
