#!/usr/bin/env python3
"""Benchmark append-compact compression backends on real Claude Code turns.

Arms, all driven through the plugin's own ``compress_text`` pipeline so that
fenced code, masking, entity forcing and fallback gating are identical:

  baseline      deterministic line-level minification only (llm="off")
  llm           LLMLingua-2 at rate 0.4
  ollama:<m>    generative summarization at rate 0.4
  chain:<m>     LLMLingua-2 at 0.4, then the same Ollama model at 0.55

Every run must pass the known-positive / known-negative controls below or it
exits loudly instead of reporting numbers.

Read-only over transcripts. Full turn outputs are written to a private results
file under ~/.cache/append-compactor-bench; only aggregates are printed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BENCH_DIR = Path(os.environ.get("APPEND_COMPACT_BENCH_DIR", "~/.cache/append-compactor-bench")).expanduser()
SCRIPT = Path(__file__).resolve().parents[1] / "hooks/scripts/append-compact.py"
_spec = importlib.util.spec_from_file_location("append_compact_backend_bench", SCRIPT)
ac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ac)

os.environ.setdefault("APPEND_COMPACT_LOG", str(BENCH_DIR / "error.log"))
# Time model work, not the memo.
os.environ["APPEND_COMPACT_CACHE"] = "0"

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA_BASE.startswith("http"):
    OLLAMA_BASE = "http://" + OLLAMA_BASE

NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?:%|ms|s|MB|GB|KB)?(?![\w.])")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")

STOPWORDS = {
    "about", "after", "again", "against", "because", "before", "being", "below",
    "between", "could", "doing", "during", "every", "first", "found", "from",
    "further", "have", "having", "here", "into", "itself", "just", "more", "most",
    "other", "over", "same", "should", "some", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "under",
    "until", "very", "want", "were", "what", "when", "where", "which", "while",
    "will", "with", "would", "your", "yours", "also", "does", "make", "made",
    "need", "needs", "like", "only", "even", "much", "many", "each", "both",
    "still", "already", "without", "within", "using", "used", "uses", "work",
    "works", "working", "thing", "things", "case", "cases", "time", "times",
}


# ---------------------------------------------------------------------------
# Scoring and controls
# ---------------------------------------------------------------------------

def _numbers(text: str) -> set[str]:
    # LLMLingua sometimes detaches the decimal point ("7.592" -> "7. 592");
    # rejoin digit runs split only by spaces around a dot before matching.
    text = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)
    return set(NUM_RE.findall(text))


def score(original: str, output: str) -> dict[str, float | int]:
    ent_in = set(ac.extract_entities(original))
    ent_out = set(ac.extract_entities(output))
    nums_in = _numbers(original)
    nums_out = _numbers(output)
    words_in = {w.lower() for w in WORD_RE.findall(original)} - STOPWORDS
    words_out = {w.lower() for w in WORD_RE.findall(output)} - STOPWORDS
    return {
        "chars_in": len(original),
        "chars_out": len(output),
        "shrink": 1 - len(output) / len(original) if original else 0.0,
        "entities_in": len(ent_in),
        "entity_recall": len(ent_in & ent_out) / len(ent_in) if ent_in else 1.0,
        "numbers_in": len(nums_in),
        "number_recall": len(nums_in & nums_out) / len(nums_in) if nums_in else 1.0,
        "content_words_in": len(words_in),
        "word_recall": len(words_in & words_out) / len(words_in) if words_in else 1.0,
    }


CONTROL_TEXT = (
    "We decided to refactor src/pricing/engine.ts so computeTotal reads the "
    "prebuilt rates table instead of rebuilding it per request. The target is "
    "42 ms at p99 and the rollout must finish in 3 days without a schema change."
)


def run_controls() -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # known-positive: identity measures as no shrink and perfect retention
    identity = score(CONTROL_TEXT, CONTROL_TEXT)
    check(
        "known-positive: identity scores 0% shrink / 100% entity / 100% number",
        abs(identity["shrink"]) < 1e-9
        and identity["entity_recall"] == 1.0
        and identity["number_recall"] == 1.0
        and identity["word_recall"] == 1.0,
        f"shrink={identity['shrink']:.3f} ent={identity['entity_recall']:.3f} num={identity['number_recall']:.3f}",
    )

    # known-negative: a dropped path/symbol/number must be caught
    damaged = CONTROL_TEXT.replace("src/pricing/engine.ts", "the engine").replace(
        "computeTotal", "the helper"
    ).replace("42 ms", "fast")
    hurt = score(CONTROL_TEXT, damaged)
    check(
        "known-negative: dropped path/symbol/number scores below 1.0",
        hurt["entity_recall"] < 1.0 and hurt["number_recall"] < 1.0,
        f"ent={hurt['entity_recall']:.3f} num={hurt['number_recall']:.3f}",
    )

    # known-positive: deterministic arm still minifies a mixed turn
    code = (
        "```python\n\"\"\"doc\"\"\"\ndef total(values):\n    \"\"\"Sum.\"\"\"\n"
        "    result = 0\n    for value in values:\n        result += value\n    return result\n```\n\n"
        "This paragraph explains the pricing refactor in enough detail to pass the "
        "prose detector and describe the cached table approach that the team chose."
    )
    baseline, _ = ac.compress_text(code, llm="off")
    check(
        "known-positive: deterministic baseline shrinks a mixed turn >= 15%",
        1 - len(baseline) / len(code) >= 0.15,
        f"{1 - len(baseline) / len(code):.1%}",
    )

    # known-negative: active errors must pass through byte-identically
    err = "Traceback (most recent call last):\n  File \"src/app.ts\", line 12\n  the pool leaked handles on retry"
    preserved, mode = ac.compress_prose(err, llm="off")
    check("known-negative: active error untouched", preserved == err and mode == "error-verbatim", mode)

    return checks


# ---------------------------------------------------------------------------
# Ollama transport
# ---------------------------------------------------------------------------

def ollama_generate(
    model: str,
    prompt: str,
    *,
    think: bool | None,
    timeout: int = 300,
    num_predict: int | None = None,
) -> str:
    options = {
        "temperature": 0,
        "seed": 42,
        "num_ctx": 8192,
        "top_k": 1,
        "top_p": 1,
    }
    if num_predict is not None:
        options["num_predict"] = num_predict
    body: dict = {"model": model, "prompt": prompt, "stream": False, "options": options}
    if think is not None:
        body["think"] = think
    request = urllib.request.Request(
        OLLAMA_BASE + "/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = payload.get("response")
    if not isinstance(text, str):
        raise RuntimeError(f"{model}: malformed ollama response keys={sorted(payload)}")
    return text.strip()


def model_uses_think(model: str) -> bool | None:
    return False if re.match(r"(?i)qwen3", model) else None


SUMMARIZE_PROMPT = """Compress the note below into dense factual shorthand for an AI coding agent's working memory.

Rules:
- Output short bullet lines of facts, not prose or full sentences.
- Keep every decision and its reason, file/directory path, function/class/variable name, number, constraint, and open question.
- Copy every placeholder of the form <<M0>> exactly as written, once each, unchanged.
- No preamble, no explanation, no markdown headings, no code fences.
- At most about {target} characters in total.

NOTE:
{text}
"""


def make_ollama_backend(model: str):
    think = model_uses_think(model)

    def compress(masked: str, rate: float, force_tokens: list[str]) -> str | None:
        target = max(120, int(len(masked) * rate))
        prompt = SUMMARIZE_PROMPT.format(target=target, text=masked)
        num_predict = max(64, int(len(masked) * rate / 3))
        start = time.monotonic()
        try:
            out = ollama_generate(model, prompt, think=think, num_predict=num_predict)
        except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
            print(f"  ! {model} call failed: {exc!r}", file=sys.stderr)
            return None
        elapsed = time.monotonic() - start
        LAST_CALL_SECONDS.append(elapsed)
        if not out:
            return None
        # Models sometimes wrap the answer in a fence despite the instruction.
        if out.startswith("```") and out.endswith("```") and out.count("```") == 2:
            out = out.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
        if len(out) >= len(masked):
            return None
        return out

    return compress


LAST_CALL_SECONDS: list[float] = []


def make_chain_backend(model: str, llm_fn, second_rate: float = 0.55):
    ollama_fn = make_ollama_backend(model)

    def compress(masked: str, rate: float, force_tokens: list[str]) -> str | None:
        first = llm_fn(masked, rate, force_tokens)
        if not first:
            return None
        second = ollama_fn(first, second_rate, force_tokens)
        return second if second else first

    return compress


# ---------------------------------------------------------------------------
# Transcript sampling
# ---------------------------------------------------------------------------

def _block_text(content) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
    return "\n".join(parts)


def prose_share(text: str) -> float:
    eligible = 0
    for para in ac.PARA_SPLIT_RE.split(text):
        stripped = para.strip()
        if (
            len(stripped) >= ac.MIN_PROSE_CHARS
            and ac._looks_like_prose(stripped)
            and not ac._ERROR_RE.search(stripped)
        ):
            eligible += len(stripped)
    return eligible / max(len(text), 1)


def sample_paths(root: Path, limit: int, max_file_bytes: int) -> list[Path]:
    paths = sorted(
        p for p in root.rglob("*.jsonl")
        if p.is_file()
        and 50_000 <= p.stat().st_size <= max_file_bytes
        and "/subagents/" not in str(p)
    )
    if len(paths) <= limit:
        return paths
    step = len(paths) / limit
    return [paths[int(i * step)] for i in range(limit)]


def collect_turns(root: Path, sessions: int, want: int) -> list[dict]:
    """Take up to two prose-heavy, entity-bearing, error-free turns per session."""
    paths = sample_paths(root, sessions, 20 * 1024 * 1024)
    picked: list[dict] = []
    for path in paths:
        if len(picked) >= want:
            break
        per_file = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if per_file >= 2 or len(picked) >= want:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(record, dict) or record.get("isSidechain"):
                        continue
                    message = record.get("message") if isinstance(record.get("message"), dict) else {}
                    role = message.get("role") or record.get("type")
                    if role not in ("user", "assistant"):
                        continue
                    text = _block_text(message.get("content")).strip()
                    if not 500 <= len(text) <= 6_000:
                        continue
                    if ac._ERROR_RE.search(text) or "```" in text:
                        continue
                    if prose_share(text) < 0.75:
                        continue
                    if len(ac.extract_entities(text)) < 1 and not NUM_RE.search(text):
                        continue
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
                    if any(t["digest"] == digest for t in picked):
                        continue
                    picked.append({"session": path.stem[:24], "role": role, "digest": digest, "text": text})
                    per_file += 1
        except OSError:
            continue
    return picked


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_arm(name: str, llm, turns: list[dict]) -> dict:
    rows = []
    for turn in turns:
        start = time.monotonic()
        before = len(LAST_CALL_SECONDS)
        output, stats = ac.compress_text(turn["text"], llm=llm)
        elapsed = time.monotonic() - start
        calls = LAST_CALL_SECONDS[before:]
        s = score(turn["text"], output)
        rows.append({
            **s,
            "digest": turn["digest"],
            "seconds": round(elapsed, 3),
            "model_seconds": round(sum(calls), 3),
            "model_calls": len(calls),
            "llm_blocks": stats.get("llm_blocks", 0),
            "fallback_blocks": stats.get("fallback_blocks", 0),
            "error_blocks": stats.get("error_blocks", 0),
            "output": output,
        })
    return {"arm": name, "rows": rows}


def aggregate(result: dict) -> dict:
    rows = result["rows"]
    n = len(rows)
    if not n:
        return {"arm": result["arm"], "n": 0}
    total_in = sum(r["chars_in"] for r in rows)
    total_out = sum(r["chars_out"] for r in rows)
    return {
        "arm": result["arm"],
        "n": n,
        "chars_in": total_in,
        "chars_out": total_out,
        "shrink": 1 - total_out / total_in if total_in else 0.0,
        "entity_recall": statistics.mean(r["entity_recall"] for r in rows),
        "number_recall": statistics.mean(r["number_recall"] for r in rows),
        "word_recall": statistics.mean(r["word_recall"] for r in rows),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
        "p90_seconds": statistics.quantiles([r["seconds"] for r in rows], n=10)[-1] if n >= 10 else max(r["seconds"] for r in rows),
        "model_seconds": round(sum(r["model_seconds"] for r in rows), 2),
        "model_calls": sum(r["model_calls"] for r in rows),
        "fallback_blocks": sum(r["fallback_blocks"] for r in rows),
        "llm_blocks": sum(r["llm_blocks"] for r in rows),
    }


def fmt_row(a: dict) -> str:
    return (
        f"{a['arm']:<22} n={a['n']:<3} shrink={a['shrink'] * 100:5.1f}%  "
        f"ent={a['entity_recall'] * 100:5.1f}%  num={a['number_recall'] * 100:5.1f}%  "
        f"word={a['word_recall'] * 100:5.1f}%  median={a['median_seconds']:6.2f}s  "
        f"p90={a['p90_seconds']:6.2f}s  llm_blocks={a['llm_blocks']:<3} "
        f"fallback={a['fallback_blocks']:<3}"
    )


def check_determinism(model: str, turn: dict) -> bool | None:
    backend = make_ollama_backend(model)
    masked, placeholders, _ = ac._mask_entities(turn["text"])
    first = backend(masked, 0.4, placeholders)
    second = backend(masked, 0.4, placeholders)
    if first is None or second is None:
        return None
    return first == second


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.home() / ".claude/projects"))
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--models", default="qwen3.5:latest,llama3.2:3b,qwen2.5:7b")
    parser.add_argument("--out", default=str(BENCH_DIR / "backend-results.json"))
    parser.add_argument("--no-chain", action="store_true")
    args = parser.parse_args()

    controls = run_controls()
    for name, ok, detail in controls:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""), file=sys.stderr)
    if not all(ok for _, ok, _ in controls):
        print("CONTROLS FAILED — refusing to report benchmark numbers", file=sys.stderr)
        return 1

    turns = collect_turns(Path(args.root), args.sessions, args.turns)
    print(f"sampled turns: {len(turns)} from {len({t['session'] for t in turns})} sessions", file=sys.stderr)
    if len(turns) < min(args.turns, 8):
        print("too few turns sampled; refusing to report", file=sys.stderr)
        return 1
    for t in turns:
        print(
            f"  {t['session']:<24} {t['role']:<9} {len(t['text']):5d} chars "
            f"prose={prose_share(t['text']) * 100:4.0f}% ent={len(ac.extract_entities(t['text']))}",
            file=sys.stderr,
        )

    # --- connectivity controls per model -----------------------------------
    models: list[str] = []
    unavailable: list[str] = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            reply = ollama_generate(model, "Reply with exactly OK and nothing else.", think=model_uses_think(model), timeout=300)
        except Exception as exc:  # noqa: BLE001 - report and skip this model
            print(f"[SKIP] {model}: transport failed: {exc!r}", file=sys.stderr)
            unavailable.append(model)
            continue
        ok = reply.strip().strip(".!\"'`").strip().upper() == "OK"
        print(f"[{'PASS' if ok else 'FAIL'}] transport control: {model} replies OK ({reply!r})", file=sys.stderr)
        if not ok:
            unavailable.append(model)
            continue
        models.append(model)

    llm_fn = ac._default_llm_compress
    if ac._get_compressor() is None:
        print("LLMLingua unavailable — refusing to report the llm arm", file=sys.stderr)
        return 1

    results: list[dict] = []
    results.append(run_arm("baseline", "off", turns))
    results.append(run_arm("llm", llm_fn, turns))

    determinism: dict[str, bool | None] = {}
    for model in models:
        determinism[model] = check_determinism(model, turns[0])
        results.append(run_arm(f"ollama:{model}", make_ollama_backend(model), turns))
        if not args.no_chain:
            results.append(run_arm(f"chain:{model}", make_chain_backend(model, llm_fn), turns))

    aggregates = [aggregate(r) for r in results]
    print()
    print("backend benchmark — real Claude Code turns")
    print(f"turns: {len(turns)}  chars: {sum(len(t['text']) for t in turns)}")
    for agg in aggregates:
        print(fmt_row(agg))
    for model, det in determinism.items():
        print(f"determinism {model}: {'identical' if det else 'NOT identical' if det is False else 'unavailable'}")
    if unavailable:
        print(f"unavailable models: {', '.join(unavailable)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "turns": len(turns),
        "models": models,
        "unavailable": unavailable,
        "determinism": determinism,
        "aggregates": aggregates,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.chmod(out, 0o600)
    print(f"full per-turn outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
