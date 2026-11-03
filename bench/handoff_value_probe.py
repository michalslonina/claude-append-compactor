#!/usr/bin/env python3
"""Does the session handoff change what the model *does*, not just what it sees?

The other benchmark measures recall: is a fact present in the injected text.
That is necessary and not sufficient.  This probe asks a question whose answer
only exists in material the injection has dropped, and compares three arms:

  none  no context at all (a guess detector)
  off   the plugin's injection with APPEND_COMPACT_HANDOFF=0
  sham  off plus the handoff's character count of irrelevant text (a placebo:
        if length alone helps, the sham wins too)
  on    the same injection with the handoff enabled
  full  every turn verbatim (the model's ceiling for the question)

The fact under test is an **open ask**, not a value. Measured on the shipped
code, a number, a path or an error code planted in a demoted turn is still in
the packed view -- the entity ledger pins it and the L3 index line keeps it --
so the handoff only duplicates what packing kept. A whole-sentence user
instruction is the one thing the ledger cannot represent and the handoff can
recover.

It runs the *shipped* code path -- ``append_turn`` then ``injection_payload``
from hooks/scripts/append-compact.py -- not a reimplementation.

Controls (exit 3 if any fails, per the standing rule for scripts that report
numbers):

  known-positive, reachability   the full-transcript arm must answer both the
                                 target and the canary fact.  If the model
                                 cannot answer with everything in front of it,
                                 the probe is measuring nothing.
  known-negative, absence        in the ``off`` arm the target value must not
                                 appear anywhere in the injection.  A scenario
                                 where it does is not testing retrieval.
  known-positive, delivery       in the ``on`` arm the target value must appear
                                 in the injection.  A miss here is a handoff
                                 bug, reported separately from a model miss.
  known-positive, canary         the canary fact is planted in the last turn, so
                                 every arm must show it; an arm that misses it
                                 is broken, not merely unhelped.
  scorer self-test               a known answer must score 1 and a known
                                 non-answer 0, on every scenario's matcher.

Usage:
    ./.venv/bin/python bench/handoff_value_probe.py [--out results.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "hooks" / "scripts" / "append-compact.py"

OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.environ.get("HC_BENCH_MODEL", "qwen2.5:7b")
MAX_CHARS = int(os.environ.get("HC_BENCH_MAX_CHARS", "6400"))
DEPTH = int(os.environ.get("HC_BENCH_DEPTH", "45"))

SYSTEM = (
    "You answer questions about a software session from the context provided. "
    "Answer only with JSON."
)


# ---------------------------------------------------------------------------
# Scenarios.  Each plants one target fact (expected to be dropped by packing)
# and one canary fact in the final turn (expected to be visible everywhere).
# ---------------------------------------------------------------------------

# Every scenario targets an *open ask*, because that is the one section that
# survives in the handoff and nowhere else.  Measured on the shipped code: a
# numeric value, a file path or an error code planted in a demoted turn is
# still visible in the packed view -- the entity ledger pins "137 s",
# "quarryfast" and "ENOSPC" as bare facts, and the L3 index line carries the
# path -- so the handoff duplicates what the packing already kept.  A
# whole-sentence user instruction has no such pin: the ledger cannot represent
# it as a token, and it is absent from the packed view.
SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "csv_writer",
        "target": "We should drop the legacy csv writer before the migration.",
        "target_re": r"\bcsv writer\b",
        "target_q": "component the user asked to drop before the migration",
        "canary": "the exporter flush interval is 743 ms",
        "canary_re": r"\b743\b",
        "canary_q": "exporter flush interval in milliseconds",
        "filler": "The export pipeline was reviewed again with the platform team",
    },
    {
        "name": "flush_daemon",
        "target": "We should rename the flush daemon before the next release.",
        "target_re": r"\bflush daemon\b",
        "target_q": "component the user asked to rename before the next release",
        "canary": "the nightly batch page limit is 219 rows",
        "canary_re": r"\b219\b",
        "canary_q": "nightly batch page limit in rows",
        "filler": "The nightly batch was discussed at length with the data team",
    },
    {
        "name": "retry_cap",
        "target": "Please add a retry cap to the importer before the migration.",
        "target_re": r"\bretry cap\b",
        "target_q": "what the user asked to add to the importer before the migration",
        "canary": "the connection pool queue depth is 58",
        "canary_re": r"\b58\b",
        "canary_q": "connection pool queue depth",
        "filler": "The connection pool behaviour was re-examined under load",
    },
    {
        "name": "signing_keystore",
        "target": "We should rotate the signing keystore before the migration.",
        "target_re": r"\bsigning keystore\b",
        "target_q": "what the user asked to rotate before the migration",
        "canary": "the search index refresh interval is 21 s",
        "canary_re": r"\b21\b",
        "canary_q": "search index refresh interval in seconds",
        "filler": "The search index layout was reviewed for the migration",
    },
    {
        "name": "manifest_loader",
        "target": "We should remove the legacy manifest loader before the migration.",
        "target_re": r"\bmanifest loader\b",
        "target_q": "component the user asked to remove before the migration",
        "canary": "the upload chunk size is 9 MB",
        "canary_re": r"\b9\b",
        "canary_q": "upload chunk size in megabytes",
        "filler": "The upload path was profiled again during the review",
    },
    {
        "name": "batch_prefetcher",
        "target": "Do not enable the batch prefetcher until the migration lands.",
        "target_re": r"\bbatch prefetcher\b",
        "target_q": "component the user said not to enable until the migration lands",
        "canary": "the edge cache key prefix is v13",
        "canary_re": r"\bv13\b",
        "canary_q": "edge cache key prefix version",
        "filler": "The edge cache behaviour was reviewed with the network team",
    },
]


# ---------------------------------------------------------------------------
# Plugin harness
# ---------------------------------------------------------------------------

# Filler words carry no digits: a stray "Turn 12:" would put the number 12 in
# the injection and quietly satisfy (or corrupt) a numeric presence check.
_WORDS = ["amber", "basalt", "cobalt", "dusk", "ember", "flint", "granite",
          "harbor", "ivory", "juniper", "kelp", "lumen", "marble", "nickel",
          "onyx", "prism", "quartz", "ridge", "slate", "tundra"]


def phrase_re(text: str) -> str:
    """Regex for ``text`` that tolerates line wrapping and repeated spaces."""
    return r"\s+".join(re.escape(word) for word in text.split())


def load_compactor():
    spec = importlib.util.spec_from_file_location("append_compact_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_state(mod, scenario: dict[str, Any], plant_at: int) -> dict[str, Any]:
    """A long synthetic session with the target fact at ``plant_at``.

    Filler carries no numbers, so a correct answer can only come from the
    planted turns.
    """
    state = mod.new_state()
    for index in range(DEPTH):
        role = "assistant"
        if index == plant_at:
            # A user turn: the handoff only treats an ask as an ask when the
            # user said it, which is also how the plugin behaves in the wild.
            text = scenario["target"]
            role = "user"
        elif index == DEPTH - 1:
            text = (
                f"Latest settings: {scenario['canary']}. "
                "Everything else in the pipeline is unchanged today."
            )
        else:
            label = f"{_WORDS[index % len(_WORDS)]} {_WORDS[(index // len(_WORDS)) % len(_WORDS)]}"
            text = (
                f"Review pass {label}: {scenario['filler']} and the retry behaviour was "
                "left unchanged; no new values were agreed in that discussion. "
            ) * 3
        mod.append_turn(state, text, llm="off", role=role)
    return state


def injection(mod, state: dict[str, Any], enabled: bool) -> str:
    os.environ["APPEND_COMPACT_HANDOFF"] = "1" if enabled else "0"
    payload = mod.injection_payload(state, max_chars=MAX_CHARS)
    os.environ.pop("APPEND_COMPACT_HANDOFF", None)
    return payload["additional_context"] if payload else ""


def full_transcript(mod, scenario: dict[str, Any], plant_at: int) -> str:
    parts = []
    for index in range(DEPTH):
        if index == plant_at:
            parts.append(f"User: {scenario['target']}")
        elif index == DEPTH - 1:
            parts.append(f"Latest settings: {scenario['canary']}.")
        else:
            label = f"{_WORDS[index % len(_WORDS)]} {_WORDS[(index // len(_WORDS)) % len(_WORDS)]}"
            parts.append(f"Review pass {label}: {scenario['filler']}, nothing changed.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Model and scoring
# ---------------------------------------------------------------------------

def ask(context: str, scenario: dict[str, Any], which: str) -> dict[str, Any]:
    """One fact per call.

    Asking for both at once made a 7B answer only one of them or put the two
    values in swapped slots, which turned a retrieval failure into a scoring
    artefact.
    """
    question = scenario["target_q"] if which == "target" else scenario["canary_q"]
    user = f'Context:\n\n{context}\n\nQuestion: report the {question}. Answer as JSON: {{"value": ...}}.'
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "stream": False,
            "format": "json",
            "options": {"num_predict": 200, "num_ctx": 8192, "seed": 42},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    raw = (payload.get("message") or {}).get("content") or ""
    latency = time.perf_counter() - started
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = {}
    return {"raw": raw, "parsed": parsed, "latency_s": round(latency, 1)}


def scored(answer: Any, regex: str) -> bool:
    """A hit anywhere in the answer's value counts.

    Models wrap the number in an object (``{"retry_budget_seconds": 1200}``),
    so the whole value subtree is searched rather than just a scalar.
    """
    if answer is None:
        return False
    return re.search(regex, json.dumps(answer) if not isinstance(answer, str) else answer) is not None


def pick_plant_turn(mod, scenario: dict[str, Any]) -> tuple[int, dict[str, bool]]:
    """First offset where the target is genuinely dropped, and recovered.

    Both conditions are measured, never assumed: the scenario has to lose the
    fact under packing (known-negative) and the handoff has to restore it
    (known-positive delivery).
    """
    offsets = [DEPTH - 2, DEPTH - 3, DEPTH - 4, DEPTH - 5, DEPTH - 6, DEPTH - 8]
    last: dict[str, bool] = {}
    present = phrase_re(scenario["target"])
    for plant_at in offsets:
        state = build_state(mod, scenario, plant_at)
        off = injection(mod, state, False)
        on = injection(mod, state, True)
        # The value alone is the right test, not the sentence: the entity
        # ledger pins a bare "137 s" even when the sentence that carried it has
        # been demoted, and a model reading that number has all it needs.
        hit = re.search(present, off) is not None or re.search(scenario["target_re"], off) is not None
        recovered = (
            re.search(present, on) is not None or re.search(scenario["target_re"], on) is not None
        )
        last = {
            "plant_at": plant_at,  # type: ignore[dict-item]
            "absent_from_off": not hit,
            "present_in_on": recovered,
        }
        if not hit and recovered:
            return plant_at, last  # type: ignore[return-value]
    return -1, last  # type: ignore[return-value]


def scorer_selftest() -> bool:
    """A known answer scores 1, a known non-answer 0, for every scenario."""
    ok = True
    for scenario in SCENARIOS:
        if not scored(f"The value is {scenario['target']}.", scenario["target_re"]):
            print(f"[control FAIL] scorer missed a known answer for {scenario['name']}", file=sys.stderr)
            ok = False
        if not scored(f"Answer: {scenario['canary']}", scenario["canary_re"]):
            print(f"[control FAIL] scorer missed a known canary for {scenario['name']}", file=sys.stderr)
            ok = False
        if scored("unknown", scenario["target_re"]):
            print(f"[control FAIL] scorer matched a non-answer for {scenario['name']}", file=sys.stderr)
            ok = False
        if scored(None, scenario["target_re"]):
            print(f"[control FAIL] scorer matched None for {scenario['name']}", file=sys.stderr)
            ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/ac-template-experiment/bench/handoff_value_results.json"))
    parser.add_argument("--scenarios", type=int, default=len(SCENARIOS))
    args = parser.parse_args(argv)

    if not scorer_selftest():
        print("CONTROL FAILED: scorer", file=sys.stderr)
        return 3

    os.environ.setdefault("APPEND_COMPACT_LOG", tempfile.mktemp())
    os.environ.setdefault("APPEND_COMPACT_CACHE_DIR", tempfile.mkdtemp())
    os.environ.setdefault("APPEND_COMPACT_L3_DIR", tempfile.mkdtemp())
    os.environ.setdefault("APPEND_COMPACT_LLM", "0")
    os.environ.setdefault("APPEND_COMPACT_BACKGROUND", "0")
    mod = load_compactor()

    rows: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    scenarios = SCENARIOS[: args.scenarios]

    def checkpoint() -> None:
        """Save after every scenario: a killed run must not lose finished arms."""
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"partial": True, "controls": controls, "rows": rows}, indent=2),
            encoding="utf-8",
        )

    for scenario in scenarios:
        plant_at, shape = pick_plant_turn(mod, scenario)
        controls.append({"scenario": scenario["name"], "shape": shape})
        if plant_at < 0:
            # Decided from the injected text alone, before any model call, so
            # this exclusion is not cherry-picking an outcome: the fact is
            # already in the packed view and the cell cannot measure recovery.
            print(
                f"{scenario['name']:16s} EXCLUDED: the fact is already in the packed "
                f"view; the handoff has nothing to add here ({shape})",
                flush=True,
            )
            continue
        state = build_state(mod, scenario, plant_at)
        off_text = injection(mod, state, False)
        on_text = injection(mod, state, True)
        # Same token budget as the handoff, none of its content: if more
        # context alone helps, the sham wins too and the handoff earns nothing.
        pad = (scenario["filler"] + ". ") * 20
        arms = {
            "none": "",
            "off": off_text,
            "sham": off_text + "\n\n" + pad[: max(0, len(on_text) - len(off_text))],
            "on": on_text,
            "full": full_transcript(mod, scenario, plant_at),
        }
        if re.search(scenario["target_re"], arms["off"]):
            print(f"[control FAIL] {scenario['name']}: target visible in the off arm", file=sys.stderr)
            return 3
        delivery = re.search(scenario["target_re"], arms["on"]) is not None
        controls.append({"scenario": scenario["name"], "delivery": delivery})
        if not delivery:
            print(f"[control FAIL] {scenario['name']}: handoff did not deliver the target", file=sys.stderr)
            return 3
        for arm, context in arms.items():
            target = ask(context, scenario, "target")
            canary = ask(context, scenario, "canary")
            rows.append(
                {
                    "scenario": scenario["name"],
                    "arm": arm,
                    "plant_at": plant_at,
                    "context_chars": len(context),
                    "target_ok": scored(target["parsed"], scenario["target_re"]),
                    "canary_ok": scored(canary["parsed"], scenario["canary_re"]),
                    "target_raw": target["raw"][:300],
                    "canary_raw": canary["raw"][:300],
                    "latency_s": round(target["latency_s"] + canary["latency_s"], 1),
                }
            )
            print(
                f"{scenario['name']:16s} {arm:5s} target={rows[-1]['target_ok']!s:5s} "
                f"canary={rows[-1]['canary_ok']!s:5s} chars={len(context):5d}",
                flush=True,
            )
        checkpoint()

    # Controls that need the model: reachability and canary.
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    failures: list[str] = []
    for row in by_arm.get("full", []):
        if not (row["target_ok"] and row["canary_ok"]):
            failures.append(f"ceiling miss: {row['scenario']} (full transcript)")
    # known-negative at the model level: with no context at all, any hit is a
    # guess, and a guessed hit would fake a win for whichever arm it lands in.
    for row in by_arm.get("none", []):
        if row["target_ok"] or row["canary_ok"]:
            failures.append(f"guess with no context: {row['scenario']}")
    # The off arm is the paired known-negative: the fact is provably not in its
    # context, so a hit there is a guess and would erase the effect this probe
    # is trying to measure.
    for arm in ("off", "sham"):
        for row in by_arm.get(arm, []):
            if row["target_ok"]:
                failures.append(f"target produced without it in context ({arm}): {row['scenario']}")
    for arm, arm_rows in by_arm.items():
        if arm == "none":
            continue
        missed = [row["scenario"] for row in arm_rows if not row["canary_ok"]]
        if missed:
            failures.append(f"canary miss in arm {arm}: {', '.join(missed)}")

    summary: dict[str, Any] = {"model": MODEL, "depth": DEPTH, "max_chars": MAX_CHARS}
    for arm, arm_rows in by_arm.items():
        summary[arm] = {
            "n": len(arm_rows),
            "target_wins": sum(row["target_ok"] for row in arm_rows),
            "target_rate": sum(row["target_ok"] for row in arm_rows) / len(arm_rows),
            "canary_rate": sum(row["canary_ok"] for row in arm_rows) / len(arm_rows),
            "mean_context_chars": sum(row["context_chars"] for row in arm_rows) / len(arm_rows),
        }
    result = {
        "summary": summary,
        "controls": controls,
        "failures": failures,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2))
    if failures:
        print("\n".join(f"[control FAIL] {item}" for item in failures), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
