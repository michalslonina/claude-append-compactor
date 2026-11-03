"""Tests for hooks/scripts/append-compact.py.

The hook file has a hyphen in its name, so it is loaded with importlib instead
of a normal import.  Every test runs against an isolated log path and tmp state;
nothing touches the real ``~/.claude`` state file.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "hooks" / "scripts" / "append-compact.py"

_spec = importlib.util.spec_from_file_location("append_compact", SCRIPT_PATH)
compactor = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(compactor)


LONG_PROSE = (
    "We should carefully inspect src/app.ts before changing buildThing, because the "
    "initialization sequence is fragile and the current implementation repeats several "
    "expensive checks on every request in the hot path.\n\n"
    "The team agreed to document each step so future readers can reproduce the result."
)

ERROR_PROSE = (
    "Traceback (most recent call last):\n"
    '  File "src/app.ts", line 12, in buildThing\n'
    "    initialize the connection pool before the first request and keep every handle "
    "closed on the failure path so nothing leaks and the process can recover cleanly."
)


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Keep logs and compressor globals test-local."""
    monkeypatch.setenv("APPEND_COMPACT_LOG", str(tmp_path / "error.log"))
    monkeypatch.setenv("APPEND_COMPACT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("APPEND_COMPACT_JOB_LOG", str(tmp_path / "jobs.log"))
    monkeypatch.setenv("APPEND_COMPACT_BACKGROUND", "0")
    monkeypatch.delenv("APPEND_COMPACT_WORKER", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_WORKERS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_DRAIN_WAIT", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_LLM", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_CACHE", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_DEDUPE", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_DEDUPE_WINDOW", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_CAPTURE_DEDUPE_SECONDS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_FACT_LEDGER", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_SENTENCE_PROTECT", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_RATE_CONTROL", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_RAW", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_RAW_DIR", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_RAW_MAX_FILES", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_CODE_SKELETON", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_DIR", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_TARGET_CHARS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_INDEX_BUDGET_CHARS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_KEEP_RECENT_MASTERS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_KEEP_RECENT_CHUNKS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_DRIFT", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_DRIFT_WINDOW", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_DRIFT_MIN_TOKENS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_DRIFT_THRESHOLD", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_DRIFT_KEEP_RECENT_MASTERS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_DRIFT_KEEP_RECENT_CHUNKS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_MAX_FILES", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_COMPRESS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L3_ZSTD_LEVEL", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_L2_DIGEST_CHARS", raising=False)
    for name in (
        "APPEND_COMPACT_GOALS",
        "APPEND_COMPACT_GOALS_EXTRACT",
        "APPEND_COMPACT_GOALS_L1",
        "APPEND_COMPACT_GOALS_L2",
        "APPEND_COMPACT_GOALS_L3",
        "APPEND_COMPACT_GOALS_PLACEMENT",
        "APPEND_COMPACT_GOALS_MAX_CHARS",
        "APPEND_COMPACT_GOALS_HYBRID_CHARS",
        "APPEND_COMPACT_GOALS_MAX_ITEMS",
        "APPEND_COMPACT_GOALS_L1_CHARS",
        "APPEND_COMPACT_GOALS_L2_CHARS",
        "APPEND_COMPACT_GOALS_L3_CHARS",
        "APPEND_COMPACT_GOALS_SHARED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENROUTER_EXTRA_HEADERS", raising=False)
    monkeypatch.setattr(compactor, "_compressor", None)
    monkeypatch.setattr(compactor, "_compressor_failed", False)
    yield


def _fresh_state() -> dict:
    return compactor.new_state()


def _append(state: dict, text: str, source: str = "user_prompt", session: str | None = None) -> dict:
    chunk = compactor.append_turn(state, text, source=source, llm="off", session=session)
    assert chunk is not None
    return chunk


# ---------------------------------------------------------------------------
# Python minification
# ---------------------------------------------------------------------------

def test_python_minifier_shrinks_removes_docstrings_and_runs():
    source = (
        '"""Module docstring that should disappear."""\n'
        "# a leading comment\n"
        "import os\n\n\n"
        "def add(a, b):\n"
        '    """Add two numbers; this docstring should disappear."""\n'
        "    # inline comment\n"
        "    return a + b\n"
    )
    mini = compactor.minify_python_source(source)
    assert mini is not None
    assert len(mini) < len(source)
    assert "docstring" not in mini
    assert "#" not in mini

    namespace: dict = {}
    exec(compile(mini, "<minified>", "exec"), namespace)
    assert namespace["add"](2, 3) == 5


def test_python_minifier_leaves_invalid_source_alone():
    assert compactor.minify_python_source("def broken(:\n") is None


# ---------------------------------------------------------------------------
# Line-level deletion
# ---------------------------------------------------------------------------

def test_line_level_deletes_debug_and_comment_only_lines():
    js = "// comment\nconsole.log('debug')\nconst keep = 1;\n"
    stripped = compactor.strip_line_level(js, "javascript")
    assert "console.log" not in stripped
    assert "// comment" not in stripped
    assert "const keep = 1;" in stripped


def test_line_level_never_empties_a_block_body():
    js = 'if (flag) {\n  print("sole child");\n}\n'
    stripped = compactor.strip_line_level(js, "javascript")
    assert 'print("sole child");' in stripped

    python = "def f():\n    # only comment before the statement\n    print('sole child')\n"
    stripped_py = compactor.strip_line_level(python, "python")
    assert "print('sole child')" in stripped_py


def test_line_level_keeps_eslint_directive():
    js = "// eslint-disable-next-line no-console\nconsole.log('debug')\nconst keep = 1;\n"
    stripped = compactor.strip_line_level(js, "javascript")
    assert "eslint-disable-next-line" in stripped
    assert "console.log" not in stripped
    assert "const keep = 1;" in stripped


def test_line_level_never_edits_inside_multiline_strings():
    js = 'const text = `\nconsole.log("inside a template literal")\n`;\nconst keep = 1;\n'
    stripped = compactor.strip_line_level(js, "javascript")
    assert 'console.log("inside a template literal")' in stripped
    assert "const keep = 1;" in stripped

    py = 'note = """\nprint("inside a python string")\n"""\nvalue = 1\n'
    stripped_py = compactor.strip_line_level(py, "python")
    assert 'print("inside a python string")' in stripped_py
    assert "value = 1" in stripped_py


def test_line_level_skips_continuations():
    js = "const value = foo(\n  // explanatory comment\n  bar\n);\n"
    stripped = compactor.strip_line_level(js, "javascript")
    assert "bar" in stripped


# ---------------------------------------------------------------------------
# LLMLingua-2 lazy import, fallback and masking
# ---------------------------------------------------------------------------

def test_llm_off_falls_back_to_deterministic_normalization():
    text, mode = compactor.compress_prose(LONG_PROSE, llm="off")
    assert mode == "fallback"
    assert text == compactor._normalize_prose(LONG_PROSE)


def test_env_flag_disables_llm(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_LLM", "0")
    _, mode = compactor.compress_prose(LONG_PROSE, llm="auto")
    assert mode == "fallback"


def test_import_failure_falls_back(monkeypatch):
    fake = types.ModuleType("llmlingua")

    class PromptCompressor:  # noqa: D401 - test double
        def __init__(self, **kwargs):
            raise RuntimeError("model unavailable in test")

    fake.PromptCompressor = PromptCompressor
    monkeypatch.setitem(sys.modules, "llmlingua", fake)
    monkeypatch.delenv("APPEND_COMPACT_LLM", raising=False)

    text, mode = compactor.compress_prose(LONG_PROSE, llm="auto")
    assert mode == "fallback"
    assert text == compactor._normalize_prose(LONG_PROSE)


def test_callable_that_raises_falls_back():
    def boom(masked, rate, force_tokens):
        raise RuntimeError("compressor blew up")

    text, mode = compactor.compress_prose(LONG_PROSE, llm=boom)
    assert mode == "fallback"
    assert text == compactor._normalize_prose(LONG_PROSE)


def test_callable_that_drops_masks_falls_back():
    def drops(masked, rate, force_tokens):
        return "short summary with no placeholders"

    text, mode = compactor.compress_prose(LONG_PROSE, llm=drops)
    assert mode == "fallback"
    assert text == compactor._normalize_prose(LONG_PROSE)


def test_working_callable_preserves_and_restores_entities():
    seen = {}

    def fake(masked, rate, force_tokens):
        seen["rate"] = rate
        seen["force_tokens"] = list(force_tokens)
        for placeholder in force_tokens:
            assert placeholder in masked
        return re.sub(r"\b(carefully|several|expensive|future|fragile) ", "", masked)

    text, mode = compactor.compress_prose(LONG_PROSE, llm=fake)
    assert mode == "llmlingua"
    assert seen["rate"] == 0.4
    assert seen["force_tokens"]
    assert "src/app.ts" in text
    assert "buildThing" in text
    assert len(text) < len(compactor._normalize_prose(LONG_PROSE))
    assert set(compactor.extract_entities(text)) == set(compactor.extract_entities(LONG_PROSE))


def test_prose_compression_keeps_decimals_and_percentages():
    text = (
        "We should carefully inspect src/app.ts before changing buildThing, because "
        "the p99 latency is 7.592s at 99% of requests and the current implementation "
        "repeats several expensive checks on every request in the hot path.\n\n"
        "The team agreed to document each step so future readers can reproduce the result."
    )

    def fake(masked, rate, force_tokens):
        assert "7.592" not in masked and "99%" not in masked
        return re.sub(r"\b(carefully|several|expensive) ", "", masked)

    output, mode = compactor.compress_prose(text, llm=fake)
    assert mode == "llmlingua"
    assert "7.592s" in output and "99%" in output
    assert len(output) < len(compactor._normalize_prose(text))


def test_numeric_literals_are_masked_and_restored_exactly():
    text = (
        "The batch takes 7.592s at 99% and uses 42ms on 3 nodes; "
        "version 2.15.0 fits in 512MB."
    )
    masked, placeholders, mapping = compactor._mask_entities(text)
    for literal in ("7.592s", "99%", "42ms", "2.15.0", "512MB"):
        assert literal not in masked
    # Bare integers stay with the compressor; masking every digit is not worth
    # the compression it costs.
    assert "3 nodes" in masked
    assert compactor._restore_entities(masked, mapping) == text
    restored_literals = {mapping[placeholder] for placeholder in placeholders}
    assert {"7.592s", "99%", "42ms", "2.15.0", "512MB"} <= restored_literals


def test_error_paragraphs_are_never_compressed():
    calls = []

    def fake(masked, rate, force_tokens):
        calls.append(masked)
        return "compressed"

    text, mode = compactor.compress_prose(ERROR_PROSE, llm=fake)
    assert mode == "error-verbatim"
    assert text == ERROR_PROSE
    assert calls == []


# ---------------------------------------------------------------------------
# Entity ledger
# ---------------------------------------------------------------------------

def test_entity_extraction_exact_and_deterministic():
    text = (
        "Please update src/app.ts and ./lib/util.py.\n"
        "export function buildThing() {}\n"
        "export { helper as renamed };\n"
        "def handler():\n"
        "    return 1\n"
        "pub fn run_task() {}\n"
    )
    expected = [
        "path: ./lib/util.py",
        "path: src/app.ts",
        "symbol: buildThing",
        "symbol: handler",
        "symbol: renamed",
        "symbol: run_task",
    ]
    assert compactor.extract_entities(text) == expected
    assert compactor.extract_entities(text) == expected  # stable across calls


def test_entity_extraction_plain_prose_and_windows_path():
    assert compactor.extract_entities("hello world") == []
    entities = compactor.extract_entities(r"C:\Users\dev\project\main.go")
    assert "path: C:\\Users\\dev\\project\\main.go" in entities


# ---------------------------------------------------------------------------
# JSON minification
# ---------------------------------------------------------------------------

def test_json_minification():
    assert compactor.minify_json_text('{"a": 1, "b": [2, 3]}') == '{"a":1,"b":[2,3]}'
    assert compactor.minify_json_text("not json") is None
    assert compactor.minify_json_text('"scalar"') is None


def test_fenced_and_standalone_json_are_compacted():
    fenced = '```json\n{"a": 1, "b": [2, 3]}\n```'
    assert compactor.compress_text(fenced, llm="off")[0] == '```json\n{"a":1,"b":[2,3]}\n```'

    standalone = '{"a": 1, "b": [2, 3]}'
    assert compactor.compress_text(standalone, llm="off")[0] == '{"a":1,"b":[2,3]}'


def test_invalid_json_is_never_touched():
    standalone = '{"a": }'
    assert compactor.compress_text(standalone, llm="off")[0] == standalone

    fenced = '```json\n{"a": }\n```'
    assert compactor.compress_text(fenced, llm="off")[0] == fenced


# ---------------------------------------------------------------------------
# Append-only L1
# ---------------------------------------------------------------------------

def test_l1_chunk_is_append_only_across_save_and_reload(tmp_path):
    state = _fresh_state()
    first = _append(state, "first turn about src/app.ts", source="user_prompt")
    block = first["block"]
    snapshot = copy.deepcopy(first)

    _append(state, "second turn about src/other.ts")
    assert state["l1_chunks"][0]["block"] == block
    assert state["l1_chunks"][0] == snapshot
    assert block.startswith("### Summary Chunk #1")
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", block) is None

    state_path = tmp_path / "state.json"
    compactor.save_state(state, state_path)
    reloaded = compactor.load_state(state_path)
    assert reloaded["l1_chunks"][0]["block"] == block
    assert reloaded["l1_chunks"][0] == snapshot


# ---------------------------------------------------------------------------
# L2 consolidation
# ---------------------------------------------------------------------------

def test_l2_consolidation_triggers_exactly_at_six():
    state = _fresh_state()
    blocks_before = []
    for index in range(5):
        chunk = _append(state, f"turn {index} touches src/module_{index}.py")
        blocks_before.append(chunk["block"])

    assert state["l2_masters"] == []
    assert [c["n"] for c in compactor.active_l1(state)] == [1, 2, 3, 4, 5]

    sixth = _append(state, "turn 5 touches src/module_5.py")
    assert len(state["l2_masters"]) == 1
    master = state["l2_masters"][0]
    assert master["covered"] == [1, 2, 3, 4, 5, 6]
    assert "Consolidated Master Summary (L2) #1" in master["block"]
    assert master["block"].startswith("### Consolidated Master Summary (L2) #1")
    assert "Covers Summary Chunks #1–#6" in master["block"]
    assert compactor.active_l1(state) == []

    # Past chunk blocks and dicts are byte-identical after consolidation.
    for index, block in enumerate(blocks_before):
        assert state["l1_chunks"][index]["block"] == block
    assert sixth["block"] == state["l1_chunks"][5]["block"]

    seventh = _append(state, "turn 6 touches src/module_6.py")
    assert len(state["l2_masters"]) == 1
    assert [c["n"] for c in compactor.active_l1(state)] == [7]
    assert seventh["n"] == 7


def test_l2_handles_multiple_batches_in_a_loop():
    state = _fresh_state()
    for index in range(12):
        _append(state, f"turn {index} touches src/module_{index}.py")

    assert [m["covered"] for m in state["l2_masters"]] == [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10, 11, 12],
    ]
    assert compactor.active_l1(state) == []


def test_l2_master_keeps_skeleton_pointer_past_digest_cut():
    body = _large_python_source()
    text = ("background " * 80) + "\n\n```python\n" + body + "```\n"
    state = _fresh_state()
    chunk = _append(state, text)
    assert chunk["stats"]["code_skeleton_blocks"] == 1
    code_id = chunk["stats"]["code_block_ids"][0]

    digest = compactor._digest(chunk["compressed_text"], compactor.L2_DIGEST_CHARS)
    assert "full code:" not in digest  # the pointer is genuinely past the cut

    for index in range(5):
        _append(state, f"follow-up {index} touches src/module_{index}.py")
    master = state["l2_masters"][0]
    assert f"Raw: {chunk['raw_id'][:12]}" in master["block"]
    assert f"Code: {code_id[:12]}" in master["block"]
    assert compactor.recall_raw(code_id) == body


def test_l2_master_pointer_lines_are_deterministic_and_scoped():
    state = _fresh_state()
    for index in range(6):
        _append(state, f"turn {index} touches src/module_{index}.py")
    master = state["l2_masters"][0]
    block, covered, entities = compactor._render_master_block(
        master["n"], state["l1_chunks"][:6]
    )
    assert block == master["block"]
    assert covered == master["covered"]
    assert entities == master["entities"]

    seventh = _append(state, "turn 6 touches src/module_6.py")
    assert seventh["raw_id"][:12] not in master["block"]


def test_l2_master_recovers_old_skeleton_pointer_from_text():
    body = _large_python_source()
    raw_id = compactor.archive_raw(body)
    assert raw_id is not None
    old_chunk = {
        "n": 1,
        "raw_id": "a" * 64,
        "compressed_text": (
            ("x" * 600)
            + "\n# [skeleton: 3 lines elided; full code: "
            + compactor._raw_file_display(raw_id)
            + "; read before editing]\n"
        ),
        "entities": [],
    }
    block, _, _ = compactor._render_master_block(1, [old_chunk])
    assert "Raw: aaaaaaaaaaaa" in block
    assert f"Code: {raw_id[:12]}" in block
    assert compactor.recall_raw(raw_id) == body


# ---------------------------------------------------------------------------
# Injection payload
# ---------------------------------------------------------------------------

def _state_with_master_and_active(session: str | None = None) -> dict:
    state = _fresh_state()
    for index in range(7):
        _append(state, f"turn {index} touches src/module_{index}.py", session=session)
    assert len(state["l2_masters"]) == 1
    assert [c["n"] for c in compactor.active_l1(state)] == [7]
    return state


def test_injection_payload_shape_order_and_cache_control(monkeypatch):
    monkeypatch.delenv("OPENROUTER_EXTRA_HEADERS", raising=False)
    state = _state_with_master_and_active()
    payload = compactor.injection_payload(state)

    assert payload is not None
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert payload["additional_context"] == payload["hookSpecificOutput"]["additionalContext"]
    assert "additional_context" in payload  # kept for custom routers

    blocks = payload["content_blocks"]
    assert len(blocks) == 2
    assert all(block["type"] == "text" for block in blocks)
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in blocks)
    assert "Consolidated Master Summary (L2) #1" in blocks[0]["text"]
    assert "Summary Chunk #7" in blocks[1]["text"]

    text = payload["additional_context"]
    assert text.index("Consolidated Master Summary (L2) #1") < text.index("Summary Chunk #7")
    assert payload["metadata"]["compactor"]["l2_masters"] == 1
    assert payload["metadata"]["compactor"]["active_l1"] == 1
    assert "openrouter_extra_headers" not in payload["metadata"]


def test_openrouter_headers_support_json_and_list_forms(monkeypatch):
    state = _state_with_master_and_active()

    monkeypatch.setenv(
        "OPENROUTER_EXTRA_HEADERS",
        '{"HTTP-Referer": "https://example.com", "X-OpenRouter-Title": "my app"}',
    )
    payload = compactor.injection_payload(state)
    assert payload["metadata"]["openrouter_extra_headers"] == {
        "HTTP-Referer": "https://example.com",
        "X-OpenRouter-Title": "my app",
    }

    monkeypatch.setenv(
        "OPENROUTER_EXTRA_HEADERS",
        "X-OpenRouter-Title: My App, HTTP-Referer: https://example.com",
    )
    assert compactor._openrouter_extra_headers() == {
        "X-OpenRouter-Title": "My App",
        "HTTP-Referer": "https://example.com",
    }


def test_injection_cap_drops_oldest_master_first(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3", "0")
    state = _fresh_state()
    filler = "implementation detail " * 40
    for index in range(7):
        _append(state, f"turn {index} {filler} src/module_{index}.py")

    payload = compactor.injection_payload(state, max_chars=700)
    assert payload is not None
    text = payload["additional_context"]
    assert len(text) <= 700
    assert payload["metadata"]["compactor"]["dropped_l2"] == 1
    assert "Consolidated Master Summary (L2) #1" not in text
    assert text.endswith("[append-compactor: older context omitted to fit the 10k hook cap]")


def test_injection_never_exceeds_hook_output_cap():
    state = _fresh_state()
    filler = "implementation detail " * 200
    for index in range(30):
        _append(state, f"turn {index} {filler} src/module_{index}.py")

    payload = compactor.injection_payload(state, max_chars=1_000_000)
    assert payload is not None
    assert len(payload["additional_context"]) <= compactor.HOOK_OUTPUT_CAP - 200
    assert payload["metadata"]["compactor"]["max_inject_chars"] <= compactor.HOOK_OUTPUT_CAP - 200


def test_injection_keeps_newest_active_chunks(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3", "0")
    state = _fresh_state()
    filler = "implementation detail " * 40
    for index in range(11):
        _append(state, f"turn {index} {filler} src/module_{index}.py")
    assert [c["n"] for c in compactor.active_l1(state)] == [7, 8, 9, 10, 11]

    payload = compactor.injection_payload(state, max_chars=2_000)
    text = payload["additional_context"]
    assert len(text) <= 2_000
    assert "Summary Chunk #11" in text       # newest turn survives
    assert "Summary Chunk #7" not in text    # oldest active turns go first
    assert payload["metadata"]["compactor"]["dropped_l1"] >= 1


# ---------------------------------------------------------------------------
# L3 index view
# ---------------------------------------------------------------------------

def test_l3_archive_round_trip_and_hash_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    block = "### Summary Chunk #1\npath: src/app.py\nbuildThing keeps the pool open."
    block_id = compactor.archive_l3(block)
    assert block_id is not None
    assert compactor.recall_l3(block_id) == block
    assert compactor.recall_l3(block_id[:12]) == block
    assert compactor.recall_l3("not-hex") is None

    # A tampered cache entry is refused, not returned as the original.
    path = compactor._l3_path(block_id)
    path.write_bytes(compactor._compress_l3_block(block + " tampered", 3))
    assert compactor.recall_l3(block_id) is None


def test_l3_archive_is_zstd_ultra_22(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    block = (
        "### Summary Chunk #9\n"
        "Files:\n- src/archive.py\n"
        "**Turn:**\nStore this with zstd ultra level twenty two. needle_42 stays exact.\n"
    )
    block_id = compactor.archive_l3(block)
    assert block_id is not None
    path = compactor._l3_path(block_id)
    assert path.name.endswith(".txt.zst")
    raw = path.read_bytes()
    assert raw[:4] == b"\x28\xb5\x2f\xfd"  # zstd frame magic
    assert b"needle_42" not in raw
    assert compactor.recall_l3(block_id) == block

    args = compactor._l3_zstd_args(22)
    assert args[:2] == ["zstd", "--ultra"]
    assert "-22" in args
    assert "--ultra" not in compactor._l3_zstd_args(19)

    # zstdgrep exercises the same decompressing scan path as `ugrep -z`.
    if shutil.which("zstdgrep"):
        scan = subprocess.run(
            ["zstdgrep", "-q", "needle_42", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert scan.returncode == 0


def test_l3_compression_kill_switch_and_legacy_plaintext(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    block = "### Summary Chunk #10\nlegacy_plaintext_payload"

    monkeypatch.setenv("APPEND_COMPACT_L3_COMPRESS", "0")
    block_id = compactor.archive_l3(block)
    assert block_id is not None
    path = compactor._l3_path(block_id)
    assert path.name.endswith(".txt") and not path.name.endswith(".zst")
    assert path.read_text(encoding="utf-8") == block
    assert compactor.recall_l3(block_id) == block

    # A legacy plaintext file is still readable while compression is enabled.
    monkeypatch.delenv("APPEND_COMPACT_L3_COMPRESS")
    legacy = "### Summary Chunk #11\nlegacy file written before zstd"
    legacy_id = compactor.l3_id_for(legacy)
    compactor._atomic_write_text(
        compactor._l3_path(legacy_id, compressed=False), legacy
    )
    assert compactor.recall_l3(legacy_id) == legacy


def test_l3_uses_plaintext_when_zstd_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    monkeypatch.setattr(compactor.shutil, "which", lambda name: None)
    block = "### Summary Chunk #12\nplaintext fallback when zstd is unavailable"
    block_id = compactor.archive_l3(block)
    assert block_id is not None
    path = compactor._l3_path(block_id)
    assert path.name.endswith(".txt")
    assert path.read_text(encoding="utf-8") == block
    assert compactor.recall_l3(block_id) == block


def test_l3_kill_switch_restores_baseline(monkeypatch, tmp_path):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    state = _fresh_state()
    filler = "implementation detail " * 40
    for index in range(7):
        _append(state, f"turn {index} {filler} src/module_{index}.py")

    monkeypatch.setenv("APPEND_COMPACT_L3", "0")
    baseline = compactor.injection_payload(state, max_chars=700)
    assert baseline["metadata"]["compactor"]["l3"] == {"enabled": False}
    assert "L3 Index" not in baseline["additional_context"]

    monkeypatch.setenv("APPEND_COMPACT_L3", "1")
    l3 = compactor.injection_payload(state, max_chars=700)
    assert l3["metadata"]["compactor"]["l3"]["enabled"] is True
    assert "L3 Index" in l3["additional_context"]


def test_l3_index_lines_carry_entities_keywords_and_readable_path(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    state = _fresh_state()
    for index in range(42):
        _append(
            state,
            f"Turn {index} about the payment retry backoff in src/payments/retry_{index}.py. "
            "The scheduleRetryBackoff helper caps the fourth attempt and returns HTTP 429.",
        )

    payload = compactor.injection_payload(state, max_chars=9_000, session=None)
    text = payload["additional_context"]
    index_lines = [line for line in text.splitlines() if line.startswith("- L2 ")]
    assert index_lines
    line = index_lines[0]
    assert "src/payments/retry_" in line
    assert "scheduleRetryBackoff" in line or "retry" in line.lower()
    assert compactor._l3_base_display() in text

    link = line.rsplit(" | ", 1)[1].strip()
    assert link.endswith(".txt.zst") and "/" in link
    archived = compactor._l3_dir() / link
    assert archived.is_file()
    content = compactor._read_l3_file(archived)
    assert content is not None
    block_id = compactor.l3_id_for(content)
    assert compactor.recall_l3(block_id[:12]) == content
    assert "ugrep -z PATTERN FILE" in text


def test_l3_numeric_signal_keeps_decimal_facts(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    # Bare integers (line counts, skeleton markers) are noise, but a decimal
    # default is an exact fact an agent may need from an old block.
    numbers = compactor._l3_numbers("line 136 then 0.25 and v2.9.1")
    assert "0.25" in numbers
    assert "136" not in numbers

    state = _fresh_state()
    for index in range(42):
        _append(
            state,
            f"Turn {index}: retry jitter defaults to 0.25 seconds in "
            f"src/payments/retry_{index}.py before the backoff cap.",
        )

    payload = compactor.injection_payload(state, max_chars=9_000, session=None)
    index_lines = [
        line for line in payload["additional_context"].splitlines()
        if line.startswith("- L2 ")
    ]
    assert index_lines
    assert any("0.25" in line for line in index_lines)


def test_l3_omitted_metadata_accounts_for_every_block(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    monkeypatch.setenv("APPEND_COMPACT_L3_INDEX_BUDGET_CHARS", "1")
    state = _fresh_state()
    filler = "implementation detail " * 60
    for index in range(30):
        _append(state, f"turn {index} {filler} src/module_{index}.py")

    payload = compactor.injection_payload(state, max_chars=3_000)
    meta = payload["metadata"]["compactor"]["l3"]
    assert meta["omitted_masters"] + meta["represented_masters"] == meta["total_masters"]
    assert meta["omitted_chunks"] + meta["represented_chunks"] == meta["total_chunks"]
    assert meta["omitted_masters"] + meta["omitted_chunks"] >= 1
    assert payload["additional_context"].endswith(
        "[append-compactor: older context omitted to fit the 10k hook cap]"
    )


def test_l2_digest_chars_is_configurable(monkeypatch):
    def master_size() -> int:
        state = _fresh_state()
        for index in range(6):
            _append(state, f"Turn {index} " + "the retry worker drains the queue " * 12)
        return len(state["l2_masters"][0]["block"])

    default = master_size()
    monkeypatch.setenv("APPEND_COMPACT_L2_DIGEST_CHARS", "120")
    assert master_size() < default - 6 * 200


def test_l3_drift_stable_drift_and_short_prompt(monkeypatch):
    state = _fresh_state()
    for index in range(3):
        _append(
            state,
            "cache eviction prefix stability scheduler ledger bitmap radix "
            f"tenancy topic{index}",
            session="s",
        )

    stable, score, compared = compactor._drift_decision(
        state, "s", "cache eviction prefix stability scheduler ledger bitmap radix tenancy"
    )
    assert stable is False
    assert compared == 3
    assert score >= compactor.DEFAULT_L3_DRIFT_THRESHOLD

    drifted, score, compared = compactor._drift_decision(
        state,
        "s",
        "photosynthesis chlorophyll stomata chloroplasts glucose oxygen mitochondria respiration",
    )
    assert drifted is True
    assert compared == 3

    short, _, compared = compactor._drift_decision(state, "s", "ok")
    assert short is False and compared == 0

    assert compactor._l3_keep_recent(True) == (
        compactor.DEFAULT_L3_DRIFT_KEEP_RECENT_MASTERS,
        compactor.DEFAULT_L3_DRIFT_KEEP_RECENT_CHUNKS,
    )


def test_l3_injection_does_not_mutate_stored_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    state = _fresh_state()
    for index in range(14):
        _append(state, f"turn {index} detail src/module_{index}.py")
    before = copy.deepcopy(state)

    compactor.injection_payload(state, max_chars=4_000, session=None)
    assert state == before


def test_recall_l3_cli_enforces_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_L3_DIR", str(tmp_path / "l3"))
    state = _fresh_state()
    chunk_a = _append(state, "session A detail src/a.py", session="A")
    _append(state, "session B detail src/b.py", session="B")
    block_id = compactor.archive_l3(chunk_a["block"])
    assert block_id is not None

    state_path = tmp_path / "state.json"
    compactor.save_state(state, state_path)
    env = os.environ.copy()

    def run(scope: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--state",
                str(state_path),
                "--recall-l3",
                block_id[:12],
                "--scope",
                scope,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

    allowed = run("A")
    assert allowed.returncode == 0
    assert allowed.stdout == chunk_a["block"] + "\n"

    denied = run("B")
    assert denied.returncode == 1
    assert "not in session" in denied.stderr

    missing = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--state",
            str(state_path),
            "--recall-l3",
            "0" * 16,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert missing.returncode == 1
    assert "missing" in missing.stderr


# ---------------------------------------------------------------------------
# Hook events
# ---------------------------------------------------------------------------

def _write_transcript(path: Path) -> None:
    records = [
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Please fix src/app.ts and export function buildThing() {}"}],
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Updating src/app.ts and buildThing now."}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_transcript_reader_skips_api_request_copies(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    records = [
        {"type": "user", "message": {"role": "user", "content": "Please fix src/app.ts"}},
        {"type": "api-request-blob", "message": {"role": "user", "content": "Please fix src/app.ts"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "Fixing src/app.ts."}},
        {"type": "api-request-blob", "message": {"role": "assistant", "content": "Fixing src/app.ts."}},
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    turns = compactor._read_transcript_turns(str(transcript), 24, 24_000)

    assert turns == [("user", "Please fix src/app.ts"), ("assistant", "Fixing src/app.ts.")]


def test_precompact_returns_plain_text_and_arms_session(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    state = _fresh_state()

    result = compactor.handle_precompact_event(
        state,
        {"session_id": "session-1", "transcript_path": str(transcript), "trigger": "auto"},
        Namespace(),
    )

    assert isinstance(result, str)
    assert not result.lstrip().startswith("{")
    assert "Entity Ledger" in result
    assert "src/app.ts" in result
    assert len(result) < 10_000
    assert state["armed_sessions"]["session-1"]["armed"] is True
    assert len(state["l1_chunks"]) == 2
    assert all(chunk["source"] == "precompact" for chunk in state["l1_chunks"])


def test_prompt_injects_history_before_appending_and_disarms(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    state = _fresh_state()
    compactor.handle_precompact_event(
        state,
        {"session_id": "session-1", "transcript_path": str(transcript)},
        Namespace(),
    )

    args = Namespace(prompt="Now also check src/other.ts", force_inject=False, inject=True)
    injected = compactor.handle_prompt_event(
        state, {"session_id": "session-1", "prompt": args.prompt}, args
    )

    assert isinstance(injected, dict)
    context = injected["hookSpecificOutput"]["additionalContext"]
    assert injected["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "src/app.ts" in context          # history is injected
    assert "src/other.ts" not in context    # current prompt is appended after injection
    assert "session-1" not in state["armed_sessions"]
    assert state["l1_chunks"][-1]["source"] == "user_prompt"
    assert "src/other.ts" in state["l1_chunks"][-1]["compressed_text"]

    # A later prompt is not injected because the session was disarmed.
    args2 = Namespace(prompt="another prompt", force_inject=False, inject=True)
    assert compactor.handle_prompt_event(state, {"session_id": "session-1"}, args2) is None


def test_main_precompact_emits_raw_stdout_and_exits_zero(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    state_path = tmp_path / "state.json"
    payload = {
        "session_id": "session-2",
        "hook_event_name": "PreCompact",
        "transcript_path": str(transcript),
        "trigger": "auto",
    }
    env = dict(os.environ)
    env["APPEND_COMPACT_LOG"] = str(tmp_path / "error.log")
    env["APPEND_COMPACT_LLM"] = "0"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--event", "precompact", "--state", str(state_path)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
    )

    assert proc.returncode == 0
    assert proc.stdout.strip()
    assert not proc.stdout.lstrip().startswith("{")
    assert "Entity Ledger" in proc.stdout

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["armed_sessions"]["session-2"]["armed"] is True


def test_prompt_flag_does_not_read_stdin(tmp_path, monkeypatch):
    """A manual --prompt run must not block waiting for hook JSON on stdin."""
    def boom(*args, **kwargs):
        raise AssertionError("stdin must not be read when --prompt is supplied")

    monkeypatch.setattr(compactor, "_read_stdin_payload", boom)
    state_path = tmp_path / "state.json"

    rc = compactor.main([
        "--state", str(state_path),
        "--event", "prompt",
        "--prompt", "Manual prompt touching src/manual.ts",
        "--no-llm",
    ])

    assert rc == 0
    saved = compactor.load_state(state_path)
    assert len(saved["l1_chunks"]) == 1
    assert "src/manual.ts" in saved["l1_chunks"][0]["block"]


# ---------------------------------------------------------------------------
# Tool-result condensation and post-compaction injection
# ---------------------------------------------------------------------------

def test_condense_tool_output_strips_ansi_collapses_repeats_and_keeps_errors():
    assert compactor.condense_tool_output("\x1b[31merror: boom\x1b[0m") == "error: boom"

    noisy = "\n".join(
        ["[INFO] waiting"] * 200
        + ["src/app.py:42: AssertionError: expected 3"]
        + ["done"] * 10
    )
    out = compactor.condense_tool_output(noisy)
    assert "(×200)" in out
    assert "src/app.py:42: AssertionError" in out
    assert len(out) <= compactor.TOOL_RESULT_CHARS


def test_condense_tool_output_is_bounded_for_long_unique_output():
    long_output = "\n".join(f"line {index}" for index in range(500))
    condensed = compactor.condense_tool_output(long_output)
    assert len(condensed) <= compactor.TOOL_RESULT_CHARS
    assert condensed.startswith("line 0")
    assert condensed.rstrip().endswith("line 499")


def test_transcript_tool_results_are_captured_and_condensed(tmp_path):
    payload = "\n".join(
        ["[INFO] polling"] * 200 + ["src/worker.ts:88: Error: socket closed"]
    )
    records = [
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Running the test suite"}],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": payload}],
            },
        },
    ]
    path = tmp_path / "transcript.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    turns = compactor._read_transcript_turns(str(path), 24, 24_000)
    tool_turns = [text for role, text in turns if role == "tool"]
    assert len(tool_turns) == 1
    assert "(×200)" in tool_turns[0]
    assert "src/worker.ts:88: Error" in tool_turns[0]
    assert len(tool_turns[0]) <= compactor.TOOL_RESULT_CHARS


def _append_compact_boundary(path: Path, timestamp: str) -> None:
    record = {"type": "system", "subtype": "compact_boundary", "timestamp": timestamp}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _armed_prompt(state, transcript: Path):
    args = Namespace(prompt="continue", force_inject=False, inject=True)
    return compactor.handle_prompt_event(
        state,
        {"session_id": "session-1", "prompt": "continue", "transcript_path": str(transcript)},
        args,
    )


def test_prompt_skips_injection_when_compaction_did_not_finish(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    state = _fresh_state()
    compactor.handle_precompact_event(
        state, {"session_id": "session-1", "transcript_path": str(transcript)}, Namespace()
    )
    state["armed_sessions"]["session-1"]["at"] = "2026-11-03T16:48:47Z"
    _append_compact_boundary(transcript, "2026-11-03T16:43:11.108Z")

    assert _armed_prompt(state, transcript) is None
    assert "session-1" not in state["armed_sessions"]


def test_prompt_injects_when_compaction_finished_after_arm(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    state = _fresh_state()
    compactor.handle_precompact_event(
        state, {"session_id": "session-1", "transcript_path": str(transcript)}, Namespace()
    )
    state["armed_sessions"]["session-1"]["at"] = "2026-11-03T16:41:02Z"
    _append_compact_boundary(transcript, "2026-11-03T16:43:11.108Z")

    injected = _armed_prompt(state, transcript)
    assert injected is not None
    assert "src/app.ts" in injected["hookSpecificOutput"]["additionalContext"]
    assert "session-1" not in state["armed_sessions"]


def test_sessionstart_injects_armed_session_once_and_disarms():
    state = _state_with_master_and_active(session="s1")
    state.setdefault("armed_sessions", {})["s1"] = {"armed": True}
    args = Namespace(force_inject=False, inject=True)

    injected = compactor.handle_sessionstart_event(
        state, {"session_id": "s1", "source": "compact"}, args
    )
    assert injected is not None
    assert injected["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "Summary Chunk #7" in injected["hookSpecificOutput"]["additionalContext"]
    assert "s1" not in state["armed_sessions"]

    assert compactor.handle_sessionstart_event(state, {"session_id": "s1"}, args) is None


def test_sessionstart_unarmed_session_is_not_injected():
    state = _state_with_master_and_active()
    args = Namespace(force_inject=False, inject=True)
    assert compactor.handle_sessionstart_event(
        state, {"session_id": "never-armed", "source": "compact"}, args
    ) is None


def test_hooks_json_registers_sessionstart_compact():
    data = json.loads((REPO_ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    entries = data["hooks"]["SessionStart"]
    assert any(entry.get("matcher") == "compact" for entry in entries)


# ---------------------------------------------------------------------------
# Content-addressed compression memo
# ---------------------------------------------------------------------------

def _fake_llm(calls: list, pattern: str = r"\b(carefully|several|expensive|future|fragile) "):
    def fake(masked, rate, force_tokens):
        calls.append(masked)
        return re.sub(pattern, "", masked)

    return fake


def test_compression_memo_reuses_identical_text(monkeypatch):
    calls = []
    monkeypatch.setattr(compactor, "_default_llm_compress", _fake_llm(calls))
    first = compactor.compress_prose(LONG_PROSE, llm="auto")
    # A different compressor must not be consulted: the memo decides the bytes.
    monkeypatch.setattr(compactor, "_default_llm_compress", _fake_llm(calls, r"\b(We|The) "))
    second = compactor.compress_prose(LONG_PROSE, llm="auto")

    assert first == second
    assert first[1] == "llmlingua"
    assert len(calls) == 1


def test_compression_memo_distinguishes_different_text(monkeypatch):
    calls = []
    monkeypatch.setattr(compactor, "_default_llm_compress", _fake_llm(calls))
    compactor.compress_prose(LONG_PROSE, llm="auto")
    compactor.compress_prose(LONG_PROSE + " One more sentence about src/other.ts.", llm="auto")
    assert len(calls) == 2


def test_compression_memo_can_be_disabled(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_CACHE", "0")
    calls = []
    monkeypatch.setattr(compactor, "_default_llm_compress", _fake_llm(calls))
    compactor.compress_prose(LONG_PROSE, llm="auto")
    compactor.compress_prose(LONG_PROSE, llm="auto")
    assert len(calls) == 2
    assert not list((Path(os.environ["APPEND_COMPACT_CACHE_DIR"])).rglob("*.json"))


def test_compression_memo_ignores_corrupt_entries(monkeypatch):
    key = compactor._compress_cache_key(
        LONG_PROSE,
        compactor.DEFAULT_RATE,
        compactor.LLMLINGUA_MODEL,
        variant="sentence" if compactor._sentence_protect_enabled() else "",
    )
    path = Path(os.environ["APPEND_COMPACT_CACHE_DIR"]) / "compressed" / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    calls = []
    monkeypatch.setattr(compactor, "_default_llm_compress", _fake_llm(calls))
    text, mode = compactor.compress_prose(LONG_PROSE, llm="auto")

    assert mode == "llmlingua"
    assert len(calls) == 1
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == compactor.COMPRESS_CACHE_SCHEMA
    assert data["text"] == text


def test_compression_memo_is_not_used_for_callables():
    calls = []
    fake = _fake_llm(calls)
    compactor.compress_prose(LONG_PROSE, llm=fake)
    compactor.compress_prose(LONG_PROSE, llm=fake)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Frozen shared block for multi-agent prefix reuse
# ---------------------------------------------------------------------------

def test_emit_shared_block_is_deterministic_and_frozen(tmp_path):
    state = _state_with_master_and_active()
    first = compactor.emit_shared_block(state, cache_dir=tmp_path / "cache")
    second = compactor.emit_shared_block(state, cache_dir=tmp_path / "cache")

    assert first["emitted"] is True
    assert first["sha256"] == second["sha256"]
    path = Path(first["shared_path"])
    frozen = path.read_text(encoding="utf-8")
    assert hashlib.sha256(frozen.encode("utf-8")).hexdigest() == first["sha256"]
    assert "Consolidated Master Summary (L2) #1" in frozen
    # Active L1 is per-agent recency and must never enter the shared block.
    assert "Summary Chunk #7" not in frozen

    for index in range(7, 12):
        _append(state, f"turn {index} touches src/module_{index}.py")
    assert len(state["l2_masters"]) == 2
    assert path.read_text(encoding="utf-8") == frozen

    third = compactor.emit_shared_block(state, cache_dir=tmp_path / "cache")
    assert third["sha256"] != first["sha256"]
    assert Path(third["shared_path"]).read_text(encoding="utf-8") != frozen


def test_emit_shared_block_without_masters_emits_nothing(tmp_path):
    result = compactor.emit_shared_block(_fresh_state(), cache_dir=tmp_path / "cache")
    assert result["emitted"] is False
    assert result["shared_path"] is None
    assert result["sha256"] is None
    assert result["l2_masters"] == 0
    assert not (tmp_path / "cache" / "shared").exists()


def test_shared_block_has_no_volatile_fields():
    state = _state_with_master_and_active()
    text = compactor.shared_block_text(state)
    twin = _state_with_master_and_active()

    assert text is not None
    assert text == compactor.shared_block_text(twin)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", text)


def test_shared_block_budget_keeps_newest_masters():
    state = _fresh_state()
    for index in range(24):
        _append(
            state,
            f"turn {index} touches src/deep/module_{index}.py " + "detail " * 60,
        )
    assert len(state["l2_masters"]) == 4

    text = compactor.shared_block_text(state, max_chars=5_000)
    assert text is not None
    assert len(text) <= 5_000
    assert "(L2) #4" in text
    assert "(L2) #1" not in text


def test_cli_emit_shared_block_prints_path_and_hash(tmp_path):
    state = _state_with_master_and_active()
    state_path = tmp_path / "state.json"
    compactor.save_state(state, state_path)

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--emit-shared-block", "--state", str(state_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "APPEND_COMPACT_CACHE_DIR": str(tmp_path / "cache")},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["emitted"] is True
    assert payload["l2_masters"] == 1
    assert Path(payload["shared_path"]).exists()


# ---------------------------------------------------------------------------
# Parallel contexts, background queue, and detached workers
# ---------------------------------------------------------------------------

def test_sessions_never_share_injected_context():
    state = _fresh_state()
    _append(state, "alpha context touches src/alpha.ts", session="A")
    _append(state, "beta context touches src/beta.ts", session="B")

    alpha = compactor.injection_payload(state, session="A")
    beta = compactor.injection_payload(state, session="B")
    assert alpha is not None and beta is not None
    assert "src/alpha.ts" in alpha["additional_context"]
    assert "src/beta.ts" not in alpha["additional_context"]
    assert "src/beta.ts" in beta["additional_context"]
    assert "src/alpha.ts" not in beta["additional_context"]
    assert alpha["metadata"]["compactor"]["pending"] == 0


def test_consolidation_is_per_session():
    state = _fresh_state()
    for index in range(3):
        _append(state, f"A turn {index} touches src/a_{index}.py", session="A")
        _append(state, f"B turn {index} touches src/b_{index}.py", session="B")
    assert state["l2_masters"] == []

    for index in range(3, 6):
        _append(state, f"A turn {index} touches src/a_{index}.py", session="A")

    assert len(state["l2_masters"]) == 1
    master = state["l2_masters"][0]
    assert master["session"] == "A"
    assert master["covered"] == [1, 3, 5, 7, 8, 9]
    # Interleaved sessions must not be described as a continuous #1–#9 range.
    assert "Covers Summary Chunks #1, #3, #5, #7–#9" in master["block"]
    assert "src/a_0.py" in master["block"]
    assert "src/b_0.py" not in master["block"]
    assert [chunk["n"] for chunk in compactor.active_l1(state, "B")] == [2, 4, 6]


def test_queue_claim_commit_round_trip():
    state = _fresh_state()
    job = compactor.enqueue_turn(state, "queued src/queued.py buildThing", session="s")
    assert job is not None
    assert compactor.pending_count(state, "s") == 1
    assert compactor.pending_count(state, "other") == 0
    assert compactor.injection_payload(state, session="s") is None

    claimed, changed = compactor.claim_job(state, "worker-1")
    assert claimed is not None and changed is True
    assert claimed["id"] == job["id"]
    assert state["pending"][0]["status"] == "working"

    compressed, stats = compactor.compress_text(claimed["text"], llm="off")
    chunk = compactor.commit_job(state, claimed["id"], compressed, stats, worker="worker-1")
    assert chunk is not None
    assert chunk["session"] == "s"
    assert chunk["n"] == 1
    assert compactor.pending_count(state, "s") == 0
    payload = compactor.injection_payload(state, session="s")
    assert payload is not None and "src/queued.py" in payload["additional_context"]


def test_claim_is_exclusive_and_stale_claims_are_reclaimed():
    state = _fresh_state()
    compactor.enqueue_turn(state, "one src/one.py", session="s")
    first, _ = compactor.claim_job(state, "worker-1")
    assert first is not None
    second, changed = compactor.claim_job(state, "worker-2")
    assert second is None and changed is False

    timeout = compactor._env_int(
        "APPEND_COMPACT_CLAIM_TIMEOUT", compactor.DEFAULT_CLAIM_TIMEOUT_SECONDS
    )
    reclaimed, changed = compactor.claim_job(
        state, "worker-2", now=first["claimed_at"] + timeout + 1
    )
    assert reclaimed is not None and changed is True
    assert state["pending"][0]["worker"] == "worker-2"


def test_commit_from_a_superseded_claim_is_rejected():
    state = _fresh_state()
    compactor.enqueue_turn(state, "work src/work.py", session="s")
    first, _ = compactor.claim_job(state, "worker-1")
    compactor._requeue_stale(state, now=first["claimed_at"] + 10_000)
    second, _ = compactor.claim_job(state, "worker-2")
    assert second is not None

    compressed = "work src/work.py"
    stats = {"bytes_in": 18, "bytes_out": 18}
    assert compactor.commit_job(state, first["id"], compressed, stats, worker="worker-1") is None
    assert compactor.commit_job(state, second["id"], compressed, stats, worker="worker-2") is not None
    assert len(state["l1_chunks"]) == 1


def test_materialize_pending_commits_deterministically():
    state = _fresh_state()
    compactor.enqueue_turn(state, LONG_PROSE, session="s")
    before = compactor.pending_count(state, "s")
    assert before == 1

    committed = compactor.materialize_pending(state, "s")
    assert committed == 1
    assert compactor.pending_count(state, "s") == 0
    chunk = state["l1_chunks"][0]
    assert chunk["session"] == "s"
    assert chunk["stats"]["llm_blocks"] == 0  # no model work on the drain path
    assert "src/app.ts" in chunk["compressed_text"]


def test_preservation_instructions_include_queued_entities():
    state = _fresh_state()
    compactor.enqueue_turn(
        state, "Please fix src/pending.ts and export function pendingThing() {}", session="s"
    )
    text = compactor._preservation_instructions(state, "s")
    assert "src/pending.ts" in text and "pendingThing" in text

    other = compactor._preservation_instructions(state, "other")
    assert "src/pending.ts" not in other


def test_background_prompt_enqueues_and_arms_nothing():
    state = _fresh_state()
    state.setdefault("armed_sessions", {})["s"] = {"armed": True}
    args = Namespace(force_inject=False, inject=True, prompt=None)
    output = compactor.handle_prompt_event(
        state,
        {"session_id": "s", "prompt": "next prompt touches src/next.py"},
        args,
        background=True,
    )
    # Nothing was ready to inject, and the current prompt is queued, not chunked.
    assert output is None
    assert state["l1_chunks"] == []
    assert compactor.pending_count(state, "s") == 1
    assert state["pending"][0]["source"] == "user_prompt"
    assert "s" not in state["armed_sessions"]


def test_max_pending_falls_back_to_deterministic_append(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_MAX_PENDING", "1")
    state = _fresh_state()
    compactor.queue_or_append(state, "first src/first.py", "user_prompt", "s", background=True)
    assert compactor.pending_count(state, "s") == 1

    compactor.queue_or_append(state, "second src/second.py", "user_prompt", "s", background=True)
    assert compactor.pending_count(state, "s") == 1
    assert len(state["l1_chunks"]) == 1
    assert "src/second.py" in state["l1_chunks"][0]["compressed_text"]


def test_total_pending_cap_bounds_the_shared_state(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_MAX_PENDING_TOTAL", "2")
    state = _fresh_state()
    for session in ("a", "b"):
        compactor.queue_or_append(state, f"{session} queued", "user_prompt", session, background=True)
    assert compactor.pending_count(state) == 2

    # A third session is over the machine-wide cap: its turn is still captured,
    # but deterministically and without growing the queue.
    compactor.queue_or_append(state, "c overflow src/c.py", "user_prompt", "c", background=True)
    assert compactor.pending_count(state) == 2
    assert len(state["l1_chunks"]) == 1
    assert state["l1_chunks"][0]["session"] == "c"


def test_emit_shared_block_can_be_scoped():
    state = _fresh_state()
    for index in range(6):
        _append(state, f"A turn {index} touches src/a_{index}.py", session="A")
        _append(state, f"B turn {index} touches src/b_{index}.py", session="B")

    a_text = compactor.shared_block_text(state, session="A")
    b_text = compactor.shared_block_text(state, session="B")
    assert a_text is not None and b_text is not None
    assert "src/a_0.py" in a_text and "src/b_0.py" not in a_text
    assert "src/b_0.py" in b_text and "src/a_0.py" not in b_text


def test_drain_session_materializes_after_the_deadline(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    state = _fresh_state()
    compactor.enqueue_turn(state, "drain me src/drain.py", session="s")
    compactor.enqueue_turn(state, "other scope src/other.py", session="other")
    compactor.save_state(state, state_path)

    remaining = compactor.drain_session(state_path, "s", wait_seconds=0)
    assert remaining == 1
    reloaded = compactor.load_state(state_path)
    assert compactor.pending_count(reloaded, "s") == 0
    assert compactor.pending_count(reloaded, "other") == 1
    assert any(chunk.get("session") == "s" for chunk in reloaded["l1_chunks"])


def test_spawn_worker_respects_background_off_and_worker_env(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    compactor.save_state(_fresh_state(), state_path)
    assert compactor.spawn_worker(state_path) is False  # fixture disables background

    monkeypatch.setenv("APPEND_COMPACT_BACKGROUND", "1")
    monkeypatch.setenv("APPEND_COMPACT_WORKER", "1")
    assert compactor.spawn_worker(state_path) is False  # a worker never spawns another

    monkeypatch.delenv("APPEND_COMPACT_WORKER")
    calls = []

    class FakePopen:
        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))

    monkeypatch.setattr(compactor.subprocess, "Popen", FakePopen)
    assert compactor.spawn_worker(state_path) is True
    assert calls and calls[0][0][-3:] == ["--worker", "--state", str(state_path)]
    assert calls[0][1]["start_new_session"] is True


def test_worker_slots_are_machine_wide(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_WORKERS", "1")
    first = compactor._acquire_worker_slot()
    assert first is not None
    try:
        assert compactor._acquire_worker_slot() is None
    finally:
        first.close()
    third = compactor._acquire_worker_slot()
    assert third is not None
    third.close()


def test_precompact_capture_is_deduplicated_across_events(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    state = _fresh_state()
    payload = {"session_id": "s", "transcript_path": str(transcript)}
    args = Namespace()

    compactor.handle_precompact_event(state, payload, args)
    first = len(state["l1_chunks"])
    assert first == 2
    compactor.handle_precompact_event(state, payload, args)
    assert len(state["l1_chunks"]) == first


def test_precompact_dedupes_queued_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_BACKGROUND", "1")
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript)
    state = _fresh_state()
    payload = {"session_id": "s", "transcript_path": str(transcript)}
    args = Namespace()

    compactor.handle_precompact_event(state, payload, args, background=True)
    assert compactor.pending_count(state, "s") == 2
    compactor.handle_precompact_event(state, payload, args, background=True)
    assert compactor.pending_count(state, "s") == 2


# ---------------------------------------------------------------------------
# Cross-source capture dedupe
# ---------------------------------------------------------------------------

DEDUPE_PROMPT = (
    "Please inspect src/dedupe.py and update buildWidget so the retry loop closes "
    "every connection, logs one concise message, and returns the collected results "
    "to the caller without repeating expensive checks in the hot path."
)


def _precompact(text: str, role: str = "user") -> str:
    return f"{role.capitalize()}:\n{text}"


def _job_log_lines() -> list[dict]:
    path = Path(os.environ["APPEND_COMPACT_JOB_LOG"])
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_capture_key_strips_precompact_role_but_preserves_indentation():
    assert compactor._capture_key(_precompact(DEDUPE_PROMPT), strip_role=True) == compactor._capture_key(
        DEDUPE_PROMPT
    )
    assert compactor._capture_key("if ready:\n    run()\n") != compactor._capture_key(
        "if ready:\nrun()\n"
    )
    assert compactor._capture_key(DEDUPE_PROMPT + "  \n\n\n") == compactor._capture_key(DEDUPE_PROMPT)


def test_precompact_then_prompt_dedupes_one_capture():
    state = _fresh_state()
    first = compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT), "precompact", "s", background=False, role="user"
    )
    assert first is not None
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "s", background=False, role="user"
    ) is None
    assert len(state["l1_chunks"]) == 1
    assert state["l1_chunks"][0]["role"] == "user"
    assert state["l1_chunks"][0]["capture_key"] == compactor._capture_key(
        DEDUPE_PROMPT
    )
    assert state["prompt_seen"]["s"] > 0
    events = _job_log_lines()
    assert any(event["event"] == "dedupe" and event["reason"] == "capture_key" for event in events)
    assert all(DEDUPE_PROMPT not in json.dumps(event) for event in events)


def test_prompt_then_precompact_dedupes_the_other_direction():
    state = _fresh_state()
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "s", background=False, role="user"
    ) is not None
    assert compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT), "precompact", "s", background=False, role="user"
    ) is None
    assert len(state["l1_chunks"]) == 1


def test_dedupe_is_scoped_by_session_role_and_source():
    state = _fresh_state()
    compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT), "precompact", "A", background=False, role="user"
    )
    # A parallel session's identical text is its own turn.
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "B", background=False, role="user"
    ) is not None
    # An assistant turn is never suppressed by a user capture.
    assert compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT, "assistant"), "precompact", "A", background=False,
        role="assistant",
    ) is not None
    # Prompt-vs-prompt is deliberately not fuzzy-deduped.
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "C", background=False, role="user"
    ) is not None
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "C", background=False, role="user"
    ) is not None
    assert len(state["l1_chunks"]) == 5


def test_prompt_seen_floor_keeps_a_deliberate_retry():
    state = _fresh_state()
    compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT), "precompact", "s", background=False, role="user"
    )
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "s", background=False, role="user"
    ) is None
    retry = compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "s", background=False, role="user"
    )
    assert retry is not None
    assert len(state["l1_chunks"]) == 2


def test_prompt_dedupe_respects_the_capture_age_window():
    state = _fresh_state()
    compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT), "precompact", "s", background=False, role="user"
    )
    old = time.time() - 200
    state["l1_chunks"][0]["created_at"] = old
    state["recent_captures"][0]["at"] = old
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "s", background=False, role="user"
    ) is not None
    assert len(state["l1_chunks"]) == 2


def test_queue_cap_fallback_still_refuses_a_duplicate(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_MAX_PENDING_TOTAL", "1")
    state = _fresh_state()
    compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT), "precompact", "s", background=False, role="user"
    )
    # Fill the machine-wide queue without touching this session's prompt floor.
    assert compactor.enqueue_turn(state, "fill the queue src/fill.py", session="other") is not None
    assert compactor.pending_count(state) == 1
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "s", background=True, role="user"
    ) is None
    assert len(state["l1_chunks"]) == 1
    assert compactor.pending_count(state, "s") == 0


def test_similarity_gate_ignores_injected_system_reminders():
    state = _fresh_state()
    wrapped = _precompact(
        "<system-reminder>injected hook context</system-reminder>\n" + DEDUPE_PROMPT
    )
    assert compactor.queue_or_append(
        state, wrapped, "precompact", "s", background=False, role="user"
    ) is not None
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "s", background=False, role="user"
    ) is None
    assert len(state["l1_chunks"]) == 1
    events = _job_log_lines()
    assert any(event["event"] == "dedupe" and event["reason"] == "substring" for event in events)


def test_similarity_gate_keeps_unrelated_and_expanded_prompts():
    state = _fresh_state()
    compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT), "precompact", "s", background=False, role="user"
    )
    unrelated = (
        "Audit the release checklist, rotate the signing keys, and document the rollback "
        "procedure for the deployment pipeline before Friday's maintenance window."
    )
    assert compactor.queue_or_append(
        state, unrelated, "user_prompt", "s", background=False, role="user"
    ) is not None
    expanded = DEDUPE_PROMPT + " Also update the documentation and changelog entries."
    assert compactor.queue_or_append(
        state, expanded, "user_prompt", "s", background=False, role="user"
    ) is not None
    assert len(state["l1_chunks"]) == 3


def test_similarity_requires_minimum_evidence():
    sim = compactor._capture_similarity("short prompt src/a.py", "short prompt src/a.py")
    assert sim["evidence"] is False


def test_recent_captures_are_bounded_and_age_pruned():
    state = _fresh_state()
    for index in range(70):
        compactor._record_recent_capture(
            state, f"prompt number {index} about src/module_{index}.py", "user_prompt", "user",
            "s", now=1_000.0 + index,
        )
    assert len(state["recent_captures"]) == 64
    assert state["recent_captures"][0]["text"].startswith("prompt number 6 ")
    compactor._record_recent_capture(
        state, "prompt after the window src/later.py", "user_prompt", "user", "s", now=2_000.0
    )
    assert len(state["recent_captures"]) == 1
    assert state["recent_captures"][0]["text"].startswith("prompt after the window")


def test_load_state_backfills_and_sanitizes_dedupe_fields(tmp_path):
    good = {
        "at": 1.0,
        "text": "kept",
        "source": "user_prompt",
        "role": "user",
        "session": "s",
        "capture_key": "abc",
    }
    raw = {
        "schema": 1,
        "recent_captures": [good, {"at": "bad"}, {"at": 1, "text": "x", "source": "other", "role": "user"}],
        "prompt_seen": {f"s{index}": index for index in range(300)},
    }
    path = tmp_path / "state.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = compactor.load_state(path)
    assert loaded["recent_captures"] == [good]
    assert len(loaded["prompt_seen"]) == compactor.PROMPT_SEEN_LIMIT
    assert "s0" not in loaded["prompt_seen"] and "s299" in loaded["prompt_seen"]


def test_dedupe_kill_switch_preserves_legacy_precompact_dedupe(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_DEDUPE", "0")
    state = _fresh_state()
    compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT), "precompact", "s", background=False, role="user"
    )
    assert compactor.queue_or_append(
        state, DEDUPE_PROMPT, "user_prompt", "s", background=False, role="user"
    ) is not None
    assert compactor.queue_or_append(
        state, _precompact(DEDUPE_PROMPT), "precompact", "s", background=False, role="user"
    ) is None
    assert len(state["l1_chunks"]) == 2


# ---------------------------------------------------------------------------
# Error heuristic and job log
# ---------------------------------------------------------------------------

def test_error_heuristic_matches_logs_but_not_prose():
    assert compactor._ERROR_RE.search("The error message was long and the retry failed.") is None
    assert compactor._ERROR_RE.search("the exception path is documented") is None
    for text in ("error: disk full", "ERROR! bad token", "Traceback (most recent call last):",
                 "panic: runtime error", "npm ERR! missing script"):
        assert compactor._ERROR_RE.search(text), text


def test_job_log_records_enqueue_and_is_text_free():
    state = _fresh_state()
    job = compactor.enqueue_turn(state, DEDUPE_PROMPT, session="s", role="user")
    assert job is not None
    events = _job_log_lines()
    enqueued = [event for event in events if event["event"] == "enqueue"]
    assert len(enqueued) == 1
    assert enqueued[0]["job_id"] == job["id"]
    assert enqueued[0]["source"] == "user_prompt"
    assert enqueued[0]["role"] == "user"
    assert enqueued[0]["bytes"] == len(DEDUPE_PROMPT)
    assert all(DEDUPE_PROMPT not in json.dumps(event) for event in events)


def test_worker_logs_commits_and_continues_after_a_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_LLM", "0")
    monkeypatch.setenv("APPEND_COMPACT_WORKER_IDLE", "1")
    state_path = tmp_path / "state.json"
    state = _fresh_state()
    compactor.enqueue_turn(state, "boom src/fail.py", session="s", role="user")
    compactor.enqueue_turn(state, "good src/good.py", session="s", role="user")
    compactor.save_state(state, state_path)

    real_compress = compactor.compress_text

    def flaky(text, llm="auto"):
        if "boom" in text:
            raise RuntimeError("simulated failure")
        return real_compress(text, llm="off")

    monkeypatch.setattr(compactor, "compress_text", flaky)
    assert compactor.worker_main(state_path) == 0
    events = _job_log_lines()
    assert any(event["event"] == "error" for event in events)
    assert any(event["event"] == "commit" for event in events)
    assert all("boom src/fail.py" not in json.dumps(event) for event in events)
    reloaded = compactor.load_state(state_path)
    assert len(reloaded["l1_chunks"]) == 1
    assert "src/good.py" in reloaded["l1_chunks"][0]["compressed_text"]


# ---------------------------------------------------------------------------
# Exact-fact ledger, sentence protection and rate control (1.7.0)
# ---------------------------------------------------------------------------

FACT_TEXT = (
    "Run cargo build --release, then verifyWebhookSignature returns HTTP 202 "
    "on 2026-12-01 after 250 ms. Keep PAYMENTS_QUEUE_URL out of logs."
)


def test_fact_ledger_extracts_literals_camel_names_and_durations():
    facts = compactor.extract_facts(FACT_TEXT)
    assert "cargo build --release" in facts
    assert "verifyWebhookSignature" in facts
    assert "HTTP 202" in facts
    assert "2026-12-01" in facts
    assert "250 ms" in facts
    assert "PAYMENTS_QUEUE_URL" in facts


def test_fact_ledger_masks_and_restores_every_fact_byte_exact():
    masked, placeholders, mapping = compactor._mask_entities(FACT_TEXT)
    for fact in compactor.extract_facts(FACT_TEXT):
        assert fact not in masked
    assert placeholders
    assert compactor._restore_entities(masked, mapping) == FACT_TEXT


def test_extract_entities_pins_facts_without_duplicating_symbols():
    entities = compactor.extract_entities(FACT_TEXT)
    assert "fact: HTTP 202" in entities
    assert "fact: verifyWebhookSignature" in entities
    assert "fact: 250 ms" in entities
    # A camelCase function defined in a fence is a symbol, not a duplicate fact.
    code = "export function verifyWebhookSignature() { return 1; }"
    entries = compactor.extract_entities(code)
    assert "symbol: verifyWebhookSignature" in entries
    assert "fact: verifyWebhookSignature" not in entries


def test_fact_ledger_can_be_disabled(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_FACT_LEDGER", "0")
    assert "fact: HTTP 202" not in compactor.extract_entities("HTTP 202")


def test_internal_raw_archive_paths_are_not_ledger_entities():
    raw_path = compactor._raw_dir() / "ab" / ("a" * 64 + ".txt")
    text = f"# full code: {raw_path}; keep src/app.ts"
    entities = compactor.extract_entities(text)
    assert not any("append-compactor/raw" in entry for entry in entities)
    assert "path: src/app.ts" in entities


def test_sentence_protection_keeps_boundaries_and_compresses_only_the_middle(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_CACHE", "0")
    text = (
        "Opening guidance must stay exactly as written here. "
        "The middle part contains expensive and repeated checks on every request. "
        "Closing guidance must also stay exactly as written here."
    )
    calls: list[str] = []

    def fake(masked, rate, force_tokens):
        calls.append(masked)
        return re.sub(r"\b(expensive|repeated|every) ", "", masked)

    monkeypatch.setattr(compactor, "_default_llm_compress", fake)
    result, mode = compactor.compress_prose(text, llm="auto")
    assert mode == "llmlingua"
    assert len(calls) == 1
    assert "Opening guidance" not in calls[0]
    assert "Closing guidance" not in calls[0]
    assert "middle part" in calls[0]
    assert result.startswith("Opening guidance must stay exactly as written here.")
    assert result.endswith("Closing guidance must also stay exactly as written here.")


def test_sentence_protection_can_be_disabled(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_CACHE", "0")
    monkeypatch.setenv("APPEND_COMPACT_SENTENCE_PROTECT", "0")
    calls: list[str] = []

    def fake(masked, rate, force_tokens):
        calls.append(masked)
        return masked

    monkeypatch.setattr(compactor, "_default_llm_compress", fake)
    compactor.compress_prose(
        "Opening guidance must stay exactly as written here. "
        "The middle part contains expensive checks on every request. "
        "Closing guidance must also stay exactly as written here.",
        llm="auto",
    )
    assert len(calls) == 1
    assert "Opening guidance" in calls[0]


def test_split_sentences_accepts_lowercase_identifier_starts():
    parts = compactor._split_sentences(
        "Use src/app.ts. scheduleRetryBackoff runs next. Done."
    )
    assert parts[1].startswith("scheduleRetryBackoff")


def test_rate_control_classifies_logs_instructions_and_plain_prose(monkeypatch):
    log = (
        "2026-12-01T10:00:00Z INFO worker started\n"
        "2026-12-01T10:00:01Z WARN retry scheduled\n"
    )
    assert compactor._prose_rate(log) == 0.25
    assert compactor._prose_rate(
        "You must ensure the worker never logs PAYMENTS_QUEUE_URL."
    ) == 0.55
    assert compactor._prose_rate(
        "The payments worker drained its queue and closed the incident."
    ) == compactor.DEFAULT_RATE
    monkeypatch.setenv("APPEND_COMPACT_RATE_CONTROL", "0")
    assert compactor._prose_rate(log) == compactor.DEFAULT_RATE


def test_rate_control_is_used_by_compress_prose(monkeypatch):
    seen: list[float] = []
    monkeypatch.setenv("APPEND_COMPACT_CACHE", "0")

    def fake(masked, rate, force_tokens):
        seen.append(rate)
        return masked

    monkeypatch.setattr(compactor, "_default_llm_compress", fake)
    log = (
        "2026-12-01T10:00:00Z INFO worker started and drained the queue completely\n"
        "2026-12-01T10:00:01Z WARN retry scheduled for the same request after timeout\n"
        "2026-12-01T10:00:02Z INFO worker stopped after the queue reached zero depth\n"
    )
    compactor.compress_prose(log, llm="auto")
    assert seen == [0.25]


# ---------------------------------------------------------------------------
# Raw archive and --recall
# ---------------------------------------------------------------------------

def test_raw_archive_round_trips_full_id_and_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_RAW_DIR", str(tmp_path / "raw"))
    text = "raw turn\nwith two lines"
    raw_id = compactor.archive_raw(text)
    assert raw_id is not None and len(raw_id) == 64
    assert compactor.recall_raw(raw_id) == text
    assert compactor.recall_raw(raw_id[:12]) == text
    assert compactor.recall_raw(raw_id[:8]) is None
    assert compactor.recall_raw("0" * 12) is None
    assert compactor.recall_raw("not-hex") is None


def test_raw_archive_refuses_tampered_content(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_RAW_DIR", str(tmp_path / "raw"))
    raw_id = compactor.archive_raw("original")
    path = tmp_path / "raw" / raw_id[:2] / f"{raw_id}.txt"
    path.write_text("tampered", encoding="utf-8")
    assert compactor.recall_raw(raw_id) is None


def test_raw_archive_prunes_oldest_files_and_obeys_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("APPEND_COMPACT_RAW_MAX_FILES", "2")
    for index in range(4):
        assert compactor.archive_raw(f"turn {index}") is not None
    assert len(list((tmp_path / "raw").glob("*/*.txt"))) == 2
    monkeypatch.setenv("APPEND_COMPACT_RAW", "0")
    assert compactor.archive_raw("disabled") is None


def test_chunk_block_prints_a_short_raw_pointer():
    block = compactor._chunk_block(3, [], "body", "user_prompt", "a" * 64)
    assert "Raw: " + "a" * 12 in block
    assert "a" * 64 not in block


def test_recall_cli_prints_archived_turn_and_enforces_scope(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("APPEND_COMPACT_RAW_DIR", str(raw_dir))
    state_path = tmp_path / "state.json"
    state = _fresh_state()
    text = "remember the exact request id req_7d41c0e2"
    chunk = _append(state, text, session="s")
    raw_id = chunk["raw_id"]
    compactor.save_state(state, state_path)
    env = os.environ.copy()
    base = [sys.executable, str(SCRIPT_PATH), "--state", str(state_path), "--recall"]

    proc = subprocess.run(base + [raw_id[:12]], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip("\n") == text

    proc = subprocess.run(base + [raw_id[:12], "--scope", "s"], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr

    proc = subprocess.run(base + [raw_id[:12], "--scope", "other"], capture_output=True, text=True, env=env)
    assert proc.returncode == 1
    assert "not in session" in proc.stderr


# ---------------------------------------------------------------------------
# Code skeletonization with raw-block placeholders
# ---------------------------------------------------------------------------

def _large_python_source() -> str:
    lines = [
        '"""Module docstring."""',
        "import os",
        "",
        'CONST = "src/app.ts"',
        "",
        "",
        "def handler(value):",
        '    """Handle one value."""',
    ]
    lines.extend(f"    x{index} = value + {index}" for index in range(120))
    lines.append("    return value")
    return "\n".join(lines) + "\n"


def test_python_skeleton_keeps_structure_and_compiles(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_CODE_SKELETON", "1")
    source = _large_python_source()
    skeleton = compactor.skeleton_python_source(source)
    assert skeleton is not None
    assert len(skeleton) < len(source) // 3
    assert "import os" in skeleton
    assert 'CONST = "src/app.ts"' in skeleton
    assert "def handler(value):" in skeleton
    assert "x0 = value + 0" not in skeleton
    compile(skeleton, "<skeleton>", "exec")


def test_braced_skeleton_keeps_signatures_and_drops_bodies():
    source = (
        "export const A = 1;\n"
        "export class Foo {\n"
        "  bar(x: number): number {\n"
        "    const y = x + 1;\n"
        "    return y;\n"
        "  }\n"
        "}\n"
    )
    skeleton = compactor.skeleton_braced_source(source)
    assert skeleton is not None
    assert "export class Foo {" in skeleton
    assert "bar(x: number): number {" in skeleton
    assert "const y" not in skeleton
    assert "// ..." in skeleton


def _fresh_fence_stats() -> dict:
    return {
        "code_blocks": 0,
        "json_blocks": 0,
        "code_skeleton_blocks": 0,
        "llm_blocks": 0,
        "fallback_blocks": 0,
        "error_blocks": 0,
        "bytes_in": 0,
        "bytes_out": 0,
    }


def test_compress_fence_skeleton_kill_switch(monkeypatch):
    source = _large_python_source()
    stats = _fresh_fence_stats()
    monkeypatch.setenv("APPEND_COMPACT_CODE_SKELETON", "1")
    result = compactor._compress_fence("python", source, stats)
    assert "def handler(value):" in result
    assert "x0 = value + 0" not in result
    assert stats["code_skeleton_blocks"] == 1

    monkeypatch.setenv("APPEND_COMPACT_CODE_SKELETON", "0")
    stats["code_skeleton_blocks"] = 0
    result = compactor._compress_fence("python", source, stats)
    assert "x0=value+0" in result
    assert stats["code_skeleton_blocks"] == 0


def test_skeleton_placeholder_points_to_exact_raw_block(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_CODE_SKELETON", "1")
    source = _large_python_source()
    stats = _fresh_fence_stats()
    result = compactor._compress_fence("python", source, stats)
    match = re.search(r"full code: (.+?); read before editing", result)
    assert match is not None
    raw_path = Path(match.group(1)).expanduser()
    assert raw_path.read_text(encoding="utf-8") == source
    assert "# [skeleton:" in result
    assert raw_path.stat().st_mode & 0o777 == 0o600


def test_skeleton_requires_raw_archive(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_CODE_SKELETON", "1")
    monkeypatch.setenv("APPEND_COMPACT_RAW", "0")
    source = _large_python_source()
    stats = _fresh_fence_stats()
    result = compactor._compress_fence("python", source, stats)
    assert "x0=value+0" in result
    assert "[skeleton:" not in result
    assert stats["code_skeleton_blocks"] == 0


def test_skeleton_placeholder_uses_language_comment_marker(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_CODE_SKELETON", "1")
    marker = compactor._skeleton_placeholder("typescript", "a" * 64, "x\n" * 10)
    assert marker.startswith("// [skeleton: 10 lines elided;")
    marker = compactor._skeleton_placeholder("python", "a" * 64, "x\n" * 10)
    assert marker.startswith("# [skeleton: 10 lines elided;")


# ---------------------------------------------------------------------------
# Grouped pinned ledger
# ---------------------------------------------------------------------------

def test_global_ledger_groups_kinds_and_is_deterministic():
    state = _fresh_state()
    state["l1_chunks"] = [{
        "session": "s",
        "entities": [
            "symbol: buildThing",
            "fact: HTTP 202",
            "path: src/app.ts",
            "fact: v3.2.1",
        ],
    }]
    first = compactor._global_ledger_block(state, 2_000, "s")
    second = compactor._global_ledger_block(state, 2_000, "s")
    assert first == second
    assert "- fact: HTTP 202; v3.2.1" in first
    assert "- path: src/app.ts" in first
    assert "- symbol: buildThing" in first
    assert len(first.splitlines()) == 4


def test_global_ledger_marks_truncation():
    state = _fresh_state()
    state["l1_chunks"] = [{
        "session": "s",
        "entities": [f"path: src/file{i}.ts" for i in range(20)],
    }]
    block = compactor._global_ledger_block(state, 80, "s")
    assert "list truncated" in block
    assert len(block) <= 120


# ---------------------------------------------------------------------------
# Goal anchors (vision / strategy / constraint preservation)
# ---------------------------------------------------------------------------

GOAL_PROMPT = (
    "Goal: Ship the CSV export.\n"
    "Constraints:\n"
    "- Must not use pandas\n"
    "- Keep memory under 200 MB\n"
    "Success criteria:\n"
    "- 10k rows in under 2 seconds"
)


def _goal_state(session: str | None = None) -> dict:
    state = _fresh_state()
    _append(state, GOAL_PROMPT, session=session)
    for index in range(6):
        _append(
            state,
            f"turn {index} implementation detail src/module_{index}.py",
            session=session,
        )
    return state


def test_goal_extraction_is_off_by_default_and_never_invents():
    state = _goal_state()
    assert state["goals"]["sessions"] == {}
    payload = compactor.injection_payload(state)
    assert "GOAL ANCHOR" not in payload["additional_context"]
    assert payload["metadata"]["compactor"]["goals"]["enabled"] is False

    # No marker in the prompt: extraction must stay empty, not guess.
    assert compactor._extract_goal_items("Please continue with the next step.") == []


def test_goal_extraction_keeps_explicit_verbatim_items(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    items = compactor._extract_goal_items(GOAL_PROMPT)
    by_kind = {(item["kind"], item["text"]) for item in items}
    assert ("goal", "Ship the CSV export.") in by_kind
    assert ("constraint", "Must not use pandas") in by_kind
    assert ("constraint", "Keep memory under 200 MB") in by_kind
    assert ("success", "10k rows in under 2 seconds") in by_kind

    # Same-line markers are split instead of swallowing the whole line.
    inline = compactor._extract_goal_items(
        "Goal: Ship CSV. Constraints: must not use pandas. "
        "Success criteria: under two seconds."
    )
    assert {item["kind"] for item in inline} == {"goal", "constraint", "success"}
    assert all("Constraints:" not in item["text"] for item in inline)


def test_goal_recording_only_uses_user_turns(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    state = _fresh_state()
    compactor.append_turn(state, GOAL_PROMPT, llm="off", role="assistant")
    assert compactor._session_goal_store(state, None, create=False) is None
    compactor.append_turn(state, GOAL_PROMPT, llm="off", role="user")
    store = compactor._session_goal_store(state, None, create=False)
    assert store is not None
    assert store["version"] == 1


def test_goal_revision_supersedes_and_versions(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    state = _fresh_state()
    _append(state, GOAL_PROMPT)
    store = compactor._session_goal_store(state, None, create=False)
    assert store["version"] == 1
    original = next(item for item in store["items"] if item["kind"] == "goal")

    _append(state, "New goal: Ship the JSON export instead.")
    assert store["version"] == 2
    assert original["status"] == "superseded"
    assert original["superseded_by"]
    active = compactor._active_goal_items(store)
    assert [item["text"] for item in active if item["kind"] == "goal"] == [
        "Ship the JSON export instead"
    ]
    anchor = compactor._render_goal_anchor(store, ["L1"], 900)
    assert "Ship the JSON export instead" in anchor
    assert "Ship the CSV export" not in anchor
    assert anchor.startswith("[GOAL ANCHOR v2")

    # Repeating the same goal refreshes provenance without a new version.
    _append(state, "Ship the JSON export instead.")
    assert store["version"] == 2
    assert len([item for item in store["items"] if item["kind"] == "goal"]) == 2


def test_explicit_goal_in_a_later_turn_replaces_the_mission(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    state = _fresh_state()
    _append(state, "Goal: Ship the CSV export.\nGoal: Document the export format.")
    store = compactor._session_goal_store(state, None, create=False)
    # Two explicit goals in one prompt are parallel, not revisions.
    assert [item["text"] for item in compactor._active_goal_items(store)] == [
        "Ship the CSV export.",
        "Document the export format.",
    ]
    _append(state, "Goal: Ship the JSON export.")
    active_goals = [
        item for item in compactor._active_goal_items(store) if item["kind"] == "goal"
    ]
    assert [item["text"] for item in active_goals] == ["Ship the JSON export."]
    superseded = [
        item for item in store["items"]
        if item["kind"] == "goal" and item["status"] == "superseded"
    ]
    assert len(superseded) == 2
    assert store["version"] == 2


def test_goal_anchor_never_mutates_stored_blocks(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    state = _goal_state()
    chunks_before = [dict(chunk) for chunk in state["l1_chunks"]]
    masters_before = [dict(master) for master in state["l2_masters"]]
    compactor.injection_payload(state, max_chars=5_000)
    assert state["l1_chunks"] == chunks_before
    assert state["l2_masters"] == masters_before


def test_goal_l3_only_knob_renders_one_l3_anchor(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_L1", "0")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_L2", "0")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_L3", "1")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_PLACEMENT", "per_level")
    state = _goal_state()
    sections = compactor._render_goal_anchors(
        state, None, {"L1": True, "L2": True, "L3": True}
    )
    assert [level for level, _ in sections] == ["L3"]
    assert "level=L3" in sections[0][1]


def test_goal_completion_does_not_create_a_complete_goal(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    state = _fresh_state()
    _append(state, GOAL_PROMPT)
    _append(state, "The goal is complete.")
    store = compactor._session_goal_store(state, None, create=False)
    assert all(item["status"] == "completed" for item in store["items"])
    assert compactor._active_goal_items(store) == []
    assert compactor._render_goal_anchor(store, ["L1"], 900) == ""


def test_goal_anchor_is_tail_placed_and_head_is_unchanged(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_PLACEMENT", "tail")
    state = _goal_state()
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "0")
    baseline = compactor.injection_payload(state)
    baseline_head = baseline["content_blocks"][0]["text"]

    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    payload = compactor.injection_payload(state, max_chars=5_000)
    text = payload["additional_context"]
    assert "[GOAL ANCHOR v1 | levels=L1,L2]" in text
    assert 'Goal (verbatim): "Ship the CSV export."' in text
    assert text.rstrip().endswith(
        'Success criteria (verbatim): "10k rows in under 2 seconds"'
    )
    # The stable head (ledger + newest L2 master) is byte-identical; only the
    # volatile tail gained the anchor.
    assert payload["content_blocks"][0]["text"] == baseline_head
    assert "GOAL ANCHOR" not in payload["content_blocks"][0]["text"]
    assert "GOAL ANCHOR" in payload["content_blocks"][1]["text"]
    meta = payload["metadata"]["compactor"]["goals"]
    assert meta["enabled"] is True
    assert meta["version"] == 1
    assert meta["anchor_count"] == 1
    assert meta["anchor_chars"] == len(text.rsplit("\n\n", 1)[-1])


def test_goal_per_level_placement_and_knobs(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_PLACEMENT", "per_level")
    state = _goal_state()
    payload = compactor.injection_payload(state, max_chars=5_000)
    text = payload["additional_context"]
    assert "[GOAL ANCHOR v1 | level=L2]" in text
    assert "[GOAL ANCHOR v1 | level=L1]" in text
    assert text.index("level=L2") < text.index("level=L1")
    assert payload["content_blocks"][0]["text"] == compactor.injection_payload(
        state, max_chars=5_000
    )["content_blocks"][0]["text"]

    monkeypatch.setenv("APPEND_COMPACT_GOALS_L1", "0")
    l1_off = compactor.injection_payload(state, max_chars=5_000)
    assert "level=L1" not in l1_off["additional_context"]
    assert "level=L2" in l1_off["additional_context"]

    monkeypatch.setenv("APPEND_COMPACT_GOALS_L2", "0")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_L3", "0")
    none_on = compactor.injection_payload(state, max_chars=5_000)
    assert "GOAL ANCHOR" not in none_on["additional_context"]
    assert none_on["metadata"]["compactor"]["goals"]["anchor_count"] == 0


def test_goal_default_placement_is_beginning(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    state = _goal_state()
    payload = compactor.injection_payload(state, max_chars=5_000)
    assert payload["additional_context"].startswith("[GOAL ANCHOR")
    assert payload["metadata"]["compactor"]["goals"]["placement"] == "beginning"


def test_goal_beginning_placement_puts_anchor_in_the_stable_head(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_PLACEMENT", "beginning")
    state = _goal_state()
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "0")
    baseline = compactor.injection_payload(state, max_chars=5_000)
    baseline_head = baseline["content_blocks"][0]["text"]
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    payload = compactor.injection_payload(state, max_chars=5_000)
    text = payload["additional_context"]
    head = payload["content_blocks"][0]["text"]
    assert head.startswith("[GOAL ANCHOR v1 | levels=")
    assert text.startswith("[GOAL ANCHOR")
    assert baseline_head in head
    assert head.index("GOAL ANCHOR") < head.index(baseline_head)
    assert "GOAL ANCHOR" not in payload["content_blocks"][1]["text"]
    assert payload["metadata"]["compactor"]["goals"]["placement"] == "beginning"


def test_goal_after_head_placement_follows_the_head(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_PLACEMENT", "after_head")
    state = _goal_state()
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "0")
    baseline = compactor.injection_payload(state, max_chars=5_000)
    baseline_head = baseline["content_blocks"][0]["text"]
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    payload = compactor.injection_payload(state, max_chars=5_000)
    assert payload["content_blocks"][0]["text"] == baseline_head
    volatile = payload["content_blocks"][1]["text"]
    assert volatile.startswith("[GOAL ANCHOR v1 | levels=")
    assert payload["metadata"]["compactor"]["goals"]["placement"] == "after_head"


def test_goal_hybrid_placement_adds_a_reminder_only_on_drift(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_PLACEMENT", "hybrid")
    state = _goal_state()
    calm = compactor.injection_payload(state, max_chars=5_000)
    assert calm["additional_context"].startswith("[GOAL ANCHOR v1 | levels=")
    assert "GOAL REMINDER" not in calm["additional_context"]

    monkeypatch.setattr(
        compactor, "_drift_decision", lambda *args, **kwargs: (True, 0.0, 5)
    )
    drifted = compactor.injection_payload(state, max_chars=5_000)
    text = drifted["additional_context"]
    assert text.startswith("[GOAL ANCHOR v1 | levels=")
    assert "[GOAL REMINDER v1]" in text
    assert text.index("GOAL ANCHOR") < text.index("GOAL REMINDER")
    assert 'Goal (verbatim): "Ship the CSV export."' in text
    meta = drifted["metadata"]["compactor"]["goals"]
    assert meta["placement"] == "hybrid"
    assert meta["anchor_count"] == 2


def test_goal_anchor_survives_a_tight_content_budget(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_PLACEMENT", "tail")
    state = _fresh_state()
    _append(state, "Goal: keep the public API stable.")
    filler = "implementation detail " * 60
    for index in range(20):
        _append(state, f"turn {index} {filler} src/module_{index}.py")

    monkeypatch.setenv("APPEND_COMPACT_GOALS", "0")
    baseline = compactor.injection_payload(state, max_chars=900)
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    payload = compactor.injection_payload(state, max_chars=900)
    text = payload["additional_context"]
    assert len(text) <= 900
    assert "[GOAL ANCHOR" in text
    assert 'Goal (verbatim): "keep the public API stable."' in text
    assert payload["metadata"]["compactor"]["goals"]["anchor_chars"] > 0
    # Reserving anchor space shrinks only the volatile tail, never the cached
    # head (ledger + masters).
    assert payload["content_blocks"][0]["text"] == baseline["content_blocks"][0]["text"]


def test_goal_beginning_anchor_survives_a_tight_content_budget(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    monkeypatch.setenv("APPEND_COMPACT_GOALS_PLACEMENT", "beginning")
    state = _fresh_state()
    _append(state, "Goal: keep the public API stable.")
    filler = "implementation detail " * 60
    for index in range(20):
        _append(state, f"turn {index} {filler} src/module_{index}.py")
    payload = compactor.injection_payload(state, max_chars=900)
    text = payload["additional_context"]
    assert len(text) <= 900
    assert text.startswith("[GOAL ANCHOR")
    assert 'Goal (verbatim): "keep the public API stable."' in text
    assert "GOAL ANCHOR" in payload["content_blocks"][0]["text"]


def test_goal_anchor_duplicate_bodies_are_deduped():
    first = "[GOAL ANCHOR v1 | level=L2]\nGoal (verbatim): \"same\""
    second = "[GOAL ANCHOR v1 | level=L3]\nGoal (verbatim): \"same\""
    other = "[GOAL ANCHOR v1 | level=L1]\nGoal (verbatim): \"different\""
    out = compactor._dedupe_goal_sections(
        [("L2", first), ("L3", second), ("L1", other)], 1
    )
    assert out == [("L2", first), ("L1", other)]


def test_goal_store_is_session_isolated(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    state = _fresh_state()
    _append(state, GOAL_PROMPT, session="alpha")
    _append(state, "turn for beta with no goal", session="beta")
    assert compactor._session_goal_store(state, "alpha", create=False)["version"] == 1
    assert compactor._session_goal_store(state, "beta", create=False) is None

    alpha = compactor.injection_payload(state, session="alpha")
    beta = compactor.injection_payload(state, session="beta")
    assert "GOAL ANCHOR" in alpha["additional_context"]
    assert "GOAL ANCHOR" not in beta["additional_context"]


def test_goal_state_round_trips_and_sanitizes(tmp_path, monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    state = _goal_state(session="s")
    path = tmp_path / "state.json"
    compactor.save_state(state, path)
    loaded = compactor.load_state(path)
    store = compactor._session_goal_store(loaded, "s", create=False)
    assert store is not None
    assert store["version"] == 1
    assert len(compactor._active_goal_items(store)) >= 3

    # Malformed persisted entries are dropped rather than rendered.
    loaded["goals"]["sessions"]["s"]["items"].append(
        {"kind": "goal", "text": "   ", "status": "active"}
    )
    loaded["goals"]["sessions"]["s"]["items"].append(
        {"kind": "not-a-kind", "text": "valid text", "status": "weird"}
    )
    cleaned = compactor._sanitize_goal_store(loaded["goals"])
    items = cleaned["sessions"]["s"]["items"]
    assert all(item["text"].strip() for item in items)
    assert items[-1]["kind"] == "goal"
    assert items[-1]["status"] == "active"


def test_shared_block_can_include_a_goal_anchor(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_GOALS", "1")
    state = _goal_state(session="s")
    without = compactor.shared_block_text(state, max_chars=5_000, session="s")
    assert without is not None
    assert "GOAL ANCHOR" not in without

    monkeypatch.setenv("APPEND_COMPACT_GOALS_SHARED", "1")
    with_goal = compactor.shared_block_text(state, max_chars=5_000, session="s")
    assert "GOAL ANCHOR v1 | shared" in with_goal
    assert 'Goal (verbatim): "Ship the CSV export."' in with_goal
    assert len(with_goal) <= 5_000
    # Deterministic between calls while the goal version is unchanged.
    assert with_goal == compactor.shared_block_text(state, max_chars=5_000, session="s")


# ---------------------------------------------------------------------------
# Deterministic handoff (additive)
# ---------------------------------------------------------------------------

def _append_role(state: dict, text: str, role: str = "user", session: str | None = None) -> dict:
    chunk = compactor.append_turn(state, text, llm="off", session=session, role=role)
    assert chunk is not None
    return chunk


def _demoting_state(turns: int = 30) -> dict:
    """A session long enough that the injection stops showing older turns in full."""
    state = _fresh_state()
    for index in range(turns):
        _append_role(
            state,
            f"Turn {index}: the exporter retries with backoff and the pool sits at "
            f"{index % 7 + 2} shards while we tune src/export/worker_{index}.py "
            "for throughput. " * 3,
            role="assistant",
        )
    return state


def test_handoff_only_carries_what_the_injection_lost(monkeypatch):
    """The section exists to restore what packing dropped, not to repeat it."""
    state = _demoting_state()
    monkeypatch.setenv("APPEND_COMPACT_HANDOFF", "0")
    rest = compactor.injection_payload(state, max_chars=2_000)["additional_context"]
    monkeypatch.setenv("APPEND_COMPACT_HANDOFF", "1")
    payload = compactor.injection_payload(state, max_chars=2_000)
    text = payload["additional_context"]

    assert "## Session handoff (key facts)" in text
    handoff = text[len(rest.split("\n\n")[0]):]  # not used for assertions
    ledger = payload["content_blocks"][0]["text"]
    rendered, meta = compactor._render_handoff(state, None, ledger, 720, rest_text=rest)
    bullets = [line[2:] for line in rendered.splitlines() if line.startswith("- ")]
    assert bullets, "a demoted session must yield at least one recovered fact"
    assert meta["items"] == len(bullets)
    compact_rest = compactor._handoff_compact(rest)
    for bullet in bullets:
        assert compactor._handoff_compact(bullet.rsplit(" (turn", 1)[0]) not in compact_rest


def test_handoff_keeps_the_stable_head_byte_identical(monkeypatch):
    state = _demoting_state()
    monkeypatch.setenv("APPEND_COMPACT_HANDOFF", "0")
    baseline = compactor.injection_payload(state, max_chars=2_000)
    monkeypatch.setenv("APPEND_COMPACT_HANDOFF", "1")
    payload = compactor.injection_payload(state, max_chars=2_000)
    assert payload["content_blocks"][0]["text"] == baseline["content_blocks"][0]["text"]


def test_handoff_is_empty_when_the_view_already_shows_everything(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_HANDOFF", "1")
    state = _fresh_state()
    _append_role(state, "please use port 8443 for the metrics listener")
    payload = compactor.injection_payload(state, max_chars=6_000)
    assert "## Session handoff (key facts)" not in payload["additional_context"]
    assert payload["metadata"]["compactor"]["handoff"]["items"] == 0


def test_handoff_is_bounded_by_its_char_cap(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_HANDOFF", "1")
    state = _demoting_state(20)
    rest = ""
    rendered, meta = compactor._render_handoff(state, None, "", 200, rest_text=rest)
    assert rendered == "" or len(rendered) <= 200


# ---------------------------------------------------------------------------
# Journal tags (bookkeeping layer, off by default)
# ---------------------------------------------------------------------------

def test_journal_is_off_by_default(monkeypatch):
    monkeypatch.delenv("APPEND_COMPACT_JOURNAL", raising=False)
    state = _fresh_state()
    chunk = _append_role(state, "port 8443 is the metrics listener.\n※v port 8443")
    assert "tags" not in chunk


def test_journal_keeps_grounded_tags_and_drops_invented(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_JOURNAL", "1")
    state = _fresh_state()
    chunk = _append_role(
        state,
        "port 8443 is the metrics listener.\n"
        "※v port 8443\n"
        "※v MERGE_STALL_TIMEOUT_S 30.0\n",
    )
    tags = chunk.get("tags") or []
    bodies = [tag.get("body") for tag in tags]
    assert "port 8443" in bodies
    assert "MERGE_STALL_TIMEOUT_S 30.0" not in bodies
    assert all(tag["grounded"] for tag in tags)


def test_journal_source_anchor_grounds_against_the_cited_turn(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_JOURNAL", "1")
    state = _fresh_state()
    _append_role(state, "the metrics listener is on port 8443")
    chunk = _append_role(state, "recap.\n※v 1 port 8443\n※v 1 port 9000")
    bodies = [tag.get("body") for tag in chunk.get("tags") or []]
    assert "port 8443" in bodies
    assert "port 9000" not in bodies


def test_journal_task_closure_is_ordered_and_reopenable(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_JOURNAL", "1")
    state = _fresh_state()
    _append_role(state, "apply the correction.\n※t 1 apply the correction")
    _append_role(state, "next.\n※t 2 rerun the benchmark")
    _append_role(state, "done with the first.\n※d 1")
    open_ids = [task["id"] for task in compactor._journal_open_tasks(state["l1_chunks"])]
    assert open_ids == ["2"]


def test_journal_open_tasks_render_in_the_injection(monkeypatch):
    monkeypatch.setenv("APPEND_COMPACT_JOURNAL", "1")
    state = _fresh_state()
    _append_role(state, "please apply the correction.\n※t 1 apply the correction")
    payload = compactor.injection_payload(state, max_chars=6_000)
    assert "[task 1] apply the correction" in payload["additional_context"]
    assert payload["metadata"]["compactor"]["handoff"]["open_tasks"] == 1


def test_journal_instruction_is_injected_only_when_enabled(monkeypatch):
    state = _fresh_state()
    _append_role(state, "port 8443 is the metrics listener")
    monkeypatch.setenv("APPEND_COMPACT_JOURNAL", "0")
    assert "※" not in compactor.injection_payload(state, max_chars=6_000)["additional_context"]
    monkeypatch.setenv("APPEND_COMPACT_JOURNAL", "1")
    payload = compactor.injection_payload(state, max_chars=6_000)
    assert compactor.JOURNAL_INSTRUCTION in payload["additional_context"]
