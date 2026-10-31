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
import subprocess
import sys
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
    monkeypatch.setenv("APPEND_COMPACT_BACKGROUND", "0")
    monkeypatch.delenv("APPEND_COMPACT_WORKER", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_WORKERS", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_DRAIN_WAIT", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_LLM", raising=False)
    monkeypatch.delenv("APPEND_COMPACT_CACHE", raising=False)
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


def test_injection_cap_drops_oldest_master_first():
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


def test_injection_keeps_newest_active_chunks():
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
        LONG_PROSE, compactor.DEFAULT_RATE, compactor.LLMLINGUA_MODEL
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
