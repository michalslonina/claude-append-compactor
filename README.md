# claude-append-compactor

A Claude Code plugin that keeps a **deterministic, append-only checkpoint** of a
conversation outside the model context window. It snapshots recent turns before
Claude Code compacts, compresses code and prose cheaply, pins file paths and
exported symbols in an entity ledger, and feeds a stable, hard-capped summary
prefix back immediately after compaction (with the next prompt as fallback).

The design goal is **stable prefixes**, not promised cache-hit rates: summary
blocks are generated once and never rewritten, master summaries are ordered
before volatile chunks, block text contains no timestamps, and JSON key order
is deterministic.

Nothing here is required for the plugin to be safe: every hook exits 0, all
state is written atomically under an `flock`, and LLMLingua is a lazy optional
dependency with a deterministic fallback.

---

## Why

Claude Code's own compaction replaces history with a summary, but it is
lossy and not reproducible across runs. This plugin keeps its own append-only
record and re-injects it as additional context, so:

- file paths and exported symbol names survive compaction verbatim;
- code blocks are minified for free (no model call);
- prose shrinks selectively while URLs, paths, inline code and identifiers are
  masked and restored byte-exact;
- L1 chunks are immutable, so already-sent prefixes stay byte-identical across
  turns (the prerequisite for provider prefix caching);
- when six active chunks accumulate they are folded into an immutable L2
  master, and the injected view is capped at 9,000 characters with a pinned
  entity ledger at its head;
- tool output is captured too, condensed to head/tail lines, collapsed
  repetition and diagnostic lines, so large command results do not vanish
  without their errors.

---

## Architecture

```
UserPromptSubmit ─┐
SessionStart      ├─► injection_payload() ──► hookSpecificOutput.additionalContext
(compact)         │        │                      (hard 9k-char cap; Claude Code's
                  │        ├─ pinned entity ledger    own wall is 10k per string)
                  │        │  (stable head, 2k)
                  │        ├─ newest L2 masters
                  │        └─ newest active L1
                  │
PreCompact ───────┘  read transcript JSONL (last N turns + condensed tool results)
        │
        ├─► compression pipeline
        │     fenced code → python-minifier / line-level / JSON
        │     prose       → selective LLMLingua-2 with entity masks
        │     ledger      → regex paths + exported/top-level symbols
        ├─► append immutable L1 chunk; every 6 L1 → immutable L2 master
        ├─► arm session
        └─► print plain-text Entity Ledger preservation instructions to stdout
```

The same state is read by all three hooks; PreCompact arms, and whichever of
`SessionStart(compact)` or the next `UserPromptSubmit` runs first consumes the
arm. The injected view is bounded and recency-prioritised; the stored L1/L2
blocks are never rewritten.

---

## Install

```sh
git clone https://github.com/michalslonina/claude-append-compactor
cd claude-append-compactor
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# from any project, load the plugin for this session:
claude --plugin-dir "$PWD"

# optional manifest/hook validation:
claude plugin validate .
```

The hook commands call `python3 ...`, so the interpreter on `PATH` must be the
one with the dependencies installed. Activating the venv before launching
Claude Code is the simplest way to guarantee that; alternatively install the
dependencies into the system/user Python that `python3` resolves to.

`claude --plugin-dir` loads `hooks/hooks.json` automatically. Do **not** add a
`hooks` field to `plugin.json`; that would register the hooks twice.

### Python and optional dependencies

- **Python 3.10+** is required.
- `python-minifier` enables Python-fence minification. Without it, Python fences
  still get conservative line-level cleanup.
- `llmlingua` (and its `torch` dependency) enables prose compression. It is
  imported lazily, only when a natural-language paragraph is large enough to
  compress. Python 3.10–3.12 has the best torch/LLMLingua support; on 3.13/3.14
  the wheel may not exist yet.
- Code minification, JSON compaction, entity extraction, L1/L2 summaries and
  the deterministic fallback all work **without** LLMLingua, torch, or a model
  download. Set `APPEND_COMPACT_LLM=0` to force that mode.

---

## Hook lifecycle

| Hook | Command | Reads | Writes | Output |
| --- | --- | --- | --- | --- |
| `PreCompact` | `append-compact.py --event precompact` | `transcript_path` JSONL, last `APPEND_COMPACT_PRECOMPACT_TURNS` text turns, capped at `APPEND_COMPACT_PRECOMPACT_CHARS`, plus condensed `tool_result` / `toolUseResult` output (6k chars total) | Appends each turn as an L1 chunk (`source: precompact`), consolidates if needed, arms the session | **Plain text** on stdout. Current Claude Code versions append successful PreCompact stdout to their compaction instructions; treat that as best-effort, not a documented injection channel. |
| `SessionStart` (matcher `compact`) | `append-compact.py --event sessionstart --inject` | Hook JSON on stdin (`session_id`, `source`) | If armed: injects once at the documented post-compaction point and disarms | JSON: `hookSpecificOutput.additionalContext` with `hookEventName: "SessionStart"` |
| `UserPromptSubmit` | `append-compact.py --event prompt --inject` | Hook JSON on stdin (`session_id`, `prompt`) | If still armed (SessionStart did not fire): builds injection from history **before** appending the prompt; always appends the prompt as an L1 chunk (`source: user_prompt`); disarms | JSON: `hookSpecificOutput.additionalContext` (functional in Claude Code), plus `additional_context`, `content_blocks` and `metadata` for custom routers |
| Manual | `append-compact.py --status` | State file | — | JSON status |
| Manual | `append-compact.py --reset` | — | Replaces state with an empty one | JSON |
| Manual | `append-compact.py --force-inject` | State file | — | Injection JSON regardless of arming |
| Manual | `append-compact.py --self-test` | — | — | PASS/FAIL control lines; exit 1 on failure |

Claude Code does not honor `hookSpecificOutput.customInstructions` on
PreCompact, and PreCompact cannot inject context, which is why the handler
returns a plain string and `main()` writes it raw. The instruction text is
capped at 4,000 characters.

**Hard output cap.** Claude Code caps each `additionalContext` string (and each
`systemMessage`, `initialUserMessage`, and plain stdout) at **10,000
characters**. Anything longer is written to a session file and replaced with
the file path plus at most the first 2,000 characters, and Claude Code does not
ask the model to read that file. There is no setting or environment variable to
raise this cap. The plugin therefore defaults `APPEND_COMPACT_MAX_INJECT_CHARS`
to 9,000 and clamps any larger value to 9,800.

On `UserPromptSubmit`, exit 2 would block the prompt, so the hook never exits
nonzero for internal errors: failures are appended to the error log and the
process exits 0.

---

## Compression layers

| Layer | Applies to | Mechanism | Cost | Fallback |
| --- | --- | --- | --- | --- |
| Python code | fenced `python`/`py`/`python3` | `python-minifier`: comments, docstrings/literal statements, `pass`, imports combined; verified by `compile()` | CPU, milliseconds | Conservative line-level pass |
| JSON | fenced `json`/`jsonc` and standalone JSON objects/arrays | `json.loads` → `json.dumps(obj, separators=(",", ":"))` | CPU, microseconds | Invalid JSON is left byte-for-byte untouched |
| Other code | fenced non-Python languages | Line-level deletion of comment-only lines, empty imports, and debug statements (`console.log/debug/trace`, `logger.debug`, `System.out.println`, `dbg!(`, `print(`); guarded against emptying block bodies, breaking continuations, and editing multi-line strings; shebangs and `eslint`/`@ts-*`/`noqa`/`pylint` directives are kept | CPU, microseconds | Lines that cannot be removed safely are kept |
| Prose | natural-language paragraphs ≥160 chars | Selective LLMLingua-2 (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, rate `0.4`). Inline code, URLs, file paths, dotted/snake_case/camelCase identifiers are masked, passed as `force_tokens`, then restored | Model call (lazy, CPU) | Whitespace normalization; also forced when a mask is lost, entity extraction changes, output is not smaller, or the compressor raises |
| Error text | paragraphs matching `Traceback`, `Error`, `Exception`, `panic`, `FAILED`, etc. | Never model-compressed; kept verbatim | none | — |
| Ledger | every chunk | Deterministic regex extraction of file paths and exported/top-level symbols; sorted and deduped | CPU, microseconds | Empty ledger → `- (none)` |
| L1 | every turn | Immutable `### Summary Chunk #N` block | none | — |
| L2 | exactly when 6 active L1 exist | Immutable `### Consolidated Master Summary (L2) #M` covering the oldest 6: union ledger + per-chunk digests bounded to ~480 chars each | CPU | — |

The mention of `Error` is intentionally broad: a paragraph containing a marker
is more valuable when preserved verbatim. Note that the LLMLingua masks are
heuristics, not a parser.

---

## State schema

Default path: `~/.claude/append_compactor_state.json` (override with `--state` or
`APPEND_COMPACT_STATE`). Writes go through a temp file + `os.replace` while an
`flock` on `<state>.lock` serializes concurrent hook invocations.

```json
{
  "schema": 1,
  "created": "2026-10-31T00:00:00Z",
  "l1_chunks": [
    {
      "n": 1,
      "created": "2026-10-31T00:01:00Z",
      "source": "user_prompt",
      "entities": ["path: src/app.ts", "symbol: buildThing"],
      "compressed_text": "Please update src/app.ts ...",
      "block": "### Summary Chunk #1\nSource: user_prompt\n**Entity Ledger:**\n- path: src/app.ts\n- symbol: buildThing\n**Turn:**\n...",
      "stats": {
        "code_blocks": 0,
        "json_blocks": 0,
        "llm_blocks": 0,
        "fallback_blocks": 0,
        "error_blocks": 0,
        "bytes_in": 120,
        "bytes_out": 110
      }
    }
  ],
  "l2_masters": [
    {
      "n": 1,
      "created": "2026-10-31T00:10:00Z",
      "covered": [1, 2, 3, 4, 5, 6],
      "entities": ["path: src/app.ts", "symbol: buildThing"],
      "block": "### Consolidated Master Summary (L2) #1\nCovers Summary Chunks #1–#6\n..."
    }
  ],
  "armed_sessions": {
    "session-id": {"armed": true, "at": "2026-10-31T00:09:00Z", "trigger": "auto", "captured_turns": 24}
  },
  "stats": {
    "turns": 7,
    "bytes_in": 900,
    "bytes_out": 700,
    "llm_blocks": 1,
    "fallback_blocks": 2,
    "error_blocks": 0
  },
  "last_event": {"event": "UserPromptSubmit", "session_id": "session-id", "at": "2026-10-31T00:11:00Z"}
}
```

Timestamps exist only in chunk/master **metadata** (`created`, `armed_sessions`,
`last_event`). They are never written into `block`, so block text is stable
across regenerations.

---

## Entity ledger example

Input turn:

```text
Please update src/app.ts and ./lib/util.py.
export function buildThing() {}
export { helper as renamed };
def handler():
    return 1
pub fn run_task() {}
```

Ledger (paths first, then symbols, each sorted):

```text
**Entity Ledger:**
- path: ./lib/util.py
- path: src/app.ts
- symbol: buildThing
- symbol: handler
- symbol: renamed
- symbol: run_task
```

Extraction covers common source extensions (`.ts`, `.tsx`, `.py`, `.go`, `.rs`,
`.java`, `.kt`, …), `./` and absolute and Windows paths, plus `export
function/class/const`, `export { a as b }`, Python `def`/`class`, Go
`func`/`type`, Rust `pub fn`/`struct`/`enum`/`trait`, and JVM
`class`/`interface`/`object`. The ledger is deterministic: the same input
always produces the same sorted entries.

On PreCompact the same entries are flattened into a plain-text instruction
(`Entity Ledger (preserve these file paths ...`) so Claude Code's own summary
keeps them verbatim.

---

## Append-only and KV-caching guarantees

What the plugin guarantees:

- **L1 chunks are immutable.** Once `### Summary Chunk #N` is written, neither
  its `block` nor its chunk dict changes when later turns arrive or after a
  save/reload. Consolidation also leaves past chunks untouched.
- **L2 masters are immutable.** A master records `covered` chunk numbers; those
  chunks are excluded from `active_l1` but remain in state.
- **Block text has no timestamps.** Generated metadata may vary; the injected
  prefix text does not.
- **Deterministic ordering.** Entity entries are sorted; selected L2 masters
  stay chronological within the window, then active L1 in chunk order.
- **Bounded injection.** `APPEND_COMPACT_MAX_INJECT_CHARS` (default **9,000**,
  clamped to 9,800) caps the payload well under Claude Code's 10,000-character
  hook-output wall. A deterministic `**Entity Ledger (pinned):**` head keeps
  every captured path and symbol alive (2,000-char budget). The newest L2
  masters and newest active L1 are kept; oldest material is dropped first, with
  an explicit notice.

These properties are what make the prefix suitable for provider-side prefix
caching: stored blocks are append-only and never rewritten, so any injected
text that remains in the conversation is byte-stable. The pinned ledger head
changes only when a genuinely new entity appears; newer material is appended
after it.

---

## cache_control, OpenRouter and model routing

### What Claude Code actually does

Claude Code reads only `hookSpecificOutput.additionalContext` from
`UserPromptSubmit` / `SessionStart` hooks and injects it as a system reminder
(at the prompt or at the start of the post-compaction conversation). It **never
forwards** the plugin's `content_blocks` and **ignores** `cache_control`. The
`content_blocks` array (each block carrying `{"type": "ephemeral"}`) is emitted
for downstream routers/proxies and tests, not for Claude Code itself.

The functional Claude Code shape is:

```json
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "..."
  }
}
```

The plugin also emits a top-level `additional_context` key for custom routers;
Claude Code does not read it.

### Model routing: DeepSeek and OpenRouter

This plugin is often used with Claude Code pointed directly at a
DeepSeek Anthropic-compatible base URL, and sometimes pointed directly at
OpenRouter. In both direct setups there is **no gateway consuming this
plugin's payload**, so the `content_blocks`/`cache_control` hints are
**informational and inert today**.

- **DeepSeek direct:** caching is automatic and provider-side; there is no
  client `cache_control` to send. The plugin's real contribution is the
  byte-stable, deduplicated, append-only summary text (stable L2 masters first,
  volatile active L1 last, no timestamps in block text, deterministic JSON key
  order). That is the design goal — not a promise that DeepSeek's prefix cache
  will hit.
- **OpenRouter direct:** caching is provider-dependent. OpenRouter passes
  `cache_control` through for Anthropic models; DeepSeek routes generally use
  automatic caching; some providers offer no caching. Do not assume hits.
- **Injected `additionalContext` is delivered as a system reminder with the
  prompt.** Whether a provider's prefix cache hits depends on the exact request
  layout; this plugin only aims to keep prefixes stable.

### `OPENROUTER_EXTRA_HEADERS`

`OPENROUTER_EXTRA_HEADERS` is **not** a Claude Code variable and **not** an
OpenRouter variable. This plugin neither reads nor sends it to either service.
If set (JSON object or comma-separated `Key: Value` list), it is emitted under
`metadata.openrouter_extra_headers` in the injection payload as a hint for a
future custom gateway that builds OpenRouter requests (for example attribution
headers such as `HTTP-Referer` or `X-OpenRouter-Title`). It is inert with direct
DeepSeek and with direct OpenRouter through Claude Code.

### When routed to Anthropic models

If a custom router does consume `content_blocks` and forward them to an
Anthropic-compatible API, the ephemeral markers tell that layer where to place
cache breakpoints (masters first, active chunks last). Claude Code itself still
ignores them.

---

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `APPEND_COMPACT_STATE` | `~/.claude/append_compactor_state.json` | State file path (overridden by `--state` when given) |
| `APPEND_COMPACT_LLM` | enabled | Set to `0`/`false`/`off`/`no`/`disabled` to disable LLMLingua entirely (deterministic mode) |
| `APPEND_COMPACT_LLM_MODEL` | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` | Override the LLMLingua-2 model |
| `APPEND_COMPACT_MAX_INJECT_CHARS` | `9000` | Hard cap on injected context, clamped to 9800 (Claude Code's per-string hook-output wall is 10,000); oldest material dropped first |
| `APPEND_COMPACT_PRECOMPACT_TURNS` | `24` | Number of recent transcript turns captured on PreCompact |
| `APPEND_COMPACT_PRECOMPACT_CHARS` | `24000` | Character cap for captured transcript text |
| `APPEND_COMPACT_DEBUG` | off | Also write errors to stderr |
| `APPEND_COMPACT_LOG` | `~/.cache/append-compactor/error.log` | Error log path (rotated at 5 MB) |
| `OPENROUTER_EXTRA_HEADERS` | unset | JSON object or `Key: Value, ...` list emitted under `metadata.openrouter_extra_headers` for a custom gateway; inert in direct setups |

---

## Performance and cold start

- **LLMLingua is lazy and heavy.** The first compressible paragraph triggers
  `torch` plus model loading (a measured cold start of ~40 s on one CPU machine,
  including the model load; gigabytes of RAM, and a one-time model download).
  Both hook timeouts are 60 s except `PreCompact` (120 s), so a cold start can
  approach the limit on a loaded machine. Consider `APPEND_COMPACT_LLM=0`, or
  pre-warm the model outside the hook.
- **Code and JSON layers are CPU-only and fast**; they are safe to run on every
  turn.
- The hook reads the transcript only on PreCompact, not on every prompt. It does
  not re-ingest the live transcript each turn; the current prompt is appended
  directly.
- State grows without bound over a long project. Injection is capped, but the
  state file accumulates all L1 chunks and L2 masters. Use `--reset` to start
  fresh.

---

## Limitations

- **Line-level heuristics, not a parser.** The fallback code pass may retain
  some removable lines (safe) but cannot guarantee it finds every debug line or
  comment. It deliberately refuses deletions that could empty a block, break a
  continuation, or touch a multi-line string.
- **Entity extraction is regex-based.** Dynamically constructed paths, aliased
  re-exports, and symbols generated at runtime are not detected.
- **L2 is extractive, not abstractive.** Masters contain a union ledger and
  bounded per-chunk digests; they do not synthesize new prose.
- **No Python minification without `python-minifier`.** The package is an
  optional dependency; without it Python fences fall back to line-level cleanup.
- **No LLMLingua without `llmlingua`/`torch`**, and no model download is
  attempted by this plugin's tests. The fallback is deterministic whitespace
  normalization.
- **Windows locking.** `flock` is unavailable on Windows, so state writes are
  not serialized there (the plugin targets macOS/Linux).
- **Caching is not guaranteed.** See above: the plugin provides stable prefixes,
  not hit-rate promises.
- **Masks are heuristics.** A compressor that loses a placeholder, changes
  entity extraction, or fails to shrink the text triggers the fallback.
- **Tool condensation is lossy by design.** Tool results are reduced to
  head/tail lines, collapsed repetition, and lines matching error/`file:line`
  diagnostics, capped at 1,200 characters each and 6,000 total per compaction.
  Unique but unremarkable output in the middle of a large result is dropped.

---

## Testing

```sh
# known-positive / known-negative controls (exit nonzero on any failure)
./.venv/bin/python hooks/scripts/append-compact.py --self-test

# full suite
./.venv/bin/pytest tests/test_compactor.py -v
```

The self-test includes controls in both directions: python-minifier shrinks
valid code but leaves invalid code alone; entity extraction finds a known path
and symbol but plain prose yields none; JSON compaction works and invalid JSON
is untouched; L2 does not trigger at five chunks and does at six; the injected
payload stays under the hook cap and keeps the newest turns; the tool condenser
collapses repeats, keeps error lines, and leaves short clean output unchanged.
The pytest suite uses `tmp_path` state, `llm="off"` and injected fake
compressors, and never touches the real `~/.claude` state.

---

## License

MIT. See [LICENSE](LICENSE).
