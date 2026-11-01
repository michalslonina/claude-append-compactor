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
  without their errors;
- compression runs in detached background workers by default, so hooks stay
  fast, and every chunk is session-tagged, so parallel contexts and agent
  swarms sharing one install never see each other's history.

---

## Architecture

```
UserPromptSubmit ─┐
SessionStart      ├─► injection_payload() ──► hookSpecificOutput.additionalContext
(compact)         │        │                      (hard 9k-char cap; Claude Code's
                  │        ├─ pinned entity ledger    own wall is 10k per string)
                  │        │  (stable head, 2k)
                  │        ├─ newest L2 masters
                  │        └─ newest active L1, this session only
                  │
PreCompact ───────┘  read transcript JSONL (last N turns + condensed tool results)
        │
        ├─► enqueue raw turns (background mode, the default)
        │     └─► detached worker process, machine-wide slot pool
        │           compression pipeline, outside the state lock
        │             fenced code → python-minifier / line-level / JSON
        │             prose       → selective LLMLingua-2 with entity masks
        │             ledger      → regex paths + exported/top-level symbols
        │           commit under the lock → immutable L1 chunk;
        │           every 6 L1 of that session → immutable L2 master
        ├─► arm session
        └─► print plain-text Entity Ledger preservation instructions to stdout
```

The same state is read by all three hooks; PreCompact arms, and whichever of
`SessionStart(compact)` or the next `UserPromptSubmit` runs first consumes the
arm. Every chunk, master and queued job carries its `session_id`; injection
filters to the calling session, so parallel Claude Code contexts sharing this
plugin never see each other's turns. The injected view is bounded and
recency-prioritised; the stored L1/L2 blocks are never rewritten.

---

## Install

### From the marketplace (recommended)

```sh
claude plugin marketplace add michalslonina/claude-append-compactor
claude plugin install claude-append-compactor@claude-append-compactor
```

From a local checkout, add the directory instead:

```sh
claude plugin marketplace add /path/to/claude-append-compactor
claude plugin install claude-append-compactor@claude-append-compactor
```

Install the Python dependencies into the interpreter the hooks resolve (see
below), then reload plugins with `/reload-plugins`, or restart Claude Code.

### From a checkout without a marketplace

```sh
git clone https://github.com/michalslonina/claude-append-compactor
cd claude-append-compactor
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# from any project, load the plugin for this session:
claude --plugin-dir "$PWD"

# optional manifest/marketplace validation:
claude plugin validate .
```

The hook commands call `python3 ...`, so the interpreter on `PATH` must be the
one with the dependencies installed. Activating the venv before launching
Claude Code is the simplest way to guarantee that; alternatively install the
dependencies into the system/user Python that `python3` resolves to:

```sh
python3 -m pip install --user -r requirements.txt   # add --break-system-packages on PEP 668 systems
```

A CPU-only PyTorch is enough — the compactor runs LLMLingua on CPU. On a
PEP 668 system, installing it first keeps pip from pulling the multi-gigabyte
CUDA wheel:

```sh
python3 -m pip install --user --break-system-packages \
  --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install --user --break-system-packages -r requirements.txt
```

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
| `PreCompact` | `append-compact.py --event precompact` | `transcript_path` JSONL, last `APPEND_COMPACT_PRECOMPACT_TURNS` text turns, capped at `APPEND_COMPACT_PRECOMPACT_CHARS`, plus condensed `tool_result` / `toolUseResult` output (6k chars total) | Queues each turn for background compression (`source: precompact`). A durable normalized key plus a bounded similarity gate suppress a turn already captured by `UserPromptSubmit`; falls back to an immediate deterministic L1 chunk if the queue is full; arms the session | **Plain text** on stdout. Current Claude Code versions append successful PreCompact stdout to their compaction instructions; treat that as best-effort, not a documented injection channel. |
| `SessionStart` (matcher `compact`) | `append-compact.py --event sessionstart --inject` | Hook JSON on stdin (`session_id`, `source`) | If armed: drains this session's queue within `APPEND_COMPACT_DRAIN_WAIT`, injects once at the documented post-compaction point and disarms | JSON: `hookSpecificOutput.additionalContext` with `hookEventName: "SessionStart"` |
| `UserPromptSubmit` | `append-compact.py --event prompt --inject` | Hook JSON on stdin (`session_id`, `prompt`) | If still armed (SessionStart did not fire): drains this session's queue within `APPEND_COMPACT_DRAIN_WAIT`, then builds injection from history **before** enqueuing the prompt; enqueues the prompt (`source: user_prompt`) unless a recent PreCompact capture already contains it; disarms | JSON: `hookSpecificOutput.additionalContext` (functional in Claude Code), plus `additional_context`, `content_blocks` and `metadata` for custom routers |
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
| Prose | natural-language paragraphs ≥160 chars | Selective LLMLingua-2 (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, rate `0.4`). Inline code, URLs, file paths, dotted/snake_case/camelCase identifiers, and decimals/versions/numbers with units (`7.592s`, `99%`, `2.15.0`, `512MB`) are masked, passed as `force_tokens`, then restored byte-exact | Model call (lazy, CPU) | Whitespace normalization; also forced when a mask is lost, entity extraction changes, output is not smaller, or the compressor raises |
| Error text | paragraphs matching `Traceback`, `fatal error`, `panic:`, `SyntaxError`, `TypeError`, `ValueError`, `ImportError`, `npm ERR!`, `error TS…`, `failed to`, `segmentation fault`, or a line starting `error:`/`exception:` | Never model-compressed; kept verbatim | none | — |
| Ledger | every chunk | Deterministic regex extraction of file paths and exported/top-level symbols; sorted and deduped | CPU, microseconds | Empty ledger → `- (none)` |
| L1 | every turn | Immutable `### Summary Chunk #N` block | none | — |
| L2 | exactly when 6 active L1 of one session exist | Immutable `### Consolidated Master Summary (L2) #M` covering the oldest 6 of that session: union ledger + per-chunk digests bounded to ~480 chars each | CPU | — |

A bare mention of the words “error” or “exception” no longer forces verbatim
handling; the marker must look like a log line or a concrete exception. This
lets prose such as “the error message was long” compress while a real traceback
stays byte-exact. The LLMLingua masks remain heuristics, not a parser.

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
      "created_at": 1793491260.0,
      "source": "user_prompt",
      "role": "user",
      "session": "session-id",
      "input_sha": "5f2c...e1",
      "capture_key": "a19d...77",
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
      "session": "session-id",
      "covered": [1, 2, 3, 4, 5, 6],
      "entities": ["path: src/app.ts", "symbol: buildThing"],
      "block": "### Consolidated Master Summary (L2) #1\nCovers Summary Chunks #1–#6\n..."
    }
  ],
  "pending": [
    {
      "id": "5c1f...a2",
      "session": "session-id",
      "source": "precompact",
      "role": "user",
      "created": "2026-10-31T00:09:01Z",
      "created_at": 1793491741.2,
      "status": "working",
      "worker": "48213",
      "claimed_at": 1793491741.4,
      "input_sha": "5f2c...e1",
      "capture_key": "a19d...77",
      "entities": ["path: src/app.ts"],
      "text": "...raw turn, truncated to 24,000 chars..."
    }
  ],
  "armed_sessions": {
    "session-id": {"armed": true, "at": "2026-10-31T00:09:00Z", "trigger": "auto", "captured_turns": 24}
  },
  "recent_captures": [
    {
      "capture_key": "a19d...77",
      "source": "precompact",
      "role": "user",
      "session": "session-id",
      "at": 1793491741.2,
      "text": "comparison-only normalized text (user role only)",
      "tokens": 42,
      "shingles": 38
    }
  ],
  "prompt_seen": {"session-id": 1793491800.0},
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

`session`, `input_sha`, `role`, `capture_key`, `created_at`, `recent_captures`,
`prompt_seen` and `pending` are additive optional fields; the schema number
stays `1`, and older readers ignore them. Three compatibility rules apply:

- A chunk, master or job **without** a `session` (written by ≤ 1.4) is treated
  as belonging to no session and is skipped by scoped injection. It remains in
  state and is still used by unscoped commands (`--status` without `--scope`,
  `--force-inject`, `--emit-shared-block` without `--scope`).
- A chunk or job **without** `role`/`capture_key` (written by ≤ 1.5) is still
  read normally, but takes no part in cross-source dedupe: only the legacy
  `input_sha` PreCompact check sees it. New captures start populating the
  fields; nothing needs migrating.
- `pending` holds raw, not-yet-compressed turns. Each job records the identity
  of the captured turn (`input_sha` and the durable `capture_key`) so re-runs
  and cross-source duplicates are deduped, and `status`/`worker`/`claimed_at`
  so exactly one worker can compress it. A `working` claim older than
  `APPEND_COMPACT_CLAIM_TIMEOUT` is reclaimed, which makes a killed worker
  harmless.
- `recent_captures` is a bounded (64-entry, `APPEND_COMPACT_DEDUPE_WINDOW`
  second) ring of **comparison-only** normalized user text; it is never
  injected and holds at most `PENDING_MAX_CHARS` characters per entry.
  `prompt_seen` maps a session id to the timestamp of its last accepted prompt
  (256 sessions, oldest pruned first). Both are validated and dropped if
  malformed when the state is loaded.

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

## Multi-agent prefix sharing

Two mechanisms let parallel agents avoid duplicating work and, where the
provider allows it, share a cached prefix.

### Content-addressed compression memo

Every successful LLMLingua result is stored by content hash under
`~/.cache/append-compactor/compressed/<xx>/<sha256>.json`:

- **Key:** `sha256(schema, rate, model, text)` — identical paragraphs map to
  the same entry and the same bytes, in any session or agent.
- **Value:** the final compressed text and mode. Only successful `llmlingua`
  output is stored; deterministic fallbacks are cheap and never cached, so a
  transient model failure cannot poison the cache.
- **Misses are safe:** unreadable, empty or stale-schema entries are ignored
  and rewritten.
- **Bounded:** pruned to `APPEND_COMPACT_CACHE_MAX_FILES` (default 4,096)
  oldest-first. `APPEND_COMPACT_CACHE=0` disables reads and writes.

This removes duplicate model work across agents and makes output byte-identical
for byte-identical input — which is what any prefix cache requires.

### Frozen shared blocks

`--emit-shared-block` renders the pinned ledger and every L2 master into one
deterministic block, writes it to
`~/.cache/append-compactor/shared/<sha256>.block`, and prints:

```json
{"emitted": true, "shared_path": ".../shared/0b3e....block", "sha256": "0b3e...", "chars": 4833, "l2_masters": 2, "pinned_entities": 12}
```

Give that file's contents to each subagent **verbatim at the head of its first
message**:

```sh
python3 hooks/scripts/append-compact.py --emit-shared-block
```

Rules that make the cache win real:

- **Fixed offset.** The block must precede every per-agent difference. Text
  after the first divergence is never shared, whatever its content.
- **Verbatim bytes.** Do not re-render, reformat or summarize it per agent.
  Compare `sha256` across agents if a hit is unexpectedly missing.
- **Frozen, not live.** Existing `<sha>.block` files are never mutated; a state
  change produces a new path, so an in-flight fan-out keeps reading the same
  bytes.
- **No active L1.** Per-agent recency is deliberately excluded; it differs by
  construction and would destroy byte alignment.
- **Cap.** Default `9,000` characters (`--shared-max-chars` or
  `APPEND_COMPACT_SHARED_CHARS`). Oldest masters are dropped first.
- **No masters, no block.** Before the first consolidation the command reports
  `"emitted": false` — there is nothing shared to freeze.

The saving is bounded and secondary: the block is billed once per agent at the
provider's cache-read rate, and only after the first agent has warmed it.
Correctness is the real cost — a digest computed for one agent's task can carry
irrelevant or private detail into another agent's context.

---

## Background compaction and parallel contexts

Background compaction is **on by default** (`APPEND_COMPACT_BACKGROUND=1`).
Hooks never wait for LLMLingua on the common path: they capture the raw turn,
return immediately, and a detached worker process compresses it a moment later.
`--no-background` (or `APPEND_COMPACT_BACKGROUND=0`) restores the synchronous
1.4 behaviour.

### Why a process, not a thread

Claude Code hooks are short-lived processes and Claude Code enforces a timeout
per hook (60 s; 120 s for PreCompact). A Python thread would die with the hook
process, so the queue is drained by a **detached worker process** started with
`start_new_session=True`, `stdin`/`stdout` on `/dev/null`, and its log in
`~/.cache/append-compactor/workers/worker.log`. Workers are therefore
independent of whichever hook spawned them and survive the hook's exit.

### The queue protocol

`pending` in the state file is the queue. Every transition happens under the
same `flock` that protects all other state writes:

1. **Capture** — `PreCompact` and `UserPromptSubmit` append a job
   (`status: pending`) with the raw text, entities, session id, input hash and
   normalized `capture_key`. A capture that duplicates an earlier one — the
   same PreCompact turn re-read, or the same user turn seen by both hooks — is
   dropped before it reaches the queue; see [Cross-source capture
   dedupe](#cross-source-capture-dedupe). Duplicates are checked **before** the
   queue-cap fallback, so a full queue cannot sneak one in as a synchronous
   chunk.
2. **Claim** — a worker takes the lock, reclaims stale `working` claims older
   than `APPEND_COMPACT_CLAIM_TIMEOUT`, then marks exactly one pending job
   `working` for itself. PreCompact jobs go first: the next injection is
   imminent after a compaction.
3. **Compress** — the worker releases the lock and runs the full pipeline
   (`llm="auto"`). This is the seconds-long part and never blocks hooks or
   other contexts.
4. **Commit** — the worker re-takes the lock, verifies it still owns the claim,
   appends an immutable L1 chunk with the next global `n`, removes the job,
   records stats, and consolidates that session's L2 if due.

If a drain materialised the job in the meantime, the job is gone and the commit
is dropped (no duplicate). If another worker reclaimed the claim, the old
worker's commit is rejected. The state file is written with the usual temp file
+ `os.replace`, so readers always see a complete JSON document.

### Cross-source capture dedupe

The same user turn legitimately arrives twice: `PreCompact` reads it out of
the transcript and prefixes a `User:` label, while `UserPromptSubmit` captures
the raw prompt. Those captures differ by a few bytes (the label, sometimes an
injected `<system-reminder>` block, whitespace), so the exact `input_sha` the
1.5 queue used could not see them as one turn. Two layers now do, with no extra
dependency and without ever modifying captured text.

**Durable key.** `capture_key` is `sha256` of the same `_normalize_prose` the
plugin already applies: trailing whitespace and blank-line runs are ignored,
line breaks and indentation are kept (so re-indented code does not collapse).
A leading `User:`/`Assistant:`/`Tool:` label is stripped only for PreCompact,
the one place that adds it. The key is stored on jobs and chunks, so it
survives queueing, consolidation, and restarts.

**Similarity gate.** Recent user captures also keep a comparison-only
normalized copy (NFKC, casefolded, whitespace-collapsed, `<system-reminder>`
blocks and role label removed). A candidate is matched on 5-word shingles
(`zlib.crc32`) and word tokens, with a minimum of 20 tokens and 8 shingles
before any fuzzy decision. The rules are deliberately asymmetric:

| Incoming | May match | Rule |
| --- | --- | --- |
| PreCompact user turn | recent `user_prompt` | prompt is contained in the turn (substring, or shingle containment ≥ 0.95 and token containment ≥ 0.98), or shingle/token Jaccard ≥ 0.90 with 0.8–1.25× length |
| Prompt | recent PreCompact user turn | prompt is contained in that capture; never the reverse, because a new prompt may add instructions |
| anything else | — | no fuzzy match (prompt-vs-prompt, PreCompact-vs-PreCompact, assistant/tool turns are always kept) |

Matching is always scoped to the same session and the user role, so parallel
contexts and agent swarms cannot suppress one another's prompts. The prompt
direction additionally requires the capture to be younger than
`APPEND_COMPACT_CAPTURE_DEDUPE_SECONDS` (default 120) **and** newer than the
session's last accepted prompt (`prompt_seen`), which means only the first
prompt after a compaction can be absorbed; a deliberate repeat is captured as a
new turn. The PreCompact direction has no age limit — chunks are append-only,
so a duplicate remains a duplicate — while fuzzy comparison only looks at
`recent_captures` inside `APPEND_COMPACT_DEDUPE_WINDOW` (default 900).

Every drop is logged as a `dedupe` job-log event with its reason (`capture_key`,
`substring`, `shingle`, `jaccard`), scores, and the matched source/role/time —
never the text. `APPEND_COMPACT_DEDUPE=0` disables the new checks and leaves
the legacy PreCompact `input_sha` dedupe in place. A missed duplicate costs one
extra compression job; a false positive would silently lose a turn, which is
why the thresholds are one-sided and conservative.

### Bounded drain at injection time

A correct injection must not be missing the turns that just happened. Therefore,
only when an injection is actually armed (`SessionStart(compact)` or the next
`UserPromptSubmit`), the hook:

1. requests a worker (the machine-wide pool caps concurrency),
2. waits up to `APPEND_COMPACT_DRAIN_WAIT` seconds (default **5**) for this
   session's pending jobs to reach zero,
3. commits whatever remains **deterministically** (`llm="off"`: code/JSON
   minification and whitespace normalisation, no model call).

The result is a complete, byte-deterministic snapshot inside both the 5-second
drain window and the hook deadline. Ordinary prompts pay no drain cost at all.
Workers also age out after `APPEND_COMPACT_WORKER_IDLE` idle seconds or
`APPEND_COMPACT_WORKER_MAX_SECONDS` total lifetime, so a queue is not kept alive
by a forgotten process.

### Parallel contexts and agent swarms

Many Claude Code contexts (and their subagents) may share one plugin install
and one state file at the same time. Safety rests on four rules:

- **One lock, short critical sections.** Model work is outside the lock. Hook
  handlers only do regex/lexical work while holding it, so a swarm contends for
  milliseconds, not seconds.
- **One machine-wide worker pool.** `APPEND_COMPACT_WORKERS` (default **2**)
  flocked slot files under `workers/` cap concurrent compression across *all*
  sessions. A 30-agent fan-out therefore gets two LLMLingua processes, not 30;
  each worker sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to `cores / slots` so two
  workers do not each spawn a full OpenMP pool. Extra turn capture stays cheap
  and queued.
- **Session isolation.** Chunks, masters and pending jobs are tagged with
  `session_id`. Injection, consolidation, entity aggregation, shared blocks and
  `--status` filter by session (or deliberately include everything when
  `--scope` is omitted). Agent A can never receive agent B's prompt text, and
  one long-running context cannot consolidate another's chunks into its L2.
- **Bounded pressure.** At most `APPEND_COMPACT_MAX_PENDING` jobs per session
  (default **32**) and `APPEND_COMPACT_MAX_PENDING_TOTAL` jobs machine-wide
  (default **64**) are queued. Past either limit, the turn is committed
  immediately with the deterministic pipeline instead of growing the queue.
  Both the state file and the worker backlog stay bounded even if the workers
  are busy or the model is unavailable, which matters because every hook saves
  the whole state under one lock.

Workers run at nice 10 (`APPEND_COMPACT_WORKER_NICE`) and are otherwise normal
processes; on a loaded box they yield to the terminal. Heartbeats
(`workers/<pid>.json`) record status and jobs done; `--status` exposes
`pending_by_session` and per-session chunk counts.

### Operating it

```sh
# what is queued, per session
python3 hooks/scripts/append-compact.py --status | jq '.pending, .pending_by_session'

# one session's view only
python3 hooks/scripts/append-compact.py --status --scope "$SESSION_ID"

# see what a worker is doing / why it failed
cat ~/.cache/append-compactor/workers/*.json
tail ~/.cache/append-compactor/workers/worker.log

# what happened to each captured turn, one JSON object per line
tail -5 ~/.cache/append-compactor/jobs.log
jq -c 'select(.event=="commit") | {id, session, source, duration, stats}' \
  ~/.cache/append-compactor/jobs.log | tail -20
```

`jobs.log` (override `APPEND_COMPACT_JOB_LOG`, rotated at 5 MB to
`jobs.log.1`) records:

| `event` | Written by | Fields |
| --- | --- | --- |
| `enqueue` | `enqueue_turn` | `id`, `session`, `source`, `role`, `bytes`, `key` (capture-key prefix) |
| `inline` | queue-full fallback | `session`, `source`, `bytes` |
| `dedupe` | capture check | `session`, `source`, `reason`, scores (`shingle_containment`, `token_containment`, `shingle_jaccard`, `token_jaccard`), `matched_source`, `age` |
| `commit` | worker | `id`, `worker`, `session`, `source`, `duration`, `stats` (llm/fallback/error blocks, bytes in/out) |
| `dropped` | worker | as `commit`; `commit_job` found the job already drained or reclaimed |
| `error` | worker | `id`, `worker`, `session`, `source`, `error` (exception type/message) |

Lines are written with `O_APPEND` so concurrent hooks and workers interleave
whole lines. The log contains no turn text — only ids, hashes, lengths and
timings. The worker wraps compression and commit in `try/except`: an unexpected
raise is logged and the process continues with the next job instead of dying
and leaving a `working` claim for the 300 s reclaim.

The queue is not a durable job system: it lives in the state file, claims are
reclaimed after the timeout, and an unfinished job is materialised
deterministically at the next armed injection. That is intentional — losing
model compression is acceptable; losing a turn is not.

Contexts that share a state file must also share `APPEND_COMPACT_CACHE_DIR`:
worker slots and the spawn gate are per cache directory. The defaults already
do this. Overriding the two independently can let two pools run against one
state; that is still safe (the state lock and claim protocol hold), but the
machine-wide worker cap no longer applies.

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
| `APPEND_COMPACT_BACKGROUND` | enabled | Compress queued turns in detached workers. Set to `0`/`false`/`off` for synchronous compression; `--no-background` does the same for one invocation |
| `APPEND_COMPACT_WORKERS` | `2` | Machine-wide concurrent worker slots (flocked files); also divides `OMP_NUM_THREADS`/`MKL_NUM_THREADS` among workers |
| `APPEND_COMPACT_WORKER_IDLE` | `120` | Seconds an idle worker waits before exiting |
| `APPEND_COMPACT_WORKER_MAX_SECONDS` | `900` | Maximum lifetime of one worker process |
| `APPEND_COMPACT_WORKER_NICE` | `10` | `os.nice` value applied by a worker (POSIX) |
| `APPEND_COMPACT_CLAIM_TIMEOUT` | `300` | Seconds after which a `working` job is reclaimed (e.g. the worker was killed) |
| `APPEND_COMPACT_MAX_PENDING` | `32` | Per-session queue cap; beyond it turns are committed deterministically instead of queued |
| `APPEND_COMPACT_MAX_PENDING_TOTAL` | `64` | Machine-wide queue cap across all sessions; bounds state-file size and worker backlog under a large swarm |
| `APPEND_COMPACT_DRAIN_WAIT` | `5` | Bounded wait before an armed injection; remaining jobs are then materialised with `llm="off"` |
| `APPEND_COMPACT_MAX_INJECT_CHARS` | `9000` | Hard cap on injected context, clamped to 9800 (Claude Code's per-string hook-output wall is 10,000); oldest material dropped first |
| `APPEND_COMPACT_PRECOMPACT_TURNS` | `24` | Number of recent transcript turns captured on PreCompact |
| `APPEND_COMPACT_PRECOMPACT_CHARS` | `24000` | Character cap for captured transcript text |
| `APPEND_COMPACT_DEDUPE` | enabled | Set to `0`/`false`/`off` to disable cross-source `capture_key` and similarity dedupe; the legacy PreCompact `input_sha` check stays on |
| `APPEND_COMPACT_DEDUPE_WINDOW` | `900` | Seconds of recent captures kept for similarity comparison; older entries are pruned |
| `APPEND_COMPACT_CAPTURE_DEDUPE_SECONDS` | `120` | How recent a PreCompact capture must be before it can absorb an incoming prompt |
| `APPEND_COMPACT_CACHE` | enabled | Set to `0`/`false`/`off` to disable the compression memo (reads and writes) |
| `APPEND_COMPACT_CACHE_DIR` | `~/.cache/append-compactor` | Root for the compression memo and frozen shared blocks |
| `APPEND_COMPACT_CACHE_MAX_FILES` | `4096` | Memo entries kept; oldest are pruned first |
| `APPEND_COMPACT_SHARED_CHARS` | `9000` | Default cap for `--emit-shared-block` |
| `APPEND_COMPACT_DEBUG` | off | Also write errors to stderr |
| `APPEND_COMPACT_LOG` | `~/.cache/append-compactor/error.log` | Error log path (rotated at 5 MB) |
| `APPEND_COMPACT_JOB_LOG` | `~/.cache/append-compactor/jobs.log` | Per-job JSONL log path (rotated at 5 MB to `.1`); contains no turn text |
| `OPENROUTER_EXTRA_HEADERS` | unset | JSON object or `Key: Value, ...` list emitted under `metadata.openrouter_extra_headers` for a custom gateway; inert in direct setups |

---

## Performance and cold start

- **LLMLingua is lazy and heavy.** The first compressible paragraph triggers
  `torch` plus model loading (a measured cold start of ~40 s on one CPU machine,
  including the model load; gigabytes of RAM, and a one-time model download).
  With background compaction (the default) that cost is paid by a detached
  worker, not by the hook; only the bounded drain at an armed injection can wait
  for it. Consider `APPEND_COMPACT_LLM=0`, or pre-warm the model outside the
  hook.
- **Code and JSON layers are CPU-only and fast**; they are safe to run on every
  turn.
- **LLMLingua-2 is the only prose backend, and that is a measured choice.**
  [`bench/`](bench/README.md) compares it with three local Ollama models and
  LLMLingua→Ollama chains on real turns: no generative arm beat its 48.0 %
  shrink or its 100 % entity/number recall, every chain was smaller but worse
  and 5–30 s slower per turn, and generative output was not reproducible across
  models or runs even at `temperature=0` (`llama3.2:3b` still differed within a
  run). Background workers, not a generative backend, are what keep that cost
  off the hook path. The benchmark also surfaced a real defect: LLMLingua was
  splitting decimals (`7.592` → `7 592`); numeric literals are now masked before
  compression and restored byte-exact.
- The hook reads the transcript only on PreCompact, not on every prompt. It does
  not re-ingest the live transcript each turn; the current prompt is captured
  directly.
- **Background workers are shared machine-wide.** Two workers normally suffice;
  increase `APPEND_COMPACT_WORKERS` only on a machine with spare cores and
  memory. Torch memory is per worker, so each additional slot costs RAM even
  when idle.
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
- **Windows locking and workers.** `flock` is unavailable on Windows, so state
  writes are not serialized and background workers are not spawned there; the
  hook falls back to synchronous compression. The plugin targets macOS/Linux.
- **The queue is best-effort, not a job system.** A job's model compression can
  be lost to a crash or timeout; the turn itself is not lost, because the next
  armed injection materialises leftovers deterministically. `pending` churn is
  visible in `--status`.
- **Queued turn text is capped at 24,000 characters.** Entities are extracted
  from the full text before truncation, so paths and symbols beyond the cap
  still reach the ledger; only the compressed turn body is bounded.
- **Legacy session-less data is invisible to scoped injection.** Chunks and
  masters from ≤ 1.4 carry no `session`; if you have such state, scoped
  `--inject`/hooks skip them until `--reset` (or you merge them deliberately).
  This is intentional: mixing an unknown context's turns into every agent is
  worse than starting fresh.
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
