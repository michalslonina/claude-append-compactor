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
- an exact-fact ledger pins versions, commands, HTTP/exit codes, dates,
  durations, environment variables, identifiers and other literals, and every
  pinned fact is masked before LLMLingua and restored byte-exact;
- code blocks are minified for free (no model call), and large fenced code is
  reduced to an AST/brace skeleton whose first line points at the exact full
  block in the raw archive, so an agent can read it before editing;
- prose shrinks selectively while URLs, paths, inline code and identifiers are
  masked and restored byte-exact; each paragraph's first and last sentence stay
  verbatim, and instruction-heavy prose compresses less than repetitive logs;
- every captured turn is archived content-addressed under a short `Raw:` id and
  can be printed again with `--recall`;
- L1 chunks are immutable, so already-sent prefixes stay byte-identical across
  turns (the prerequisite for provider prefix caching);
- when six active chunks accumulate they are folded into an immutable L2
  master, and the injected view is capped at 9,000 characters with a pinned
  entity ledger at its head;
- older masters and chunks are replaced in the *new* injection by one-line L3
  index entries with entities, exact numeric literals, topic keywords and a
  verified file link; the agent can read the full block back with
  `--recall-l3` or directly from the archive;
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
                  │        ├─ newest L2 masters (full)
                  │        ├─ L3 index of older blocks (1 line each)
                  │        └─ newest active L1, this session only (full)
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
| Manual | `append-compact.py --recall <id>` | Raw archive (+ state with `--scope`) | — | Full original turn for a 12+ hex `Raw:` id; exit 1 if missing, tampered, or out of scope |
| Manual | `append-compact.py --recall-l3 <id>` | L3 archive (+ state with `--scope`) | — | Full L2 master or L1 chunk for a 12+ hex L3 link; re-hashes the file, exit 1 if missing, tampered, or out of scope |
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
| Code skeleton | fenced code ≥2,000 chars, on by default (`APPEND_COMPACT_CODE_SKELETON=0` disables) | Python `ast` keeps imports, decorators, signatures, first docstring line, simple constants and `...` bodies; brace languages keep depth-0 lines, declarations, method signatures and `// ...` markers. The skeleton starts with a comment naming the exact raw-archive file that holds the full block. Used only when strictly smaller than normal minification | CPU, milliseconds + one raw write | Normal minification (also when the raw archive is off or the block write fails) |
| Prose | natural-language paragraphs ≥160 chars | Selective LLMLingua-2 (`microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`). Exact facts, inline code, URLs, paths, identifiers and literals are masked, passed as `force_tokens`, then restored byte-exact. The first and last sentence of a ≥3-sentence paragraph are kept verbatim; the middle is compressed at a content-type rate | Model call (lazy, CPU) | Whitespace normalization; also forced when a mask is lost, entity extraction changes, output is not smaller, or the compressor raises |
| Error text | paragraphs matching `Traceback`, `fatal error`, `panic:`, `SyntaxError`, `TypeError`, `ValueError`, `ImportError`, `npm ERR!`, `error TS…`, `failed to`, `segmentation fault`, or a line starting `error:`/`exception:` | Never model-compressed; kept verbatim | none | — |
| Ledger | every chunk | Deterministic regex extraction of file paths, exported/top-level symbols and exact facts; sorted and deduped | CPU, microseconds | Empty ledger → `- (none)` |
| Raw archive | every accepted turn, plus every skeletonized code block | Full text stored content-addressed at `~/.cache/append-compactor/raw/<xx>/<sha256>.txt`, mode 600; chunk gets a short `Raw: <12hex>` pointer, and each skeleton points at its own block file | disk write, milliseconds | If disabled or the write fails, the turn/chunk is stored without a pointer and code is kept minified rather than skeletonized |
| L1 | every turn | Immutable `### Summary Chunk #N` block | none | — |
| L2 | exactly when 6 active L1 of one session exist | Immutable `### Consolidated Master Summary (L2) #M` covering the oldest 6 of that session: union ledger + per-chunk digests bounded to ~480 chars each | CPU | — |
| L3 index | every injection, on by default (`APPEND_COMPACT_L3=0` restores the 1.7 view) | Older L2 masters and active L1 chunks become one line each: up to 2 paths, 2 symbols, 2 facts, 2 exact numeric literals and 7 per-session TF-IDF keywords, plus a zstd `--ultra -22` archive link. The newest 3 masters / 2 chunks stay full. The stable ledger and full masters stay at the head; index lines follow them in the volatile part of the new injection | CPU, milliseconds + one zstd archive write per indexed block | Blocks that do not fit the index budget keep the explicit omission note; the full state is never rewritten |

A bare mention of the words “error” or “exception” no longer forces verbatim
handling; the marker must look like a log line or a concrete exception. This
lets prose such as “the error message was long” compress while a real traceback
stays byte-exact. The LLMLingua masks remain heuristics, not a parser.

### Goal anchors (opt-in)

Compaction keeps recent work but can lose the mission. Long-context evidence
shows that every model degrades as input grows, with information in the middle
of a long prompt hardest to use ([Context Rot](https://research.trychroma.com/context-rot),
[Lost in the Middle](https://arxiv.org/abs/2307.03172)); long-horizon agents
also drift from their original objective as context grows
([Arike et al., 2025](https://arxiv.org/abs/2505.02709)). The goal anchor is a
small, render-time block that restates the user's own mission words at the head
of the compacted context (placement is selectable; see the table below).

It is **off by default** (`APPEND_COMPACT_GOALS=1` enables it). The extractor is
deterministic and CPU-only (no model call). It recognises explicit
`Goal:`/`Vision:`/`Strategy:`/`Target:`/`Constraints:`/`Success criteria:`/
`Non-goals:` lines (including bullet sections), inline “our goal is …”
sentences, and hard-constraint sentences. Text is copied verbatim; only a
per-level budget fallback truncates it. Extraction runs only on user-authored
turns (`role` empty or `user`), so a tool or assistant summary cannot silently
establish a goal.

State is versioned and session-scoped:

```json
"goals": {
  "sessions": {
    "<session>": {
      "version": 3,
      "updated_at": "2026-11-01T10:00:00Z",
      "items": [{"id": "g4", "kind": "goal", "text": "Ship the JSON export.",
                 "status": "active", "source_chunks": [12],
                 "created_version": 3, "superseded_by": null}],
      "history": [{"version": 3, "event": "revise", "superseded": ["g1"]}]
    }
  }
}
```

An explicit `Goal:` (or `Vision:`/`Strategy:`/`Target:`) line in a later user
turn deliberately supersedes the previous active item of that kind and
increments the session version. Two explicit goal lines in the same prompt stay
parallel. Natural-language revisions (“instead”, “new goal”, “no longer”,
“from now on”, …) and “the goal is complete” are handled too. Old items remain
in `items` with `superseded`/`completed` status, and `history` records every
version.

The anchor is built inside `injection_payload()` from that state; stored L1
chunks and L2 masters are never edited:

```text
[GOAL ANCHOR v3 | levels=L1,L2,L3]
Use as the current mission; older summaries may be stale.
Goal (verbatim): "Ship the JSON export."
Constraints (verbatim):
- "Must not use pandas"
- "Keep memory under 200 MB"
Success criteria (verbatim): "10k rows in under 2 seconds"
```

Placement is empirical, not fixed. Five modes are available; all reserve the
anchor budget before history packing, so a full history cannot squeeze the
mission out:

| Mode | Where the anchor goes | Cache property | Measured attention |
| --- | --- | --- | --- |
| `beginning` (default) | before the ledger, masters, index and active blocks | anchor leads `content_blocks[0]`, the cached head; survives head churn | 1.00 recall at 2.2k and 3.8k chars |
| `after_head` | after ledger + newest masters, before index/active | cached while the head is stable; lost when a new master changes the head | 1.00 recall; no cache gain under head churn |
| `tail` | after all context (the original design) | prior turn's injection is cached as its own user message; the anchor is not reused | 1.00 at 2.2k, 0.75 goal recall at 3.8k |
| `per_level` | an anchor after each present level (L2, L3, L1); duplicate bodies deduped | L2 anchor can cache; L3/L1 anchors sit after changing material | 1.00 at 3.8k, 0.96 quality at 2.2k; costs 1.3–1.7x anchor bytes |
| `hybrid` | compact anchor at the beginning, plus a short `[GOAL REMINDER]` at the end only when the L3 drift detector fires | beginning anchor caches in stable-head mode; its 209-character compact anchor added no block under head churn | 1.00; reminder showed no additional benefit in the benchmark |

`per_level` is retained for block-local attention and `tail` for users who
prefer recency. The default changed from `tail` to `beginning` because the
longer-context benchmark showed both primacy and head churn favour it, and the
cache probe found no case where it was worse. Per-call metadata reports
version, active count, anchor levels and exact anchor characters.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APPEND_COMPACT_GOALS` | off | Enable extraction and anchors |
| `APPEND_COMPACT_GOALS_EXTRACT` | on | Disable extraction while still rendering an existing store |
| `APPEND_COMPACT_GOALS_PLACEMENT` | `beginning` | `beginning`, `after_head`, `tail`, `per_level`, or `hybrid` |
| `APPEND_COMPACT_GOALS_L1_CHARS` | `240` | L1-only anchor cap |
| `APPEND_COMPACT_GOALS_L2_CHARS` | `520` | L2 anchor cap |
| `APPEND_COMPACT_GOALS_L3_CHARS` | `760` | L3 anchor cap |
| `APPEND_COMPACT_GOALS_MAX_CHARS` | `900` | Hard total anchor cap |
| `APPEND_COMPACT_GOALS_HYBRID_CHARS` | `520` | Beginning-anchor cap in `hybrid` mode |
| `APPEND_COMPACT_GOALS_MAX_ITEMS` | `8` | Active items rendered per anchor |
| `APPEND_COMPACT_GOALS_L1/L2/L3` | on | Per-level kill switches |
| `APPEND_COMPACT_GOALS_SHARED` | off | Append a session-scoped anchor to `--emit-shared-block`; it changes only on a deliberate goal revision |

**Measured attention.** `bench/goal_drift.py` builds four synthetic sessions,
then asks a local `qwen2.5:7b` (temperature 0, JSON mode, seeds 42 and 43) to
restate the mission and choose the next action. The 2,200-character run uses
18 distraction turns; the longer run uses 120 distraction turns and a 6,000
character target, which rendered a 3,815-character baseline (the synthetic
corpus saturated the available full/index budget). Only placement differs
within the anchor conditions. All five controls passed on every run: gold
answer scores 1.0, empty and shuffled answers score 0, a verbatim goal in the
current prompt is recalled 1.0/1.0, and a no-goal prompt yields no invented
goal.

Longer context, n = 8 scenarios (seeds 42 and 43):

| Condition | Goal recall | Constraint recall | Success recall | Drift | Quality | Context chars | Anchor chars |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (no anchor) | 0.00 | 0.00 | 0.00 | 1.00 | 0.25 | 3,815 | 0 |
| all levels, tail | 0.75 | 0.92 | 1.00 | 0.25 | 0.92 | 4,164 | 349 |
| all levels, middle (control) | 0.88 | 1.00 | 1.00 | 0.12 | 0.97 | 4,166 | 351 |
| all levels, after head | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 4,164 | 349 |
| all levels, **beginning** | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 4,164 | 349 |
| all levels, per level | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 4,274 | 459 |
| hybrid (compact beginning) | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 4,164 | 349 |
| hybrid + forced reminder | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 4,236 | 421 |
| shuffled anchor (control) | 0.00 | 0.00 | 0.00 | 1.00 | 0.06 | 4,103 | 288 |

At the shorter 2,200-character context (n = 4, seed 42), every real placement
except per-level scored 1.00 on goal, constraint and success recall; per-level
scored 0.83 on constraints and 0.96 quality because its level caps truncate.
The longer run is the discriminating one: tail loses the mission in two of
eight scenario-runs and middle in one, while beginning and after-head are
complete. This is the primacy effect the placement change is meant to capture;
it is a synthetic stress test, not a production effect size. Decision accuracy
was 1.00 in every real-anchor condition and did not discriminate; the shuffled
control fell to 0.25.

**Measured prefix cache.** `bench/goal_cache_probe.py` sends synthetic two-turn
requests to DeepSeek V4.1-Flash and reads `prompt_cache_hit_tokens`. Two
deployment models are tested: `append` (Claude Code's UserPromptSubmit hook adds
each injection as a new user message) and `inplace` (a custom router replaces
one injected block after a stable system prefix). All controls passed: an
identical repeat hit 1,920 of 2,093 warm-up tokens, and a fully unique prompt
hit exactly 0.

| Model | Head | tail hit / miss | beginning hit / miss | after_head hit / miss | per_level hit / miss |
| --- | --- | ---: | ---: | ---: | ---: |
| append | stable | 1,920 / 1,997 | 1,920 / 1,981 | 1,920 / 1,993 | 1,920 / 2,091 |
| inplace | stable | 1,920 / 1,091 | 2,048 / 968 | 2,048 / 949 | 2,048 / 1,002 |
| inplace | new master each turn | 896 / 2,100 | 1,024 / 1,988 | 896 / 2,089 | 896 / 2,159 |

In the append model, placement does not change the injected anchor's cache
reuse at all: the previous turn's injection is already a complete cached user
message, so a new user message starts after it. In the in-place model, a
beginning anchor adds 128 cached tokens and removes 112–123 miss tokens even
when the head changes every turn; after-head and per-level lose that gain under
head churn, and the 209-character hybrid anchor was too small to add a block in
that case. At off-peak prices, beginning saves about **$0.000016 per turn
($0.0016 per 100 turns)** versus tail in the in-place/churn case, and is neutral
in the append case. That is a real but negligible saving; the default is chosen
because beginning was never worse and was better on the longer-context attention
test. Per-level gained the same 128 tokens in the stable-head case but paid
+197 characters for it.

**Measured cost.** Extraction is 0.14 ms per turn / 0 API calls in the harness.
The complete all-level anchor is 349 characters at both measured context sizes
(about 87 tokens at 4 chars/token, estimated). Per-level grows to 459 characters
long-context and 579 at 2,200 characters because two truncated level anchors
survive deduplication. Hybrid adds a 72-character reminder when drift fires.
The hard cap is 900 characters (225 tokens). Model latency is not attributed to
the anchor: each condition made two calls (restate, then decide).

At DeepSeek V4.1-Flash list prices (USD per 1M tokens; off-peak/peak
cache-miss $0.15/$0.30, cache-hit $0.003/$0.006; source
[Modellix](https://www.modellix.ai/blog/deepseek-api-pricing), retrieved
2026-11-01), the estimated **incremental** anchor cost per 100 turns is:
all-level beginning/tail/after-head $0.0013/$0.0026 (cache miss); per-level
$0.0017–$0.0022 / $0.0034–$0.0043; hybrid reminder +$0.0003/$0.0006. If the
beginning anchor is cached, its 100-turn cost is about $0.00003/$0.00005
instead. These are estimates, not measured spend (the local benchmark spent
$0). For scale, a 6k-token context over 100 turns is about $0.09/$0.18
cache-miss or $0.0018/$0.0036 cache-hit; 9k is $0.135/$0.27 cache-miss or
$0.0027/$0.0054 cache-hit.

**Recommendation.** Keep the feature opt-in until a workload is known to lose
its goal. When enabled, use the default single all-level `beginning` anchor:
it was complete at both context sizes, robust to head churn, and never worse on
cache. Choose `tail` only if the deployment shows a recency failure that
primacy does not fix, and `per_level` only when block-local repetition is worth
1.3–1.7x the anchor bytes. `hybrid` is available for long-running sessions
where the L3 drift detector should trigger a recency reminder, but the benchmark
showed no additional benefit over beginning alone. L1 alone remains a
deliberate low-budget mode that can truncate success criteria.

### Exact facts (1.7.0)

The fact ledger recognises the literals that a summary must not blur:

| Kind | Examples |
| --- | --- |
| URL / path / file:line | `https://api.example.com/v2`, `src/app.ts:12`, `target/release/payments` |
| Package / version / range | `zod@3.23.8`, `v4.2.0`, `PostgreSQL 16.3`, `>=2.4.0 <3.0.0` |
| HTTP / exit / error code | `HTTP 503`, `exit code 137`, `EADDRINUSE` |
| Command / flag / setting | `cargo build --release`, `retry.backoff.enabled` |
| Environment variable | `STRIPE_WEBHOOK_SECRET` |
| Identifier | `verifyWebhookSignature`, `verify_chain`, `amount_cents` |
| Date / timestamp / duration | `2026-12-01`, `2026-12-01T10:14:22Z`, `250 ms` |
| Header / IP:port / SHA / CVE | `X-Request-Id`, `10.0.4.17:8443`, `c0ffee1`, `CVE-2026-12345` |

Each captured turn is capped per kind, so a code-heavy turn cannot crowd out
paths. Facts are masked before LLMLingua and restored byte-exact, then pinned
as `fact:` entries in the chunk and global ledgers. Masking is what makes the
ledger safe: without it, LLMLingua could drop a fact and the entity-equality
guard would reject its output, forcing an uncompressed fallback.

The global `**Entity Ledger (pinned):**` head groups entries by kind
(`fact:`, `path:`, `symbol:`) on one line each, so a long session fits far more
entries in the same 2,000-character budget. If even that overflows, it prints an
explicit truncation marker rather than silently dropping entries.

### Boundary protection and content-type rates

For a prose paragraph with at least three sentences, the first and last
sentence are kept byte-for-byte and only the middle is sent to LLMLingua.
Opening context and closing instructions therefore survive intact. The middle's
compression rate is selected by content type:

| Text looks like | Rate | Effect |
| --- | --- | --- |
| two or more log lines, or >18% digits | `0.25` | prunes repetitive log prose harder |
| two or more instruction words (`must`, `never`, `ensure`, `run`, …) | `0.55` | keeps more instruction wording |
| everything else | `0.40` | default |

Exact facts are masked at every rate, so facts survive even when the
surrounding prose is pruned aggressively.

### Raw archive and `--recall`

Every accepted turn is written once under its SHA-256 and never rewritten.
Chunks carry only a 12-hex pointer, so the state file stays bounded:

```text
### Summary Chunk #7
Source: user_prompt
Raw: 28063c97e2a3
**Entity Ledger:**
...
```

```sh
append-compact.py --recall 28063c97e2a3
append-compact.py --recall 28063c97e2a3 --scope <session-id>
```

`--recall` accepts a 12–64 character hex prefix, re-hashes the file before
printing it, and refuses a file whose content no longer matches its id. With
`--scope`, the id must also belong to a chunk in that session. Archive writes
are mode `600`, and the archive is pruned oldest-first beyond
`APPEND_COMPACT_RAW_MAX_FILES` (default 4,096).

A skeletonized block carries its own pointer as the first line of the fence:

```python
# [skeleton: 214 lines elided; full code: ~/.cache/append-compactor/raw/41/41fba64b….txt; read before editing]
import os
...
def reconcile_batch(...):
    ...
```

The target file is the exact fenced body (no fence lines), so it can be read
directly or recovered with `--recall <id>`. Brace-language skeletons use `//`
instead of `#`. Skeletonization requires the raw archive: with
`APPEND_COMPACT_RAW=0`, or when the block write fails, the block is left as
minified code rather than losing its body without a reference.

### L3 index (1.9.0, on by default)

The injected view used to be a strict recency window: once a master fell out of
the budget, the agent could no longer see that it existed. L3 replaces each
older L2 master (and each older active L1 chunk) with one line in the *new*
injection:

```text
- L2 #34 [chunks 199–204] bench/real_log_check.py, hooks/scripts/append-compact.py, _env_bool | 10k, raw.strip, cap, tool_use_error, default:, name: | 81/81243374626a8f94.txt.zst
```

The line carries up to two paths, two symbols, two ledger facts, two exact
numeric literals (`0.25`, `250ms`, `v3.7.1`) and seven per-session TF-IDF
keywords. Keywords are what make the line useful when a block has no path or
symbol: entity-only lines retrieved the right old block 13.6% / 33.3% of the
time on the two real sessions below, while entity + keyword lines retrieved it
77.3% / 66.7%.

Every linked block is stored as a **zstd `--ultra -22`** frame
(`<xx>/<sha256-prefix-16>.txt.zst`, mode 600). The index header names the
archive root and the two scan commands, so an agent can search without writing
a decompressed copy to disk:

```sh
ugrep -z 'pattern' ~/.cache/append-compactor/l3/81/81243374626a8f94.txt.zst
zstdgrep 'pattern' ~/.cache/append-compactor/l3/81/81243374626a8f94.txt.zst
zstd -dc ~/.cache/append-compactor/l3/81/81243374626a8f94.txt.zst
```

`ugrep -z` decompresses gzip/bzip2/xz/lzma/zstd/brotli/lz4 streams in place;
`zstdgrep` ships with zstd and works without ugrep installed. `--recall-l3` also
reads both compressed and legacy plaintext `.txt` files, verifies the full
SHA-256 of the decompressed bytes, and refuses a mismatch.

Defaults:

| Knob | Default | Meaning |
| --- | --- | --- |
| `APPEND_COMPACT_L3` | enabled | `0` restores the 1.7 recency-only view |
| `APPEND_COMPACT_L3_TARGET_CHARS` | `6400` | Cap for the L3 view (the hard hook cap remains 9,000) |
| `APPEND_COMPACT_L3_KEEP_RECENT_MASTERS` | `3` | Newest masters left full |
| `APPEND_COMPACT_L3_KEEP_RECENT_CHUNKS` | `2` | Newest active chunks left full |
| `APPEND_COMPACT_L3_INDEX_BUDGET_CHARS` | `4000` | Space reserved for index lines before full blocks |
| `APPEND_COMPACT_L3_DIR` | `~/.cache/append-compactor/l3` | Content-addressed archive root |
| `APPEND_COMPACT_L3_COMPRESS` | enabled | Store blocks as `zstd --ultra -22`; `0` writes plaintext `.txt` (legacy files remain readable) |
| `APPEND_COMPACT_L3_ZSTD_LEVEL` | `22` | zstd level, clamped to 1–22; levels >19 use `--ultra` |
| `APPEND_COMPACT_L3_MAX_FILES` | `4096` | Archive files kept; oldest-first prune |

The index sits in the **volatile part** of the new injection, after the
byte-stable ledger and the newest full masters (the newest active chunks come
last). The stored L1/L2 blocks, prior messages and system prompt are never
rewritten, so the provider's prefix cache still sees the same prefix and only
the new turn's tail changes. Full blocks are archived as zstd `--ultra -22`
frames under `<xx>/<sha256-prefix-16>.txt.zst`, mode 600. Every link is verified
against the full SHA-256 before it is written; an existing file with a different
hash is reported as a collision rather than reused. If `zstd` is not on `PATH`,
the archive falls back to plaintext `.txt` and the index links say so.
`--recall-l3 <12+ hex>` prints the full block (and `--scope <session-id>`
enforces that the block belongs to that session).

Drift-aware aggressiveness uses the local token overlap between the current
prompt and the last four substantive user chunks. A score below `0.12` (with at
least 8 tokens compared) means the conversation moved topics; L3 then keeps only
one full master and one full chunk and indexes the rest. Short follow-up
prompts are ignored.

Measured on the real snapshots (read-only copies under `/tmp`):

| Session | Injection bytes | Blocks represented | Entity values | Numeric literals | Archive links |
| --- | --- | --- | --- | --- | --- |
| `e0910473` | 6,655 → **5,146** | 4/15 → **15/15** | 110 → 103 | 7 → 9 | 13/13 exact |
| `fadf2bc6` | 8,753 → **6,314** | 3/23 → **23/23** | 74 → 71 | 1 → 37 | 22/22 exact |

The ledger prefix is byte-identical in both views, every index line is after the
full-master head, and no stored block is mutated. The remaining entity delta is
the intended trade-off: a value that does not fit on an index line is still in
the linked block, one `--recall-l3` away. `bench/l3_real_check.py --snapshot
/tmp/ac-reallog` prints those numbers and exits 2 if any control fails.

Held-one-sentence-out retrieval (`bench/l3_retrieval.py`): the held-out sentence
is the query, and the benchmark asks whether the block it came from ranks first.
Controls require the index line to retrieve its own block, absent tokens to
score exactly zero, random/scrambled signals not to retrieve, and every archive
link to round-trip byte-identically.

| Dataset | Baseline top-1 / coverage / chars | L3 top-1 / coverage / chars | Index-only top-1 | Entity-only top-1 |
| --- | --- | --- | --- | --- |
| real `fadf2bc6` | 4.5% / 3-of-23 / 8,753 | **77.3% / 23-of-23 / 6,319** | 77.3% | 13.6% |
| real `e0910473` | 6.7% / 4-of-15 / 6,655 | **66.7% / 15-of-15 / 5,151** | 73.3% | 33.3% |
| synthetic 10-block | 10.0% / 6-of-10 / 7,608 | 40.0% / 10-of-10 / 3,558 | 50.0% | 50.0% |

The synthetic corpus deliberately includes prose-only turns with no path or
symbol, so an index line has to earn its keep from keywords. The index-only
variant is the retrieval upper bound; the actual L3 view also keeps the newest
masters full, which is better for continuity but can dilute a held-out-sentence
query in a small repetitive corpus.

`bench/recall.py` retains exact facts at 100% with `fact+l3` (the default
combination). L3 alone, with the fact ledger explicitly disabled, falls to
34.1% because the index cannot carry every exact literal; the fact ledger is
what pins them, so keep it on (the default) when using L3.

DeepSeek prefix-cache probe (`bench/cache_probe.py`, synthetic prompts only):
a changing L3 block at the start of the **new turn** hit 92.8% of the current
prompt / 98.6% of the prior prompt on `deepseek-flash`, and 94.2% / 99.9% on
`deepseek-v4-pro`. Rewriting the first historical turn, the middle, the last
turn, or a block near the system start produced 0% on `deepseek-v4-pro`. That is
why L3 only changes the new injection and why it sits after the stable head. A
hook cannot delete an injection already in the live context window; L3 shrinks
the next injection, not already-sent history.

### Measured effect

`bench/recall.py` builds a 21-turn synthetic corpus (88 planted facts across 26
classes, prose, Python/TypeScript/Go fences, a traceback and a log turn) and
runs the real `append_turn` → L1/L2 → `injection_payload` pipeline. It enforces
known-positive/known-negative controls and exits nonzero if any control fails.
With LLMLingua enabled:

| Arm | Chunk shrink | Facts in injection | Boundary sentences verbatim | Instruction verbs |
| --- | --- | --- | --- | --- |
| baseline (all 1.7 features off) | 32.7% | 73.9% | 0% | 36% |
| exact facts | 32.5% | **100%** | 0% | 36% |
| exact facts + rate control | 31.6% | 100% | 0% | 43% |
| exact facts + sentence protection | 23.0% | 100% | **100%** | 64% |
| exact facts + code skeleton | 45.8% | 100% | 0% | 36% |
| all of the above | 35.9% | 100% | 100% | 64% |

The numbers are a regression harness, not a promise about a particular
conversation. They are why the defaults are: facts, sentence protection, rate
control, raw archive and code skeletonization **on**. Code-skeleton arms turn
the raw archive on too, because the placeholder is what makes the elided body
recoverable. `bench/recall.py --llm off` exercises the deterministic path and
prints a note that the sentence/rate/code arms only change the LLMLingua path.

### Session handoff (1.10.0, on by default)

Every injection already carries the ledger, the newest L2 masters and the
newest live chunks. The handoff is one more section, **added to** that block
rather than replacing any of it:

```
## Session handoff (key facts)

### Open questions / pending asks
- please make the export worker stream rows instead of buffering (turn 1)

### Errors and fixes
- Fix: close the file handle in a finally block. (turn 5)
- OSError: [Errno 24] too many open files (turn 3)

### Files, symbols, exact references
- line 64 (turn 3)

### Critical exact values
- Retry budget is 1200 s. (turn 5)
- I changed stream_export() to flush every 500 rows and set the pool to 8 shards. (turn 2)
```

It is built with regexes from the raw captured turns at commit time -- no
generator call, no extra latency on the hook path, and nothing invented. Each
bullet carries the turn it came from, so a reader can check it. It is rendered
into the **volatile** half of the block, after the cached head, so the head
stays byte-identical from turn to turn (see the KV-caching section). Items
already pinned by the ledger are dropped rather than paid for twice.

Why these four sections and no others: a benchmark on branch
`experiment/handoff-template-bench` measured every section against a full
transcript control on six synthetic sessions. **Open questions / pending asks**
carries the open-loop result on its own (remove it and open-loop recall falls
from 1.00 to 0.17). **Critical exact values** carries exact-value and fact
recall. Files and errors are the cheapest way to keep a path and a traceback
reachable. A success-criteria section changed task success by exactly 0.000
across 12 paired cells, so it is not included. The full evidence, including the
rendering bug that voided the first ablation run, is in
`bench/HANDOFF_TEMPLATE_FINDINGS.md`.

The handoff is skipped below a 1,200-character injection budget: at that size
the recent turns it summarises are still in the injection themselves.

### Journal tags (opt-in)

`APPEND_COMPACT_JOURNAL=1` adds a line to the injection asking the model to
drop tagged lines as it works:

| Tag | Meaning |
| --- | --- |
| `※v <fact>` | a value or fact, quoted from the session |
| `※e <error>` | an error line |
| `※f <fix>` | the fix applied |
| `※t <id> <task>` | open a task |
| `※d <id>` | close that task |

The marker is `※` (U+203B) and the entry runs to end of line. A `v`/`e`/`f`
body is kept **only if it appears contiguously in the turn it cites** -- the
grounding check is what separates a copied fact from a plausible invention, and
in the benchmark an ungrounded model invented 6 of 21 tags while a grounded one
kept 14 of 14 (with the source-turn form `※v 12 …`, all anchors valid, at 23%
fewer tags than free-form). Ungrounded tags are dropped, never rendered.
`※t`/`※d` are pure bookkeeping: closure is a replay over the immutable chunks,
so a closed task needs no state rewrite.

Off by default. The grounding result prices the mechanism, not the usefulness
of the task tags, and several `t`/`d` pairs in the probe simply echoed an
instruction and closed it immediately.

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
      "raw_id": "28063c97e2a3...",
      "entities": ["path: src/app.ts", "symbol: buildThing", "fact: HTTP 202"],
      "compressed_text": "Please update src/app.ts ...",
      "block": "### Summary Chunk #1\nSource: user_prompt\nRaw: 28063c97e2a3\n**Entity Ledger:**\n- path: src/app.ts\n- symbol: buildThing\n- fact: HTTP 202\n**Turn:**\n...",
      "handoff": {
        "critical_exact_values": [["Set the pool to 8 shards.", 1]],
        "files_symbols_references": [["src/app.ts", 1], ["buildThing", 1]]
      },
      "tags": [
        {"kind": "v", "body": "HTTP 202", "turn": 1, "source": 1, "grounded": true},
        {"kind": "t", "id": "1", "body": "apply the correction", "turn": 1, "grounded": true}
      ],
      "stats": {
        "code_blocks": 0,
        "json_blocks": 0,
        "code_skeleton_blocks": 0,
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
      "raw_id": "28063c97e2a3...",
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
stays `1`, and older readers ignore them. Four compatibility rules apply:

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
Please update src/app.ts and ./lib/util.py on port 8080.
export function buildThing() {}
export { helper as renamed };
def handler():
    return 1
pub fn run_task() {}
```

Ledger (paths, symbols, then exact facts; each group sorted):

```text
**Entity Ledger:**
- path: ./lib/util.py
- path: src/app.ts
- symbol: buildThing
- symbol: handler
- symbol: renamed
- symbol: run_task
- fact: port 8080
```

Extraction covers common source extensions (`.ts`, `.tsx`, `.py`, `.go`, `.rs`,
`.java`, `.kt`, …), `./` and absolute and Windows paths, plus `export
function/class/const`, `export { a as b }`, Python `def`/`class`, Go
`func`/`type`, Rust `pub fn`/`struct`/`enum`/`trait`, and JVM
`class`/`interface`/`object`. The fact layer adds the literals listed in
[Exact facts](#exact-facts-170), skipping a value already pinned as a path or
symbol. The ledger is deterministic: the same input always produces the same
sorted entries.

The per-chunk ledger uses one entry per line. The injected global
`**Entity Ledger (pinned):**` head groups each kind on a single line
(`- fact: v4.2.0; HTTP 202; …`) to fit more entries in its 2,000-character
budget; it is deterministic and append-only in the same way.

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
- **The L3 view is not a rewrite.** It is a different rendering of the *new*
  injection. The stored chunks and masters are unchanged, prior messages and the
  system prompt are untouched, and the index lines sit in the volatile tail
  after the stable ledger and full masters. `APPEND_COMPACT_L3=0` restores the
  old view.
- **Block text has no timestamps.** Generated metadata may vary; the injected
  prefix text does not.
- **Deterministic ordering.** Entity entries are sorted; selected L2 masters
  stay chronological within the window, then active L1 in chunk order.
- **Bounded injection.** `APPEND_COMPACT_MAX_INJECT_CHARS` (default **9,000**,
  clamped to 9,800) caps the payload well under Claude Code's 10,000-character
  hook-output wall. A deterministic `**Entity Ledger (pinned):**` head keeps
  captured paths, symbols and exact facts alive (2,000-char budget, grouped by
  kind). The newest L2 masters and newest active L1 are kept full; older
  material becomes one L3 index line with a verified file link. If even that
  does not fit, it is dropped with an explicit notice.

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
| `APPEND_COMPACT_FACT_LEDGER` | enabled | Mask and pin exact facts (versions, commands, codes, dates, durations, identifiers, …). Set `0` to use only the original path/symbol masking |
| `APPEND_COMPACT_SENTENCE_PROTECT` | enabled | Keep the first and last sentence of each ≥3-sentence prose paragraph byte-exact; set `0` to compress the whole paragraph |
| `APPEND_COMPACT_RATE_CONTROL` | enabled | Choose the LLMLingua rate by content type (`0.25` logs, `0.55` instructions, `0.4` default); set `0` to force `0.4` |
| `APPEND_COMPACT_RAW` | enabled | Archive every accepted turn (and every skeletonized block) content-addressed for `--recall`; set `0` to disable writes, pointers and code skeletonization |
| `APPEND_COMPACT_RAW_DIR` | `~/.cache/append-compactor/raw` | Raw archive root |
| `APPEND_COMPACT_RAW_MAX_FILES` | `4096` | Archive files kept; oldest-first prune on write |
| `APPEND_COMPACT_CODE_SKELETON` | enabled | AST/brace skeletonization for fenced code ≥2,000 chars, with a first-line pointer to the exact raw block. Set `0` to keep minified code; requires `APPEND_COMPACT_RAW` |
| `APPEND_COMPACT_L3` | enabled | Replace older blocks in the new injection with one-line index entries plus a verified archive link; `0` restores the 1.7 recency-only view |
| `APPEND_COMPACT_L3_TARGET_CHARS` | `6400` | Cap for the L3 view; the hook's hard cap (`APPEND_COMPACT_MAX_INJECT_CHARS`, 9,000) still applies |
| `APPEND_COMPACT_L3_KEEP_RECENT_MASTERS` | `3` | Newest full L2 masters retained in the L3 view |
| `APPEND_COMPACT_L3_KEEP_RECENT_CHUNKS` | `2` | Newest full active L1 chunks retained in the L3 view |
| `APPEND_COMPACT_L3_INDEX_BUDGET_CHARS` | `4000` | Space reserved for L3 index lines before allocating full-block budget |
| `APPEND_COMPACT_L3_DIR` | `~/.cache/append-compactor/l3` | Content-addressed L3 archive root |
| `APPEND_COMPACT_L3_COMPRESS` | enabled | Store blocks as zstd `--ultra -22`; `0` writes plaintext `.txt` |
| `APPEND_COMPACT_L3_ZSTD_LEVEL` | `22` | zstd level clamped to 1–22; levels >19 pass `--ultra` |
| `APPEND_COMPACT_L3_MAX_FILES` | `4096` | L3 archive files kept; oldest-first prune on write |
| `APPEND_COMPACT_L3_DRIFT` | enabled | Enable drift-aware aggressiveness (fewer full blocks when the topic changes) |
| `APPEND_COMPACT_L3_DRIFT_WINDOW` | `4` | Recent substantive prompts compared for drift |
| `APPEND_COMPACT_L3_DRIFT_MIN_TOKENS` | `8` | Shorter prompts are treated as follow-ups and skipped |
| `APPEND_COMPACT_L3_DRIFT_THRESHOLD` | `0.12` | Token-overlap score below which drift is declared |
| `APPEND_COMPACT_L3_DRIFT_KEEP_RECENT_MASTERS` | `1` | Full masters retained while drifting |
| `APPEND_COMPACT_L3_DRIFT_KEEP_RECENT_CHUNKS` | `1` | Full active chunks retained while drifting |
| `APPEND_COMPACT_SHARED_CHARS` | `9000` | Default cap for `--emit-shared-block` |
| `APPEND_COMPACT_GOALS` | off | Enable deterministic goal extraction and render-time mission anchors |
| `APPEND_COMPACT_GOALS_PLACEMENT` | `beginning` | `beginning`, `after_head`, `tail`, `per_level`, or `hybrid` |
| `APPEND_COMPACT_GOALS_MAX_CHARS` | `900` | Hard cap on total anchor characters per injection |
| `APPEND_COMPACT_GOALS_HYBRID_CHARS` | `520` | Beginning-anchor cap in `hybrid` mode |
| `APPEND_COMPACT_GOALS_L1_CHARS` / `_L2_CHARS` / `_L3_CHARS` | `240` / `520` / `760` | Per-level anchor caps |
| `APPEND_COMPACT_GOALS_L1` / `_L2` / `_L3` | on | Per-level kill switches while goals are enabled |
| `APPEND_COMPACT_GOALS_SHARED` | off | Include a session-scoped anchor in emitted shared blocks |
| `APPEND_COMPACT_HANDOFF` | enabled | Add the deterministic four-section handoff (`open asks`, `errors`, `files/symbols`, `exact values`) to every injection; `0` disables it |
| `APPEND_COMPACT_HANDOFF_CHARS` | `720` | Character cap for the handoff section (skipped entirely below a 1,200-char injection budget) |
| `APPEND_COMPACT_HANDOFF_PER_SECTION` | per-section defaults | Override the per-section item cap (`3` asks/errors, `4` files/values) for every section at once |
| `APPEND_COMPACT_HANDOFF_RECENT_CHUNKS` | `8` | How many recent chunks the handoff is built from |
| `APPEND_COMPACT_JOURNAL` | off | Inject the `※` tag protocol and render open `※t` tasks; ungrounded fact tags are always dropped |
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
  from the full text before truncation, so paths, symbols and facts beyond the
  cap still reach the ledger; only the compressed turn body is bounded.
- **Legacy session-less data is invisible to scoped injection.** Chunks and
  masters from ≤ 1.4 carry no `session`; if you have such state, scoped
  `--inject`/hooks skip them until `--reset` (or you merge them deliberately).
  This is intentional: mixing an unknown context's turns into every agent is
  worse than starting fresh.
- **Caching is not guaranteed.** See above: the plugin provides stable prefixes,
  not hit-rate promises.
- **Masks are heuristics.** A compressor that loses a placeholder, changes
  entity extraction, or fails to shrink the text triggers the fallback. Fact
  patterns likewise recognise common shapes, not arbitrary identifiers.
- **Boundary protection needs three sentences.** A paragraph with fewer, or one
  whose middle is under 40 characters, is compressed as a whole; its boundaries
  are not guaranteed byte-exact.
- **The raw archive is plaintext on disk.** It is written mode `600` under the
  cache directory and pruned to a bounded file count, but anyone who can read
  the cache can read the archived turns. `APPEND_COMPACT_RAW=0` disables it.
  Pruning is oldest-first, so in a very long-running swarm an old `Raw:` id or
  skeleton pointer can outlive its file; raise `APPEND_COMPACT_RAW_MAX_FILES`
  if that matters, and expect `--recall` to refuse a missing id.
- **Skeletonized code is not the code.** Bodies, local logic and comments are
  replaced by `...`/`// ...`; the placeholder points at the exact block, so read
  it before editing. `APPEND_COMPACT_CODE_SKELETON=0` restores minified code.
- **An L3 index line is a pointer, not the block.** It carries entities, exact
  numbers and keywords, not the full prose or code. The linked file is the
  fallback; if the L3 archive is cleared or pruned past
  `APPEND_COMPACT_L3_MAX_FILES`, an old link stops resolving even though the
  line remains in the conversation. Raise that cap or keep the cache directory
  if old links matter.
- **L3 archive files are zstd frames, not text.** A plain `cat` shows binary.
  Read with `zstd -dc FILE`, or scan without writing a copy with
  `ugrep -z PATTERN FILE` / `zstdgrep PATTERN FILE`. `ugrep` must be installed
  separately; `zstdgrep` ships with zstd. If `zstd` itself is missing, new
  archives fall back to plaintext `.txt` and the links say so. Compression is
  at-rest size reduction, not encryption; the files are mode `600` and rely on
  filesystem permissions.
- **L3 assumes the fact ledger.** With `APPEND_COMPACT_FACT_LEDGER=0`, the index
  has fewer exact literals to carry and the `l3`-only recall arm drops to 34.1%;
  the default combination (`fact+l3`) is 100% in the same harness. Disable L3
  as well if you deliberately run without the fact ledger.
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

### Recall harness

```sh
# deterministic path (no model download)
python3 bench/recall.py --llm off --arms baseline,l3,fact,fact+l3,all

# real LLMLingua-2 comparison; use mnice so the model load yields the terminal
mnice python3 bench/recall.py --llm auto --arms baseline,fact,sentence,rate,code,all

# L3 retrieval and real-snapshot retention (each script has its own controls)
python3 bench/l3_retrieval.py --synthetic --real
python3 bench/l3_real_check.py --snapshot /tmp/ac-reallog
```

The harness builds a fixed 21-turn corpus with 88 planted facts, runs the real
append/L1/L2/injection pipeline, and reports survival in the compressed chunk,
in stored state, and in the bounded injection payload. It also reports
paragraph-boundary survival and instruction-verb retention, plus raw-archive
and skeleton-placeholder round-trips (each placeholder path is read back and
its SHA-256 checked against the filename).

It refuses to print arm numbers unless its controls pass: every planted fact
must be found in the source, unplanted facts must stay absent, damaged paths
must not match, entity extraction must find a known path/symbol and none in
plain prose, six chunks must make one L2 master while five must not, and a
one-chunk state must inject while an empty state must not. On any failure it
exits 2.

---

## License

MIT. See [LICENSE](LICENSE).
