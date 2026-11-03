#!/usr/bin/env python3
"""Controlled experiment: structured handoff template vs append-compactor injection.

The experiment builds six synthetic long coding sessions with decisive facts
placed early and in the middle, then compares eight context arms at depths
24/60/96 with a local Ollama model (qwen2.5:7b by default).

Everything is synthetic.  The plugin is loaded with importlib; no repository
file is modified and no live state is touched.  Controls run before any
measurement and abort with exit code 2 if any control fails.

Usage:
    ~/.local/bin/mnice .venv/bin/python bench/template_compaction.py --pilot
    ~/.local/bin/mnice .venv/bin/python bench/template_compaction.py
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = ROOT / "hooks" / "scripts" / "append-compact.py"
MODEL = os.environ.get("TC_BENCH_MODEL", "qwen2.5:7b")
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
PROVIDER = os.environ.get("TC_BENCH_PROVIDER", "ollama").strip().lower()
DEEPSEEK_URL = os.environ.get("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_ENV = Path.home() / ".config/model-router/env"
REASONING_HEADROOM = int(os.environ.get("TC_BENCH_REASONING_HEADROOM", "3000"))
SEED = int(os.environ.get("TC_BENCH_SEED", "42"))
# Minimum decision/correction keyword coverage a generated template must carry.
# 0.0 means "any mention at all"; a negative value disables the check.
COVERAGE_MIN = float(os.environ.get("TC_BENCH_COVERAGE_MIN", "0.0"))
NUM_CTX = int(os.environ.get("TC_BENCH_NUM_CTX", "24576"))
OUT_PATH = Path(
    os.environ.get("TC_BENCH_OUT", str(ROOT / "bench" / "template_compaction_results.json"))
)
SUMMARY_PATH = Path(
    os.environ.get("TC_BENCH_SUMMARY", str(ROOT / "bench" / "template_compaction_summary.md"))
)
# Template generation is deterministic (temperature 0, fixed seed), so caching it
# lets a re-run after a control or scoring change skip the multi-hour generation
# phase without changing what the templates say.  Keyed by model + seed + the
# depths requested, so a different model never reuses another's templates.
TEMPLATE_CACHE_PATH = Path(
    os.environ.get(
        "TC_BENCH_TEMPLATE_CACHE",
        str(ROOT / "bench" / "template_compaction_templates_cache.json"),
    )
)
DEPTHS = (24, 60, 96)
ARMS = (
    "full",
    "current",
    "current_goals",
    "current_goals_template",
    "template_minimal",
    # Section ablations: each variant drops or adds one section so the set is
    # earned by measurement rather than chosen by argument.
    "template_minimal_anchors",
    "template_minimal_minus_files",
    "template_minimal_minus_errors",
    "template_minimal_minus_open",
    "template_minimal_minus_values",
    "template_minimal_plus_success",
    # No model in the loop: regex extraction, ledger dedupe, per-section caps.
    "template_det",
    "template_only",
    "template_ledger",
    "shuffled_template",
    "headers_only",
)
ARM_RANK = ARMS + ("template_regen_only",)

# DeepSeek V4.1-Flash public list price, USD per 1M tokens (same reference
# used by bench/goal_drift.py).  All figures are estimates; local spend $0.
PRICE_MISS_LOW = 0.15
PRICE_MISS_HIGH = 0.30
PRICE_HIT_LOW = 0.003
PRICE_HIT_HIGH = 0.006

# ---------------------------------------------------------------------------
# Synthetic scenarios
# ---------------------------------------------------------------------------
#
# Every scenario places the decisive facts at the same turns so that depth
# only adds later distractions: goal (1), decision (3), exact values (6),
# artifact (8), error/fix (12), correction (16), open ask (19), ack (20).
# Turns 22..depth are distractions.  Nothing here is real session text.

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "csv_export",
        "goal_prompt": (
            "Goal: Ship the billing CSV export.\n"
            "Vision: a boring, reliable download that finance can trust.\n"
            "Constraints:\n"
            "- Must not use pandas\n"
            "- Keep memory under 200 MB\n"
            "- Preserve the public API\n"
            "Success criteria:\n"
            "- 10k rows exported in under 2 seconds\n"
            "- 10M rows complete under 5 minutes\n"
            "- Non-goal: no new dashboard UI"
        ),
        "goal": "Ship the billing CSV export.",
        "constraints": [
            "Must not use pandas",
            "Keep memory under 200 MB",
            "Preserve the public API",
            "Non-goal: no new dashboard UI",
        ],
        "success_criteria": [
            "10k rows exported in under 2 seconds",
            "10M rows complete under 5 minutes",
        ],
        "decision": "Use the Python standard-library csv reader with a bounded heap for top-N mode",
        "decision_reason": "pandas is forbidden and a bounded heap keeps memory under 200 MB",
        "correction": (
            "Do not use the bounded heap for the full export: top-N mode keeps the heap, "
            "but the full export must stream rows into temporary shards and merge them in order"
        ),
        "artifact_path": "src/export/csv_writer.py",
        "artifact_symbol": "stream_export()",
        "artifact_line": "line 218",
        "error": "OSError: [Errno 24] Too many open files",
        "fix": "reuse a pool of at most 8 file handles and close on rotation",
        "open_ask": "Add a --dry-run flag before finishing.",
        "next_step": (
            "Reuse a pool of 8 file handles while streaming shards inside "
            "stream_export() at src/export/csv_writer.py line 218, keeping the stdlib path"
        ),
        "exact_values": [
            "metrics port 8443",
            "runtime version 2.7.1",
            "coverage threshold 0.85",
            "command python -m exporter.csv --shards 8",
        ],
        "keywords": {
            "goal": [["csv", "export"], ["billing"]],
            "constraints": [
                ["pandas"],
                ["200", "mb"],
                ["public", "api"],
                ["dashboard", "ui"],
            ],
            "success": [["10k"], ["2", "second"], ["10m"], ["5", "minute"]],
            "decision": [["standard", "library"], ["bounded", "heap"], ["pandas"]],
            "correction": [["full", "export"], ["temporary", "shards"], ["top", "mode"]],
            "artifact": [
                ["src/export/csv_writer.py"],
                ["stream_export"],
                ["line 218"],
            ],
            "error": [
                ["errno", "24"],
                ["too", "many", "open", "files"],
                ["pool", "8", "file", "handles"],
            ],
            "open_loop": [["dry-run"], ["before", "finishing"]],
            "next_step": [["pool", "8", "file", "handles"], ["stream_export"], ["line 218"]],
            "values": [["8443"], ["2.7.1"], ["0.85"], ["shards", "8"]],
        },
        # Every option is an equally plausible next step; only the values stated
        # in the transcript (handle-pool size, whether the heap still covers the
        # full export) separate them.  Options stay near-equal in length so
        # "pick the longest, most detailed one" is not a usable cue.
        "probe": (
            "The export job just failed under load. Choose the next implementation step.\n"
            "A) Reuse a pool of 8 file handles while streaming shards in stream_export() at "
            "src/export/csv_writer.py line 218, and keep the bounded heap for the full "
            "export.\n"
            "B) Reuse a pool of 12 file handles while streaming shards in stream_export() at "
            "src/export/csv_writer.py line 218, and drop the bounded heap for the full "
            "export.\n"
            "C) Reuse a pool of 8 file handles while streaming shards in stream_export() at "
            "src/export/csv_writer.py line 218, and drop the bounded heap for the full "
            "export.\n"
            "D) Reuse a pool of 12 file handles while streaming shards in stream_export() at "
            "src/export/csv_writer.py line 218, and keep the bounded heap for the full "
            "export."
        ),
        "correct": "C",
        "events": {
            1: ("user", "goal_prompt"),
            3: (
                "assistant",
                "Decision: use the Python standard-library csv reader with a bounded heap "
                "for top-N mode, rather than loading the file with pandas. Rationale: pandas "
                "is forbidden and a bounded heap keeps memory under 200 MB.",
            ),
            6: (
                "user",
                "Deploy target for this work: metrics port 8443, runtime version 2.7.1, "
                "coverage threshold 0.85. The export command is "
                "python -m exporter.csv --shards 8.",
            ),
            8: (
                "assistant",
                "The exporter entry point is src/export/csv_writer.py, function "
                "stream_export(), at line 218. That is where shard output is assembled.",
            ),
            12: (
                "assistant",
                "Hit OSError: [Errno 24] Too many open files when the shard writer opened "
                "one handle per shard. Fix: reuse a pool of at most 8 file handles and close "
                "on rotation.",
            ),
            16: (
                "user",
                "Correction to the earlier decision: do not use the bounded heap for the "
                "full export. Top-N mode keeps the heap, but the full export must stream "
                "rows into temporary shards and merge them in order.",
            ),
            19: (
                "user",
                "One more ask: add a --dry-run flag before finishing. Do not start it yet; "
                "wait until the export is green.",
            ),
            20: (
                "assistant",
                "Understood. I will leave the --dry-run flag pending until the export is "
                "green and keep the stdlib streaming path.",
            ),
        },
    },
    {
        "id": "oidc_migration",
        "goal_prompt": (
            "Vision: make sign-in boring and safe.\n"
            "Goal: Move the auth service to OIDC.\n"
            "Constraints:\n"
            "- Zero downtime during cutover\n"
            "- No schema migrations in business hours\n"
            "- Keep audit logs for every login\n"
            "- Non-goal: do not change the password policy\n"
            "Success criteria:\n"
            "- All active sessions survive the migration"
        ),
        "goal": "Move the auth service to OIDC.",
        "constraints": [
            "Zero downtime during cutover",
            "No schema migrations in business hours",
            "Keep audit logs for every login",
            "Non-goal: do not change the password policy",
        ],
        "success_criteria": ["All active sessions survive the migration"],
        "decision": "Dual-run old session cookies and new OIDC tokens, migrating cohorts by 5 percent",
        "decision_reason": "a dual run preserves zero downtime during cutover",
        "correction": (
            "The 5 percent cohort showed audit gaps, so do not dual-write session state: "
            "keep the cohort at 5 percent and mirror audit events synchronously to both "
            "stores before committing a session"
        ),
        "artifact_path": "services/auth/session_bridge.py",
        "artifact_symbol": "mirror_audit_event()",
        "artifact_line": "line 87",
        "error": "AUDIT_GAP_409: session has no audit record",
        "fix": "flush the audit mirror before committing the session and retry once",
        "open_ask": "Add a rollback runbook for the cohort migration; do not write it yet.",
        "next_step": (
            "Flush the audit mirror in mirror_audit_event() at "
            "services/auth/session_bridge.py line 87 before committing the session, "
            "then cut the 5 percent cohort"
        ),
        "exact_values": [
            "server port 9443",
            "auth version 1.14.2",
            "error budget 0.5 percent",
            "command authctl migrate --cohort 5",
        ],
        "keywords": {
            "goal": [["oidc"], ["auth", "service"]],
            "constraints": [
                ["zero", "downtime"],
                ["business", "hours"],
                ["audit", "logs"],
                ["password", "policy"],
            ],
            "success": [["active", "sessions"], ["survive", "migration"]],
            "decision": [["dual", "run"], ["cohort"], ["zero", "downtime"]],
            "correction": [["audit", "gap"], ["mirror", "audit"], ["5", "percent"]],
            "artifact": [
                ["services/auth/session_bridge.py"],
                ["mirror_audit_event"],
                ["line 87"],
            ],
            "error": [["audit_gap_409"], ["no", "audit", "record"], ["flush", "audit", "mirror"]],
            "open_loop": [["rollback", "runbook"], ["do", "not", "write", "yet"]],
            "next_step": [["flush", "audit", "mirror"], ["mirror_audit_event"], ["line 87"]],
            "values": [["9443"], ["1.14.2"], ["0.5", "percent"], ["cohort", "5"]],
        },
        "probe": (
            "The cohort migration is about to start. Choose the next step.\n"
            "A) Flush the audit mirror in mirror_audit_event() at "
            "services/auth/session_bridge.py line 87 before committing the session, then "
            "cut the 1 percent cohort.\n"
            "B) Flush the audit mirror in mirror_audit_event() at "
            "services/auth/session_bridge.py line 87 before committing the session, then "
            "cut the 5 percent cohort.\n"
            "C) Flush the audit mirror in mirror_audit_event() at "
            "services/auth/session_bridge.py line 87 after committing the session, then "
            "cut the 5 percent cohort.\n"
            "D) Flush the audit mirror in mirror_audit_event() at "
            "services/auth/session_bridge.py line 87 after committing the session, then "
            "cut the 1 percent cohort."
        ),
        "correct": "B",
        "events": {
            1: ("user", "goal_prompt"),
            3: (
                "assistant",
                "Decision: dual-run old session cookies and new OIDC tokens, migrating "
                "cohorts by 5 percent, rather than a big-bang cutover. Rationale: a dual "
                "run preserves zero downtime during cutover.",
            ),
            6: (
                "user",
                "Deploy target for this work: server port 9443, auth version 1.14.2, error "
                "budget 0.5 percent. The migration command is "
                "authctl migrate --cohort 5.",
            ),
            8: (
                "assistant",
                "The session bridge is services/auth/session_bridge.py, function "
                "mirror_audit_event(), at line 87. Every login must pass through it.",
            ),
            12: (
                "assistant",
                "Hit AUDIT_GAP_409: session has no audit record when a cohort moved before "
                "the audit write landed. Fix: flush the audit mirror before committing the "
                "session and retry once.",
            ),
            16: (
                "user",
                "Correction: the 5 percent cohort showed audit gaps, so do not dual-write "
                "session state. Keep the cohort at 5 percent and mirror audit events "
                "synchronously to both stores before committing a session.",
            ),
            19: (
                "user",
                "One more ask: add a rollback runbook for the cohort migration. Do not "
                "write it yet; wait for the migration to finish.",
            ),
            20: (
                "assistant",
                "Understood. The rollback runbook stays pending and the cohort remains at "
                "5 percent.",
            ),
        },
    },
    {
        "id": "latency",
        "goal_prompt": (
            "Goal: Reduce p95 checkout latency below 120 ms.\n"
            "Vision: keep checkout fast on low-end phones.\n"
            "Constraints:\n"
            "- No new runtime dependencies\n"
            "- Keep the JS bundle under 200 KB\n"
            "- Do not break Safari 15\n"
            "- Non-goal: do not redesign the checkout UI\n"
            "Success criteria:\n"
            "- p95 under 120 ms on the checkout route"
        ),
        "goal": "Reduce p95 checkout latency below 120 ms.",
        "constraints": [
            "No new runtime dependencies",
            "Keep the JS bundle under 200 KB",
            "Do not break Safari 15",
            "Non-goal: do not redesign the checkout UI",
        ],
        "success_criteria": ["p95 under 120 ms on the checkout route"],
        "decision": "Code-split the existing bundles and lazy-load route chunks",
        "decision_reason": "it adds no new runtime dependency and keeps the bundle under 200 KB",
        "correction": (
            "Code-splitting alone did not fix LCP: do not lazy-load the checkout route "
            "chunk; preload that chunk in the HTML head instead"
        ),
        "artifact_path": "web/routes/checkout.tsx",
        "artifact_symbol": "preloadCheckoutChunk()",
        "artifact_line": "line 64",
        "error": "ChunkLoadError: Loading chunk checkout failed",
        "fix": "add a modulepreload link for the checkout chunk and retry hydration once",
        "open_ask": "Add a Safari 15 fallback test; wait until p95 is green.",
        "next_step": (
            "Preload the checkout chunk with a modulepreload link in the HTML head via "
            "preloadCheckoutChunk() at web/routes/checkout.tsx line 64, keeping the bundle "
            "under 200 KB"
        ),
        "exact_values": [
            "p95 threshold 120 ms",
            "bundle cap 200 KB",
            "Safari 15",
            "command pnpm build --analyze",
        ],
        "keywords": {
            "goal": [["p95"], ["checkout", "latency"]],
            "constraints": [
                ["new", "runtime", "depend"],
                ["200", "kb"],
                ["safari", "15"],
                ["redesign", "checkout", "ui"],
            ],
            "success": [["p95", "120"], ["checkout", "route"]],
            "decision": [["code", "split"], ["lazy", "load"], ["runtime", "depend"]],
            "correction": [["preload", "html", "head"], ["not", "lazy", "load"]],
            "artifact": [
                ["web/routes/checkout.tsx"],
                ["preloadcheckoutchunk"],
                ["line 64"],
            ],
            "error": [["chunkloaderror"], ["loading", "chunk", "checkout", "failed"], ["modulepreload"]],
            "open_loop": [["safari", "15", "fallback", "test"], ["p95", "green"]],
            "next_step": [["modulepreload"], ["preloadcheckoutchunk"], ["line 64"]],
            "values": [["120", "ms"], ["200", "kb"], ["safari", "15"], ["pnpm", "analyze"]],
        },
        "probe": (
            "After the route split, Safari 15 sometimes throws ChunkLoadError. Choose the "
            "next step.\n"
            "A) Preload the checkout chunk with a modulepreload link in "
            "preloadCheckoutChunk() at web/routes/checkout.tsx line 64, keeping the bundle "
            "under 200 KB.\n"
            "B) Preload the checkout chunk with a modulepreload link in "
            "preloadCheckoutChunk() at web/routes/checkout.tsx line 64, keeping the bundle "
            "under 150 KB.\n"
            "C) Preload the checkout chunk with a rel=preload link in "
            "preloadCheckoutChunk() at web/routes/checkout.tsx line 64, keeping the bundle "
            "under 150 KB.\n"
            "D) Preload the checkout chunk with a rel=preload link in "
            "preloadCheckoutChunk() at web/routes/checkout.tsx line 64, keeping the bundle "
            "under 200 KB."
        ),
        "correct": "A",
        "events": {
            1: ("user", "goal_prompt"),
            3: (
                "assistant",
                "Decision: code-split the existing bundles and lazy-load route chunks, "
                "rather than adding a virtualization library. Rationale: it adds no new "
                "runtime dependency and keeps the bundle under 200 KB.",
            ),
            6: (
                "user",
                "Deploy target for this work: p95 threshold 120 ms, bundle cap 200 KB, "
                "Safari 15 support. The analysis command is pnpm build --analyze.",
            ),
            8: (
                "assistant",
                "The checkout route is web/routes/checkout.tsx; the helper "
                "preloadCheckoutChunk() is at line 64. That helper decides how the route "
                "chunk is fetched.",
            ),
            12: (
                "assistant",
                "Hit ChunkLoadError: Loading chunk checkout failed on Safari 15. Fix: add "
                "a modulepreload link for the checkout chunk and retry hydration once.",
            ),
            16: (
                "user",
                "Correction: code-splitting alone did not fix LCP. Do not lazy-load the "
                "checkout route chunk; preload that chunk in the HTML head instead.",
            ),
            19: (
                "user",
                "One more ask: add a Safari 15 fallback test. Wait until p95 is green "
                "before starting it.",
            ),
            20: (
                "assistant",
                "Understood. The Safari 15 fallback test stays pending and the bundle cap "
                "remains 200 KB.",
            ),
        },
    },
    {
        "id": "retention",
        "goal_prompt": (
            "Goal: Enforce 30-day data retention for user events.\n"
            "Vision: privacy by default, auditable deletion.\n"
            "Constraints:\n"
            "- Encrypt PII at rest\n"
            "- Delete user rows within 24 hours of a request\n"
            "- No third-party analytics\n"
            "- Non-goal: do not build a new consent UI\n"
            "Success criteria:\n"
            "- 30-day TTL verified by an audit"
        ),
        "goal": "Enforce 30-day data retention for user events.",
        "constraints": [
            "Encrypt PII at rest",
            "Delete user rows within 24 hours of a request",
            "No third-party analytics",
            "Non-goal: do not build a new consent UI",
        ],
        "success_criteria": ["30-day TTL verified by an audit"],
        "decision": "Partition events by day and drop whole partitions older than 30 days",
        "decision_reason": "partition drops are cheaper and safer than per-row deletes",
        "correction": (
            "Partition drop alone is insufficient: keep the 30-day partition drop but add "
            "a deletion queue that tombstones user rows within 24 hours"
        ),
        "artifact_path": "jobs/retention/partition_drop.py",
        "artifact_symbol": "drop_expired_partitions()",
        "artifact_line": "line 51",
        "error": "PARTITION_LOCK_TIMEOUT: could not drop partition within 30s",
        "fix": "retry with lock_timeout=5s and process partitions oldest-first",
        "open_ask": "Document the audit evidence format; do not start until the TTL job is green.",
        "next_step": (
            "Retry oldest-first with lock_timeout=5s in drop_expired_partitions() at "
            "jobs/retention/partition_drop.py line 51, keeping the 24-hour user deletion queue"
        ),
        "exact_values": [
            "retention window 30 days",
            "user deletion window 24 hours",
            "lock timeout 30s",
            "command retentionctl drop --older-than 30d",
        ],
        "keywords": {
            "goal": [["30", "day"], ["retention"], ["user", "events"]],
            "constraints": [
                ["encrypt", "pii"],
                ["24", "hour"],
                ["third", "party"],
                ["consent", "ui"],
            ],
            "success": [["30", "day", "ttl"], ["audit"]],
            "decision": [["partition"], ["drop"], ["per", "row", "delete"]],
            "correction": [["deletion", "queue"], ["24", "hour"], ["insufficient"]],
            "artifact": [
                ["jobs/retention/partition_drop.py"],
                ["drop_expired_partitions"],
                ["line 51"],
            ],
            "error": [["partition_lock_timeout"], ["could", "not", "drop", "partition"], ["lock_timeout", "5s"]],
            "open_loop": [["audit", "evidence", "format"], ["ttl", "job", "green"]],
            "next_step": [["oldest", "first"], ["lock_timeout", "5s"], ["drop_expired_partitions"]],
            "values": [["30", "day"], ["24", "hour"], ["30s"], ["retentionctl"]],
        },
        "probe": (
            "The partition job just hit a lock timeout. Choose the next step.\n"
            "A) Retry newest-first with lock_timeout=30s in drop_expired_partitions() at "
            "jobs/retention/partition_drop.py line 51, keeping the 48-hour deletion queue.\n"
            "B) Retry newest-first with lock_timeout=5s in drop_expired_partitions() at "
            "jobs/retention/partition_drop.py line 51, keeping the 48-hour deletion queue.\n"
            "C) Retry oldest-first with lock_timeout=30s in drop_expired_partitions() at "
            "jobs/retention/partition_drop.py line 51, keeping the 24-hour deletion queue.\n"
            "D) Retry oldest-first with lock_timeout=5s in drop_expired_partitions() at "
            "jobs/retention/partition_drop.py line 51, keeping the 24-hour deletion queue."
        ),
        "correct": "D",
        "events": {
            1: ("user", "goal_prompt"),
            3: (
                "assistant",
                "Decision: partition events by day and drop whole partitions older than "
                "30 days, rather than per-row deletes. Rationale: partition drops are "
                "cheaper and safer than per-row deletes.",
            ),
            6: (
                "user",
                "Deploy target for this work: retention window 30 days, user deletion "
                "window 24 hours, lock timeout 30s. The command is "
                "retentionctl drop --older-than 30d.",
            ),
            8: (
                "assistant",
                "The partition job is jobs/retention/partition_drop.py, function "
                "drop_expired_partitions(), at line 51.",
            ),
            12: (
                "assistant",
                "Hit PARTITION_LOCK_TIMEOUT: could not drop partition within 30s during "
                "the audit freeze. Fix: retry with lock_timeout=5s and process partitions "
                "oldest-first.",
            ),
            16: (
                "user",
                "Correction: partition drop alone is insufficient. Keep the 30-day "
                "partition drop but add a deletion queue that tombstones user rows within "
                "24 hours.",
            ),
            19: (
                "user",
                "One more ask: document the audit evidence format. Do not start until the "
                "TTL job is green.",
            ),
            20: (
                "assistant",
                "Understood. The audit evidence document stays pending and the 24-hour "
                "deletion queue remains in scope.",
            ),
        },
    },
    {
        "id": "search_index",
        "goal_prompt": (
            "Goal: Replace ad-hoc SQL LIKE search with a local full-text index.\n"
            "Vision: fast, local search with no network hop.\n"
            "Constraints:\n"
            "- No external services\n"
            "- p95 search under 80 ms\n"
            "- Index build under 10 minutes\n"
            "- Non-goal: do not add semantic or vector search\n"
            "Success criteria:\n"
            "- 1M docs searchable with no external network"
        ),
        "goal": "Replace ad-hoc SQL LIKE search with a local full-text index.",
        "constraints": [
            "No external services",
            "p95 search under 80 ms",
            "Index build under 10 minutes",
            "Non-goal: do not add semantic or vector search",
        ],
        "success_criteria": ["1M docs searchable with no external network"],
        "decision": "Use SQLite FTS5 as the index",
        "decision_reason": "FTS5 needs no external service and fits the local constraint",
        "correction": (
            "The default unicode61 tokenizer mishandles product codes: use the trigram "
            "tokenizer for the product-code column and keep unicode61 for descriptions"
        ),
        "artifact_path": "search/index_build.py",
        "artifact_symbol": "build_fts_index()",
        "artifact_line": "line 133",
        "error": "SQLITE_ERROR: no such tokenizer: trigram",
        "fix": "enable the trigram tokenizer build flag and recreate the FTS table once",
        "open_ask": "Benchmark 1M docs on the staging box; do not run it yet.",
        "next_step": (
            "Enable the trigram tokenizer build flag and recreate the FTS table in "
            "build_fts_index() at search/index_build.py line 133, keeping the index local"
        ),
        "exact_values": [
            "p95 80 ms",
            "1M docs",
            "10 minute build",
            "command searchctl build --docs 1000000",
        ],
        "keywords": {
            "goal": [["full", "text", "index"], ["sql", "like"]],
            "constraints": [
                ["no", "external", "service"],
                ["80", "ms"],
                ["10", "minute"],
                ["vector", "search"],
            ],
            "success": [["1m", "docs"], ["no", "external", "network"]],
            "decision": [["sqlite", "fts5"], ["no", "external", "service"]],
            "correction": [["trigram"], ["unicode61"], ["product", "code"]],
            "artifact": [
                ["search/index_build.py"],
                ["build_fts_index"],
                ["line 133"],
            ],
            "error": [["sqlite_error"], ["no", "such", "tokenizer", "trigram"], ["trigram", "build", "flag"]],
            "open_loop": [["benchmark", "1m", "staging"], ["do", "not", "run", "yet"]],
            "next_step": [["trigram", "tokenizer", "build", "flag"], ["recreate", "fts", "table"]],
            "values": [["80", "ms"], ["1m", "docs"], ["10", "minute"], ["searchctl"]],
        },
        "probe": (
            "build_fts_index() fails with SQLITE_ERROR: no such tokenizer: trigram. Choose "
            "the next step.\n"
            "A) Enable the trigram tokenizer build flag and recreate the FTS table in "
            "build_fts_index() at search/index_build.py line 133.\n"
            "B) Enable the unicode61 tokenizer build flag and recreate the FTS table in "
            "build_fts_index() at search/index_build.py line 133.\n"
            "C) Enable the trigram tokenizer build flag and skip recreating the FTS table "
            "in build_fts_index() at search/index_build.py line 133.\n"
            "D) Enable the trigram tokenizer build flag and recreate the FTS table in "
            "rebuild_search_index() at search/index_build.py line 214."
        ),
        "correct": "A",
        "events": {
            1: ("user", "goal_prompt"),
            3: (
                "assistant",
                "Decision: use SQLite FTS5 as the index, rather than Elasticsearch. "
                "Rationale: FTS5 needs no external service and fits the local constraint.",
            ),
            6: (
                "user",
                "Deploy target for this work: p95 80 ms, 1M docs, 10 minute build. The "
                "command is searchctl build --docs 1000000.",
            ),
            8: (
                "assistant",
                "The index builder is search/index_build.py, function build_fts_index(), "
                "at line 133.",
            ),
            12: (
                "assistant",
                "Hit SQLITE_ERROR: no such tokenizer: trigram on the product-code column. "
                "Fix: enable the trigram tokenizer build flag and recreate the FTS table "
                "once.",
            ),
            16: (
                "user",
                "Correction: the default unicode61 tokenizer mishandles product codes. Use "
                "the trigram tokenizer for the product-code column and keep unicode61 for "
                "descriptions.",
            ),
            19: (
                "user",
                "One more ask: benchmark 1M docs on the staging box. Do not run it yet; "
                "wait until the build is green.",
            ),
            20: (
                "assistant",
                "Understood. The staging benchmark stays pending and the index remains "
                "local.",
            ),
        },
    },
    {
        "id": "billing_webhook",
        "goal_prompt": (
            "Goal: Make Stripe webhook processing idempotent and replay-safe.\n"
            "Vision: never double-charge a customer.\n"
            "Constraints:\n"
            "- Verify the signature on every request\n"
            "- No duplicate invoices\n"
            "- At-least-once delivery with safe retries\n"
            "- Non-goal: do not change pricing\n"
            "Success criteria:\n"
            "- Replaying 10k events creates zero duplicates"
        ),
        "goal": "Make Stripe webhook processing idempotent and replay-safe.",
        "constraints": [
            "Verify the signature on every request",
            "No duplicate invoices",
            "At-least-once delivery with safe retries",
            "Non-goal: do not change pricing",
        ],
        "success_criteria": ["Replaying 10k events creates zero duplicates"],
        "decision": "Store processed event IDs in Postgres with a unique constraint",
        "decision_reason": "Postgres survives restarts, unlike an in-memory dedupe cache",
        "correction": (
            "A unique constraint alone races under concurrent workers. The processed-event "
            "insert and the invoice creation must commit atomically in the same "
            "transaction: use INSERT ... ON CONFLICT DO NOTHING in that transaction and "
            "treat zero rows as a duplicate. A separate transaction for the event insert "
            "is not acceptable"
        ),
        "artifact_path": "services/billing/webhook_handler.py",
        "artifact_symbol": "handle_event()",
        "artifact_line": "line 204",
        "error": 'IntegrityError: duplicate key value violates unique constraint "processed_event_pkey"',
        "fix": "catch the conflict via ON CONFLICT and return 200 without creating an invoice",
        "open_ask": "Add a replay tool for the last 24h; do not run it until the idempotency test is green.",
        "next_step": (
            "Use INSERT ... ON CONFLICT DO NOTHING inside the invoice transaction in "
            "handle_event() at services/billing/webhook_handler.py line 204"
        ),
        "exact_values": [
            "replay window 24h",
            "10k events",
            "signature header Stripe-Signature",
            "command billingctl replay --since 24h",
        ],
        "keywords": {
            "goal": [["stripe", "webhook"], ["idempotent", "replay"]],
            "constraints": [
                ["verify", "signature"],
                ["no", "duplicate", "invoice"],
                ["at", "least", "once"],
                ["pricing"],
            ],
            "success": [["10k", "events"], ["zero", "duplicates"]],
            "decision": [["postgres"], ["unique", "constraint"], ["survives", "restarts"]],
            "correction": [["on", "conflict"], ["same", "transaction"], ["zero", "rows"]],
            "artifact": [
                ["services/billing/webhook_handler.py"],
                ["handle_event"],
                ["line 204"],
            ],
            "error": [["integrityerror"], ["duplicate", "key", "value"], ["on", "conflict"]],
            "open_loop": [["replay", "tool"], ["idempotency", "test", "green"]],
            "next_step": [["on", "conflict", "do", "nothing"], ["handle_event"], ["line 204"]],
            "values": [["24h"], ["10k", "events"], ["stripe", "signature"], ["billingctl"]],
        },
        "probe": (
            "Concurrent replays just produced two invoices and the log shows "
            "IntegrityError on processed_event_pkey. Choose the next step.\n"
            "A) Use INSERT ... ON CONFLICT DO NOTHING in a separate transaction before "
            "the invoice in handle_event() at services/billing/webhook_handler.py line 204.\n"
            "B) Use INSERT ... ON CONFLICT DO NOTHING inside the invoice transaction in "
            "handle_event() at services/billing/webhook_handler.py line 204.\n"
            "C) Use INSERT ... ON CONFLICT DO NOTHING inside the invoice transaction in "
            "replay_event() at services/billing/webhook_handler.py line 204.\n"
            "D) Use SELECT ... FOR UPDATE inside the invoice transaction in handle_event() "
            "at services/billing/webhook_handler.py line 204."
        ),
        "correct": "B",
        "events": {
            1: ("user", "goal_prompt"),
            3: (
                "assistant",
                "Decision: store processed event IDs in Postgres with a unique constraint, "
                "rather than an in-memory LRU dedupe. Rationale: Postgres survives "
                "restarts, unlike an in-memory dedupe cache.",
            ),
            6: (
                "user",
                "Deploy target for this work: replay window 24h, 10k events, signature "
                "header Stripe-Signature. The command is billingctl replay --since 24h.",
            ),
            8: (
                "assistant",
                "The webhook handler is services/billing/webhook_handler.py, function "
                "handle_event(), at line 204.",
            ),
            12: (
                "assistant",
                "Hit IntegrityError: duplicate key value violates unique constraint "
                '"processed_event_pkey". Fix: catch the conflict via ON CONFLICT and '
                "return 200 without creating an invoice.",
            ),
            16: (
                "user",
                "Correction: a unique constraint alone races under concurrent workers. "
                "The processed-event insert and the invoice creation must commit atomically "
                "in the same transaction. Use INSERT ... ON CONFLICT DO NOTHING in that "
                "transaction and treat zero rows as a duplicate. A separate transaction "
                "for the event insert is not acceptable.",
            ),
            19: (
                "user",
                "One more ask: add a replay tool for the last 24h. Do not run it until the "
                "idempotency test is green.",
            ),
            20: (
                "assistant",
                "Understood. The replay tool stays pending and signature verification "
                "remains mandatory.",
            ),
        },
    },
]

DISTRACTIONS = [
    "Tool output: pytest -q -> 214 passed, 3 skipped in 8.4s. Warnings: deprecated typing.Tuple in tests.",
    "Reviewed the cache module and renamed a local variable; no behaviour changed.",
    "Upgraded the linter config and fixed 17 line-length warnings in helper files.",
    "Profiled the local build; the slowest step is still the source map pass at 1.9 seconds.",
    "Tool output: git status -> modified README.md, modified docs/architecture.md.",
    "The nightly container image rebuilt successfully; digest sha256:4f2a...; no code changed.",
    "Discussed naming for a metrics counter and updated the dashboard draft only.",
    "Tool output: grep -R returns 12 references to a deprecated helper in docs and tests.",
    "Refactored the temporary-directory test fixture so it cleans up on failure.",
    "Benchmarked JSON parsing on a synthetic 5 MB file; the median stayed at 41 ms.",
    "Updated the changelog entry for the documentation refresh and fixed two links.",
    "Tool output: type checker reports two unused imports in scripts/release.py.",
    "Read the incident postmortem from last quarter; action items are already tracked.",
    "Cleaned up stale branches in the local clone; no remote branches changed.",
    "Tuned the CI cache key for the dependency layer; the next build should skip install.",
    "Tool output: coverage -> 87.4 percent statements, 81.1 percent branches, unchanged.",
    "Reviewed the error message for a malformed config file and mentioned the exact path.",
    "Sketched a diagram of the request path for the onboarding document; no code edited.",
    'Tool output: curl health check -> 200 OK, 14 ms, body {"status":"ok"}.',
    "Discussed whether to rename the internal queue module; deferred to the next review.",
    "Updated the developer setup notes to mention the pinned toolchain.",
    "Tool output: dependency audit reports no high-severity findings in the lockfile.",
    "Wrote a short note about a flaky timing test and marked it for follow-up.",
    "Checked the release calendar; the next maintenance window is two weeks away.",
    "Tool output: ruff check -> 3 unused imports fixed in scratch helpers.",
    "Re-read the style guide and aligned two docstrings with it.",
    "Tool output: make docs -> 42 pages built, 0 broken links.",
    "Spent a few minutes tidying the scratch directory and deleting old patch files.",
    "Reviewed a draft PR description and suggested a clearer summary line.",
    "Tool output: git diff --stat -> 6 files changed, 41 insertions, 18 deletions.",
    "Noted that the staging box was rebooted overnight; no action needed.",
    "Tool output: docker images -> three old images pruned, 1.2 GB reclaimed.",
    "Walked through the on-call handoff notes and clarified one timer.",
    "Tool output: shellcheck -> no issues in the helper script.",
    "Sketched a small state diagram for the worker queue on paper; nothing committed.",
    "Checked the metrics dashboard and confirmed the request counter is stable.",
    "Tool output: mypy -> success, no issues found in 24 source files.",
    "Reread the review checklist and ticked off the documentation item.",
    "Updated a comment that referred to an old directory name.",
    "Tool output: npm audit -> 0 vulnerabilities after the lockfile refresh.",
    "Discussed the naming of a temporary table in a planning note; deferred.",
    "Tool output: du -sh -> build cache is 2.1 GB, logs are 340 MB.",
    "Cleaned up a stale TODO and linked it to the tracking issue.",
    "Tool output: find . -name '*.pyc' -> removed 118 cache files.",
    "Reviewed the accessibility notes for the internal admin page; no changes yet.",
    "Tool output: git log --oneline -5 -> five recent documentation commits.",
    "Reordered the import block in a scratch file and left the rest alone.",
    "Tool output: bandit -> no high-confidence findings in the changed files.",
    "Wrote a short comment explaining why a retry uses a fixed delay.",
    "Tool output: pytest --collect-only -> 431 tests collected.",
    "Updated the team wiki index with a link to the current runbook.",
    "Tool output: terraform plan -> no changes; state is up to date.",
    "Reviewed a snippet of shell history for a repeated command and saved it to notes.",
    "Tool output: git stash list -> one old stash from last week; left it in place.",
    "Spent time reading the logging conventions and fixed one label.",
    "Tool output: pre-commit run -> all hooks passed.",
    "Confirmed the pinned Python version in the dev container matches CI.",
    "Tool output: licenses -> no new or unknown licenses detected.",
    "Refactored a helper name in the scratch notebook; not part of the deliverable.",
    "Tool output: grep TODO -> 9 TODOs, all assigned to existing tickets.",
    "Read the release notes for the patch update and noted one behaviour change.",
    "Tool output: git remote -v -> origin and upstream point at the expected hosts.",
    "Rebuilt the local docs preview and confirmed the sidebar renders.",
    "Tool output: curl -I -> 204 No Content for the internal health endpoint.",
    "Reviewed the naming conventions for environment variables in the setup doc.",
    "Tool output: codeowners -> two owners listed for the docs directory.",
    "Moved a scratch note into the team folder and renamed it for clarity.",
    "Tool output: pytest -q tests/unit -> 188 passed in 4.1s.",
    "Checked that the local clock is synced; no drift reported.",
    "Tool output: df -h -> 38 percent used on the workspace volume.",
    "Reviewed a diff that only reordered dictionary keys; no functional change.",
    "Tool output: git branch -vv -> local branch tracks the expected upstream.",
    "Read through the contributing guide and confirmed the commit format.",
    "Tool output: make lint -> 0 errors, 2 informational notes.",
    "Tested the docs search box locally; results look reasonable.",
    "Tool output: pip check -> no broken requirements found.",
    "Cleaned a stale entry from the scratch changelog.",
    "Tool output: go test ./internal/... -> ok, 3 packages cached.",
    "Confirmed the fixture data is generated deterministically from the seed.",
    "Tool output: git grep deprecated -> 4 matches, all in archived notes.",
    "Reviewed the error taxonomy notes and left them unchanged.",
    "Tool output: node --version -> v20.11.1, matching the toolchain pin.",
    "Updated a comment to point at the current design note.",
    "Tool output: make fmt -> 11 files left unchanged, 2 reformatted.",
    "Read the database backup schedule and confirmed the next window.",
    "Tool output: kubectl get pods -> all pods running, zero restarts.",
    "Noted a minor inconsistency in the docs and filed a tracking issue.",
    "Tool output: pytest -q tests/integration -> 27 passed, 1 skipped.",
    "Reviewed the retry policy notes and confirmed the jitter setting.",
    "Tool output: git diff --check -> no whitespace errors.",
    "Cleaned up two stale scratch scripts; nothing referenced them.",
    "Tool output: make typecheck -> success in 31 seconds.",
    "Read a short article on deterministic test fixtures and saved a note.",
]


# ---------------------------------------------------------------------------
# Plugin loading and environment
# ---------------------------------------------------------------------------

def _load_plugin():
    spec = importlib.util.spec_from_file_location("append_compact_tcbench", PLUGIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load plugin at {PLUGIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def patched_env(**values: str):
    keys = list(values)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


BASE_ENV = {
    "APPEND_COMPACT_L3": "1",
    "APPEND_COMPACT_L3_COMPRESS": "1",
    "APPEND_COMPACT_MAX_INJECT_CHARS": "9000",
    "APPEND_COMPACT_L3_TARGET_CHARS": "6400",
}
GOAL_ENV = {
    "APPEND_COMPACT_GOALS": "1",
    "APPEND_COMPACT_GOALS_EXTRACT": "1",
    "APPEND_COMPACT_GOALS_L1": "1",
    "APPEND_COMPACT_GOALS_L2": "1",
    "APPEND_COMPACT_GOALS_L3": "1",
    "APPEND_COMPACT_GOALS_PLACEMENT": "beginning",
    "APPEND_COMPACT_GOALS_MAX_CHARS": "900",
    "APPEND_COMPACT_GOALS_MAX_ITEMS": "8",
    "APPEND_COMPACT_GOALS_L1_CHARS": "240",
    "APPEND_COMPACT_GOALS_L2_CHARS": "520",
    "APPEND_COMPACT_GOALS_L3_CHARS": "760",
}
OFF_ENV = {**BASE_ENV, "APPEND_COMPACT_GOALS": "0"}


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

CALLS: list[dict[str, Any]] = []
USAGE: dict[str, int] = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "reasoning_tokens": 0,
}


def _deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if DEEPSEEK_ENV.exists():
        for line in DEEPSEEK_ENV.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("DEEPSEEK_API_KEY not found")


def _post_deepseek(
    messages: list[dict[str, str]],
    *,
    num_predict: int,
    fmt: str | None = "json",
    attempt: int = 0,
) -> dict[str, Any]:
    # deepseek-flash bills its reasoning tokens against max_tokens, so a
    # num_predict-sized budget gets spent thinking and the JSON body is cut off
    # (finish_reason="length": 624-759 reasoning tokens on a 1200 budget, and
    # one 560-token answer burned 3539).  Each retry doubles the headroom.
    budget = num_predict + REASONING_HEADROOM * (2 ** min(attempt, 4))
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": budget,
        "stream": False,
    }
    if fmt:
        body["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        DEEPSEEK_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_deepseek_key()}",
        },
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    latency = time.perf_counter() - started
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message", {})
    usage = payload.get("usage") or {}
    if choice.get("finish_reason") == "length":
        raise ValueError(
            f"deepseek response truncated at max_tokens={budget} "
            f"(completion={usage.get('completion_tokens')}, "
            f"reasoning={(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')})"
        )
    return {
        "raw": str(message.get("content", "")),
        "latency_s": latency,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": str(choice.get("finish_reason") or ""),
        "reasoning_tokens": int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        ),
    }


def _post_chat(
    messages: list[dict[str, str]],
    *,
    num_predict: int,
    num_ctx: int | None = None,
    fmt: str | None = "json",
    attempt: int = 0,
) -> dict[str, Any]:
    if PROVIDER == "deepseek":
        return _post_deepseek(
            messages, num_predict=num_predict, fmt=fmt, attempt=attempt
        )
    if PROVIDER != "ollama":
        raise RuntimeError(f"unknown TC_BENCH_PROVIDER={PROVIDER!r}")
    # A retry that re-sends an identical request to a temperature-0 model is
    # deterministic: it fails the same way.  Raise the output budget on each
    # attempt so a truncated JSON body has room to finish.
    budget = int(num_predict * (1.5 ** min(attempt, 3)))
    body: dict[str, Any] = {
        "model": MODEL,
        "stream": False,
        "keep_alive": "40m",
        "options": {
            "temperature": 0,
            # Vary the seed per attempt: seed+0 reproduces the same malformed
            # JSON forever, which is what killed two runs on the same cell.
            "seed": SEED + attempt * 1000,
            "num_predict": budget,
            "num_ctx": num_ctx or NUM_CTX,
        },
        "messages": messages,
    }
    if fmt:
        body["format"] = fmt
    request = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        payload = json.load(response)
    latency = time.perf_counter() - started
    return {
        "raw": str(payload.get("message", {}).get("content", "")),
        "latency_s": latency,
        "prompt_tokens": int(payload.get("prompt_eval_count") or 0),
        "completion_tokens": int(payload.get("eval_count") or 0),
    }


def _parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Only the first "{" can be the top-level object.  Falling back to
        # later braces would let a truncated response masquerade as a small
        # nested object (seen with a template update cut off at num_predict).
        start = text.find("{")
        if start < 0:
            raise
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("model returned a non-object JSON value")
    return parsed


def call_model(
    system: str,
    user: str,
    *,
    num_predict: int = 400,
    attempts: int = 2,
    tag: str = "",
) -> dict[str, Any]:
    if PROVIDER == "deepseek":
        # Reasoning budget is unpredictable; give truncation room to escalate.
        attempts = max(attempts, 4)
    elif PROVIDER == "ollama":
        # Truncated JSON is the common local failure; retry with more room.
        attempts = max(attempts, 3)
    last_error: Exception | None = None
    # A temperature-0 model given the identical request returns the identical
    # malformed JSON, so retrying verbatim cannot recover.  Each retry changes
    # the seed (in _post_chat) and, after a parse failure, the prompt.
    user_text = user
    json_repair = (
        "\n\nYour previous reply was not valid JSON. Return one complete, valid "
        "JSON object and nothing else -- no prose, no code fence, no trailing text."
    )
    for attempt in range(attempts):
        try:
            result = _post_chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
                num_predict=num_predict,
                attempt=attempt,
            )
            result["parsed"] = _parse_json_response(result["raw"])
            USAGE["calls"] += 1
            USAGE["prompt_tokens"] += result["prompt_tokens"]
            USAGE["completion_tokens"] += result["completion_tokens"]
            USAGE["reasoning_tokens"] += int(result.get("reasoning_tokens") or 0)
            CALLS.append(
                {
                    "tag": tag,
                    "latency_s": result["latency_s"],
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "reasoning_tokens": result.get("reasoning_tokens", 0),
                    "attempt": attempt + 1,
                }
            )
            return result
        except json.JSONDecodeError as exc:
            last_error = exc
            user_text = user + json_repair
            time.sleep(1.0)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"model call {tag!r} failed after {attempts} attempts: {last_error!r}")


# ---------------------------------------------------------------------------
# Synthetic session construction
# ---------------------------------------------------------------------------

def turns_for(scenario: dict[str, Any], depth: int) -> list[dict[str, Any]]:
    """Return exactly ``depth`` turns; decisive events occupy fixed early turns."""
    events = scenario["events"]
    turns: list[dict[str, Any]] = []
    start = 0
    for index in range(1, depth + 1):
        if index in events:
            role, text = events[index]
            if text == "goal_prompt":
                text = scenario["goal_prompt"]
            turns.append({"n": index, "role": role, "text": text})
        else:
            text = DISTRACTIONS[start % len(DISTRACTIONS)]
            start += 1
            turns.append({"n": index, "role": "assistant", "text": f"Turn {index}: {text}"})
    return turns


def build_state(plugin, turns: list[dict[str, Any]], session: str) -> dict[str, Any]:
    state = plugin.new_state()
    with patched_env(**GOAL_ENV):
        for turn in turns:
            plugin.append_turn(
                state,
                turn["text"],
                llm="off",
                role=turn["role"],
                session=session,
            )
    return state


def render_full_transcript(turns: list[dict[str, Any]]) -> str:
    lines = ["Full session transcript:"]
    for turn in turns:
        lines.append(f"Turn {turn['n']} ({turn['role']}): {turn['text']}")
    return "\n\n".join(lines)


def plugin_context(plugin, state: dict[str, Any], session: str, goals: bool) -> str:
    env = {**BASE_ENV, **(GOAL_ENV if goals else {"APPEND_COMPACT_GOALS": "0"})}
    with patched_env(**env):
        payload = plugin.injection_payload(state, session=session)
    if payload is None:
        return ""
    return str(payload.get("additional_context") or "")


def ledger_block(plugin, state: dict[str, Any], session: str) -> str:
    return plugin._global_ledger_block(state, 2000, session)


# ---------------------------------------------------------------------------
# Handoff template
# ---------------------------------------------------------------------------

TEMPLATE_KEYS = [
    "goal_intent",
    "constraints_preferences",
    "success_criteria",
    "decisions_and_rationale",
    "progress",
    "files_symbols_references",
    "errors_and_fixes",
    "open_questions_pending_asks",
    "remaining_work_next_step",
    "critical_exact_values",
]
TEMPLATE_SHAPE = (
    '{"goal_intent":"copy the Goal/Vision sentence verbatim",'
    '"constraints_preferences":["..."],'
    '"success_criteria":["acceptance criteria copied verbatim, with their numbers"],'
    '"decisions_and_rationale":["choice + rationale; include corrections that supersede"],'
    '"progress":{"done":["..."],"in_progress":["..."],"blocked":["..."]},'
    '"files_symbols_references":["path, symbol and line exactly"],'
    '"errors_and_fixes":["exact error text + exact fix"],'
    '"open_questions_pending_asks":["..."],'
    '"remaining_work_next_step":["..."],'
    '"critical_exact_values":["port, version, threshold, command verbatim"]}'
)
TEMPLATE_SYSTEM = (
    "You write a structured handoff summary for a long software engineering session. "
    "You are not continuing the task: do not use tools and do not answer the user's "
    "questions. Copy exact file paths, symbol names, line numbers, numbers, commands and "
    "error text verbatim from the transcript. Never invent a fact. If a field has no "
    "information, use \"(none)\" for a string or [] for a list. Return one JSON object "
    "and nothing else."
)
TEMPLATE_HINT = (
    "Required JSON shape (all keys required):\n"
    f"{TEMPLATE_SHAPE}\n"
    "goal_intent MUST contain the exact Goal/Vision sentence from turn 1, copied "
    "character for character; do not paraphrase it, shorten it, or move it to another "
    "field. For decisions, give the choice and its rationale. For progress, "
    "separate done, in progress and blocked. Preserve any correction that supersedes an "
    "earlier decision and say what it supersedes. Keep every exact artifact reference in "
    "files_symbols_references. success_criteria MUST copy every acceptance or success "
    "criterion stated in the session verbatim, including its numbers; do not summarize "
    "it away. open_questions_pending_asks MUST list every ask the user deferred, with "
    "its exact trigger condition. Be concise: at most 5 bullets per list and at most 2 "
    "sentences per string."
)


def _template_request(turns: list[dict[str, Any]]) -> str:
    return (
        "Session transcript:\n"
        f"{render_full_transcript(turns)}\n\n"
        "---\n"
        "Produce the structured handoff for this session.\n"
        f"{TEMPLATE_HINT}"
    )


def _update_request(
    previous: dict[str, Any], new_turns: list[dict[str, Any]]
) -> str:
    return (
        "Previous handoff JSON:\n"
        f"{json.dumps(previous, ensure_ascii=False, indent=2)}\n\n"
        "New turns since that handoff:\n"
        f"{render_full_transcript(new_turns)}\n\n"
        "---\n"
        "Produce the updated structured handoff. Never discard a prior entry that is "
        "still relevant; correct or supersede an entry only when a new turn does so, and "
        "say what superseded it. Copy exact paths, numbers, commands and error text "
        "verbatim. Do not invent. Mark missing fields \"(none)\" or [].\n"
        f"{TEMPLATE_HINT}"
    )


def _template_corrective(missing: list[str]) -> str:
    return (
        "Your previous JSON failed validation. It must contain the exact goal, at "
        "least one exact file path or number from the transcript, the stated success "
        "criteria, every deferred user ask, the stated constraints, the early decision "
        "with its rationale, and any later correction that supersedes it. Missing or "
        f"weak: {', '.join(missing) or 'goal, exact reference, success criteria, pending ask'}. "
        "Copy success criteria, constraints, pending asks and corrections verbatim from "
        "the transcript. Return the corrected JSON object only, using the required shape."
    )


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "(none)", "none", "n/a", "[]"}
    if isinstance(value, (list, tuple)):
        return not value
    if isinstance(value, dict):
        return not any(_is_empty(v) for v in value.values())
    return False


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [] if _is_empty(value) else [value]
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out.extend(f"{key}: {text}" for text in _as_list(item))
        return out
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if not _is_empty(item)]
    return [] if _is_empty(value) else [str(value)]


def render_template(obj: dict[str, Any]) -> str:
    sections = [
        ("Goal / intent", _as_list(obj.get("goal_intent"))),
        ("Constraints / preferences", _as_list(obj.get("constraints_preferences"))),
        ("Success criteria", _as_list(obj.get("success_criteria"))),
        ("Decisions and rationale", _as_list(obj.get("decisions_and_rationale"))),
        ("Progress (done / in progress / blocked)", _as_list(obj.get("progress"))),
        ("Files, symbols, exact references", _as_list(obj.get("files_symbols_references"))),
        ("Errors and fixes", _as_list(obj.get("errors_and_fixes"))),
        ("Open questions / pending asks", _as_list(obj.get("open_questions_pending_asks"))),
        ("Remaining work / next step", _as_list(obj.get("remaining_work_next_step"))),
        ("Critical exact values", _as_list(obj.get("critical_exact_values"))),
    ]
    lines = ["## Session handoff"]
    for heading, items in sections:
        lines.append("")
        lines.append(f"### {heading}")
        if not items:
            lines.append("- (none)")
        else:
            lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


# Sections kept by the minimal arm.  The v8 run showed the whole template's
# measurable wins came from exactly these: pending asks (+0.62), exact values
# and files (+0.38 on facts) and errors.  Everything else is carried by the
# goals injection the minimal arm still includes.
MINIMAL_TEMPLATE_KEYS = (
    ("Files, symbols, exact references", "files_symbols_references"),
    ("Errors and fixes", "errors_and_fixes"),
    ("Open questions / pending asks", "open_questions_pending_asks"),
    ("Critical exact values", "critical_exact_values"),
    ("Remaining work / next step", "remaining_work_next_step"),
)


# Variants of the minimal arm, so each section can be earned or dropped by
# measurement instead of argument.  ``drop`` names object keys to leave out,
# ``add`` appends a section the minimal set omits, ``use_anchors`` appends the
# turn a bullet came from.
MINIMAL_ARM_KWARGS: dict[str, dict[str, Any]] = {
    "template_minimal": {},
    "template_minimal_anchors": {"use_anchors": True},
    "template_minimal_minus_files": {"drop": ("files_symbols_references",)},
    "template_minimal_minus_errors": {"drop": ("errors_and_fixes",)},
    "template_minimal_minus_open": {"drop": ("open_questions_pending_asks",)},
    "template_minimal_minus_values": {"drop": ("critical_exact_values",)},
    "template_minimal_plus_success": {"add": (("Success criteria", "success_criteria"),)},
}


def _locate_turn(item: str, turns: list[dict[str, Any]]) -> str:
    """Best-effort turn number for a bullet.

    Attribution is a string search, not model output: the generator never says
    which turn a fact came from, and asking it would add another thing a weak
    model can get wrong.
    """
    needle = _compact(item)
    if not needle:
        return "?"
    words = [word for word in needle.split() if len(word) >= 4][:6]
    for turn in turns:
        haystack = _compact(turn.get("text", ""))
        if needle in haystack or (words and all(word in haystack for word in words)):
            return str(turn.get("n", "?"))
    return "?"


def render_minimal_template(
    obj: dict[str, Any],
    *,
    drop: tuple[str, ...] = (),
    add: tuple[tuple[str, str], ...] = (),
    turns: list[dict[str, Any]] | None = None,
    anchors: dict[str, str] | None = None,
    max_per_section: int = 0,
) -> str:
    """The full template cut down to the sections that earned their tokens.

    ``max_per_section`` is the cap arm: a chatty generation cannot spend the
    whole prefix on one section.  ``anchors`` carries exact turn numbers (the
    deterministic arm knows them); ``turns`` instead locates each bullet by
    search, which is what the generated arms have to do.
    """
    sections = [pair for pair in MINIMAL_TEMPLATE_KEYS if pair[1] not in drop]
    sections.extend(add)
    lines = ["## Session handoff (key facts)"]
    for heading, key in sections:
        items = _as_list(obj.get(key))
        if not items:
            continue
        if max_per_section:
            items = items[:max_per_section]
        lines.append("")
        lines.append(f"### {heading}")
        for item in items:
            if anchors and item in anchors:
                lines.append(f"- {item} (turn {anchors[item]})")
            elif turns:
                lines.append(f"- {item} (turn {_locate_turn(item, turns)})")
            else:
                lines.append(f"- {item}")
    return "\n".join(lines)


def render_minimal_arm(
    arm: str, obj: dict[str, Any], turns: list[dict[str, Any]] | None = None
) -> str:
    """Render one of the MINIMAL_ARM_KWARGS variants."""
    kwargs = dict(MINIMAL_ARM_KWARGS.get(arm, {}))
    if kwargs.pop("use_anchors", False) and turns:
        kwargs["turns"] = turns
    return render_minimal_template(obj, **kwargs)


def _minimal_arm_keys(arm: str) -> tuple[str, ...]:
    """Object keys a minimal variant actually renders."""
    kwargs = MINIMAL_ARM_KWARGS.get(arm, {})
    drop = kwargs.get("drop", ())
    keys = [key for _, key in MINIMAL_TEMPLATE_KEYS if key not in drop]
    keys.extend(key for _, key in kwargs.get("add", ()))
    return tuple(keys)


# --- deterministic extraction ------------------------------------------------
# Every section that measurably earns its tokens is a copying task -- a path, a
# value, an error line, a user's request -- not a summarising one.  This arm
# removes the model from the loop entirely for those sections: no generation
# call, no hallucination, and no silent omission of the kind that left all 30
# generated templates without decisions.
_DET_ERROR_RE = re.compile(
    r"^(?:Traceback|panic:|npm ERR!|error TS\d+|SyntaxError|TypeError|ValueError|"
    r"error:|fatal error|segmentation fault)",
    re.I,
)
_DET_PATH_RE = re.compile(
    r"(?:[\w.-]+/)*[\w.-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|ya?ml|go|rs|java|lua|sql|sh|css|html)"
    r"(?::\d+)?"
)
_DET_VALUE_RE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s?(?:ms|s|%|MB|GB|KB|rps|days?|hours?|minutes?|turns?)\b",
    re.I,
)
_DET_ASK_RE = re.compile(
    r"\b(?:please|let's|lets|add|remove|rename|make sure|don't|do not|wait until|one more|"
    r"need to|must|should|instead|avoid|ensure)\b",
    re.I,
)
DET_CAPS = {
    "files_symbols_references": 12,
    "critical_exact_values": 8,
    "errors_and_fixes": 6,
    "open_questions_pending_asks": 8,
}


def _dedupe_against_ledger(items: list[str], ledger_compact: str) -> list[str]:
    """Drop bullets the injected ledger already carries verbatim-ish."""
    if not ledger_compact:
        return items
    kept = []
    for item in items:
        compact = _compact(item)
        if compact and compact in ledger_compact:
            continue
        kept.append(item)
    return kept


def _tidy(item: str, limit: int = 160) -> str:
    """Strip list markers and cut on a word boundary.

    A transcript line often starts with "- ", which would render as a double
    bullet, and a hard slice mid-word reads as a typo rather than a truncation.
    """
    text = re.sub(r"^[-*+]\s+", "", str(item).strip())
    if len(text) > limit:
        text = text[: limit - 3].rsplit(" ", 1)[0] + "..."
    return text


def extract_deterministic_facts(
    turns: list[dict[str, Any]], ledger_text: str = ""
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build a handoff from the transcript with regexes, no model call.

    Returns the template object and an {item: turn} map, so the render carries
    exact anchors rather than a search result.
    """
    ledger_compact = _compact(ledger_text)
    found: dict[str, list[str]] = {key: [] for key in DET_CAPS}
    anchors: dict[str, str] = {}
    for turn in turns:
        role = str(turn.get("role", "")).lower()
        text = str(turn.get("text", ""))
        where = str(turn.get("n", "?"))
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if _DET_ERROR_RE.match(stripped):
                tidy = _tidy(stripped)
                found["errors_and_fixes"].append(tidy)
                anchors.setdefault(tidy, where)
            for path in _DET_PATH_RE.findall(stripped):
                found["files_symbols_references"].append(path)
                anchors.setdefault(path, where)
            for value in _DET_VALUE_RE.findall(stripped):
                found["critical_exact_values"].append(value.strip())
                anchors.setdefault(value.strip(), where)
            # Only user turns carry asks; an assistant sentence that happens to
            # say "should" is commentary, not a request to act on.
            if role == "user" and _DET_ASK_RE.search(stripped):
                for sentence in re.split(r"(?<=[.!?])\s+", stripped):
                    if _DET_ASK_RE.search(sentence):
                        ask = _tidy(sentence)
                        found["open_questions_pending_asks"].append(ask)
                        anchors.setdefault(ask, where)
    obj: dict[str, Any] = {}
    for key, items in found.items():
        deduped = _dedupe_against_ledger(items, ledger_compact)
        # Preserve first-seen order, drop exact repeats.
        unique = list(dict.fromkeys(deduped))[: DET_CAPS[key]]
        if unique:
            obj[key] = unique
    return obj, anchors


def render_headers_only() -> str:
    headings = [
        "Goal / intent",
        "Constraints / preferences",
        "Success criteria",
        "Decisions and rationale",
        "Progress (done / in progress / blocked)",
        "Files, symbols, exact references",
        "Errors and fixes",
        "Open questions / pending asks",
        "Remaining work / next step",
        "Critical exact values",
    ]
    lines = ["## Session handoff"]
    for heading in headings:
        lines.append("")
        lines.append(f"### {heading}")
        lines.append("- (none)")
    return "\n".join(lines)


def gold_template_obj(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal_intent": scenario["goal"],
        "constraints_preferences": scenario["constraints"],
        "success_criteria": scenario["success_criteria"],
        "decisions_and_rationale": [
            f"{scenario['decision']} because {scenario['decision_reason']}",
            f"Correction: {scenario['correction']}",
        ],
        "progress": {
            "done": ["Initial implementation in place"],
            "in_progress": ["Apply the correction and exact fix"],
            "blocked": [scenario["error"]],
        },
        "files_symbols_references": [
            f"{scenario['artifact_path']} {scenario['artifact_symbol']} {scenario['artifact_line']}"
        ],
        "errors_and_fixes": [f"{scenario['error']} -- fix: {scenario['fix']}"],
        "open_questions_pending_asks": [scenario["open_ask"]],
        "remaining_work_next_step": [scenario["next_step"]],
        "critical_exact_values": scenario["exact_values"],
    }


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _recall(groups: list[list[str]], text: str) -> float:
    if not groups:
        return 1.0
    compact = _compact(text)
    hits = sum(1 for group in groups if all(_compact(term) in compact for term in group))
    return hits / len(groups)


def template_valid(obj: dict[str, Any], scenario: dict[str, Any], rendered: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    goal_text = _as_list(obj.get("goal_intent"))
    goal_recall = _recall(scenario["keywords"]["goal"], " ".join(goal_text))
    if goal_recall < 1.0:
        missing.append("exact goal")
    exact_refs = (
        _as_list(obj.get("files_symbols_references"))
        + _as_list(obj.get("critical_exact_values"))
        + _as_list(obj.get("errors_and_fixes"))
    )
    compact_refs = _compact(" ".join(exact_refs))
    has_path = _compact(scenario["artifact_path"]) in compact_refs
    has_number = any(
        _compact(term) in compact_refs
        for group in scenario["keywords"]["values"]
        for term in group
    )
    if not (has_path or has_number):
        missing.append("exact path or number")
    success_text = " ".join(_as_list(obj.get("success_criteria")))
    if _recall(scenario["keywords"]["success"], success_text) < 1.0:
        missing.append("success criteria")
    open_text = " ".join(_as_list(obj.get("open_questions_pending_asks")))
    if _recall(scenario["keywords"]["open_loop"], open_text) < 1.0:
        missing.append("pending ask")
    # Decision and correction are what the replacement arms lost: every 7B
    # template dropped both fields entirely, so a template that replaced the
    # ledger lost corrections the ledger still carried (-0.38 in the v8 run).
    # Enforcing full keyword recall here rejected all 60 of that model's
    # attempts, so the bar is "mentions it at all" -- enough to stop the field
    # being silently empty, loose enough that a paraphrase passes.
    # TC_BENCH_COVERAGE_MIN=-1 restores the old accept-anything behaviour.
    coverage = template_coverage(obj, scenario)
    for dim, label in (
        ("decision", "the early decision with its rationale"),
        ("correction", "the later correction that supersedes it"),
    ):
        if scenario["keywords"].get(dim) and coverage[dim] <= COVERAGE_MIN:
            missing.append(f"{label} (coverage {coverage[dim]:.0%})")
    return (not missing, missing)


def template_coverage(obj: dict[str, Any], scenario: dict[str, Any]) -> dict[str, float]:
    """Diagnostic coverage for fields the generator tends to drop.

    These are reported, not enforced.  Making them blocking turned every one of
    a 7B model's 60 generation attempts into a failure, which leaves the
    template arms empty and measures nothing.  The benchmark's job is to
    quantify what the template carries, so the number is recorded beside the
    result instead of gating it.

    Constraint: keyword groups present in constraints_preferences.
    Decision / correction: scored against decisions_and_rationale only -- the
    spec puts a superseding correction there, and scoring the whole object lets
    a template pass because next_step happens to repeat the same words.
    """
    decision_text = " ".join(_as_list(obj.get("decisions_and_rationale")))
    return {
        "constraints": _recall(
            scenario["keywords"]["constraints"],
            " ".join(_as_list(obj.get("constraints_preferences"))),
        ),
        "decision": _recall(scenario["keywords"]["decision"], decision_text),
        "correction": _recall(scenario["keywords"]["correction"], decision_text),
    }


def generate_template(
    turns: list[dict[str, Any]], scenario: dict[str, Any], tag: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate and validate one template, retrying with a corrective prompt."""
    base = _template_request(turns)
    prompt = base
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, 4):
        result = call_model(TEMPLATE_SYSTEM, prompt, num_predict=1200, tag=f"{tag}:gen{attempt}")
        obj = result["parsed"]
        rendered = render_template(obj)
        ok, missing = template_valid(obj, scenario, rendered)
        coverage = template_coverage(obj, scenario)
        attempts.append(
            {
                "attempt": attempt,
                "ok": ok,
                "missing": missing,
                "coverage": coverage,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "raw": result["raw"][:4000],
            }
        )
        print(
            f"[template {tag} attempt {attempt}] ok={ok} missing={missing} "
            f"cov={coverage} chars={len(rendered)} ptok={result['prompt_tokens']} "
            f"ctok={result['completion_tokens']} lat={result['latency_s']:.1f}s",
            flush=True,
        )
        if ok:
            return obj, {
                "attempts": attempts,
                "valid": True,
                "missing": [],
                "coverage": coverage,
            }
        prompt = base + "\n\n" + _template_corrective(missing)
    return {}, {"attempts": attempts, "valid": False, "missing": attempts[-1]["missing"]}


def update_template(
    previous: dict[str, Any],
    previous_turns: list[dict[str, Any]],
    new_turns: list[dict[str, Any]],
    scenario: dict[str, Any],
    tag: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _update_request(previous, new_turns)
    prompt = base
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, 4):
        result = call_model(TEMPLATE_SYSTEM, prompt, num_predict=1200, tag=f"{tag}:upd{attempt}")
        obj = result["parsed"]
        rendered = render_template(obj)
        ok, missing = template_valid(obj, scenario, rendered)
        coverage = template_coverage(obj, scenario)
        attempts.append(
            {
                "attempt": attempt,
                "ok": ok,
                "missing": missing,
                "coverage": coverage,
                "prompt_tokens": result["prompt_tokens"],
                "completion_tokens": result["completion_tokens"],
                "raw": result["raw"][:4000],
            }
        )
        print(
            f"[template {tag} update {attempt}] ok={ok} missing={missing} "
            f"cov={coverage} chars={len(rendered)} ptok={result['prompt_tokens']} "
            f"ctok={result['completion_tokens']} lat={result['latency_s']:.1f}s",
            flush=True,
        )
        if ok:
            return obj, {
                "attempts": attempts,
                "valid": True,
                "missing": [],
                "coverage": coverage,
            }
        prompt = base + "\n\n" + _template_corrective(missing)
    return {}, {"attempts": attempts, "valid": False, "missing": attempts[-1]["missing"]}


def _load_template_cache() -> dict[str, Any]:
    try:
        data = json.loads(TEMPLATE_CACHE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[template cache] ignoring unreadable cache: {exc}", flush=True)
        return {}
    return data if isinstance(data, dict) else {}


def _save_template_cache(cache: dict[str, Any]) -> None:
    try:
        TEMPLATE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = TEMPLATE_CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(TEMPLATE_CACHE_PATH)
    except OSError as exc:
        # A cache write must never kill a multi-hour run.
        print(f"[template cache] write failed: {exc}", flush=True)


def _freeze_templates(result: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy: depth int keys become strings, 'turns' is dropped."""
    frozen: dict[str, Any] = {}
    for field in ("anchored", "anchored_meta", "regen", "regen_meta"):
        frozen[field] = {str(depth): value for depth, value in result[field].items()}
    return frozen


def _thaw_templates(frozen: dict[str, Any]) -> dict[str, Any]:
    thawed: dict[str, Any] = {"turns": {}}
    for field in ("anchored", "anchored_meta", "regen", "regen_meta"):
        section = frozen[field]
        if not isinstance(section, dict):
            raise TypeError(f"{field} is not a mapping")
        thawed[field] = {int(depth): value for depth, value in section.items()}
    if not thawed["anchored"]:
        raise ValueError("cached generation has no anchored templates")
    return thawed


def generate_templates_for_scenario(
    scenario: dict[str, Any],
    depths: tuple[int, ...],
    regen_depths: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Return anchored and regenerated templates at the requested depths.

    ``depths`` drives the anchored chain (so a 24->60->96 chain can be built
    even when only 24 and 96 are measured).  ``regen_depths`` defaults to
    ``depths`` and controls which from-scratch summaries are generated.
    """
    regen_depths = tuple(regen_depths or depths)
    all_depths = tuple(sorted(set(depths) | set(regen_depths)))
    # The validation policy is part of the identity: a cache keyed only by
    # model and depths would serve templates generated under a looser rule
    # after the rule was tightened.
    cache_key = (
        f"{scenario['id']}|{PROVIDER}:{MODEL}|seed={SEED}|d={list(depths)}"
        f"|r={list(regen_depths)}|cov={COVERAGE_MIN}"
    )
    cache = _load_template_cache()
    if cache_key in cache:
        try:
            cached = _thaw_templates(cache[cache_key])
            # Turns are pure functions of scenario+seed; rebuild rather than
            # trust a cached copy so a scenario edit cannot be masked.
            cached["turns"] = {depth: turns_for(scenario, depth) for depth in all_depths}
            print(f"[template {scenario['id']}] cache hit", flush=True)
            return cached
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[template {scenario['id']}] cache unusable ({exc}); regenerating", flush=True)
    turns_by_depth = {depth: turns_for(scenario, depth) for depth in all_depths}
    anchored: dict[int, dict[str, Any]] = {}
    anchored_meta: dict[int, dict[str, Any]] = {}
    regen: dict[int, dict[str, Any]] = {}
    regen_meta: dict[int, dict[str, Any]] = {}
    previous: dict[str, Any] | None = None
    previous_depth = 0
    for depth in depths:
        turns = turns_by_depth[depth]
        tag = f"{scenario['id']}/{depth}"
        # Anchored: summarize depth 1, then update with only the new turns.
        if previous is None:
            anchored[depth], anchored_meta[depth] = generate_template(
                turns, scenario, f"{tag}/anchored"
            )
        else:
            try:
                anchored[depth], anchored_meta[depth] = update_template(
                    previous,
                    turns_by_depth[previous_depth][:previous_depth],
                    turns[previous_depth:],
                    scenario,
                    f"{tag}/anchored",
                )
            except RuntimeError as exc:
                # An update whose JSON never parses must not kill a multi-hour
                # run: regenerate from the full transcript instead, which is a
                # valid way to build the same summary, and record the fallback.
                anchored[depth], anchored_meta[depth] = generate_template(
                    turns, scenario, f"{tag}/anchored-fallback"
                )
                anchored_meta[depth]["fallback"] = f"update failed, regenerated: {exc}"[:300]
                print(
                    f"[template {tag}/anchored] update failed; regenerated instead",
                    flush=True,
                )
        previous = anchored[depth]
        previous_depth = depth
    for depth in regen_depths:
        turns = turns_by_depth[depth]
        try:
            regen[depth], regen_meta[depth] = generate_template(
                turns, scenario, f"{scenario['id']}/{depth}/regen"
            )
        except RuntimeError as exc:
            regen[depth], regen_meta[depth] = {}, {
                "attempts": [],
                "valid": False,
                "missing": ["generation failed"],
                "fallback": f"regen failed: {exc}"[:300],
            }
            print(f"[template {scenario['id']}/{depth}/regen] failed outright", flush=True)
    result = {
        "turns": turns_by_depth,
        "anchored": anchored,
        "anchored_meta": anchored_meta,
        "regen": regen,
        "regen_meta": regen_meta,
    }
    # Only cache a generation whose JSON actually parsed: a failed generation
    # recorded as valid=False would otherwise be frozen forever.
    if all(meta.get("valid") for meta in anchored_meta.values()) and all(
        meta.get("valid") for meta in regen_meta.values()
    ):
        cache[cache_key] = _freeze_templates(result)
        _save_template_cache(cache)
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

RECALL_SHAPE = (
    '{"goal":"one sentence or (none)",'
    '"constraints":["..."],'
    '"success_criteria":["..."],'
    '"decisions":["early decision + rationale"],'
    '"corrections":["later correction that supersedes a decision"],'
    '"artifacts":["exact path, symbol and line"],'
    '"errors":["exact error text and fix"],'
    '"open_loops":["pending user asks"],'
    '"next_step":"the next implementation step",'
    '"exact_values":["port, version, threshold, command"]}'
)
COMBINED_SHAPE = RECALL_SHAPE[:-1] + ',"decision":"A|B|C|D","reason":"one sentence"}'
SYSTEM = (
    "You are continuing a long software session. The context below is the compacted "
    "record of earlier turns. Answer the next request using only that context. If the "
    "context does not state a field, leave it empty; never invent one. Return one JSON "
    "object and nothing else."
)


def make_recall_prompt(context: str) -> str:
    return (
        "Session context:\n"
        f"{context}\n\n"
        "---\n"
        "Restate everything the context records about this session. Be exhaustive:\n"
        "- Report every constraint and preference as a separate item, copied verbatim, "
        "including any non-goal; do not summarize, merge or omit them.\n"
        "- Report the early decision and its rationale separately from any later correction.\n"
        "- Report every file path, symbol and line reference under artifacts.\n"
        "- Report every error message and its exact fix under errors.\n"
        "- Report every exact number, port, version, threshold and command under "
        "exact_values.\n"
        "- Report every pending user ask under open_loops.\n"
        "Copy exact paths, symbols, line numbers, error text and values verbatim. Do not "
        "choose an action in this call.\n"
        f"Return JSON exactly in this shape: {RECALL_SHAPE}"
    )


def make_combined_prompt(context: str, scenario: dict[str, Any]) -> str:
    """One structured call: recall fields first, then the 4-option choice.

    The two components are scored independently and deterministically.  They
    share one HTTP call only to keep the local-7B run inside the time budget;
    the JSON fields are kept distinct so the choice cannot leak into a recall
    field.
    """
    return (
        "Session context:\n"
        f"{context}\n\n"
        "---\n"
        "Part 1 -- recall. Restate everything the context records about this "
        "session. Be exhaustive:\n"
        "- Report every constraint and preference as a separate item, copied verbatim, "
        "including any non-goal; do not summarize, merge or omit them.\n"
        "- Report the early decision and its rationale separately from any later correction.\n"
        "- Report every file path, symbol and line reference under artifacts.\n"
        "- Report every error message and its exact fix under errors.\n"
        "- Report every exact number, port, version, threshold and command under "
        "exact_values.\n"
        "- Report every pending user ask under open_loops.\n"
        "Copy exact paths, symbols, line numbers, error text and values verbatim.\n\n"
        "Part 2 -- next action. Then answer this request:\n"
        f"{scenario['probe']}\n\n"
        "Choose exactly one option (A, B, C or D) that best serves the session's active "
        "mission and constraints, including any later correction. Put the choice in the "
        "decision field and nothing else there.\n"
        f"Return one JSON object exactly in this shape: {COMBINED_SHAPE}"
    )


def make_decision_prompt(context: str, scenario: dict[str, Any]) -> str:
    return (
        "Session context:\n"
        f"{context}\n\n"
        "---\n"
        f"Next request:\n{scenario['probe']}\n\n"
        "Choose exactly one option (A, B, C or D) that best serves the session's active "
        "mission and constraints, including any later correction. Return JSON exactly in "
        'this shape: {"decision":"A|B|C|D","reason":"one sentence"}'
    )


def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{k} {_as_text(v)}" for k, v in value.items())
    return "" if value is None else str(value)


def _as_decision(value: Any) -> str:
    text = _as_text(value).strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    return match.group(1) if match else text[:1]


def score_recall(parsed: dict[str, Any], scenario: dict[str, Any]) -> dict[str, float]:
    """Deterministic keyword-group recall over the whole recall response.

    Scoring the whole object rather than one field makes the metric insensitive
    to which JSON field the model chose (small models often put a correction
    under ``decisions`` or a value under ``errors``).  The keyword groups are
    disjoint enough that this measures recall of the fact, not field discipline.
    """
    keywords = scenario["keywords"]
    full_text = _as_text(parsed)
    goal = _recall(keywords["goal"], full_text)
    constraints = _recall(keywords["constraints"], full_text)
    success = _recall(keywords["success"], full_text)
    decision = _recall(keywords["decision"], full_text)
    correction = _recall(keywords["correction"], full_text)
    artifact = _recall(keywords["artifact"], full_text)
    error = _recall(keywords["error"], full_text)
    open_loop = _recall(keywords["open_loop"], full_text)
    next_step = _recall(keywords["next_step"], full_text)
    values = _recall(keywords["values"], full_text)
    return {
        "goal_recall": goal,
        "constraint_recall": constraints,
        "success_recall": success,
        "decision_recall": decision,
        "correction_recall": correction,
        "artifact_recall": artifact,
        "error_recall": error,
        "values_recall": values,
        "fact_recall": statistics.fmean([error, values]),
        "open_loop_recall": open_loop,
        "next_step_recall": next_step,
    }


def gold_answer(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        "goal": scenario["goal"],
        "constraints": scenario["constraints"],
        "success_criteria": scenario["success_criteria"],
        "decisions": [f"{scenario['decision']} because {scenario['decision_reason']}"],
        "corrections": [scenario["correction"]],
        "artifacts": [
            f"{scenario['artifact_path']} {scenario['artifact_symbol']} {scenario['artifact_line']}"
        ],
        "errors": [f"{scenario['error']} -- fix: {scenario['fix']}"],
        "open_loops": [scenario["open_ask"]],
        "next_step": scenario["next_step"],
        "exact_values": scenario["exact_values"],
    }


def task_success(row: dict[str, Any]) -> bool:
    return bool(
        row.get("decision_correct")
        and row["goal_recall"] >= 0.5
        and row["constraint_recall"] >= 2 / 3
        and (row["artifact_recall"] >= 2 / 3 or row["error_recall"] >= 2 / 3)
    )


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def _run_scorer_controls(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    dims = [
        "goal_recall",
        "constraint_recall",
        "success_recall",
        "decision_recall",
        "correction_recall",
        "artifact_recall",
        "error_recall",
        "values_recall",
        "open_loop_recall",
        "next_step_recall",
    ]
    for scenario in scenarios:
        gold = score_recall(gold_answer(scenario), scenario)
        checks.append(
            {
                "name": f"scorer known-positive: gold answer scores 1.0 -- {scenario['id']}",
                "ok": all(gold[dim] == 1.0 for dim in dims),
                "detail": gold,
            }
        )
        empty = score_recall({}, scenario)
        checks.append(
            {
                "name": f"scorer known-negative: empty answer scores 0 -- {scenario['id']}",
                "ok": all(empty[dim] == 0.0 for dim in dims),
                "detail": empty,
            }
        )
        donor = SCENARIOS[(SCENARIOS.index(scenario) + 1) % len(SCENARIOS)]
        shuffled = score_recall(gold_answer(donor), scenario)
        checks.append(
            {
                "name": f"scorer known-negative: other-scenario gold scores 0 -- {scenario['id']}",
                "ok": all(shuffled[dim] == 0.0 for dim in dims),
                "detail": shuffled,
            }
        )
        # The gold answer is hand-written, so scoring it 1.0 only proves the
        # keyword lists match that hand-written text.  It does not prove the
        # terms are reachable from the session the model actually sees.  A
        # keyword that never appears in the transcript makes its group
        # unanswerable except by accident and silently deflates every arm.
        unreachable: dict[str, list[list[str]]] = {}
        for depth in (24, 96):
            compact = _compact(render_full_transcript(turns_for(scenario, depth)))
            for field, groups in scenario["keywords"].items():
                for group in groups:
                    if any(_compact(term) not in compact for term in group):
                        unreachable.setdefault(f"{field}@{depth}", []).append(group)
        checks.append(
            {
                "name": f"scorer known-positive: every keyword is reachable in the transcript -- {scenario['id']}",
                "ok": not unreachable,
                "detail": unreachable,
            }
        )
    failures = [check for check in checks if not check["ok"]]
    return {"checks": checks, "passed": not failures}


def _run_full_context_control(
    plugin, scenarios: list[dict[str, Any]], depth: int
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for scenario in scenarios:
        turns = turns_for(scenario, depth)
        full = render_full_transcript(turns)
        result = call_model(
            SYSTEM,
            make_combined_prompt(full, scenario),
            num_predict=560,
            tag=f"ctrl-full/{scenario['id']}",
        )
        scored = score_recall(result["parsed"], scenario)
        chosen = _as_decision(result["parsed"].get("decision"))
        # The known-positive gates on what the decision metric actually uses:
        # can the model pick the right option when it has the whole transcript?
        # Constraint recall is recorded but not gated per scenario.  A 7B
        # under-recalls the constraint groups even with the full transcript in
        # front of it (measured 0.50-0.75 at depth 24, all four decisions
        # still correct), which is a model limit, not a broken item -- and
        # gating on it blocked every local run from reporting at all.  A floor
        # is applied across scenarios below so a real scorer break (all zeros)
        # still fails loudly.
        ok = chosen == scenario["correct"]
        checks.append(
            {
                "name": f"full-context known-positive -- {scenario['id']}",
                "ok": ok,
                "detail": {
                    "chosen": chosen,
                    "expected": scenario["correct"],
                    "goal_recall": scored["goal_recall"],
                    "constraint_recall": scored["constraint_recall"],
                    # Full recall vector: with the whole transcript in front of
                    # it this is the model's ceiling for each metric, which is
                    # what arms should be read against when the model is weak.
                    "scored": scored,
                    "raw": result["raw"],
                },
            }
        )
        print(
            f"[control full {scenario['id']}] chosen={chosen} expected={scenario['correct']} "
            f"goal={scored['goal_recall']:.2f} constraints={scored['constraint_recall']:.2f}",
            flush=True,
        )
    # Cross-scenario floor: proves the recall scorer is actually reading the
    # transcript.  A broken scorer (renamed keys, empty gold lists) reports
    # zeros everywhere and would still let the decision check pass by luck, so
    # fail loudly instead.  The bar is deliberately low -- it catches a dead
    # scorer, not a weak model.
    mean_goal = sum(c["detail"]["goal_recall"] for c in checks) / max(len(checks), 1)
    mean_constraints = sum(c["detail"]["constraint_recall"] for c in checks) / max(
        len(checks), 1
    )
    checks.append(
        {
            "name": "full-context recall floor (scorer alive)",
            "ok": bool(mean_goal >= 0.5 and mean_constraints >= 0.4),
            "detail": {
                "mean_goal_recall": mean_goal,
                "mean_constraint_recall": mean_constraints,
                "floor_goal": 0.5,
                "floor_constraints": 0.4,
            },
        }
    )
    print(
        f"[control full recall-floor] goal={mean_goal:.2f} "
        f"constraints={mean_constraints:.2f}",
        flush=True,
    )
    failures = [check for check in checks if not check["ok"]]
    return {"checks": checks, "passed": not failures, "depth": depth}


def _probe_options(probe: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for line in probe.splitlines()[1:]:
        match = re.match(r"^([ABCD])\)\s*(.*)$", line.strip())
        if match:
            options[match.group(1)] = match.group(2).strip()
    return options


def _run_probe_shape_control(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Structural checks on the multiple-choice probes.

    A probe only measures session memory if the answer cannot be read off the
    options themselves.  Two cues did exactly that in the first DeepSeek run:
    the correct option was the longest and most detailed one, and the correct
    letter was biased toward C.  Both are checked here so a regression fails
    the run instead of quietly inflating the decision metric.
    """
    checks: list[dict[str, Any]] = []
    letters: list[str] = []
    for scenario in scenarios:
        options = _probe_options(scenario["probe"])
        lengths = [len(text) for text in options.values()]
        ratio = (max(lengths) / min(lengths)) if lengths and min(lengths) else 0.0
        checks.append(
            {
                "name": f"probe shape -- {scenario['id']}",
                "ok": len(options) == 4 and scenario["correct"] in options and ratio <= 1.30,
                "detail": {
                    "n_options": len(options),
                    "correct_present": scenario["correct"] in options,
                    "length_ratio": round(ratio, 2),
                },
            }
        )
        letters.append(scenario["correct"])
    counts = {letter: letters.count(letter) for letter in "ABCD"}
    limit = (len(scenarios) + 1) // 2
    checks.append(
        {
            "name": "probe answer balance",
            "ok": max(counts.values()) <= limit,
            "detail": {"counts": counts, "limit": limit},
        }
    )
    return {"checks": checks, "passed": all(check["ok"] for check in checks)}


def _run_no_context_control(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    correct = 0
    for scenario in scenarios:
        decision = call_model(
            SYSTEM,
            make_decision_prompt("(no session context available)", scenario),
            num_predict=120,
            tag=f"ctrl-nocontext/{scenario['id']}",
        )
        chosen = _as_decision(decision["parsed"].get("decision"))
        hit = chosen == scenario["correct"]
        correct += 1 if hit else 0
        checks.append(
            {
                "name": f"no-context known-negative -- {scenario['id']}",
                # Known-negative, per scenario.  It does NOT gate on its own:
                # one correct guess out of four is what chance looks like, and
                # gating per scenario made the run abort on a single hit while
                # the group as a whole was at chance.  The aggregate check
                # below is the gate; these entries are the evidence behind it.
                # (Hardcoding ok=True here once hid a real failure -- 4/4 with
                # no context -- behind "All controls passed: True".)
                "ok": not hit,
                "gating": False,
                "detail": {"chosen": chosen, "expected": scenario["correct"], "correct": hit},
            }
        )
    accuracy = correct / len(scenarios)
    passed = accuracy <= 0.5
    # The gate: without any context the model must be at or near chance.  A
    # probe set that scores high here is measuring the options, not memory.
    checks.append(
        {
            "name": "no-context accuracy at chance (gate)",
            "ok": passed,
            "detail": {
                "accuracy": accuracy,
                "chance_accuracy": 1 / 4,
                "correct": correct,
                "n": len(scenarios),
                "bar": 0.5,
            },
        }
    )
    return {
        "checks": checks,
        "passed": passed,
        "accuracy": accuracy,
        "chance_accuracy": 1 / 4,
        "correct": correct,
        "n": len(scenarios),
    }


def _run_template_controls(scenario: dict[str, Any]) -> dict[str, Any]:
    gold_text = render_template(gold_template_obj(scenario))
    donor = gold_template_obj(SCENARIOS[(SCENARIOS.index(scenario) + 1) % len(SCENARIOS)])
    shuffled_text = render_template(donor)
    gold_recall = call_model(
        SYSTEM, make_recall_prompt(gold_text), num_predict=450, tag="ctrl-template-gold"
    )
    shuffled_recall = call_model(
        SYSTEM, make_recall_prompt(shuffled_text), num_predict=450, tag="ctrl-template-shuffled"
    )
    gold_scored = score_recall(gold_recall["parsed"], scenario)
    shuffled_scored = score_recall(shuffled_recall["parsed"], scenario)
    dims = [
        "goal_recall",
        "constraint_recall",
        "success_recall",
        "decision_recall",
        "correction_recall",
        "artifact_recall",
        "error_recall",
        "values_recall",
        "open_loop_recall",
        "next_step_recall",
    ]
    gold_mean = statistics.fmean(gold_scored[dim] for dim in dims)
    shuffled_mean = statistics.fmean(shuffled_scored[dim] for dim in dims)
    gold_obj = gold_template_obj(scenario)
    gold_valid, gold_missing = template_valid(gold_obj, scenario, gold_text)
    no_success = {**gold_obj, "success_criteria": []}
    success_valid, success_missing = template_valid(
        no_success, scenario, render_template(no_success)
    )
    no_open = {**gold_obj, "open_questions_pending_asks": []}
    open_valid, open_missing = template_valid(no_open, scenario, render_template(no_open))
    gold_cov = template_coverage(gold_obj, scenario)
    # Keep the early decision and drop only the correction that supersedes it.
    decision_only = {
        **gold_obj,
        "decisions_and_rationale": [
            f"{scenario['decision']} because {scenario['decision_reason']}",
        ],
    }
    decision_only_cov = template_coverage(decision_only, scenario)
    no_decision = {**gold_obj, "decisions_and_rationale": []}
    no_decision_cov = template_coverage(no_decision, scenario)
    checks = [
        {
            "name": "template known-positive: hand-authored gold template has high recall",
            "ok": gold_mean >= 0.8,
            "detail": {"mean_recall": gold_mean, "dimensions": gold_scored, "raw": gold_recall["raw"]},
        },
        {
            "name": "validator known-positive: gold template passes coverage checks",
            "ok": gold_valid,
            "detail": {"missing": gold_missing},
        },
        {
            "name": "validator known-negative: missing success criteria is rejected",
            "ok": not success_valid and "success criteria" in success_missing,
            "detail": {"missing": success_missing},
        },
        {
            "name": "validator known-negative: missing pending ask is rejected",
            "ok": not open_valid and "pending ask" in open_missing,
            "detail": {"missing": open_missing},
        },
        {
            "name": "coverage known-positive: gold template carries constraints, decision and correction",
            "ok": min(gold_cov.values()) >= 1.0,
            "detail": gold_cov,
        },
        {
            "name": "coverage known-negative: dropping the decision entry zeroes decision coverage",
            "ok": no_decision_cov["decision"] == 0.0,
            "detail": no_decision_cov,
        },
        {
            "name": "coverage known-negative: dropping the correction lowers correction coverage",
            "ok": decision_only_cov["correction"] < decision_only_cov["decision"],
            "detail": decision_only_cov,
        },
        {
            "name": "template known-negative: shuffled gold template has low recall",
            "ok": shuffled_mean <= 0.25,
            "detail": {
                "mean_recall": shuffled_mean,
                "dimensions": shuffled_scored,
                "raw": shuffled_recall["raw"],
            },
        },
    ]
    failures = [check for check in checks if not check["ok"]]
    return {"checks": checks, "passed": not failures}


def _run_generation_control(templates: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for scenario_id, data in templates.items():
        for strategy in ("anchored", "regen"):
            for depth, meta in data[f"{strategy}_meta"].items():
                checks.append(
                    {
                        "name": f"template generation sanity -- {scenario_id}/{depth}/{strategy}",
                        "ok": bool(meta["valid"]),
                        "detail": {
                            "missing": meta["missing"],
                            "attempts": len(meta["attempts"]),
                        },
                    }
                )
    failures = [check for check in checks if not check["ok"]]
    return {"checks": checks, "passed": not failures}


# ---------------------------------------------------------------------------
# Arm contexts
# ---------------------------------------------------------------------------

# Fixed decoy for the shuffled-template arm: same shape as a real template,
# unrelated subject matter, no term any benchmark scenario scores.
DECOY_TEMPLATE = (
    "# Session Handoff\n\n"
    "## Goal Intent\nRedesign the marketing newsletter signup page so the brand "
    "palette is applied consistently across browsers.\n\n"
    "## Constraints and Preferences\n"
    "- The brand palette is fixed for this quarter.\n"
    "- Copy tone stays informal.\n"
    "- Non-goal: do not touch the checkout funnel.\n\n"
    "## Success Criteria\n"
    "- Signup form renders correctly in all supported browsers.\n\n"
    "## Files, Symbols and References\n"
    "- theming/palette.css brand_tokens() line 42\n\n"
    "## Decisions and Rationale\n"
    "- Introduce a palette token file rather than hardcoding colours, because "
    "the brand palette changes quarterly.\n\n"
    "## Constraints Still Open\n"
    "- Confirm the informal copy tone with the brand reviewer.\n"
)


def shuffled_template(
    real_text: str, donor_texts: list[str], scenario: dict[str, Any]
) -> str:
    """Same-length template from another scenario, padded or trimmed.

    Tries each donor until one does not leak this scenario's gold keyword
    groups.  Raises if every donor leaks, rather than silently shipping a
    contaminated token control.
    """
    target = len(real_text)
    filler = (
        "\n- Note: unrelated background material kept only to preserve length."
    )
    for donor_text in donor_texts:
        if len(donor_text) > target:
            cut = donor_text.rfind("\n", 0, target)
            if cut < target // 2:
                cut = target
            text = donor_text[:cut]
        else:
            text = donor_text
        while len(text) < target:
            text += filler
        text = text[:target]
        if all(_recall(groups, text) == 0 for groups in scenario["keywords"].values()):
            return text
    # No cross-scenario donor is clean (small scenario sets share vocabulary),
    # so fall back to a fixed decoy written to mention nothing any scenario
    # scores.  The decoy is verified with the same check and raises if it ever
    # leaks, so this cannot silently ship a contaminated control.
    text = DECOY_TEMPLATE
    while len(text) < target:
        text += filler
    text = text[:target]
    if all(_recall(groups, text) == 0 for groups in scenario["keywords"].values()):
        return text
    raise RuntimeError(
        f"no donor template and no decoy is keyword-free for scenario "
        f"{scenario['id']!r}"
    )


# Arms whose context is built from a generated template.  If generation yields
# {} these arms cannot be measured honestly (see _refuse_empty_template).
TEMPLATE_DEPENDENT_ARMS = (
    "current_goals_template",
    "template_minimal",
    "template_only",
    "template_ledger",
    "shuffled_template",
)


def _carries_content(obj: Any) -> bool:
    """True when a template object holds at least one non-empty section."""
    return isinstance(obj, dict) and any(_as_list(value) for value in obj.values())


def _refuse_empty_template(
    arm_list: tuple[str, ...],
    template_obj: dict[str, Any],
    regen_obj: dict[str, Any] | None,
    scenario_id: str,
    depth: int,
) -> None:
    """Exit rather than measure an arm whose template came out empty.

    A template that fails validation is returned as {}, which renders to a full
    skeleton with every section marked "(none)".  The additive arms would then
    be a header plus their ledger baseline, and the run would report "the
    template added nothing" -- a null caused by a failed generation, not by the
    template.  Testing the object rather than the rendered text is deliberate:
    render_template({}) is never the empty string.
    """
    missing = [
        arm
        for arm in arm_list
        if arm in TEMPLATE_DEPENDENT_ARMS and not _carries_content(template_obj)
    ]
    # The minimal variants render only a subset of the sections, so a template
    # carrying content elsewhere still leaves those arms equal to their
    # baseline -- and a minus-* variant whose remaining sections are all empty
    # degenerates the same way.
    for arm in arm_list:
        if arm not in MINIMAL_ARM_KWARGS or not _carries_content(template_obj):
            continue
        if not any(_as_list(template_obj.get(key)) for key in _minimal_arm_keys(arm)):
            if arm not in missing:
                missing.append(arm)
    if "template_regen_only" in arm_list and not _carries_content(regen_obj):
        missing.append("template_regen_only")
    if not missing:
        return
    raise SystemExit(
        f"[{scenario_id}/{depth}] refusing to measure {missing}: template "
        f"generation produced no summary at all ({PROVIDER}:{MODEL}, "
        f"TC_BENCH_COVERAGE_MIN={COVERAGE_MIN}). Measuring it would report a "
        f"null result caused by a failed generation rather than by the "
        f"template. Use a stronger generator, or set TC_BENCH_COVERAGE_MIN=-1 "
        f"to accept whatever structure the model does produce -- and say so "
        f"when reporting."
    )


def build_contexts(
    plugin,
    scenario: dict[str, Any],
    depth: int,
    turns: list[dict[str, Any]],
    state: dict[str, Any],
    session: str,
    template_obj: dict[str, Any],
    regen_obj: dict[str, Any] | None,
    donor_templates: list[str],
    arms: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, int], dict[str, str]]:
    """Build every requested context, with its template size and provenance.

    Returns (contexts, template_chars, generation).  Size and provenance are
    computed here rather than re-derived by the caller: a second derivation is
    a second chance for the reported cost to disagree with the context that was
    actually measured.
    """
    contexts: dict[str, str] = {}
    sizes: dict[str, int] = {}
    generation: dict[str, str] = {}
    full = render_full_transcript(turns)
    current = plugin_context(plugin, state, session, goals=False)
    current_goals = plugin_context(plugin, state, session, goals=True)
    template_text = render_template(template_obj) if template_obj else ""
    regen_text = render_template(regen_obj) if regen_obj else ""
    _refuse_empty_template(arms, template_obj, regen_obj, scenario["id"], depth)
    ledger = ledger_block(plugin, state, session)
    if "full" in arms:
        contexts["full"] = full
    if "current" in arms:
        contexts["current"] = current
    if "current_goals" in arms:
        contexts["current_goals"] = current_goals
    if "current_goals_template" in arms:
        contexts["current_goals_template"] = template_text + "\n\n" + current_goals
        sizes["current_goals_template"] = len(template_text)
        generation["current_goals_template"] = "anchored"
    # Every minimal variant shares the additive shape, so the only difference
    # between them is which sections they carry.
    for arm in MINIMAL_ARM_KWARGS:
        if arm not in arms:
            continue
        text = render_minimal_arm(arm, template_obj, turns)
        contexts[arm] = text + "\n\n" + current_goals
        sizes[arm] = len(text)
        generation[arm] = "anchored-minimal"
    if "template_det" in arms:
        # No model call: extracted from the transcript, deduped against the
        # ledger, capped per section, and anchored to the turn it came from.
        det_obj, det_anchors = extract_deterministic_facts(turns, ledger)
        text = render_minimal_template(
            det_obj, anchors=det_anchors, max_per_section=6
        )
        contexts["template_det"] = text + "\n\n" + current_goals
        sizes["template_det"] = len(text)
        generation["template_det"] = "deterministic"
    if "template_only" in arms:
        contexts["template_only"] = template_text
        sizes["template_only"] = len(template_text)
        generation["template_only"] = "anchored"
    if "template_ledger" in arms:
        contexts["template_ledger"] = template_text + "\n\n" + ledger
        sizes["template_ledger"] = len(template_text)
        generation["template_ledger"] = "anchored"
    if "shuffled_template" in arms:
        contexts["shuffled_template"] = (
            shuffled_template(template_text, donor_templates, scenario)
            + "\n\n"
            + current_goals
        )
        sizes["shuffled_template"] = len(contexts["shuffled_template"]) - len(current_goals) - 2
        generation["shuffled_template"] = "anchored"
    if "headers_only" in arms:
        header = render_headers_only()
        contexts["headers_only"] = header + "\n\n" + current_goals
        sizes["headers_only"] = len(header)
        generation["headers_only"] = "headers"
    if "template_regen_only" in arms:
        contexts["template_regen_only"] = regen_text
        sizes["template_regen_only"] = len(regen_text)
        generation["template_regen_only"] = "regen"
    return contexts, sizes, generation


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_row(
    scenario: dict[str, Any],
    depth: int,
    arm: str,
    context: str,
    template_chars: int,
    generation: str,
) -> dict[str, Any]:
    result = call_model(
        SYSTEM,
        make_combined_prompt(context, scenario),
        num_predict=560,
        tag=f"{scenario['id']}/{depth}/{arm}/combined",
    )
    scored = score_recall(result["parsed"], scenario)
    chosen = _as_decision(result["parsed"].get("decision"))
    row: dict[str, Any] = {
        "scenario": scenario["id"],
        "depth": depth,
        "arm": arm,
        "generation": generation,
        "context_chars": len(context),
        "template_chars": template_chars,
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "latency_s": result["latency_s"],
        "decision": chosen,
        "expected": scenario["correct"],
        "decision_correct": chosen == scenario["correct"],
        "reason": result["parsed"].get("reason", ""),
        "raw_recall": result["raw"],
        "raw_decision": result["raw"],
    }
    row.update(scored)
    row["task_success"] = task_success(row)
    print(
        f"[{scenario['id']}/{depth}/{arm}] "
        f"goal={row['goal_recall']:.2f} con={row['constraint_recall']:.2f} "
        f"dec={row['decision_recall']:.2f} cor={row['correction_recall']:.2f} "
        f"art={row['artifact_recall']:.2f} err={row['error_recall']:.2f} "
        f"open={row['open_loop_recall']:.2f} choice={row['decision']}"
        f"{'*' if row['decision_correct'] else ''} "
        f"chars={row['context_chars']} lat={row['latency_s']:.1f}s",
        flush=True,
    )
    return row


def aggregate(rows: list[dict[str, Any]], key: str, order: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    metrics = [
        "goal_recall",
        "constraint_recall",
        "success_recall",
        "decision_recall",
        "correction_recall",
        "artifact_recall",
        "error_recall",
        "values_recall",
        "fact_recall",
        "open_loop_recall",
        "next_step_recall",
        "decision_correct",
        "task_success",
        "context_chars",
        "template_chars",
        "prompt_tokens",
        "completion_tokens",
        "latency_s",
    ]
    for value in order:
        group = [row for row in rows if row[key] == value]
        if not group:
            continue
        stats: dict[str, Any] = {"n": len(group)}
        for metric in metrics:
            values = [float(row.get(metric) or 0.0) for row in group]
            stats[metric] = statistics.fmean(values)
        out[value] = stats
    return out


def paired_differences(
    rows: list[dict[str, Any]], baseline: str, metrics: list[str]
) -> dict[str, Any]:
    by_cell: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((row["scenario"], row["depth"]), {})[row["arm"]] = row
    out: dict[str, Any] = {}
    arms = sorted({row["arm"] for row in rows} - {baseline})
    for arm in arms:
        stats: dict[str, Any] = {"n_pairs": 0}
        metric_out: dict[str, Any] = {}
        for metric in metrics:
            diffs: list[float] = []
            wins = losses = ties = 0
            for cell, arm_rows in by_cell.items():
                if baseline not in arm_rows or arm not in arm_rows:
                    continue
                diff = float(arm_rows[arm][metric]) - float(arm_rows[baseline][metric])
                diffs.append(diff)
                if diff > 1e-9:
                    wins += 1
                elif diff < -1e-9:
                    losses += 1
                else:
                    ties += 1
            if diffs:
                metric_out[metric] = {
                    "mean_diff": statistics.fmean(diffs),
                    "min_diff": min(diffs),
                    "max_diff": max(diffs),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "n": len(diffs),
                }
                stats["n_pairs"] = max(stats["n_pairs"], len(diffs))
        stats["metrics"] = metric_out
        out[arm] = stats
    return out


def cost_projection(aggregates: dict[str, Any], baseline: str) -> dict[str, Any]:
    base_chars = aggregates[baseline]["context_chars"]
    out: dict[str, Any] = {}
    for arm, stats in aggregates.items():
        extra_chars = stats["context_chars"] - base_chars
        extra_tokens = extra_chars / 4.0
        out[arm] = {
            "context_chars": stats["context_chars"],
            "template_chars": stats["template_chars"],
            "prompt_tokens_measured": stats["prompt_tokens"],
            "extra_chars_vs_current_goals": extra_chars,
            "extra_tokens_per_turn_estimated": extra_tokens,
            "extra_cost_100_turns_usd_offpeak_miss_estimated": (
                extra_tokens * 100 * PRICE_MISS_LOW / 1_000_000
            ),
            "extra_cost_100_turns_usd_peak_miss_estimated": (
                extra_tokens * 100 * PRICE_MISS_HIGH / 1_000_000
            ),
            "extra_cost_100_turns_usd_offpeak_hit_estimated": (
                extra_tokens * 100 * PRICE_HIT_LOW / 1_000_000
            ),
            "extra_cost_100_turns_usd_peak_hit_estimated": (
                extra_tokens * 100 * PRICE_HIT_HIGH / 1_000_000
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_controls(
    plugin,
    scenarios: list[dict[str, Any]],
    templates: dict[str, Any],
    *,
    pilot: bool,
) -> dict[str, Any]:
    control_scenarios = scenarios[:1] if pilot else scenarios
    controls: dict[str, Any] = {}
    controls["scorer"] = _run_scorer_controls(control_scenarios)
    print(
        f"[controls] scorer passed={controls['scorer']['passed']} "
        f"checks={len(controls['scorer']['checks'])}",
        flush=True,
    )
    controls["full_context"] = _run_full_context_control(plugin, control_scenarios, 24)
    print(
        f"[controls] full-context passed={controls['full_context']['passed']}",
        flush=True,
    )
    # Scenario design is global: check no-context discrimination on every
    # scenario even in pilot mode, where the other controls use one scenario.
    controls["no_context"] = _run_no_context_control(scenarios)
    print(
        f"[controls] no-context accuracy={controls['no_context']['accuracy']:.2f} "
        f"passed={controls['no_context']['passed']}",
        flush=True,
    )
    controls["probe_shape"] = _run_probe_shape_control(scenarios)
    print(
        f"[controls] probe-shape passed={controls['probe_shape']['passed']} "
        f"checks={len(controls['probe_shape']['checks'])}",
        flush=True,
    )
    controls["template"] = _run_template_controls(control_scenarios[0])
    print(
        f"[controls] template passed={controls['template']['passed']}",
        flush=True,
    )
    controls["generation_sanity"] = _run_generation_control(templates)
    print(
        f"[controls] generation-sanity passed={controls['generation_sanity']['passed']} "
        f"checks={len(controls['generation_sanity']['checks'])}",
        flush=True,
    )
    all_checks: list[dict[str, Any]] = []
    for key in (
        "scorer",
        "full_context",
        "no_context",
        "probe_shape",
        "template",
        "generation_sanity",
    ):
        # A group's "passed" flag and the flattened per-check result must agree;
        # they diverged once (a per-scenario no-context check flagged a failure
        # while its group passed at chance).  Checks marked gating=False are
        # evidence for their group's own verdict, not independent gates.
        gating = [check for check in controls[key]["checks"] if check.get("gating", True)]
        all_checks.extend(gating)
        if bool(controls[key].get("passed")) != all(check["ok"] for check in gating):
            # A group that says "passed" while one of its own gating checks
            # failed (or the reverse) is a harness bug, and one that has already
            # produced a run claiming success with a failing control.  Stop
            # rather than report numbers nobody can trust.
            print(
                f"[controls] INTERNAL INCONSISTENCY in {key}: "
                f"passed={controls[key].get('passed')} "
                f"gating_failures={[c['name'] for c in gating if not c['ok']]}",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(3)
    controls["all_passed"] = all(check["ok"] for check in all_checks)
    controls["n_checks"] = len(all_checks)
    controls["failures"] = [check for check in all_checks if not check["ok"]]
    return controls


def _write_checkpoint(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    depths: tuple[int, ...],
    arms: tuple[str, ...],
) -> None:
    """Persist measured rows so a killed run can resume instead of restarting.

    The identity recorded is model/seed/depths/arms/coverage policy -- not the
    harness source. A change to scoring or prompt construction is therefore
    invisible to resume: old rows keep the old scorer's numbers, so discard the
    checkpoint by hand when either changes.
    """
    path.write_text(
        json.dumps(
            {
                "partial": True,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                # Identity, so a later --resume can prove the checkpoint belongs
                # to a run with the same model, seed, depths and arms.
                "settings_snapshot": {
                    "model": MODEL,
                    "provider": PROVIDER,
                    "seed": SEED,
                    "depths": list(depths),
                    "arms": list(arms),
                    "coverage_min": COVERAGE_MIN,
                },
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[checkpoint] {path} rows={len(rows)}", flush=True)


def _scope_note(scenarios: list[dict[str, Any]], depths: tuple[int, ...]) -> str:
    """Describe the scope this run actually had.

    Derived rather than hardcoded: a fixed "4 of 6 scenarios" string was true
    for the v8 run and false for the v9 run, which used all six -- the summary
    would have repeated a claim about its own scope that nothing measured.
    """
    planned_scenarios = len(SCENARIOS)
    planned_depths = (24, 60, 96)
    parts = [
        f"Planned design was {planned_scenarios} scenarios x depths "
        f"{'/'.join(str(d) for d in planned_depths)}."
    ]
    deviations = []
    if len(scenarios) < planned_scenarios:
        deviations.append(f"{len(scenarios)} of {planned_scenarios} scenarios")
    if tuple(depths) != planned_depths:
        deviations.append(
            f"depths {'/'.join(str(d) for d in depths)} instead of "
            f"{'/'.join(str(d) for d in planned_depths)}"
        )
    if deviations:
        parts.append(
            "This run used "
            + " and ".join(deviations)
            + " — a reduction I made to fit available time, NOT a user-approved "
            "scope change."
        )
    else:
        parts.append("This run used the full planned scope.")
    parts.append(
        "Anchored templates are still built through an unmeasured 60-turn step; "
        "the regen arm is measured at every measured depth."
    )
    return " ".join(parts)


def write_summary(
    report: dict[str, Any], rows: list[dict[str, Any]], aggregates: dict[str, Any]
) -> None:
    lines: list[str] = []
    lines.append("# Template compaction experiment -- summary")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("## Controls")
    lines.append("")
    lines.append(f"- All controls passed: **{report['controls']['all_passed']}**")
    lines.append(f"- Checks: {report['controls']['n_checks']}")
    lines.append(
        f"- No-context decision accuracy: {report['controls']['no_context']['accuracy']:.2f} "
        f"(chance {report['controls']['no_context']['chance_accuracy']:.2f})"
    )
    if report["controls"]["failures"]:
        lines.append("")
        lines.append("### Failures")
        for failure in report["controls"]["failures"]:
            lines.append(f"- {failure['name']}")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    header = (
        "| arm | n | goal | constr | success | decision | correction | artifact | "
        "fact | open-loop | choice | task | chars | template chars | prompt tok | latency s |"
    )
    lines.append(header)
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for arm in ARM_RANK:
        stats = aggregates.get(arm)
        if not stats:
            continue
        lines.append(
            f"| {arm} | {stats['n']} | {stats['goal_recall']:.2f} | "
            f"{stats['constraint_recall']:.2f} | {stats['success_recall']:.2f} | "
            f"{stats['decision_recall']:.2f} | {stats['correction_recall']:.2f} | "
            f"{stats['artifact_recall']:.2f} | {stats['fact_recall']:.2f} | "
            f"{stats['open_loop_recall']:.2f} | {stats['decision_correct']:.2f} | "
            f"{stats['task_success']:.2f} | {stats['context_chars']:.0f} | "
            f"{stats['template_chars']:.0f} | {stats['prompt_tokens']:.0f} | "
            f"{stats['latency_s']:.1f} |"
        )
    lines.append("")
    lines.append("## Paired differences vs current_goals")
    lines.append("")
    lines.append("| arm | metric | mean diff | wins | losses | ties | n |")
    lines.append("|---|---|--:|--:|--:|--:|--:|")
    for arm, data in report["paired_differences"].items():
        for metric, value in data["metrics"].items():
            if metric in {
                "decision_correct",
                "task_success",
                "goal_recall",
                "constraint_recall",
                "decision_recall",
                "correction_recall",
                "artifact_recall",
                "fact_recall",
                "open_loop_recall",
            }:
                lines.append(
                    f"| {arm} | {metric} | {value['mean_diff']:+.3f} | "
                    f"{value['wins']} | {value['losses']} | {value['ties']} | {value['n']} |"
                )
    lines.append("")
    lines.append("## Per-depth arm means")
    lines.append("")
    lines.append("| arm | depth | n | goal | constr | correction | artifact | fact | open-loop | choice | task |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for arm in ARM_RANK:
        for depth in report["settings"]["depths"]:
            key = f"{arm}@{depth}"
            stats = report["aggregates_by_depth"].get(key)
            if not stats:
                continue
            lines.append(
                f"| {arm} | {depth} | {stats['n']} | {stats['goal_recall']:.2f} | "
                f"{stats['constraint_recall']:.2f} | {stats['correction_recall']:.2f} | "
                f"{stats['artifact_recall']:.2f} | {stats['fact_recall']:.2f} | "
                f"{stats['open_loop_recall']:.2f} | {stats['decision_correct']:.2f} | "
                f"{stats['task_success']:.2f} |"
            )
    lines.append("")
    lines.append("## Cost projection (estimated)")
    lines.append("")
    lines.append(
        "| arm | extra chars vs current_goals | extra tok/turn | $/100 turns off-peak miss | $/100 turns off-peak hit |"
    )
    lines.append("|---|--:|--:|--:|--:|")
    for arm, data in report["cost_projection"]["arms"].items():
        lines.append(
            f"| {arm} | {data['extra_chars_vs_current_goals']:+.0f} | "
            f"{data['extra_tokens_per_turn_estimated']:+.1f} | "
            f"{data['extra_cost_100_turns_usd_offpeak_miss_estimated']:.4f} | "
            f"{data['extra_cost_100_turns_usd_offpeak_hit_estimated']:.4f} |"
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    usage = report["settings"].get("usage") or {}
    where = "local" if PROVIDER == "ollama" else "remote API"
    lines.append(
        f"- Synthetic sessions, {MODEL} ({where}), small n. No p-values or significance claims."
    )
    lines.append(
        f"- Template generation uses {MODEL}; template quality is measured, not assumed."
    )
    if usage.get("prompt_tokens"):
        lines.append(
            f"- Measured model usage: {usage['calls']} calls, "
            f"{usage['prompt_tokens']:,} prompt / {usage['completion_tokens']:,} completion "
            f"tokens ({usage['reasoning_tokens']:,} of them reasoning)."
        )
    elif usage.get("calls"):
        lines.append(
            f"- {usage['calls']} model calls made; token split was not recorded in this run."
        )
    if PROVIDER == "ollama":
        lines.append("- All dollar figures are list-price estimates; local benchmark spend is $0.")
    elif usage.get("prompt_tokens"):
        lines.append(
            "- All dollar figures are list-price estimates of injection cost, not the "
            f"{PROVIDER} bill; the token usage above is what was actually sent."
        )
    else:
        lines.append(
            "- All dollar figures are list-price estimates of injection cost, not the "
            f"{PROVIDER} bill; this run did not record token usage."
        )
    # Derived, never asserted: the v8 run used 4 of 6 scenarios and this line
    # hardcoded that, so the v9 run that used all six still claimed "4 of 6" --
    # a false statement in its own summary. See _scope_note.
    scope = (report.get("settings") or {}).get("runtime_reduction") or ""
    if scope:
        lines.append(f"- Scope: {scope}")
    lines.append(
        "- All decisive facts sit in turns 1-20 so the 24-turn variant is complete; "
        "the 96-turn variant tests retention over a long distraction tail, not "
        "late-session event placement."
    )
    lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--scenarios", type=int, default=0, help="limit scenario count")
    parser.add_argument("--depths", default="24,96", help="depths that get measured")
    parser.add_argument(
        "--generation-depths",
        default="24,60,96",
        help="depths the anchored template chain is built through (default includes "
        "an unmeasured 60-turn step so the 96-turn anchor is genuinely incremental)",
    )
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_PATH))
    parser.add_argument("--regen-arm", action="store_true", help="also measure template_regen_only")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse cells from the .partial.json checkpoint next to --out",
    )
    args = parser.parse_args(argv)

    global SEED
    SEED = args.seed
    depths = tuple(int(item) for item in args.depths.split(",") if item.strip())
    generation_depths = tuple(
        int(item) for item in args.generation_depths.split(",") if item.strip()
    )
    for depth in depths:
        if depth not in generation_depths:
            generation_depths = tuple(sorted(set(generation_depths) | {depth}))
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if args.regen_arm:
        arms = arms + ("template_regen_only",)
    scenarios = SCENARIOS[: args.scenarios] if args.scenarios else list(SCENARIOS)
    if args.pilot:
        scenarios = scenarios[:1]
        depths = (24,)
        generation_depths = (24,)
        arms = ARMS
    out_path = Path(args.out)
    checkpoint_path = out_path.with_suffix(out_path.suffix + ".partial.json")
    # A run is hours long on a slow local model, and a killed process loses
    # everything measured since the last checkpoint. Resume reuses those cells
    # -- but only cells this run would measure itself, and only when the
    # checkpoint is provably from the same configuration.
    scope_keys = {
        (scenario["id"], depth, arm)
        for scenario in scenarios
        for depth in depths
        for arm in arms
    }
    reused_rows: list[dict[str, Any]] = []
    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"--resume given but no checkpoint at {checkpoint_path}")
        partial = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        snapshot = partial.get("settings_snapshot") or {}
        if snapshot:
            expected = {
                "model": MODEL,
                "provider": PROVIDER,
                "seed": SEED,
                "depths": list(depths),
                "arms": list(arms),
                "coverage_min": COVERAGE_MIN,
            }
            if snapshot != expected:
                raise SystemExit(
                    "checkpoint was written for a different run:\n"
                    f"  checkpoint: {snapshot}\n  this run:   {expected}"
                )
        else:
            # Older checkpoints carry no settings; the arm set is the only
            # identity available, so say plainly what was and was not checked.
            seen_arms = {row.get("arm") for row in partial.get("rows") or []}
            extra = sorted(arm for arm in seen_arms - set(arms) if arm)
            if extra:
                raise SystemExit(
                    f"checkpoint contains arms this run does not measure: {extra}"
                )
            print(
                "[resume] WARNING: checkpoint has no settings snapshot; only the "
                f"arm names were verified ({len(seen_arms) or 0} arms). Confirm it "
                "came from the same model and provider before trusting the result.",
                flush=True,
            )
        reused_rows = [
            row
            for row in partial.get("rows") or []
            if (row.get("scenario"), row.get("depth"), row.get("arm")) in scope_keys
        ]
        print(
            f"[resume] reusing {len(reused_rows)} of {len(scope_keys)} cells from "
            f"{checkpoint_path}",
            flush=True,
        )
    done_keys = {
        (row.get("scenario"), row.get("depth"), row.get("arm")) for row in reused_rows
    }
    started = time.time()
    plugin = _load_plugin()

    with tempfile.TemporaryDirectory(prefix="ac-template-bench-") as temp_dir:
        with patched_env(
            APPEND_COMPACT_L3_DIR=str(Path(temp_dir) / "l3"),
            APPEND_COMPACT_CACHE_DIR=str(Path(temp_dir) / "cache"),
            APPEND_COMPACT_LOG=str(Path(temp_dir) / "bench.log"),
            APPEND_COMPACT_BACKGROUND="0",
        ):
            # Build sessions and templates before controls.
            sessions: dict[tuple[str, int], dict[str, Any]] = {}
            templates: dict[str, Any] = {}
            for scenario in scenarios:
                print(f"[build] scenario {scenario['id']}", flush=True)
                data = generate_templates_for_scenario(
                    scenario, generation_depths, regen_depths=depths
                )
                templates[scenario["id"]] = data
                for depth in depths:
                    session = f"{scenario['id']}-{depth}"
                    state = build_state(plugin, data["turns"][depth], session)
                    sessions[(scenario["id"], depth)] = {
                        "state": state,
                        "session": session,
                        "turns": data["turns"][depth],
                    }
                if scenario["id"] == scenarios[0]:
                    print(
                        f"[build] sample contexts for {scenario['id']}/{depths[0]}: "
                        f"full={len(render_full_transcript(data['turns'][depths[0]]))} chars, "
                        f"current={len(plugin_context(plugin, sessions[(scenario['id'], depths[0])]['state'], sessions[(scenario['id'], depths[0])]['session'], goals=False))} chars, "
                        f"current_goals={len(plugin_context(plugin, sessions[(scenario['id'], depths[0])]['state'], sessions[(scenario['id'], depths[0])]['session'], goals=True))} chars, "
                        f"template={len(render_template(data['anchored'][depths[0]]))} chars",
                        flush=True,
                    )

            # Fail here, not after the controls: an empty template makes every
            # template arm a measurement of the baseline under another name, and
            # both the controls and the measured cells cost more than the
            # generation that would have revealed it.
            for scenario in scenarios:
                data = templates[scenario["id"]]
                for depth in depths:
                    _refuse_empty_template(
                        arms,
                        data["anchored"][depth],
                        data["regen"].get(depth),
                        scenario["id"],
                        depth,
                    )

            controls = run_controls(plugin, scenarios, templates, pilot=args.pilot)
            if not controls["all_passed"]:
                print(json.dumps({"controls": controls}, indent=2, sort_keys=True))
                print("CONTROL FAILED: refusing to report benchmark numbers", file=sys.stderr)
                return 2

            rows: list[dict[str, Any]] = list(reused_rows)
            for scenario in scenarios:
                data = templates[scenario["id"]]
                for depth in depths:
                    cell = sessions[(scenario["id"], depth)]
                    anchored_obj = data["anchored"][depth]
                    regen_obj = data["regen"][depth]
                    donor_templates = [
                        render_template(templates[other["id"]]["anchored"][depth])
                        for other in scenarios
                        if other["id"] != scenario["id"]
                    ]
                    contexts, arm_sizes, arm_generation = build_contexts(
                        plugin,
                        scenario,
                        depth,
                        cell["turns"],
                        cell["state"],
                        cell["session"],
                        anchored_obj,
                        regen_obj,
                        donor_templates,
                        arms,
                    )
                    for arm in arms:
                        context = contexts.get(arm)
                        if context is None:
                            continue
                        if (scenario["id"], depth, arm) in done_keys:
                            continue
                        row = measure_row(
                            scenario,
                            depth,
                            arm,
                            context,
                            arm_sizes.get(arm, 0),
                            arm_generation.get(arm, "none"),
                        )
                        rows.append(row)
                    elapsed = time.time() - started
                    done = len(rows)
                    print(
                        f"[progress] rows={done} elapsed={elapsed/60:.1f}min "
                        f"avg_row={statistics.fmean(r['latency_s'] for r in rows):.1f}s",
                        flush=True,
                    )
                    # Checkpoint per depth, not per scenario: a kill between
                    # checkpoints is what a resume has to redo, and a depth is
                    # 11 rows rather than 22.
                    _write_checkpoint(checkpoint_path, rows, depths=depths, arms=arms)

    aggregates = aggregate(rows, "arm", list(ARM_RANK))
    depth_rows: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy["arm_depth"] = f"{row['arm']}@{row['depth']}"
        depth_rows.append(copy)
    aggregates_by_depth = aggregate(
        depth_rows,
        "arm_depth",
        [f"{arm}@{depth}" for arm in ARM_RANK for depth in depths],
    )
    paired = paired_differences(
        rows,
        "current_goals",
        [
            "goal_recall",
            "constraint_recall",
            "success_recall",
            "decision_recall",
            "correction_recall",
            "artifact_recall",
            "fact_recall",
            "open_loop_recall",
            "next_step_recall",
            "decision_correct",
            "task_success",
        ],
    )
    costs = cost_projection(aggregates, "current_goals") if "current_goals" in aggregates else {}
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "model": MODEL,
            "ollama": OLLAMA,
            "temperature": 0,
            "seed": SEED,
            "format": "json",
            "num_ctx": NUM_CTX,
            "depths": list(depths),
            "generation_depths": list(generation_depths),
            "arms": list(arms),
            "scenarios": [scenario["id"] for scenario in scenarios],
            "scenario_count": len(scenarios),
            "measurement_rows": len(rows),
            "model_calls": len(CALLS),
            "api_calls": len(CALLS),
            "provider": PROVIDER,
            "usage": dict(USAGE),
            # Local inference costs nothing; a remote provider's bill is not
            # reconstructed from list prices here, only the tokens are recorded.
            "spend_usd": 0.0 if PROVIDER == "ollama" else None,
            "runtime_s": time.time() - started,
            "template_strategy": "anchored for arms 4-6; regen for template_regen_only",
            # Validation policy. 0.0 (the default) requires the decision and
            # correction fields to be non-empty; -1 disables that check and
            # accepts whatever structure the model produces. Results from a -1
            # run measured loose templates and must be reported as such.
            "template_coverage_min": COVERAGE_MIN,
            "runtime_reduction": _scope_note(scenarios, depths),
            "plugin_injection": {
                "l3_target_chars": 6400,
                "max_inject_chars": 9000,
                "goal_anchor_placement": "beginning",
                "goal_anchor_max_chars": 900,
            },
        },
        "controls": controls,
        "aggregates": aggregates,
        "aggregates_by_depth": aggregates_by_depth,
        "paired_differences": paired,
        "cost_projection": {
            "price_source": (
                "DeepSeek V4.1-Flash list price, USD per 1M tokens, off-peak/peak; "
                "https://www.modellix.ai/blog/deepseek-api-pricing (retrieved 2026-11-01)"
            ),
            "price_miss_low": PRICE_MISS_LOW,
            "price_miss_high": PRICE_MISS_HIGH,
            "price_hit_low": PRICE_HIT_LOW,
            "price_hit_high": PRICE_HIT_HIGH,
            "token_estimate": "chars / 4 for projections; prompt_tokens are measured",
            "arms": costs,
        },
        "template_validation": {
            scenario_id: {
                strategy: {
                    str(depth): {
                        "valid": meta["valid"],
                        "attempts": len(meta["attempts"]),
                        "missing": meta["missing"],
                        # Non-blocking diagnostic: what the template actually
                        # carried for constraints, the early decision and the
                        # superseding correction.
                        "coverage": meta.get("coverage")
                        or (meta["attempts"][-1].get("coverage") if meta["attempts"] else None),
                    }
                    for depth, meta in data[f"{strategy}_meta"].items()
                }
                for strategy in ("anchored", "regen")
            }
            for scenario_id, data in templates.items()
        },
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary(report, rows, aggregates)

    print("\narm              n  goal  constr  success  decision  corr  artifact  fact  open  choice  task  chars  tmpl  tok  lat")
    print("-" * 118)
    for arm in ARM_RANK:
        stats = aggregates.get(arm)
        if not stats:
            continue
        print(
            f"{arm:<16} {stats['n']:>2}  {stats['goal_recall']:>4.2f}  "
            f"{stats['constraint_recall']:>6.2f}  {stats['success_recall']:>7.2f}  "
            f"{stats['decision_recall']:>8.2f}  {stats['correction_recall']:>4.2f}  "
            f"{stats['artifact_recall']:>8.2f}  {stats['fact_recall']:>4.2f}  "
            f"{stats['open_loop_recall']:>4.2f}  {stats['decision_correct']:>6.2f}  "
            f"{stats['task_success']:>4.2f}  {stats['context_chars']:>5.0f}  "
            f"{stats['template_chars']:>4.0f}  {stats['prompt_tokens']:>4.0f}  "
            f"{stats['latency_s']:>4.1f}s"
        )
    print(f"\nresults: {out_path}")
    print(f"summary: {SUMMARY_PATH}")
    print(f"model calls: {len(CALLS)}  runtime: {(time.time() - started)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
