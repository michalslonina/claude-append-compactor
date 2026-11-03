#!/usr/bin/env python3
"""Multilevel append-only context compactor for Claude Code.

The compactor keeps a deterministic, append-only checkpoint of conversation
turns outside the model context window.  It combines three compression layers:

* zero-cost code minification (python-minifier, JSON compaction, line-level
  deletion of debug output and comment-only lines);
* selective LLMLingua-2 compression applied *only* to natural-language prose,
  with entities masked so paths, symbols and inline code survive byte-exact;
* deterministic entity pinning and L1/L2 summary blocks that are never
  rewritten, so the injected prefix stays stable for provider KV caching.

Hooks capture turns into a queue and return; detached worker processes (one
machine-wide pool, serialised by flock) compress them off the hook path.  Every
chunk, master and queued job carries its session id, so parallel Claude Code
contexts sharing this state never inject each other's history.

State lives in ``~/.claude/append_compactor_state.json`` (override with
``--state`` or ``APPEND_COMPACT_STATE``).

Run ``append-compact.py --self-test`` to verify the known-positive and
known-negative controls used by the compression pipeline.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

try:  # POSIX only; the plugin targets Claude Code on macOS/Linux.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


VERSION = "1.10.0"
SCHEMA_VERSION = 1
PLUGIN_NAME = "claude-append-compactor"

DEFAULT_STATE_PATH = "~/.claude/append_compactor_state.json"
DEFAULT_CACHE_DIR = "~/.cache/append-compactor"
L1_CONSOLIDATE_AT = 6
DEFAULT_RATE = 0.4
LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Compression results are content-addressed so parallel agents can share the
# expensive LLMLingua work and receive byte-identical output for identical
# input.  Bump the schema when the compression algorithm changes.
COMPRESS_CACHE_SCHEMA = 2
COMPRESS_CACHE_MAX_FILES = 4_096

# Claude Code caps each hook output string (additionalContext, systemMessage,
# plain stdout) at 10,000 characters and replaces anything longer with a file
# path plus a 2,000-character preview; there is no setting to raise it.  Keep
# the whole payload comfortably under that wall.
HOOK_OUTPUT_CAP = 10_000
DEFAULT_MAX_INJECT_CHARS = 9_000
LEDGER_BUDGET_CHARS = 2_000
TOOL_RESULT_CHARS = 1_200
PRECOMPACT_TOOL_CHARS = 6_000
SHARED_BLOCK_BANNER = (
    "**Shared compacted context (append-compactor snapshot)**\n"
    "Reuse this block verbatim; do not rewrite it."
)
SHARED_BLOCK_NOTE = "\n\n[append-compactor: older masters omitted to fit the shared block budget]"

DEFAULT_PRECOMPACT_TURNS = 24
DEFAULT_PRECOMPACT_CHARS = 24_000
MIN_PROSE_CHARS = 160
L2_DIGEST_CHARS = 480
ENTITY_HEADER_LIMIT = 200

# L3 placeholders.  Older L1 chunks and L2 masters stay recoverable through a
# content-addressed file and a one-line index entry, while the newest material
# is injected in full.  The index is always after the pinned ledger, so the
# stable head of the injected block never moves.  This is deliberately a view:
# no stored chunk or master is ever rewritten, and no conversation message is
# touched, which is what keeps provider prefix caches useful (see README).
DEFAULT_L3_TARGET_CHARS = 6_400
DEFAULT_L3_KEEP_RECENT_MASTERS = 3
DEFAULT_L3_KEEP_RECENT_CHUNKS = 2
DEFAULT_L3_DRIFT_KEEP_RECENT_MASTERS = 1
DEFAULT_L3_DRIFT_KEEP_RECENT_CHUNKS = 1
DEFAULT_L3_INDEX_BUDGET_CHARS = 4_000
DEFAULT_L3_SUMMARY_CHARS = 80
DEFAULT_L3_KEYWORDS = 7
DEFAULT_L3_ENTITY_CHARS = 72
DEFAULT_L3_DRIFT_WINDOW = 4
DEFAULT_L3_DRIFT_MIN_TOKENS = 8
DEFAULT_L3_DRIFT_THRESHOLD = 0.12
DEFAULT_L3_MAX_FILES = 4_096
# L3 blocks are stored as zstd --ultra -22 files.  The agent can scan them
# without decompressing to disk: `ugrep -z PATTERN FILE` or `zstdgrep PATTERN
# FILE`.  Level 22 is the maximum zstd level; >19 requires --ultra.
DEFAULT_L3_ZSTD_LEVEL = 22

# Goal / vision / strategy anchors.  Goals are extracted deterministically
# from user-authored turns and kept as a versioned, session-scoped state.  The
# anchor is rendered only into the *current* injection, never into a stored L1
# chunk, L2 master or conversation message, so a goal revision cannot rewrite
# history.  The defaults are deliberately conservative: the feature is off
# until measured in a deployment (see README "Goal anchors").
GOAL_STATE_SCHEMA = 1
DEFAULT_GOAL_ANCHOR_L1_CHARS = 240
DEFAULT_GOAL_ANCHOR_L2_CHARS = 520
DEFAULT_GOAL_ANCHOR_L3_CHARS = 760
DEFAULT_GOAL_ANCHOR_MAX_CHARS = 900
DEFAULT_GOAL_ANCHOR_HYBRID_CHARS = 520
DEFAULT_GOAL_REMINDER_CHARS = 160
DEFAULT_GOAL_MAX_ITEMS = 8
GOAL_ITEM_LIMIT = 48
GOAL_SOURCE_LIMIT = 64
GOAL_HISTORY_LIMIT = 64

# Background compaction.  Hooks enqueue raw turns and return immediately; a
# small, machine-wide pool of detached workers compresses them outside the hook
# deadline.  Slots are global (not per session) so a swarm of parallel Claude
# contexts cannot start one model process each.
DEFAULT_WORKERS = 2
DEFAULT_WORKER_IDLE_SECONDS = 120
DEFAULT_WORKER_MAX_SECONDS = 900
DEFAULT_CLAIM_TIMEOUT_SECONDS = 300
DEFAULT_MAX_PENDING = 32
DEFAULT_MAX_PENDING_TOTAL = 64
DEFAULT_DRAIN_WAIT_SECONDS = 5
PENDING_MAX_CHARS = 24_000
PENDING_TEXT_HEADER = "raw"

# Cross-source capture dedupe.  A PreCompact capture prefixes the turn with a
# role label while UserPromptSubmit does not, so the raw hashes never match.
DEFAULT_DEDUPE_WINDOW_SECONDS = 900
DEFAULT_CAPTURE_DEDUPE_SECONDS = 120
RECENT_CAPTURE_LIMIT = 64
PROMPT_SEEN_LIMIT = 256
DEDUPE_MIN_TOKENS = 20
DEDUPE_MIN_SHINGLES = 8
DEDUPE_SHINGLE_WORDS = 5
DEDUPE_CONTAINMENT = 0.95
DEDUPE_TOKEN_CONTAINMENT = 0.98
DEDUPE_JACCARD = 0.90
DEDUPE_LENGTH_RATIO = 0.8
JOBS_LOG_MAX_BYTES = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _l2_digest_chars() -> int:
    return max(80, _env_int("APPEND_COMPACT_L2_DIGEST_CHARS", L2_DIGEST_CHARS))


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def default_state_path() -> Path:
    configured = os.environ.get("APPEND_COMPACT_STATE", DEFAULT_STATE_PATH)
    return Path(configured).expanduser()


def _log_path() -> Path:
    configured = os.environ.get("APPEND_COMPACT_LOG", "~/.cache/append-compactor/error.log")
    return Path(configured).expanduser()


def _log_error(message: str) -> None:
    """Append to a bounded log; never raise from a hook."""
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 5 * 1024 * 1024:
            path.replace(path.with_name(path.name + ".1"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{_now()} {message}\n")
    except Exception:
        pass
    if _env_bool("APPEND_COMPACT_DEBUG", False):
        print(f"[append-compact] {message}", file=sys.stderr)


def _job_log_path() -> Path:
    configured = os.environ.get("APPEND_COMPACT_JOB_LOG")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return default_cache_dir() / "jobs.log"


def _log_job(event: str, **fields: Any) -> None:
    """Append one JSONL job event; never raise and never log raw turn text."""
    try:
        path = _job_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > JOBS_LOG_MAX_BYTES:
            with contextlib.suppress(OSError):
                path.replace(path.with_name(path.name + ".1"))
        record: dict[str, Any] = {"at": _now(), "ts": round(time.time(), 3), "event": event}
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


@contextlib.contextmanager
def state_lock(path: Path) -> Iterator[None]:
    """Serialise load-modify-save across concurrent hook invocations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# Content-addressed compression cache
# ---------------------------------------------------------------------------

def default_cache_dir() -> Path:
    configured = os.environ.get("APPEND_COMPACT_CACHE_DIR", DEFAULT_CACHE_DIR)
    return Path(configured).expanduser()


def _cache_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_CACHE", True)


def _compress_cache_key(text: str, rate: float, model: str, variant: str = "") -> str:
    digest = hashlib.sha256()
    digest.update(f"append-compactor/compressed/v{COMPRESS_CACHE_SCHEMA}\0".encode("utf-8"))
    digest.update(f"rate={rate:g}\0model={model}\0".encode("utf-8"))
    # Only non-default strategies add a marker, so existing cache entries stay
    # valid when sentence protection is off.
    if variant:
        digest.update(f"variant={variant}\0".encode("utf-8"))
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Raw-text archive
#
# Compaction is lossy.  Archiving the pre-compression turn makes every loss
# recoverable with ``--recall <id>`` while keeping the state file bounded.
# Files are content-addressed and pruned oldest-first at a fixed cap, so a long
# session cannot grow the cache without bound.
# ---------------------------------------------------------------------------

_RAW_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")


def _raw_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_RAW", True)


def _raw_dir() -> Path:
    configured = os.environ.get("APPEND_COMPACT_RAW_DIR")
    return Path(configured).expanduser() if configured else default_cache_dir() / "raw"


def _raw_max_files() -> int:
    return max(1, _env_int("APPEND_COMPACT_RAW_MAX_FILES", 4096))


def _prune_raw_archive() -> None:
    root = _raw_dir()
    limit = _raw_max_files()
    try:
        files = [path for path in root.glob("*/*.txt") if path.is_file()]
    except OSError:
        return
    if len(files) <= limit:
        return
    try:
        files.sort(key=lambda path: path.stat().st_mtime)
    except OSError:
        return
    for path in files[: len(files) - limit]:
        with contextlib.suppress(OSError):
            path.unlink()


def archive_raw(text: str) -> str | None:
    """Store ``text`` under its SHA-256 and return the id, or None if disabled."""
    if not _raw_enabled() or not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = _raw_dir() / digest[:2] / f"{digest}.txt"
    if not path.exists():
        try:
            _atomic_write_text(path, text)
        except OSError as exc:
            _log_error(f"raw archive write failed: {exc!r}")
            return None
        _prune_raw_archive()
    return digest


def recall_raw(raw_id: str) -> str | None:
    """Return the archived text for a full id or a >=12 hex prefix.

    The content hash is re-verified against the id, so a truncated, edited or
    misplaced cache file is refused rather than returned as the original.
    """
    raw_id = raw_id.strip().lower()
    if not _RAW_ID_RE.fullmatch(raw_id):
        return None
    root = _raw_dir() / raw_id[:2]
    candidates = (
        [root / f"{raw_id}.txt"]
        if len(raw_id) == 64
        else sorted(root.glob(f"{raw_id}*.txt"))
    )
    for path in candidates:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if hashlib.sha256(content.encode("utf-8")).hexdigest().startswith(raw_id):
            return content
    return None


def _raw_file_display(raw_id: str) -> str:
    """Absolute raw-archive path, abbreviated with ``~`` when under $HOME."""
    path = _raw_dir() / raw_id[:2] / f"{raw_id}.txt"
    text = str(path)
    home = str(Path.home())
    if text.startswith(home + os.sep):
        return "~" + text[len(home):]
    return text


# ---------------------------------------------------------------------------
# L3 view archive
#
# L3 never deletes a chunk.  It stores the exact rendered chunk/master block in
# a content-addressed file and injects a short index line instead.  The file is
# verified against its id on recall, so an edited or truncated cache entry is
# refused rather than presented as the original.
# ---------------------------------------------------------------------------

_L3_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")
_L3_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _l3_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_L3", True)


def _l3_dir() -> Path:
    configured = os.environ.get("APPEND_COMPACT_L3_DIR")
    return Path(configured).expanduser() if configured else default_cache_dir() / "l3"


def _l3_max_files() -> int:
    return max(1, _env_int("APPEND_COMPACT_L3_MAX_FILES", DEFAULT_L3_MAX_FILES))


def _l3_zstd_level() -> int:
    return max(1, min(22, _env_int("APPEND_COMPACT_L3_ZSTD_LEVEL", DEFAULT_L3_ZSTD_LEVEL)))


def _l3_compress_enabled() -> bool:
    return (
        _env_bool("APPEND_COMPACT_L3_COMPRESS", True)
        and shutil.which("zstd") is not None
    )


def _l3_zstd_args(level: int) -> list[str]:
    args = ["zstd"]
    if level > 19:
        args.append("--ultra")
    args.extend([f"-{level}", "-q", "-c"])
    return args


def _compress_l3_block(block: str, level: int) -> bytes:
    proc = subprocess.run(
        _l3_zstd_args(level),
        input=block.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
        raise OSError(f"zstd failed ({proc.returncode}): {detail}")
    return proc.stdout


def _read_l3_file(path: Path) -> str | None:
    if path.suffix == ".zst":
        try:
            proc = subprocess.run(
                ["zstd", "-d", "-q", "-c", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        try:
            return proc.stdout.decode("utf-8")
        except UnicodeDecodeError:
            return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _prune_l3_archive() -> None:
    root = _l3_dir()
    limit = _l3_max_files()
    try:
        files = [
            path
            for pattern in ("*/*.txt.zst", "*/*.txt")
            for path in root.glob(pattern)
            if path.is_file()
        ]
    except OSError:
        return
    if len(files) <= limit:
        return
    try:
        files.sort(key=lambda path: path.stat().st_mtime)
    except OSError:
        return
    for path in files[: len(files) - limit]:
        with contextlib.suppress(OSError):
            path.unlink()


def l3_id_for(block: str) -> str:
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def _l3_path(block_id: str, compressed: bool | None = None) -> Path:
    # A 64-bit prefix keeps file links short; recall still verifies the full
    # SHA-256 of the stored bytes, so a shortened name is not a shortened id.
    if compressed is None:
        compressed = _l3_compress_enabled()
    suffix = ".txt.zst" if compressed else ".txt"
    return _l3_dir() / block_id[:2] / f"{block_id[:16]}{suffix}"


def archive_l3(block: str, block_id: str | None = None) -> str | None:
    """Store a zstd --ultra -22 block and return its content id.

    ``block_id`` lets the caller compute the id (and therefore the stable path)
    before deciding which index entries fit the budget.  The write itself only
    happens for selected entries.
    """
    if not block:
        return None
    block_id = block_id or l3_id_for(block)
    compressed = _l3_compress_enabled()
    path = _l3_path(block_id, compressed=compressed)
    if path.exists():
        existing = _read_l3_file(path)
        if existing is not None and hashlib.sha256(
            existing.encode("utf-8")
        ).hexdigest() == block_id:
            return block_id
        _log_error(f"L3 archive name collision for {block_id[:16]}")
        return None
    try:
        if compressed:
            _atomic_write_bytes(path, _compress_l3_block(block, _l3_zstd_level()))
        else:
            _atomic_write_text(path, block)
    except (OSError, subprocess.SubprocessError) as exc:
        _log_error(f"L3 archive write failed: {exc!r}")
        return None
    _prune_l3_archive()
    return block_id


def recall_l3(block_id: str) -> str | None:
    """Return a verified full block for a full id or a >=12 hex prefix."""
    block_id = block_id.strip().lower()
    if not _L3_ID_RE.fullmatch(block_id):
        return None
    root = _l3_dir() / block_id[:2]
    prefix = block_id[:16] if len(block_id) > 16 else block_id
    candidates = sorted({
        path
        for pattern in (f"{prefix}*.txt.zst", f"{prefix}*.txt")
        for path in root.glob(pattern)
    })
    for path in candidates:
        content = _read_l3_file(path)
        if content is None:
            continue
        if hashlib.sha256(content.encode("utf-8")).hexdigest().startswith(block_id):
            return content
    return None


def _l3_file_display(block_id: str) -> str:
    """Absolute L3-archive path, abbreviated with ``~`` when under $HOME."""
    text = str(_l3_path(block_id))
    home = str(Path.home())
    if text.startswith(home + os.sep):
        return "~" + text[len(home):]
    return text


def _l3_base_display() -> str:
    """L3 archive root, abbreviated with ``~`` when under $HOME."""
    text = str(_l3_dir())
    home = str(Path.home())
    if text.startswith(home + os.sep):
        return "~" + text[len(home):]
    return text


def _l3_file_link(block_id: str) -> str:
    """Short link relative to the archive root (the header names the root)."""
    suffix = ".txt.zst" if _l3_compress_enabled() else ".txt"
    return f"{block_id[:2]}/{block_id[:16]}{suffix}"


def _is_l3_archive_path(value: str) -> bool:
    """True for internal ``~/.cache/append-compactor/l3/...`` pointers."""
    normalized = value.replace("\\", "/")
    root = _l3_dir().as_posix()
    home = Path.home().as_posix()
    prefixes = {root}
    if root.startswith(home + "/"):
        prefixes.add("~" + root[len(home):])
    return any(normalized == prefix or normalized.startswith(prefix + "/") for prefix in prefixes)


def _read_cached_compression(path: Path) -> tuple[str, str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != COMPRESS_CACHE_SCHEMA:
        return None
    text, mode = data.get("text"), data.get("mode")
    if not isinstance(text, str) or not text or not isinstance(mode, str):
        return None
    return text, mode


def _write_cached_compression(path: Path, text: str, mode: str) -> None:
    try:
        _atomic_write_json(path, {"schema": COMPRESS_CACHE_SCHEMA, "text": text, "mode": mode})
    except OSError as exc:  # a full disk must never break compaction
        _log_error(f"compression cache write failed: {exc!r}")
        return
    _prune_compression_cache(path.parent.parent)


def _prune_compression_cache(root: Path) -> None:
    limit = _env_int("APPEND_COMPACT_CACHE_MAX_FILES", COMPRESS_CACHE_MAX_FILES)
    try:
        files = [entry for entry in root.rglob("*.json") if entry.is_file()]
    except OSError:
        return
    if len(files) <= limit:
        return
    try:
        files.sort(key=lambda entry: entry.stat().st_mtime)
    except OSError:
        return
    for stale in files[: len(files) - limit]:
        with contextlib.suppress(OSError):
            stale.unlink()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def new_goal_store() -> dict[str, Any]:
    """Empty goal state.

    Goals are stored per Claude Code session.  The store is additive: old
    items are never deleted on a revision, only marked ``superseded`` or
    ``completed``, so the anchor's provenance and version history survive a
    later extraction error.
    """
    return {"schema": GOAL_STATE_SCHEMA, "sessions": {}}


def new_state() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "created": _now(),
        "l1_chunks": [],
        "l2_masters": [],
        "pending": [],
        "armed_sessions": {},
        "recent_captures": [],
        "prompt_seen": {},
        "goals": new_goal_store(),
        "stats": {
            "turns": 0,
            "bytes_in": 0,
            "bytes_out": 0,
            "llm_blocks": 0,
            "fallback_blocks": 0,
            "error_blocks": 0,
        },
    }


def load_state(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else default_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return new_state()
    except (OSError, json.JSONDecodeError) as exc:
        _log_error(f"unreadable state {path}: {exc!r}; starting a fresh state")
        return new_state()

    if not isinstance(data, dict):
        _log_error(f"state {path} is not an object; starting a fresh state")
        return new_state()
    schema = data.get("schema")
    if isinstance(schema, int) and schema > SCHEMA_VERSION:
        raise RuntimeError(
            f"state schema {schema} is newer than supported {SCHEMA_VERSION}; refusing to rewrite {path}"
        )

    base = new_state()
    for key, value in base.items():
        data.setdefault(key, value)
    for key in ("l1_chunks", "l2_masters", "pending"):
        if not isinstance(data.get(key), list):
            data[key] = []
    if not isinstance(data.get("armed_sessions"), dict):
        data["armed_sessions"] = {}
    recent = data.get("recent_captures")
    if not isinstance(recent, list):
        recent = []
    data["recent_captures"] = [
        entry
        for entry in recent
        if isinstance(entry, dict)
        and isinstance(entry.get("at"), (int, float))
        and isinstance(entry.get("text"), str)
        and entry.get("source") in ("precompact", "user_prompt")
        and entry.get("role") == "user"
    ][-RECENT_CAPTURE_LIMIT:]
    prompt_seen = data.get("prompt_seen")
    if not isinstance(prompt_seen, dict):
        prompt_seen = {}
    data["prompt_seen"] = {
        str(key): float(value)
        for key, value in prompt_seen.items()
        if isinstance(value, (int, float))
    }
    _cap_prompt_seen(data)
    data["goals"] = _sanitize_goal_store(data.get("goals"))
    if not isinstance(data.get("stats"), dict):
        data["stats"] = base["stats"]
    return data


def _cap_prompt_seen(state: dict[str, Any]) -> None:
    seen = state.get("prompt_seen")
    if not isinstance(seen, dict) or len(seen) <= PROMPT_SEEN_LIMIT:
        return
    ordered = sorted(seen.items(), key=lambda item: float(item[1]))
    state["prompt_seen"] = dict(ordered[-PROMPT_SEEN_LIMIT:])


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    path = Path(path) if path is not None else default_state_path()
    _atomic_write_json(path, state)


def _belongs(item: dict[str, Any], session: str | None) -> bool:
    return session is None or item.get("session") == session


def active_l1(state: dict[str, Any], session: str | None = None) -> list[dict[str, Any]]:
    """L1 chunks not yet absorbed into an L2 master.

    With ``session`` set, only that Claude context's chunks are returned, so
    parallel sessions never see each other's turns.
    """
    covered: set[int] = set()
    for master in state.get("l2_masters", []):
        if not _belongs(master, session):
            continue
        covered.update(int(n) for n in master.get("covered", []) if isinstance(n, int))
    return [
        chunk for chunk in state.get("l1_chunks", [])
        if int(chunk.get("n", 0)) not in covered and _belongs(chunk, session)
    ]


def status_of(
    state: dict[str, Any],
    path: Path | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    active = active_l1(state, session)
    stats = state.get("stats", {})
    sessions: dict[str, dict[str, int]] = {}
    for chunk in state.get("l1_chunks", []):
        key = str(chunk.get("session") or "legacy")
        entry = sessions.setdefault(key, {"chunks": 0, "active": 0})
        entry["chunks"] += 1
    for chunk in active:
        key = str(chunk.get("session") or "legacy")
        sessions.setdefault(key, {"chunks": 0, "active": 0})["active"] += 1
    pending_by_session: dict[str, int] = {}
    for job in state.get("pending", []):
        key = str(job.get("session") or "legacy")
        pending_by_session[key] = pending_by_session.get(key, 0) + 1
    scoped_masters = [
        master for master in state.get("l2_masters", []) if _belongs(master, session)
    ]
    return {
        "plugin": PLUGIN_NAME,
        "version": VERSION,
        "state_file": str(path) if path else str(default_state_path()),
        "schema": state.get("schema"),
        "created": state.get("created"),
        "scope": session,
        "l1_chunks": sum(
            1 for chunk in state.get("l1_chunks", []) if _belongs(chunk, session)
        ),
        "active_l1": [c.get("n") for c in active],
        "l2_masters": len(scoped_masters),
        "pending": pending_count(state, session),
        "pending_by_session": pending_by_session,
        "sessions": sessions,
        "armed_sessions": sorted(state.get("armed_sessions", {})),
        "stats": stats,
        "last_event": state.get("last_event"),
    }


# ---------------------------------------------------------------------------
# Goal / vision / strategy state and render-time anchors
# ---------------------------------------------------------------------------
#
# Design constraints, in priority order:
#
# 1. Never rewrite an L1 chunk, an L2 master or a conversation message.  The
#    anchor is built fresh inside injection_payload().  With the default
#    beginning placement it leads the stable prefix; with tail/per-level
#    placement the ledger + masters prefix is byte-identical whether goals are
#    off, on, or revised.
# 2. Keep the user's words.  The deterministic extractor copies clauses and
#    bullet text; it never paraphrases.  Only the budget fallback truncates.
# 3. Make revisions deliberate and visible.  A revision supersedes the previous
#    active item and increments the session's anchor version; history is kept.
# 4. Keep the anchor small and measurable.  Per-level caps plus one hard total
#    cap bound the added characters; the injection metadata reports the exact
#    cost on every call.

_GOAL_KINDS = ("goal", "vision", "strategy", "target", "constraint", "non_goal", "success")
_GOAL_KIND_ALIASES = {
    "objective": "goal",
    "objectives": "goal",
    "goals": "goal",
    "targets": "target",
    "success": "success",
    "success criteria": "success",
    "acceptance criteria": "success",
    "definition of done": "success",
    "constraints": "constraint",
    "constraint": "constraint",
    "requirements": "constraint",
    "requirement": "constraint",
    "non-goal": "non_goal",
    "non-goals": "non_goal",
    "non goal": "non_goal",
    "non goals": "non_goal",
    "vision": "vision",
    "strategy": "strategy",
    "target": "target",
    "goal": "goal",
}
_GOAL_KIND_ORDER = ("goal", "constraint", "success", "target", "strategy", "vision", "non_goal")
_GOAL_KIND_LABEL = {
    "goal": "Goal",
    "constraint": "Constraints",
    "success": "Success criteria",
    "target": "Targets",
    "strategy": "Strategy",
    "vision": "Vision",
    "non_goal": "Non-goals",
}

_GOAL_LINE_RE = re.compile(
    r"^[ \t]*(?:[-*+•]|\d+[.)])?[ \t]*(?:\*\*|__)?"
    r"(goal|goals|objective|objectives|vision|strategy|target|targets|"
    r"success criteria|success|acceptance criteria|definition of done|"
    r"constraints?|requirements?|non[- ]?goals?)"
    r"(?:\*\*|__)?[ \t]*(?::|#|[-–—])[ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_GOAL_SEGMENT_RE = re.compile(
    r"(?i)\s+(?=(?:goal|goals|objective|objectives|vision|strategy|target|targets|"
    r"success criteria|success|acceptance criteria|definition of done|"
    r"constraints?|requirements?|non[- ]?goals?)\s*(?::|#|[-–—])\s+)"
)
_GOAL_INLINE_RE = re.compile(
    r"(?i)\b(?:the\s+|our\s+|my\s+)?"
    r"(goal|objective|vision|strategy|target|success criteria|acceptance criteria|definition of done)"
    r"(?:\s+(?:is|are|was|were)|:)\s+(.{3,500}?)(?=(?:[.!?](?:\s|$))|$)"
)
_GOAL_CONSTRAINT_RE = re.compile(
    r"(?i)\b(?:must\s+not|must\s+never|never|do\s+not|don't|should\s+not|shouldn't|"
    r"without|only|keep|preserve|avoid|no\s+more\s+than|at\s+most|at\s+least|"
    r"required\s+to|ensure|make\s+sure)\b"
)
_GOAL_REVISION_RE = re.compile(
    r"(?i)\b(?:instead|rather\s+than|change\s+of\s+plan|new\s+(?:goal|objective|vision|"
    r"strategy|target|plan|direction|priority)|actually|scratch\s+that|no\s+longer|"
    r"from\s+now\s+on|revise[ds]?|supersed\w*|replace\s+(?:the|our|that)|"
    r"switch(?:ing)?\s+to|not\s+.{0,80}?\s+anymore|now\s+we\s+(?:want|need)|"
    r"the\s+new\s+(?:goal|target|priority))\b"
)
_GOAL_REPLACE_RE = re.compile(
    r"(?i)\b(?:instead|rather\s+than|no\s+longer|not\s+.{0,80}?\s+anymore|"
    r"replace|drop|remove|forget|scrap|abandon)\b"
)
_GOAL_COMPLETION_RE = re.compile(
    r"(?i)\b(?:goal|objective|task|project|mission)\s+(?:is\s+)?"
    r"(?:complete|completed|done|finished)\b|"
    r"\bmark\s+(?:the\s+)?(?:goal|task|objective)\s+(?:as\s+)?"
    r"(?:complete|completed|done)\b"
)


def _goals_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_GOALS", False)


def _goal_extraction_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_GOALS_EXTRACT", True)


def _goal_level_enabled(level: str) -> bool:
    if not _goals_enabled():
        return False
    return _env_bool(f"APPEND_COMPACT_GOALS_{level.upper()}", True)


def _goal_placement() -> str:
    value = os.environ.get("APPEND_COMPACT_GOALS_PLACEMENT", "beginning").strip().lower()
    if value in {"per_level", "per-level", "per_chunk", "per-chunk", "chunk", "chunks"}:
        return "per_level"
    if value in {"beginning", "start", "prefix", "head", "front"}:
        return "beginning"
    if value in {"after_head", "after-head", "afterhead"}:
        return "after_head"
    if value in {"hybrid", "beginning+reminder", "prefix+reminder"}:
        return "hybrid"
    return "tail"


def _goal_level_cap(level: str) -> int:
    default = {
        "L1": DEFAULT_GOAL_ANCHOR_L1_CHARS,
        "L2": DEFAULT_GOAL_ANCHOR_L2_CHARS,
        "L3": DEFAULT_GOAL_ANCHOR_L3_CHARS,
    }.get(level, DEFAULT_GOAL_ANCHOR_L3_CHARS)
    return max(80, _env_int(f"APPEND_COMPACT_GOALS_{level.upper()}_CHARS", default))


def _goal_total_cap() -> int:
    return max(120, _env_int("APPEND_COMPACT_GOALS_MAX_CHARS", DEFAULT_GOAL_ANCHOR_MAX_CHARS))


def _goal_max_items() -> int:
    return max(1, _env_int("APPEND_COMPACT_GOALS_MAX_ITEMS", DEFAULT_GOAL_MAX_ITEMS))


def _goal_session_key(session: str | None) -> str:
    return str(session) if session else "default"


def _new_session_goal() -> dict[str, Any]:
    return {
        "version": 0,
        "updated_at": None,
        "next_id": 1,
        "items": [],
        "history": [],
    }


def _session_goal_store(
    state: dict[str, Any],
    session: str | None,
    *,
    create: bool = False,
) -> dict[str, Any] | None:
    goals = state.get("goals")
    if not isinstance(goals, dict):
        goals = new_goal_store()
        state["goals"] = goals
    sessions = goals.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
        goals["sessions"] = sessions
    key = _goal_session_key(session)
    store = sessions.get(key)
    if isinstance(store, dict):
        return store
    if not create:
        return None
    store = _new_session_goal()
    sessions[key] = store
    return store


def _clean_goal_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(?:[-*+•]|\d+[.)])\s+", "", text).strip()
    return text[:600].rstrip()


def _goal_norm(value: str) -> str:
    text = re.sub(r"[^\w\s]+", " ", value.lower(), flags=re.UNICODE)
    return " ".join(text.split())[:300]


def _strip_goal_role(text: str) -> str:
    return re.sub(r"^(?:User|Human|Assistant|Tool)\s*:\s*", "", text.strip(), count=1, flags=re.I)


def _goal_kind(value: str) -> str:
    return _GOAL_KIND_ALIASES.get(" ".join(value.lower().split()).replace("_", "-"), "goal")


def _extract_goal_items(text: str) -> list[dict[str, Any]]:
    """Deterministic, verbatim extraction from one user-authored turn.

    It recognises explicit ``Goal:``/``Constraint:``-style lines, bullet lists
    under an empty marker line, inline "our goal is ..." sentences and hard
    constraint sentences.  It deliberately does not call a model: extraction
    must be cheap, deterministic and available inside a hook.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    clean = _strip_goal_role(text)
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(
        kind: str,
        value: str,
        confidence: float,
        *,
        explicit: bool = False,
    ) -> None:
        cleaned = _clean_goal_text(value)
        if len(cleaned) < 4 or cleaned.lower() in {"none", "n/a", "na", "tbd", "todo"}:
            return
        norm = _goal_norm(cleaned)
        if not norm:
            return
        key = (kind, norm)
        if key in seen:
            return
        seen.add(key)
        items.append({
            "kind": kind,
            "text": cleaned,
            "norm": norm,
            "confidence": round(float(confidence), 2),
            "explicit": explicit,
        })

    def add_segmented(kind: str, value: str) -> None:
        """Split ``Goal: A. Constraints: B`` on a second inline marker."""
        parts = _GOAL_SEGMENT_RE.split(value)
        add(kind, parts[0], 0.95, explicit=True)
        for part in parts[1:]:
            marker = _GOAL_LINE_RE.match(part.strip())
            if marker:
                add(_goal_kind(marker.group(1)), marker.group(2), 0.95, explicit=True)
            else:
                add(kind, part, 0.9)

    current_kind: str | None = None
    in_fence = False
    for raw_line in clean.splitlines():
        raw = raw_line.rstrip()
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            current_kind = None
            continue
        if in_fence:
            continue
        if not stripped:
            current_kind = None
            continue
        if stripped.lower() in {"user:", "human:", "assistant:", "tool:"}:
            continue
        # Markdown headings other than a recognised marker end a section.
        had_heading = bool(re.match(r"^#{1,6}\s+", stripped))
        line = re.sub(r"^#{1,6}\s+", "", stripped).strip()
        marker = _GOAL_LINE_RE.match(line)
        if marker:
            kind = _goal_kind(marker.group(1))
            value = marker.group(2).strip()
            if value:
                add_segmented(kind, value)
                current_kind = None
            else:
                current_kind = kind
            continue
        if current_kind is not None:
            if had_heading or re.match(r"^[A-Za-z][\w /-]{0,48}:\s*$", line):
                current_kind = None
                continue
            add(current_kind, line, 0.9, explicit=True)
            continue

    # Inline goal and constraint sentences.  Fenced code is excluded above;
    # remove it here so a code comment cannot become a goal.
    prose = re.sub(r"(?ms)^```.*?^```", " ", clean)
    prose = _strip_goal_role(prose)
    for sentence in _split_sentences(prose):
        if sentence.endswith("?"):
            continue
        # Explicit marker lines were parsed above; do not re-add the whole
        # sentence as a second, differently-normalised goal.
        if _GOAL_LINE_RE.match(sentence.strip()):
            continue
        for match in _GOAL_INLINE_RE.finditer(sentence):
            value = match.group(2)
            if _goal_norm(value) in {"complete", "completed", "done", "finished"}:
                continue
            add(_goal_kind(match.group(1)), value, 0.8)
        if _GOAL_CONSTRAINT_RE.search(sentence):
            add("constraint", sentence, 0.7)
        if len(items) >= GOAL_ITEM_LIMIT:
            break
    return items[:GOAL_ITEM_LIMIT]


def _sanitize_goal_store(raw: Any) -> dict[str, Any]:
    store = new_goal_store()
    if not isinstance(raw, dict):
        return store
    sessions = raw.get("sessions")
    if not isinstance(sessions, dict):
        return store
    for session_key, raw_session in sessions.items():
        if not isinstance(raw_session, dict):
            continue
        raw_items = raw_session.get("items")
        if not isinstance(raw_items, list):
            continue
        items: list[dict[str, Any]] = []
        max_version = 0
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            text = _clean_goal_text(raw_item.get("text"))
            if not text:
                continue
            kind = str(raw_item.get("kind") or "goal").lower()
            if kind not in _GOAL_KINDS:
                kind = "goal"
            status = str(raw_item.get("status") or "active").lower()
            if status not in {"active", "superseded", "completed"}:
                status = "active"
            version = int(raw_item.get("updated_version") or raw_item.get("created_version") or 0)
            max_version = max(max_version, version)
            source_chunks = []
            for value in raw_item.get("source_chunks") or []:
                if isinstance(value, int) and value > 0 and value not in source_chunks:
                    source_chunks.append(value)
            items.append({
                "id": str(raw_item.get("id") or f"g{len(items) + 1}"),
                "kind": kind,
                "text": text,
                "norm": _goal_norm(text),
                "status": status,
                "confidence": float(raw_item.get("confidence") or 0.5),
                "source_chunks": source_chunks[-GOAL_SOURCE_LIMIT:],
                "created_version": int(raw_item.get("created_version") or version),
                "updated_version": version,
                "created": str(raw_item.get("created") or ""),
                "last_seen": str(raw_item.get("last_seen") or ""),
                "superseded_by": raw_item.get("superseded_by"),
            })
        if not items:
            continue
        history = raw_session.get("history")
        if not isinstance(history, list):
            history = []
        store["sessions"][str(session_key)] = {
            "version": max(max_version, int(raw_session.get("version") or 0)),
            "updated_at": raw_session.get("updated_at"),
            "next_id": max(
                int(raw_session.get("next_id") or 1),
                len(items) + 1,
            ),
            "items": items[-GOAL_ITEM_LIMIT:],
            "history": [entry for entry in history if isinstance(entry, dict)][-GOAL_HISTORY_LIMIT:],
        }
    return store


def _cap_goal_store(store: dict[str, Any]) -> None:
    items = store.get("items")
    if not isinstance(items, list) or len(items) <= GOAL_ITEM_LIMIT:
        return
    active = [item for item in items if item.get("status") == "active"]
    inactive = [item for item in items if item.get("status") != "active"]
    # Active items always survive; only old superseded/completed records are
    # pruned once the per-session bound is reached.
    room = max(0, GOAL_ITEM_LIMIT - len(active))
    store["items"] = active + inactive[-room:] if room else active
    history = store.get("history")
    if isinstance(history, list):
        store["history"] = history[-GOAL_HISTORY_LIMIT:]


def _active_goal_items(store: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not store:
        return []
    items = [
        item for item in store.get("items", [])
        if isinstance(item, dict) and item.get("status") == "active" and item.get("text")
    ]
    items.sort(key=lambda item: (
        _GOAL_KIND_ORDER.index(item.get("kind")) if item.get("kind") in _GOAL_KIND_ORDER else 99,
        int(item.get("created_version") or 0),
        str(item.get("id") or ""),
    ))
    return items


def _update_goals_from_text(
    state: dict[str, Any],
    text: str,
    session: str | None,
    source_chunk: int | None = None,
) -> dict[str, Any] | None:
    """Extract goal items from a user turn and apply deliberate revisions.

    Returns the appended history entry when the session version changed, else
    ``None``.  A repeated verbatim goal only refreshes provenance; it does not
    create a new version or a second item.
    """
    if not _goals_enabled() or not _goal_extraction_enabled():
        return None
    items = _extract_goal_items(text)
    completed = bool(_GOAL_COMPLETION_RE.search(text))
    if not items and not completed:
        return None
    store = _session_goal_store(state, session, create=True)
    if store is None:
        return None

    revision = bool(_GOAL_REVISION_RE.search(text))
    replacement = bool(_GOAL_REPLACE_RE.search(text))
    active = _active_goal_items(store)
    preexisting = {id(item) for item in active}
    existing = {(item["kind"], item["norm"]): item for item in active}
    now = _now()
    added: list[str] = []
    superseded: list[str] = []
    finished: list[str] = []
    changed = False
    version = int(store.get("version") or 0)

    if completed:
        for item in active:
            item["status"] = "completed"
            item["updated_version"] = version + 1
            item["last_seen"] = now
            finished.append(str(item.get("id")))
        if finished:
            changed = True

    for candidate in items:
        kind = candidate["kind"]
        norm = candidate["norm"]
        prior = existing.get((kind, norm))
        if prior is not None and prior.get("status") == "active":
            prior["last_seen"] = now
            if source_chunk is not None and source_chunk not in prior.setdefault("source_chunks", []):
                prior["source_chunks"] = (prior.get("source_chunks") or [])[-GOAL_SOURCE_LIMIT + 1:] + [source_chunk]
            continue
        new_id = f"g{int(store.get('next_id') or 1)}"
        store["next_id"] = int(store.get("next_id") or 1) + 1
        # An explicit ``Goal:``/``Vision:``/... line in a later user turn is a
        # deliberate statement of the current mission, so it replaces the
        # earlier active item of that kind even without the word "instead".
        # Items added in this same turn are not touched: two explicit goal
        # lines in one prompt remain two parallel goals.
        mission_kind = kind in {"goal", "vision", "strategy", "target"}
        replace_this_kind = (
            (revision or bool(candidate.get("explicit"))) and mission_kind
        ) or (revision and replacement)
        if replace_this_kind:
            for old in active:
                if (
                    id(old) in preexisting
                    and old.get("kind") == kind
                    and old.get("norm") != norm
                    and old.get("status") == "active"
                ):
                    old["status"] = "superseded"
                    old["superseded_by"] = new_id
                    old["updated_version"] = version + 1
                    old["last_seen"] = now
                    superseded.append(str(old.get("id")))
        new_item = {
            "id": new_id,
            "kind": kind,
            "text": candidate["text"],
            "norm": norm,
            "status": "active",
            "confidence": candidate["confidence"],
            "source_chunks": [source_chunk] if source_chunk is not None else [],
            "created_version": version + 1,
            "updated_version": version + 1,
            "created": now,
            "last_seen": now,
            "superseded_by": None,
        }
        store.setdefault("items", []).append(new_item)
        active.append(new_item)
        existing[(kind, norm)] = new_item
        added.append(new_id)
        changed = True

    if not changed:
        return None
    version += 1
    store["version"] = version
    store["updated_at"] = now
    entry = {
        "version": version,
        "at": now,
        "source_chunk": source_chunk,
        "event": "complete" if completed and not added else ("revise" if (superseded or replacement) else "extract"),
        "added": added,
        "superseded": superseded,
        "completed": finished,
    }
    history = store.setdefault("history", [])
    history.append(entry)
    store["history"] = history[-GOAL_HISTORY_LIMIT:]
    _cap_goal_store(store)
    return entry


def _record_goals_for_turn(
    state: dict[str, Any],
    text: str,
    session: str | None,
    source_chunk: int | None,
    role: str | None,
) -> dict[str, Any] | None:
    """Only user-authored turns may establish or revise a goal."""
    if role not in (None, "user"):
        return None
    return _update_goals_from_text(state, text, session, source_chunk)


def _truncate_goal_text(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    head = text[: max(1, budget - 1)]
    cut = head.rfind(" ")
    if cut > max(8, budget // 2):
        head = head[:cut]
    return head.rstrip(" ,;:-") + "…"


def _goal_anchor_body(store: dict[str, Any], cap: int, header: str) -> str:
    active = _active_goal_items(store)[:_goal_max_items()]
    if not active or cap < 80:
        return ""
    lines = [header]
    used = len(header) + 1
    has_item = False
    note = "Use as the current mission; older summaries may be stale."
    if used + len(note) + 1 <= cap - 40:
        lines.append(note)
        used += len(note) + 1

    def append_line(line: str) -> bool:
        nonlocal used
        if used + len(line) + 1 > cap:
            return False
        lines.append(line)
        used += len(line) + 1
        return True

    for kind in _GOAL_KIND_ORDER:
        group = [item for item in active if item.get("kind") == kind]
        if not group:
            continue
        label = _GOAL_KIND_LABEL.get(kind, kind.title())
        if len(group) == 1:
            text = str(group[0]["text"])
            line = f'{label} (verbatim): "{text}"'
            if append_line(line):
                has_item = True
                continue
            remaining = cap - used - 1
            prefix = f'{label} (verbatim): "'
            if remaining > len(prefix) + 12:
                clipped = _truncate_goal_text(text, remaining - len(prefix) - 2)
                has_item = append_line(prefix + clipped + '"') or has_item
            # If the highest-priority group cannot fit, lower-priority groups
            # will not fit either; stop rather than emit a misleading fragment.
            break
        header_line = f"{label} (verbatim):"
        if not append_line(header_line):
            break
        for item in group:
            bullet = f'- "{item["text"]}"'
            if append_line(bullet):
                has_item = True
                continue
            remaining = cap - used - 1
            if remaining > 18:
                clipped = _truncate_goal_text(str(item["text"]), remaining - 4)
                has_item = append_line(f'- "{clipped}"') or has_item
            else:
                has_item = append_line("- …") or has_item
            break
    return "\n".join(lines) if has_item else ""


def _render_goal_anchor(
    store: dict[str, Any],
    levels: list[str],
    cap: int,
    origin: str = "tail",
) -> str:
    version = int(store.get("version") or 0)
    if origin == "shared":
        header = f"[GOAL ANCHOR v{version} | shared]"
    elif origin == "tail":
        header = f"[GOAL ANCHOR v{version} | levels={','.join(levels)}]"
    else:
        header = f"[GOAL ANCHOR v{version} | level={levels[0]}]"
    return _goal_anchor_body(store, max(80, cap), header)


def _render_goal_reminder(store: dict[str, Any], cap: int | None = None) -> str:
    """One-line recency reminder for the hybrid placement.

    The beginning anchor is the durable copy.  This reminder exists only to
    counter recency loss when the drift detector says the current prompt is
    unrelated to the recent thread; it never replaces the stable anchor.
    """
    active = _active_goal_items(store)
    if not active:
        return ""
    cap = max(60, cap if cap is not None else DEFAULT_GOAL_REMINDER_CHARS)
    priority = [
        item for kind in ("goal", "vision", "strategy", "target")
        for item in active if item.get("kind") == kind
    ]
    goal = (priority or active)[0]
    version = int(store.get("version") or 0)
    header = f"[GOAL REMINDER v{version}]"
    label = _GOAL_KIND_LABEL.get(str(goal.get("kind")), "Goal")
    prefix = f'{label} (verbatim): "'
    suffix = '"'
    room = cap - len(header) - 1 - len(prefix) - len(suffix)
    if room < 12:
        return header
    text = _truncate_goal_text(str(goal.get("text")), room)
    return f"{header}\n{prefix}{text}{suffix}"


def _dedupe_goal_sections(sections: list[tuple[str, str]], version: int) -> list[tuple[str, str]]:
    """Drop duplicate bodies, keeping the earliest (highest-priority) copy.

    Per-level anchors differ only in their header; repeating the same body at
    two offsets costs bytes and adds no information, so one copy is enough.
    The kept header still names its level, which is what placement needs.
    """
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for level, text in sections:
        body = "\n".join(text.splitlines()[1:])
        if body and body in seen:
            continue
        out.append((level, text))
        if body:
            seen.add(body)
    return out


def _render_goal_anchors(
    state: dict[str, Any],
    session: str | None,
    present: dict[str, bool],
    *,
    max_total: int | None = None,
    drift: bool = False,
) -> list[tuple[str, str]]:
    """Render anchors for one injection; returns ``[(level, text), ...]``."""
    if not _goals_enabled():
        return []
    store = _session_goal_store(state, session, create=False)
    if not _active_goal_items(store):
        return []
    levels = [
        level for level in ("L1", "L2", "L3")
        if present.get(level) and _goal_level_enabled(level)
    ]
    if not levels:
        return []
    total_cap = max_total if max_total is not None else _goal_total_cap()
    total_cap = max(80, total_cap)
    placement = _goal_placement()
    if placement == "hybrid":
        hybrid_cap = max(
            80,
            _env_int(
                "APPEND_COMPACT_GOALS_HYBRID_CHARS",
                DEFAULT_GOAL_ANCHOR_HYBRID_CHARS,
            ),
        )
        cap = min(total_cap, hybrid_cap)
        text = _render_goal_anchor(store, levels, cap, "tail")
        sections = [("+".join(levels), text)] if text else []
        if drift and sections:
            remaining = total_cap - len(text) - 2
            if remaining >= 60:
                reminder = _render_goal_reminder(
                    store, min(DEFAULT_GOAL_REMINDER_CHARS, remaining)
                )
                if reminder:
                    sections.append(("reminder", reminder))
        return sections
    if placement in {"tail", "beginning", "after_head"}:
        cap = total_cap if len(levels) > 1 else min(total_cap, _goal_level_cap(levels[0]))
        text = _render_goal_anchor(store, levels, cap, "tail")
        return [("+".join(levels), text)] if text else []

    order = [level for level in ("L2", "L3", "L1") if level in levels]
    caps = {level: min(total_cap, _goal_level_cap(level)) for level in order}
    sections: list[tuple[str, str]] = []
    for level in order:
        text = _render_goal_anchor(store, [level], caps[level], "per_level")
        if text:
            sections.append((level, text))

    def cost(entries: list[tuple[str, str]]) -> int:
        return sum(len(text) for _, text in entries) + 2 * max(0, len(entries) - 1)

    if cost(sections) > total_cap:
        scale = total_cap / max(1, cost(sections))
        scaled: list[tuple[str, str]] = []
        for level, _ in sections:
            cap = max(80, int(caps[level] * scale))
            text = _render_goal_anchor(store, [level], cap, "per_level")
            if text:
                scaled.append((level, text))
        sections = scaled
    # If headers alone overflow a very small total, prefer the deepest level
    # (L3 is the oldest, least recoverable material) and drop L1, then L2.
    while cost(sections) > total_cap and len(sections) > 1:
        drop = "L1" if any(level == "L1" for level, _ in sections) else "L2"
        sections = [(level, text) for level, text in sections if level != drop]
    return _dedupe_goal_sections(sections, int(store.get("version") or 0))


def _render_shared_goal_anchor(
    state: dict[str, Any],
    session: str | None,
    cap: int,
) -> str:
    if not _env_bool("APPEND_COMPACT_GOALS_SHARED", False) or session is None:
        return ""
    if not _goals_enabled():
        return ""
    store = _session_goal_store(state, session, create=False)
    if not _active_goal_items(store):
        return ""
    return _render_goal_anchor(store, ["shared"], min(cap, _goal_total_cap()), "shared")


def _goal_metadata(
    state: dict[str, Any],
    session: str | None,
    sections: list[tuple[str, str]],
) -> dict[str, Any]:
    store = _session_goal_store(state, session, create=False)
    active = _active_goal_items(store)
    return {
        "enabled": _goals_enabled(),
        "placement": _goal_placement(),
        "version": int((store or {}).get("version") or 0),
        "active": len(active),
        "levels": [level for level, _ in sections],
        "anchor_count": len(sections),
        "anchor_chars": sum(len(text) for _, text in sections),
    }


# ---------------------------------------------------------------------------
# Handoff extraction (deterministic, additive)
# ---------------------------------------------------------------------------
# A benchmark on branch experiment/handoff-template-bench measured a
# five-section handoff template against the current injection.  Two results
# shape what is implemented here:
#
#   1. It has to be *additive*.  Replacing the ledger with a template loses
#      what the template never wrote; adding sections to the ledger does not.
#   2. The sections that earn their tokens are copying tasks -- a path, an
#      exact value, an error line, a user's pending ask -- not summarising
#      ones.  So they are built with regexes from the captured turns: no
#      generator call, no latency on the hook path, and nothing invented.
#
# Generated handoffs recalled more per token, but generation belongs on the
# background path and is deliberately not wired in here.  This is the fallback
# that always works, and it is what ships.
DEFAULT_HANDOFF_CHARS = 720
DEFAULT_HANDOFF_PER_SECTION = 3
DEFAULT_HANDOFF_RECENT_CHUNKS = 8
DEFAULT_JOURNAL_MAX_OPEN = 4

HANDOFF_HEADINGS = {
    "open_questions_pending_asks": "Open questions / pending asks",
    "errors_and_fixes": "Errors and fixes",
    "files_symbols_references": "Files, symbols, exact references",
    "critical_exact_values": "Critical exact values",
}
# Open asks render first: a resumed agent needs the unfinished instruction
# before it needs the facts that surround it.
HANDOFF_SECTION_ORDER = (
    "open_questions_pending_asks",
    "errors_and_fixes",
    "files_symbols_references",
    "critical_exact_values",
)
# Extraction caps: how many candidates survive ranking before rendering.
HANDOFF_SECTION_CAPS = {
    "open_questions_pending_asks": 8,
    "errors_and_fixes": 6,
    "files_symbols_references": 12,
    "critical_exact_values": 8,
}
# Render caps: how many of them are worth showing every turn.  One section
# must not starve the others, and the total character budget is the real
# bound -- these are only a per-section share of it.
HANDOFF_RENDER_CAPS = {
    "open_questions_pending_asks": 3,
    "errors_and_fixes": 3,
    "files_symbols_references": 4,
    "critical_exact_values": 4,
}

_HANDOFF_ERROR_RE = re.compile(
    r"(?:Traceback|panic:|npm ERR!|error TS\d+|SyntaxError|TypeError|ValueError|"
    r"OSError|Errno|too many open files|error:|fatal error|segmentation fault|"
    r"^\s*(?:Fix|Fixed|Resolution)\s*:|"
    # Raised codes: SQLITE_ERROR:, AUDIT_GAP_409:. Case-sensitive on purpose,
    # so "file_name:" in prose does not read as an error.
    r"(?-i:^\s*(?:Hit\s+)?[A-Z][A-Z0-9]*_[A-Z0-9_]+\s*:))",
    re.I,
)
# A command the next turn will want to rerun carries no file extension, so no
# other rule sees it.
_HANDOFF_FLAG_RE = re.compile(r"(?:^|\s)--[a-z][\w-]*\b")
# Longest extension first, plus a token boundary after it: without the
# boundary "web/routes/checkout.tsx" matches as "checkout.ts".
_HANDOFF_PATH_RE = re.compile(
    r"(?:[\w.-]+/)*[\w.-]+\.(?:tsx|ts|jsx|js|json|toml|ya?ml|py|md|go|rs|java|lua|sql|sh|css|html)"
    r"(?![\w])(?::\d+)?"
)
# Call-shaped symbols only, both spellings: real code is as often
# preloadCheckoutChunk() as stream_export(), and snake_case alone silently
# drops every camelCase symbol.
_HANDOFF_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(")
_HANDOFF_CALL_STOP = frozenset(
    {"if", "for", "while", "switch", "catch", "return", "print", "and", "or", "not",
     # "Traceback (most recent call last):" reads as a call and names nothing.
     "traceback"}
)
_HANDOFF_LINE_RE = re.compile(r"\bline\s+\d+\b", re.I)
# Strong numbers carry a unit, a version shape, or an explicit label.  A bare
# integer is weak: transcripts are full of "Turn 12" and "17 warnings", and
# ranking every integer equally buries the port number under noise.
_HANDOFF_STRONG_NUM_RE = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)+|"
    r"(?<![\w.])\d+(?:\.\d+)?\s?(?:ms|s|%|MB|GB|KB|rps|days?|hours?|minutes?|h|hrs?)\b|"
    r"(?<![\w.])\d+(?:\.\d+)?\s?[kKMB]\b|"
    r"(?:port|version|v)\s?\d+|"
    r"(?<![\w.])\d+(?=\s*(?:shards|handles|retries|workers|seconds|lines))",
    re.I,
)
# A trailing period must not hide a number: "--shards 8." is a fact, and
# requiring a non-word character after the digit dropped it at clause end.
_HANDOFF_WEAK_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\w)")
_HANDOFF_TURN_PREFIX_RE = re.compile(r"^\s*Turn\s+\d+\s*:\s*", re.I)
_HANDOFF_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+")
_HANDOFF_ASK_RE = re.compile(
    r"\b(?:please|let's|lets|add|remove|rename|make sure|don't|do not|wait until|one more|"
    r"need to|must|should|instead|avoid|ensure)\b",
    re.I,
)
# Output of a command is not a fact: "pytest -q -> 214 passed, 3 skipped in
# 8.4s" crowded the value slots in the benchmark and hurt delivery.  Drop the
# clause rather than rank it.
_HANDOFF_CHATTER_RE = re.compile(
    r"(?:\b\d+\s+(?:passed|failed|skipped|errors?|warnings?)\b|"
    r"\bcollected\s+\d+\b|"
    r"\bcoverage\b|\bTOTAL\s+\d|"
    r"(?:->|=>)\s*\d|"
    r"\bin\s+\d+(?:\.\d+)?s\b|"
    r"\bnpm (?:WARN|notice)\b)",
    re.I,
)

# Journal tags.  The marker is ※ (U+203B) -- two tokens, no collisions in real
# transcripts, and visible in a terminal.  The entry runs from the marker to
# end of line; every punctuation candidate occurs inside the values we care
# about ("2.7.1", "src/export/csv_writer.py", "Hit OSError: [Errno 24]").
JOURNAL_MARKER = "※"
JOURNAL_INSTRUCTION = (
    "Journal: start a line with ※v, ※e or ※f for a fact, error or fix "
    "(quote it word for word from the session), ※t <id> <task> to open a task, "
    "※d <id> to close it."
)
_JOURNAL_LINE_RE = re.compile(r"(?m)^[ \t]*※([veftd])[ \t]?(.*)$")
_JOURNAL_ID_RE = re.compile(r"^([A-Za-z0-9_.-]{1,32})(?:\s+(.*))?$")
_JOURNAL_TURN_RE = re.compile(r"^(\d{1,5})\s+(.+)$")


def _handoff_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_HANDOFF", True)


def _journal_enabled() -> bool:
    # Off by default: the benchmark priced only the emission/verification
    # mechanism, and a usefulness audit of the task tags has not been run yet.
    return _env_bool("APPEND_COMPACT_JOURNAL", False)


def _handoff_cap() -> int:
    return max(200, _env_int("APPEND_COMPACT_HANDOFF_CHARS", DEFAULT_HANDOFF_CHARS))


def _handoff_per_section(key: str) -> int:
    """Per-section item count, overridable for every section at once."""
    override = _env_int("APPEND_COMPACT_HANDOFF_PER_SECTION", 0)
    if override:
        return max(1, override)
    return HANDOFF_RENDER_CAPS.get(key, DEFAULT_HANDOFF_PER_SECTION)


def _handoff_recent_chunks() -> int:
    return max(1, _env_int("APPEND_COMPACT_HANDOFF_RECENT_CHUNKS", DEFAULT_HANDOFF_RECENT_CHUNKS))


def _handoff_compact(text: str) -> str:
    """Case- and punctuation-blind form, for dedupe only."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _handoff_tidy(item: str, limit: int = 160) -> str:
    """Strip list markers and cut on a word boundary, never mid-word."""
    text = re.sub(r"^[-*+]\s+", "", str(item).strip())
    # "Turn 39: ..." is the transcript's own numbering, not part of the fact;
    # keeping it both wastes a bullet and defeats duplicate detection.
    text = _HANDOFF_TURN_PREFIX_RE.sub("", text, count=1)
    # A clause split on ";" leaves the separator behind, which renders as a
    # stray semicolon at the end of a bullet.
    text = text.rstrip(";,").strip()
    if len(text) > limit:
        text = text[: limit - 3].rsplit(" ", 1)[0] + "..."
    return text


def _handoff_clauses(text: str) -> list[str]:
    """Sentence-sized clauses, without a ``Turn N:`` prefix.

    Clause granularity is the point: "8" alone could be a port, a count or a
    timeout, but the sentence around it carries the meaning the resumed agent
    needs.  The prefix is removed first or every entry looks like it mentions
    a number.
    """
    text = _HANDOFF_TURN_PREFIX_RE.sub("", str(text), count=1)
    clauses: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for clause in _HANDOFF_CLAUSE_SPLIT_RE.split(stripped):
            clause = clause.strip()
            if clause:
                clauses.append(clause)
    return clauses


def _handoff_rank(scored: list[tuple[float, int, int, str]], cap: int) -> list[str]:
    """Strongest facts first, transcript order as the tie-break."""
    scored = sorted(scored, key=lambda item: (-item[0], item[1], item[2]))
    out: list[str] = []
    for _score, _turn, _clause, text in scored:
        if text not in out:
            out.append(text)
        if len(out) >= cap:
            break
    return out


def _handoff_from_turn(text: str, turn: int, role: str | None = None) -> dict[str, list[list[Any]]]:
    """Copy exact facts out of one raw turn; no model involved.

    Returns ``{section: [[item, turn], ...]}`` with the section priority
    errors, asks, files, values, so one clause is paid for once.  Nothing is
    inferred: every item is a substring of the turn it is anchored to.
    """
    errors: list[tuple[float, int, int, str]] = []
    values: list[tuple[float, int, int, str]] = []
    asks: list[tuple[float, int, int, str]] = []
    files: list[tuple[float, int, int, str]] = []
    role_l = str(role or "").lower()
    for c_index, clause in enumerate(_handoff_clauses(text)):
        if _HANDOFF_CHATTER_RE.search(clause):
            continue
        is_error = bool(_HANDOFF_ERROR_RE.search(clause))
        strong_hits = len(_HANDOFF_STRONG_NUM_RE.findall(clause))
        weak = bool(_HANDOFF_WEAK_NUM_RE.search(clause))
        tidy = _handoff_tidy(clause)
        # Two signals separate a stated fact from a tool log: a user turn
        # stated it, and the clause carries several exact values rather than
        # one incidental count.
        rank = 1.0 + (2.0 if role_l == "user" else 0.0) + min(strong_hits, 3) * 0.5
        if is_error:
            errors.append((rank + 3.0, 0, c_index, tidy))
        elif strong_hits or weak:
            values.append((rank, 0, c_index, tidy))
        # Only user turns carry asks; an assistant sentence that happens to
        # say "should" is commentary, not a request to act on.
        elif role_l == "user" and _HANDOFF_ASK_RE.search(clause):
            asks.append((1.0, 0, c_index, tidy))
        for path in _HANDOFF_PATH_RE.findall(clause):
            files.append((2.0, 0, c_index, path))
        for ident in _HANDOFF_CALL_RE.findall(clause):
            if ident.lower() not in _HANDOFF_CALL_STOP:
                files.append((2.0, 0, c_index, ident))
        for line_ref in _HANDOFF_LINE_RE.findall(clause):
            files.append((2.0, 0, c_index, line_ref))
        if _HANDOFF_FLAG_RE.search(clause) and not is_error:
            files.append((2.5, 0, c_index, tidy))
    payload = {
        "errors_and_fixes": (errors, HANDOFF_SECTION_CAPS["errors_and_fixes"]),
        "open_questions_pending_asks": (asks, HANDOFF_SECTION_CAPS["open_questions_pending_asks"]),
        "files_symbols_references": (files, HANDOFF_SECTION_CAPS["files_symbols_references"]),
        "critical_exact_values": (values, HANDOFF_SECTION_CAPS["critical_exact_values"]),
    }
    out: dict[str, list[list[Any]]] = {}
    placed: set[str] = set()
    for key, (scored, cap) in payload.items():
        items = [item for item in _handoff_rank(scored, cap) if item not in placed]
        placed.update(items)
        if items:
            out[key] = [[item, turn] for item in items]
    return out


def _ground_text(text: str) -> str:
    """Comparison form for the journal verifier: case, whitespace, edge
    punctuation normalised.  Word overlap deliberately does not count."""
    return re.sub(r"\s+", " ", str(text).casefold()).strip(" \t.,;:!?\"'`()[]{}")


def _strip_journal_lines(text: str) -> str:
    return _JOURNAL_LINE_RE.sub("", str(text))


def _grounded(body: str, source: str | None) -> bool:
    """True when ``body`` appears contiguously in ``source``.

    This is the fact checker: a regex cannot tell a copied fact from a
    plausible invention, but a contiguous match can.  Derived reasoning fails
    by design, which is why the journal only carries copied facts and state.
    """
    body_norm = _ground_text(body)
    if len(body_norm) < 3 or not source:
        return False
    return body_norm in _ground_text(_strip_journal_lines(source))


def _turn_source_text(state: dict[str, Any], session: str | None, number: int) -> str | None:
    """The raw text of turn ``number``, for grounding a source-anchored tag."""
    for chunk in reversed(state.get("l1_chunks", [])):
        if int(chunk.get("n", 0)) != number or not _belongs(chunk, session):
            continue
        raw = recall_raw(str(chunk.get("raw_id") or "")) if chunk.get("raw_id") else None
        return raw or str(chunk.get("compressed_text") or "")
    return None


def _harvest_tags(
    text: str,
    turn: int,
    state: dict[str, Any] | None = None,
    session: str | None = None,
) -> list[dict[str, Any]]:
    """Parse ``※`` journal lines, keeping only what the session can verify.

    Kinds: ``v`` value, ``e`` error, ``f`` fix carry an optional source turn
    (``※v 12 port 8443``); ``t`` opens a task by id and ``d`` closes it.  An
    ungrounded tag is dropped, never rendered.
    """
    tags: list[dict[str, Any]] = []
    for match in _JOURNAL_LINE_RE.finditer(str(text)):
        kind = match.group(1)
        body = match.group(2).strip()
        if not body:
            continue
        tag: dict[str, Any] = {"kind": kind, "turn": turn, "grounded": False}
        if kind in {"t", "d"}:
            ident = _JOURNAL_ID_RE.match(body)
            if not ident:
                continue
            tag["id"] = ident.group(1)
            tag["body"] = (ident.group(2) or "").strip()
            # Task state is the model's own bookkeeping, not a copied fact;
            # there is nothing in the transcript to ground it against.
            tag["grounded"] = True
            tags.append(tag)
            continue
        source_turn = turn
        anchor = _JOURNAL_TURN_RE.match(body)
        if anchor:
            source_turn = int(anchor.group(1))
            body = anchor.group(2).strip()
        source = (
            _turn_source_text(state, session, source_turn)
            if state is not None and source_turn != turn
            else str(text)
        )
        if not _grounded(body, source):
            continue
        tag["body"] = body
        tag["source"] = source_turn
        tag["grounded"] = True
        tags.append(tag)
    return tags


def _journal_open_tasks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay ``t``/``d`` tags in chunk order; return the still-open tasks.

    Pure function of immutable chunks, so a reopened id is correct and nothing
    has to be rewritten when a task closes.
    """
    order: list[str] = []
    open_tasks: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for tag in chunk.get("tags") or []:
            if not isinstance(tag, dict):
                continue
            ident = tag.get("id")
            if not ident:
                continue
            if tag.get("kind") == "t":
                if ident not in open_tasks:
                    order.append(ident)
                open_tasks[ident] = {
                    "id": ident,
                    "text": str(tag.get("body") or ""),
                    "turn": tag.get("turn"),
                }
            elif tag.get("kind") == "d":
                open_tasks.pop(ident, None)
    return [open_tasks[ident] for ident in order if ident in open_tasks]


def _render_handoff(
    state: dict[str, Any],
    session: str | None,
    ledger_text: str,
    cap: int | None = None,
    rest_text: str = "",
) -> tuple[str, dict[str, Any]]:
    """Render the additive handoff section, bounded to ``cap`` characters.

    Items are gathered newest-first so the freshest statement of a fact wins,
    deduped against the ledger (the ledger already pins paths and symbols) and
    capped per section.  The journal, when enabled, contributes only task
    state -- the one thing neither a template nor a copier can know.
    """
    cap = _handoff_cap() if cap is None else max(0, cap)
    chunks = [chunk for chunk in state.get("l1_chunks", []) if _belongs(chunk, session)]
    recent = chunks[-_handoff_recent_chunks():]
    # A bullet is worth its tokens only if the rest of the injection does not
    # already say it.  ``rest_text`` is the assembled view without this section
    # (ledger, masters, index lines, live chunks), so this is what makes the
    # handoff additive rather than a second copy of the transcript.
    already_shown = _handoff_compact(ledger_text) + _handoff_compact(rest_text)
    sections: dict[str, list[str]] = {key: [] for key in HANDOFF_SECTION_ORDER}
    seen: set[str] = set()
    for chunk in reversed(recent):
        handoff = chunk.get("handoff")
        if not isinstance(handoff, dict):
            continue
        for key in HANDOFF_SECTION_ORDER:
            if len(sections[key]) >= _handoff_per_section(key):
                continue
            for entry in handoff.get(key) or []:
                if not isinstance(entry, (list, tuple)) or not entry:
                    continue
                item = str(entry[0])
                compact = _handoff_compact(item)
                if not compact or compact in seen or compact in already_shown:
                    continue
                turn = entry[1] if len(entry) > 1 else "?"
                sections[key].append(f"{item} (turn {turn})")
                seen.add(compact)
    open_tasks: list[dict[str, Any]] = []
    if _journal_enabled():
        for task in _journal_open_tasks(chunks)[-DEFAULT_JOURNAL_MAX_OPEN:]:
            label = f"- [task {task['id']}] {task['text']}".rstrip()
            compact = _handoff_compact(label)
            if compact and compact not in seen:
                sections["open_questions_pending_asks"].append(
                    f"{label} (turn {task.get('turn')})"
                )
                open_tasks.append(task)
    def compose() -> str:
        lines: list[str] = []
        if any(sections.values()):
            lines.append("## Session handoff (key facts)")
            for key in HANDOFF_SECTION_ORDER:
                if not sections[key]:
                    continue
                lines.append("")
                lines.append(f"### {HANDOFF_HEADINGS[key]}")
                lines.extend(f"- {item}" for item in sections[key])
        return "\n".join(lines)

    text = compose()
    # Trim items from the least load-bearing end first, so the section order
    # keeps its priority under a tight budget; drop nothing silently.
    while len(text) > cap and any(sections[key] for key in reversed(HANDOFF_SECTION_ORDER)):
        for key in reversed(HANDOFF_SECTION_ORDER):
            if sections[key]:
                sections[key].pop()
                break
        text = compose()
    if len(text) > cap or not text:
        return "", {"enabled": _handoff_enabled(), "sections": 0, "items": 0, "chars": 0}
    if _journal_enabled() and len(text) + len(JOURNAL_INSTRUCTION) + 1 <= cap:
        text = text + "\n" + JOURNAL_INSTRUCTION
    return text, {
        "enabled": _handoff_enabled(),
        "sections": sum(1 for key in HANDOFF_SECTION_ORDER if sections[key]),
        "items": sum(len(items) for items in sections.values()),
        "chars": len(text),
        "open_tasks": len(open_tasks),
    }



CODE_EXTENSIONS = (
    "ts|tsx|mts|cts|js|jsx|mjs|cjs|py|pyi|pyw|go|rs|java|kt|kts|c|h|cc|cpp|cxx|hpp|hh|"
    "cs|rb|php|swift|m|mm|sh|bash|zsh|fish|ps1|bat|sql|yaml|yml|toml|json|jsonc|md|mdx|"
    "html|htm|xml|css|scss|sass|less|vue|svelte|astro|lua|ex|exs|erl|dart|scala|clj|cljs|"
    "proto|graphql|gql|tf|hcl|nix|gradle|cfg|ini|conf|env|lock|txt|log|csv|tsv|ipynb"
)

PATH_RE = re.compile(
    r"(?<![\w@$.-])"
    r"(?:(?:~|\.{1,2})?[\\/]|[A-Za-z]:\\)?"
    r"(?:[\w@.+-]+[\\/])*"
    r"[\w@.+-]+\.(?:" + CODE_EXTENSIONS + r")"
    r"(?![\w])"
)

SPECIAL_FILES_RE = re.compile(
    r"(?<![\w/])(?:Dockerfile|Makefile|CMakeLists\.txt|\.gitignore|\.dockerignore|"
    r"\.env(?:\.[\w-]+)?)(?![\w])"
)

_EXPORT_SYMBOL_RE = re.compile(
    r"\bexport\s+(?:default\s+)?(?:declare\s+)?(?:async\s+)?"
    r"(?:function\*?|class|interface|type|enum|const|let|var)\s+([A-Za-z_$][\w$]*)"
)
_EXPORT_LIST_RE = re.compile(r"\bexport\s*(?:type\s*)?\{([^}]*)\}")
_PY_SYMBOL_RE = re.compile(
    r"(?m)^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\("
    r"|^[ \t]*class[ \t]+([A-Za-z_]\w*)\b"
)
_GO_SYMBOL_RE = re.compile(
    r"(?m)^[ \t]*func[ \t]+(?:\([^)\n]*\)[ \t]*)?([A-Za-z_]\w*)"
    r"|^[ \t]*type[ \t]+([A-Za-z_]\w*)[ \t]*(?:struct|interface|func|map|chan|\[|int|string|bool|float|byte|rune)\b"
)
_RS_SYMBOL_RE = re.compile(
    r"(?m)^[ \t]*(?:pub(?:\([^)\n]*\))?[ \t]+)?(?:async[ \t]+)?(?:unsafe[ \t]+)?"
    r"(?:extern[ \t]+\"[^\"]+\"[ \t]+)?"
    r"(?:fn|struct|enum|trait|type|const|static|mod)[ \t]+([A-Za-z_]\w*)"
)
_JVM_SYMBOL_RE = re.compile(
    r"(?m)^[ \t]*(?:public|protected|internal|private)?[ \t]*"
    r"(?:abstract[ \t]+|final[ \t]+|sealed[ \t]+|data[ \t]+|open[ \t]+|partial[ \t]+)*"
    r"(?:class|interface|enum|record|object)[ \t]+([A-Za-z_]\w*)"
)


def extract_paths(text: str) -> set[str]:
    paths = {m.group(0) for m in PATH_RE.finditer(text)}
    paths.update(m.group(0) for m in SPECIAL_FILES_RE.finditer(text))
    return paths


def extract_symbols(text: str) -> set[str]:
    symbols: set[str] = set()
    for match in _EXPORT_SYMBOL_RE.finditer(text):
        symbols.add(match.group(1))
    for match in _EXPORT_LIST_RE.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if " as " in part:
                part = part.split(" as ", 1)[1].strip()
            if part.startswith("type "):
                part = part[5:].strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", part):
                symbols.add(part)
    for pattern in (_PY_SYMBOL_RE, _GO_SYMBOL_RE, _RS_SYMBOL_RE, _JVM_SYMBOL_RE):
        for match in pattern.finditer(text):
            for group in match.groups():
                if group:
                    symbols.add(group)
    return symbols


def _is_raw_archive_path(value: str) -> bool:
    """True for internal ``~/.cache/append-compactor/raw/...`` pointers.

    Skeleton placeholders name the raw file so an agent can read the block
    back.  That path is plumbing, not a fact worth pinning: it is long, it is
    unique per block, and left in the ledger it evicts real symbols from the
    injection budget.
    """
    normalized = value.replace("\\", "/")
    root = _raw_dir().as_posix()
    home = Path.home().as_posix()
    prefixes = {root}
    if root.startswith(home + "/"):
        prefixes.add("~" + root[len(home):])
    return any(normalized == prefix or normalized.startswith(prefix + "/") for prefix in prefixes)


def extract_entities(text: str) -> list[str]:
    """Deterministic ledger entries: paths, symbols, then exact facts.

    Fact entries are skipped when the same value is already pinned as a path or
    symbol, so the ledger does not grow a duplicate line per capture.
    """
    paths = {
        path for path in extract_paths(text)
        if not _is_raw_archive_path(path) and not _is_l3_archive_path(path)
    }
    symbols = extract_symbols(text)
    entries = [f"path: {p}" for p in sorted(paths)]
    entries.extend(f"symbol: {s}" for s in sorted(symbols))
    if _fact_ledger_enabled():
        covered = paths | symbols
        entries.extend(
            f"fact: {value}"
            for value in sorted(extract_facts(text))
            if value not in covered
            and not _is_raw_archive_path(value)
            and not _is_l3_archive_path(value)
        )
    return entries


# ---------------------------------------------------------------------------
# Zero-cost minification
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"(?ms)^```[ \t]*([^\n`]*)[ \t]*\n(.*?)^```[ \t]*$")
PARA_SPLIT_RE = re.compile(r"(?:[ \t]*\r?\n){2,}")

_HASH_LANGS = {
    "python", "py", "python3", "bash", "sh", "shell", "zsh", "fish", "yaml", "yml",
    "toml", "ruby", "rb", "perl", "pl", "r", "makefile", "make", "dockerfile", "ini",
    "cfg", "conf", "nginx", "terraform", "hcl", "nix", "powershell", "ps1",
}
_SLASH_LANGS = {
    "javascript", "js", "jsx", "typescript", "ts", "tsx", "java", "c", "cpp", "c++", "h",
    "hpp", "cs", "csharp", "go", "golang", "rust", "rs", "swift", "kotlin", "kt", "scala",
    "dart", "php", "scss", "sass", "less", "jsonc", "objective-c", "objc", "m", "protobuf",
    "proto", "graphql", "gql",
}
_DASH_LANGS = {"sql", "lua", "haskell", "hs", "elm", "ada", "vhdl"}

_DIRECTIVE_MARKERS = (
    "noqa", "type:", "coding:", "pylint:", "eslint", "prettier-ignore", "ts-ignore",
    "ts-nocheck", "ts-expect-error", "@ts-", "shellcheck", "nosec", "nolint",
    "coverage:", "pytest:", "fmt:", "isort:", "yapf:", "pragma", "go:build",
    "region", "endregion",
)

_DEBUG_STMT_RE = re.compile(
    r"^[ \t]*(?:"
    r"console\.(?:log|debug|trace)\s*\("
    r"|(?:logger|log|logging)\.debug\s*\("
    r"|System\.(?:out|err)\.println\s*\("
    r"|fmt\.Println\s*\("
    r"|dbg!\s*\("
    r"|print\s*\("
    r")"
)
_EMPTY_IMPORT_RE = re.compile(
    r"^[ \t]*(?:import[ \t]*;?|from[ \t]+[\w.]+[ \t]+import[ \t]*)$"
)

_ERROR_RE = re.compile(
    r"(?im)(?:"
    r"traceback \(most recent call last\)|fatal error|panic:|syntaxerror|typeerror|"
    r"referenceerror|valueerror|modulenotfounderror|importerror|failed to|npm err!|"
    r"error ts\d+|segmentation fault|core dumped"
    r"|^\s*(?:error|exception)\s*[:!]"
    r")"
)

_FILE_PATH_PATTERN = (
    r"(?:[A-Za-z]:[\\/]|~[\\/]|\.{1,2}[\\/]|[\\/])?(?:[\w@.+-]+[\\/])+[\w@.+-]+\.(?:"
    + CODE_EXTENSIONS
    + r")"
    r"|\b[\w@.+-]+\.(?:" + CODE_EXTENSIONS + r")\b"
)

_NUMERIC_LITERAL_PATTERN = (
    # Decimals, versions and numbers with units.  LLMLingua preserves digits
    # (force_reserve_digit) but can drop the punctuation that makes them one
    # number ("7.592" -> "7 592"), so mask the whole literal and restore it
    # byte-exact.  Bare integers are left to the compressor: masking every "3"
    # costs more compression than it saves.
    r"\b\d+(?:[.,]\d+)+(?:%|ns|µs|us|ms|s|KiB|MiB|GiB|TiB|KB|MB|GB|TB|PB)?(?![\w])"
    r"|\b\d+(?:%|ns|µs|us|ms|s|KiB|MiB|GiB|TiB|KB|MB|GB|TB|PB)(?![\w])"
)

# ---------------------------------------------------------------------------
# Exact-fact patterns
#
# These are the identifiers that compaction must never lose: a wrong path,
# version or exit code is worse than a dropped adjective.  The same patterns
# feed two mechanisms: masked literals are restored byte-exact after the model
# runs, and extracted facts are pinned in the entity ledger so they survive L2
# digest truncation and the injection budget.  Matching is deliberately
# context-anchored; bare integers are left alone.
# ---------------------------------------------------------------------------

_FILELINE_PATTERN = (
    r"(?<![\w/])(?:" + _FILE_PATH_PATTERN + r"):\d+(?::\d+)?"
)
_BARE_PATH_PATTERN = r"(?<![\w/])(?:[\w.@+-]+/){2,}[\w.@+-]+\b"
_PKG_VERSION_PATTERN = (
    r"\b(?:@[a-z0-9._-]+/)?[a-z0-9][a-z0-9._-]*@\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.]+)?"
)
_SEMVER_RANGE_PATTERN = (
    r"(?:[<>]=?\s*)?\d+\.\d+\.\d+(?:\s+(?:[<>]=?\s*)\d+\.\d+\.\d+)+"
)
_SOFTWARE_VERSION_PATTERN = (
    r"\b(?:PostgreSQL|Postgres|Node|node|Python|python3?|OpenSSL|openssl|Go|Rust|"
    r"Java|React|CUDA|Ubuntu|Debian|Alpine|macOS|Windows|Chrome|Firefox|nginx|"
    r"Redis|MySQL|SQLite|Ruby|PHP|Kotlin|Swift|Scala|Elixir|Erlang|Terraform|"
    r"Docker|Kubernetes)\s+v?\d+(?:\.\d+)+"
)
_IP_PORT_PATTERN = r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?\b"
_VERSION_PATTERN = r"\bv?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.]+)?\b"
_SHORT_VERSION_PATTERN = r"\bv\d+(?:\.\d+)*\b"
_PORT_PATTERN = r"\b(?i:port)\s+\d{1,5}\b|\blocalhost:\d{1,5}\b"
_HTTP_STATUS_PATTERN = r"\b(?i:HTTP(?:/[0-9.]+)?\s+[1-5]\d{2})\b"
_EXIT_CODE_PATTERN = (
    r"\b(?i:exit(?:ed)?(?:\s+with)?(?:\s+code)?|status(?:\s+code)?)"
    r"\s*[:=]?\s*[1-9]\d{0,2}\b"
)
_COMMAND_PATTERN = (
    r"\b(?:cargo|npm|pnpm|yarn|npx|go|make|cmake|docker|podman|kubectl|helm|"
    r"git|pytest|python3?|pip3?|uv|poetry|rustc|gcc|g\+\+|clang|mvn|gradle|bazel)"
    r"\s+(?:build|test|run|check|install|ci|add|remove|publish|exec|fmt|clippy|"
    r"bench|apply|get|describe|logs|create|delete|push|pull|commit|status|diff)\b"
    r"(?:\s+--?[\w-]+(?:=\S+)?)*"
)
_CVE_PATTERN = r"\b(?i:CVE-\d{4}-\d{4,})\b"
_HEADER_PATTERN = (
    r"\b(?:X-[A-Za-z0-9-]{2,}|Content-(?:Type|Length|Encoding)|User-Agent|"
    r"Authorization|Accept-Encoding|Accept-Language|Cache-Control|If-None-Match|ETag)\b"
)
_TIMESTAMP_PATTERN = (
    r"\b20\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_DATE_PATTERN = r"\b20\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
_SHA_PATTERN = (
    r"\b(?i:(?=[0-9a-f]{7,40}\b)(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)"
    r"[0-9a-f]{7,40})\b"
)
_ENV_PATTERN = r"\b(?=[A-Z0-9_]{6,}\b)[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"
_SNAKE_PATTERN = r"\b(?=[a-z0-9_]{6,}\b)[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"
_CAMEL_PATTERN = (
    r"(?<![\w$])(?=[\w$]{6,}(?![\w$]))"
    r"(?:[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+"
    r"|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"
    r"|(?:[A-Z]{2,}[a-z0-9]+)+[A-Za-z0-9]*)"
    r"(?![\w$])"
)
_DURATION_PATTERN = r"\b\d+(?:\.\d+)?\s?(?:ns|µs|us|ms|min|s|h)\b"
_FLAG_PATTERN = r"\b[a-z][a-z0-9]*(?:\.[a-z][a-z0-9_-]+){2,}\b"
_ERRCODE_PATTERN = r"\bE[A-Z0-9]{3,}\b"

_FACT_MASK_PATTERNS = (
    _FILELINE_PATTERN,
    _BARE_PATH_PATTERN,
    _PKG_VERSION_PATTERN,
    _SEMVER_RANGE_PATTERN,
    _SOFTWARE_VERSION_PATTERN,
    _IP_PORT_PATTERN,
    _VERSION_PATTERN,
    _SHORT_VERSION_PATTERN,
    _PORT_PATTERN,
    _HTTP_STATUS_PATTERN,
    _EXIT_CODE_PATTERN,
    _COMMAND_PATTERN,
    _CVE_PATTERN,
    _HEADER_PATTERN,
    _TIMESTAMP_PATTERN,
    _DATE_PATTERN,
    _SHA_PATTERN,
    _DURATION_PATTERN,
)

# (kind, pattern) in priority order.  A later pattern never re-matches a span
# an earlier one claimed, so "v3.7.1" is one fact, not also "v3".
_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("url", re.compile(r"https?://[^\s<>\"')\]]+")),
    ("fileline", re.compile(_FILELINE_PATTERN)),
    ("path", re.compile(_BARE_PATH_PATTERN)),
    ("pkg", re.compile(_PKG_VERSION_PATTERN)),
    ("semver", re.compile(_SEMVER_RANGE_PATTERN)),
    ("version", re.compile(_SOFTWARE_VERSION_PATTERN)),
    ("ipport", re.compile(_IP_PORT_PATTERN)),
    ("version", re.compile(_VERSION_PATTERN)),
    ("version", re.compile(_SHORT_VERSION_PATTERN)),
    ("port", re.compile(_PORT_PATTERN)),
    ("http", re.compile(_HTTP_STATUS_PATTERN)),
    ("exit", re.compile(_EXIT_CODE_PATTERN)),
    ("command", re.compile(_COMMAND_PATTERN)),
    ("cve", re.compile(_CVE_PATTERN)),
    ("header", re.compile(_HEADER_PATTERN)),
    ("date", re.compile(_TIMESTAMP_PATTERN)),
    ("date", re.compile(_DATE_PATTERN)),
    ("sha", re.compile(_SHA_PATTERN)),
    ("duration", re.compile(_DURATION_PATTERN)),
    ("env", re.compile(_ENV_PATTERN)),
    ("setting", re.compile(_SNAKE_PATTERN)),
    ("camel", re.compile(_CAMEL_PATTERN)),
    ("flag", re.compile(_FLAG_PATTERN)),
    ("errcode", re.compile(_ERRCODE_PATTERN)),
)

# Per-kind cap per captured turn.  A code-heavy turn can contain dozens of
# snake_case identifiers; pinning all of them would crowd out paths and facts
# in the 2k ledger.
_FACT_CAP_PER_KIND = {
    "url": 8,
    "fileline": 8,
    "path": 16,
    "pkg": 8,
    "semver": 4,
    "version": 12,
    "ipport": 8,
    "port": 6,
    "http": 8,
    "exit": 6,
    "command": 8,
    "cve": 4,
    "header": 8,
    "date": 6,
    "sha": 8,
    "duration": 8,
    "env": 16,
    "setting": 16,
    "camel": 12,
    "flag": 8,
    "errcode": 6,
}

_BASE_MASKABLE_RE = re.compile(
    r"`[^`\n]+`"
    r"|https?://[^\s<>\"')\]]+"
    r"|" + _FILE_PATH_PATTERN +
    r"|" + _NUMERIC_LITERAL_PATTERN +
    r"|\b[A-Za-z_$][\w$]*(?:_[A-Za-z0-9_$]+)+\b"
    r"|\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+\b"
    r"|\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b"
    r"|\b[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*\b"
)

# Fact patterns go before the generic file-path/numeric alternatives:
# otherwise "src/app.ts:12" masks only "src/app.ts", and "10.0.4.17:8443"
# masks "10.0.4.17" plus a separate numeric literal.
_FACT_MASKABLE_RE = re.compile(
    r"`[^`\n]+`"
    r"|https?://[^\s<>\"')\]]+"
    r"|" + "|".join(_FACT_MASK_PATTERNS) +
    r"|" + _FILE_PATH_PATTERN +
    r"|" + _NUMERIC_LITERAL_PATTERN +
    r"|\b[A-Za-z_$][\w$]*(?:_[A-Za-z0-9_$]+)+\b"
    r"|\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+\b"
    r"|\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b"
    r"|\b[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*\b"
)


def _fact_ledger_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_FACT_LEDGER", True)


def _active_maskable_re() -> re.Pattern[str]:
    return _FACT_MASKABLE_RE if _fact_ledger_enabled() else _BASE_MASKABLE_RE


def extract_facts(text: str) -> set[str]:
    """Extract exact facts in priority order, skipping already-claimed spans."""
    claimed: list[tuple[int, int]] = []
    by_kind: dict[str, set[str]] = {}
    for kind, pattern in _FACT_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(start < other_end and other_start < end for other_start, other_end in claimed):
                continue
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            if not value:
                continue
            claimed.append((start, end))
            by_kind.setdefault(kind, set()).add(value)
    facts: set[str] = set()
    for kind, values in by_kind.items():
        facts.update(sorted(values)[: _FACT_CAP_PER_KIND.get(kind, 12)])
    return facts


def _language(info: str) -> str:
    token = info.strip().lower()
    if token.startswith("{") and token.endswith("}"):
        token = token[1:-1].strip()
    token = token.split("{", 1)[0].strip()
    if not token:
        return ""
    return token.split()[0].strip(".>,")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def minify_json_text(text: str) -> str | None:
    """Compact a JSON object/array; return None if the text is not strict JSON."""
    try:
        obj = json.loads(text, parse_constant=_reject_json_constant)
    except (TypeError, ValueError):
        return None
    if not isinstance(obj, (dict, list)):
        return None
    try:
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except ValueError:
        return None


def minify_python_source(source: str, filename: str = "<snippet>") -> str | None:
    """Return minified Python or None when unavailable / unsafe / not smaller.

    python-minifier changed its keyword surface across major versions (3.x
    removed ``remove_comments``/``remove_docstrings`` and folds docstring
    removal into ``remove_literal_statements``), so build the kwargs from the
    installed signature instead of hard-coding them.
    """
    try:
        import python_minifier
    except Exception:
        return None
    try:
        params = inspect.signature(python_minifier.minify).parameters
    except (TypeError, ValueError):
        params = {}

    kwargs: dict[str, Any] = {}

    def offer(name: str, value: Any) -> None:
        if not params or name in params:
            kwargs[name] = value

    offer("remove_comments", True)
    if "remove_docstrings" in params:
        offer("remove_docstrings", True)
        offer("remove_literal_statements", False)
    else:
        # 3.x: bare string-literal statements are docstrings/comment-like and
        # have no runtime effect, so removing them is safe for a summary.
        offer("remove_literal_statements", True)
    offer("remove_pass", True)
    offer("combine_imports", True)
    offer("hoist_literals", False)
    offer("rename_locals", False)
    offer("rename_globals", False)
    offer("remove_annotations", False)
    offer("preserve_shebang", True)

    try:
        result = python_minifier.minify(source, filename=filename, **kwargs)
    except Exception:
        return None
    if not result or len(result) >= len(source):
        return None
    try:
        compile(result, filename, "exec")
    except (SyntaxError, ValueError):
        return None
    return result


def _comment_style(language: str) -> tuple[str, ...]:
    if language in _HASH_LANGS:
        return ("#",)
    if language in _DASH_LANGS:
        return ("--",)
    if language in {"css"}:
        return ("/*",)
    if language in _SLASH_LANGS:
        return ("//", "/*")
    if language in {"html", "xml", "vue", "svelte", "astro"}:
        return ("<!--",)
    return ()


def _has_directive(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in _DIRECTIVE_MARKERS)


def _is_comment_only(stripped: str, styles: tuple[str, ...]) -> bool:
    for style in styles:
        if style == "#":
            if stripped.startswith("#") and not stripped.startswith(("#!", "#[")):
                if not _has_directive(stripped):
                    return True
        elif style == "--":
            if stripped.startswith("--") and not stripped.startswith("-->"):
                return True
        elif style == "//":
            if stripped.startswith("//") and not _has_directive(stripped):
                return True
        elif style == "/*":
            if stripped.startswith("/*") and stripped.endswith("*/") and not _has_directive(stripped):
                return True
        elif style == "<!--":
            if stripped.startswith("<!--") and stripped.endswith("-->"):
                return True
    return False


_CONTINUATION_ENDINGS = (
    ",", "(", "[", "\\", "+", "-", "*", "/", "=", "=>", "&&", "||", "|", "&", ".",
)


def _delete_candidate(line: str, styles: tuple[str, ...]) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if styles and _is_comment_only(stripped, styles):
        return True
    if _DEBUG_STMT_RE.match(line):
        return True
    if _EMPTY_IMPORT_RE.match(line):
        return True
    return False


def _block_opener(lines: list[str], idx: int, indent: int) -> int | None:
    """Nearest enclosing block opener (a less-indented line ending in ':' or '{')."""
    j = idx - 1
    while j >= 0:
        candidate = lines[j]
        if not candidate.strip():
            j -= 1
            continue
        cand_indent = len(candidate) - len(candidate.lstrip())
        if cand_indent < indent and candidate.rstrip().endswith((":", "{")):
            return j
        if cand_indent < indent:
            return None
        j -= 1
    return None


def _would_empty_block(
    lines: list[str],
    idx: int,
    indent: int,
    candidates: set[int],
) -> bool:
    opener = _block_opener(lines, idx, indent)
    if opener is None:
        return False
    opener_indent = len(lines[opener]) - len(lines[opener].lstrip())
    for j in range(opener + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent <= opener_indent:
            break
        if j not in candidates:
            # Any remaining statement/string keeps the block non-empty.
            return False
    return True


def _continuation_unsafe(lines: list[str], idx: int) -> bool:
    j = idx - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    if j >= 0:
        previous = lines[j].rstrip()
        if previous.endswith(_CONTINUATION_ENDINGS):
            return True
    k = idx + 1
    while k < len(lines) and not lines[k].strip():
        k += 1
    if k < len(lines):
        nxt = lines[k].lstrip()
        if nxt.startswith((".", ")", "]", "}", "?.")):
            return True
    return False


def strip_line_level(source: str, language: str = "") -> str:
    """Delete comment-only lines, debug statements and empty imports by line.

    The pass is intentionally conservative: a line is kept when removing it
    could empty a block body, break a continuation, or sit inside a multi-line
    string.  Remaining blank-line runs collapse to one.
    """
    lang = language.lower().strip()
    styles = _comment_style(lang)
    lines = source.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    protected: set[int] = set()
    candidates: set[int] = set()
    multiline_quote: str | None = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if multiline_quote is not None:
            protected.add(idx)
            if stripped.count(multiline_quote) % 2 == 1:
                multiline_quote = None
            continue

        # Detect entry into a multi-line string so its contents stay verbatim.
        opened: str | None = None
        for quote in ('"""', "'''", "`"):
            if quote not in stripped or stripped.count(quote) % 2 == 0:
                continue
            if quote == "`" and lang not in {"javascript", "js", "jsx", "typescript", "ts", "tsx", "vue", "svelte"}:
                continue
            opened = quote
            break
        if opened is not None:
            multiline_quote = opened
            protected.add(idx)
            continue

        if _delete_candidate(line, styles):
            candidates.add(idx)

    deletable: set[int] = set()
    for idx in sorted(candidates):
        if idx in protected:
            continue
        indent = len(lines[idx]) - len(lines[idx].lstrip())
        if _continuation_unsafe(lines, idx):
            continue
        if _would_empty_block(lines, idx, indent, candidates):
            continue
        deletable.add(idx)

    out: list[str] = []
    for idx, line in enumerate(lines):
        if idx in deletable:
            continue
        out.append(line.rstrip())

    collapsed: list[str] = []
    blank = False
    for line in out:
        if not line.strip():
            if blank:
                continue
            collapsed.append("")
            blank = True
        else:
            collapsed.append(line)
            blank = False
    return "\n".join(collapsed).strip("\n")


def _render_fence(info: str, body: str) -> str:
    info = info.strip()
    body = body.rstrip("\n")
    if not body:
        return f"```{info}```"
    return f"```{info}\n{body}\n```"


# ---------------------------------------------------------------------------
# Code skeletonization with raw-block placeholders
#
# A large fenced code block often exists to show structure: imports, classes,
# signatures, constants.  Replacing bodies with ``...`` keeps that structure
# at a fraction of the size.  It is off by default because bodies can hold the
# fact you actually needed; the recall harness measures the trade-off.
# ---------------------------------------------------------------------------

CODE_SKELETON_MIN_CHARS = 2_000
_BRACED_DECL_RE = re.compile(
    r"\b(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(?:function|class|interface|type|enum|struct|impl|trait|fn|func|constructor|"
    r"namespace|module)\b"
)
_BRACED_METHOD_RE = re.compile(
    r"^\s*(?:[\w$<>[\],.?]+\s+)*\w+\s*\([^;]*\)\s*(?::[^{]+)?\{\s*$"
)
_BRACED_CONTROL_RE = re.compile(
    r"^\s*(?:if|for|while|switch|catch|with|else|try|do|return)\b"
)


def _code_skeleton_enabled() -> bool:
    # Skeleton bodies are recoverable from the raw archive, so this can be on
    # by default.  If raw archiving is disabled, _compress_fence falls back to
    # minified code rather than dropping a body with no reference.
    return _env_bool("APPEND_COMPACT_CODE_SKELETON", True)


def _python_doc_first_line(node: ast.AST) -> str:
    body = getattr(node, "body", None)
    if not body:
        return ""
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        text = first.value.value.strip()
        if text:
            return text.splitlines()[0][:120]
    return ""


def skeleton_python_source(source: str) -> str | None:
    """AST-based Python skeleton: imports, signatures, constants, first doc line."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError):
        return None
    lines = source.splitlines()
    out: list[str] = []

    def header_end(node: ast.AST) -> int:
        body = getattr(node, "body", None) or []
        return body[0].lineno if body else getattr(node, "lineno", 1)

    def emit(node: ast.AST, indent: int) -> None:
        pad = " " * indent
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.extend(lines[node.lineno - 1: node.end_lineno])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                out.extend(lines[dec.lineno - 1: dec.end_lineno])
            body_line = header_end(node)
            if body_line <= node.lineno:
                out.extend(lines[node.lineno - 1: node.end_lineno])
                return
            out.extend(lines[node.lineno - 1: body_line - 1])
            doc = _python_doc_first_line(node)
            if doc:
                out.append(pad + "    " + '"""' + doc.replace('"""', "'''") + '"""')
            out.append(pad + "    ...")
        elif isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                out.extend(lines[dec.lineno - 1: dec.end_lineno])
            body_line = header_end(node)
            if body_line <= node.lineno:
                out.extend(lines[node.lineno - 1: node.end_lineno])
                return
            out.extend(lines[node.lineno - 1: body_line - 1])
            doc = _python_doc_first_line(node)
            if doc:
                out.append(pad + "    " + '"""' + doc.replace('"""', "'''") + '"""')
            emitted = 0
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    emit(child, child.col_offset)
                    emitted += 1
                elif isinstance(child, (ast.Assign, ast.AnnAssign)) and _simple_value(child.value):
                    out.extend(lines[child.lineno - 1: child.end_lineno])
                    emitted += 1
            if not emitted and not doc:
                out.append(pad + "    ...")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and indent == 0:
            out.extend(lines[node.lineno - 1: node.end_lineno])

    for statement in tree.body:
        emit(statement, 0)
    skeleton = "\n".join(line for line in out if line.strip())
    return skeleton or None


def _simple_value(value: ast.AST) -> bool:
    return isinstance(
        value,
        (ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.UnaryOp, ast.BinOp),
    )


def skeleton_braced_source(source: str) -> str | None:
    """Conservative signature skeleton for brace languages (JS/TS/Go/Rust/Java)."""
    lines = source.splitlines()
    out: list[str] = []
    depth = 0
    skipped = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        is_method = (
            bool(_BRACED_METHOD_RE.match(line))
            and not _BRACED_CONTROL_RE.match(line)
        )
        keep = (
            depth == 0
            or bool(_BRACED_DECL_RE.search(stripped))
            or is_method
            or stripped in {"{", "}", "};", "})", "});"}
        )
        if keep:
            out.append(line)
            skipped = False
        elif not skipped:
            indent = line[: len(line) - len(line.lstrip())]
            out.append(indent + "// ...")
            skipped = True
        depth = max(0, depth + line.count("{") - line.count("}"))
    skeleton = "\n".join(out)
    return skeleton or None


_SKELETON_LANGS = {
    "javascript", "js", "jsx", "typescript", "ts", "tsx", "go", "golang",
    "rust", "rs", "java", "kotlin", "kt", "scala", "c", "cpp", "c++", "h",
    "hpp", "cs", "csharp", "swift", "dart", "php",
}


def _skeletonize_fence(info: str, body: str, stats: dict[str, int]) -> str | None:
    if not _code_skeleton_enabled() or len(body) < CODE_SKELETON_MIN_CHARS:
        return None
    lang = _language(info)
    if lang in {"python", "py", "python3"}:
        skeleton = skeleton_python_source(body)
    elif lang in _SKELETON_LANGS:
        skeleton = skeleton_braced_source(body)
    else:
        return None
    if skeleton and len(skeleton.strip()) < len(body.strip()):
        return skeleton.strip()
    return None


_SKELETON_RAW_ID_RE = re.compile(r"([0-9a-f]{64})\.txt; read before editing")


def _skeleton_placeholder(lang: str, raw_id: str, body: str) -> str:
    """One-line pointer to the full block in the raw archive.

    The raw file is the exact fenced body (no fence lines), so an agent can
    read it directly or pass the id to ``--recall``.  Skeletonization is only
    used when this pointer can be written: losing a body without leaving a
    recoverable reference is worse than keeping the minified code.
    """
    marker = "#" if lang in {"python", "py", "python3"} else "//"
    count = max(1, len(body.splitlines()))
    return (
        f"{marker} [skeleton: {count} lines elided; "
        f"full code: {_raw_file_display(raw_id)}; read before editing]"
    )


def _compress_fence(info: str, body: str, stats: dict[str, int]) -> str:
    lang = _language(info)
    if lang in {"json", "jsonc"}:
        compact = minify_json_text(body)
        if compact is not None:
            stats["json_blocks"] += 1
            return _render_fence(info, compact)
        # Invalid JSON must be left byte-for-byte alone.
        return _render_fence(info, body.rstrip("\n"))
    normal = body.rstrip("\n")
    if lang in {"python", "py", "python3"}:
        mini = minify_python_source(body)
        if mini is not None:
            normal = mini
    else:
        stripped = strip_line_level(body, lang)
        if stripped != normal:
            normal = stripped
    if normal != body.rstrip("\n"):
        stats["code_blocks"] += 1
    skeleton = _skeletonize_fence(info, body, stats)
    if skeleton is not None and len(skeleton) < len(normal):
        raw_id = archive_raw(body)
        if raw_id is not None:
            annotated = _skeleton_placeholder(lang, raw_id, body) + "\n" + skeleton
            if len(annotated) < len(normal):
                stats["code_skeleton_blocks"] = stats.get("code_skeleton_blocks", 0) + 1
                stats.setdefault("code_block_ids", []).append(raw_id)
                return _render_fence(info, annotated)
    return _render_fence(info, normal)


# ---------------------------------------------------------------------------
# Selective LLMLingua-2 prose compression
# ---------------------------------------------------------------------------

_compressor: Any = None
_compressor_failed = False


def _get_compressor() -> Any:
    global _compressor, _compressor_failed
    if _compressor is not None or _compressor_failed:
        return _compressor
    model = os.environ.get("APPEND_COMPACT_LLM_MODEL", LLMLINGUA_MODEL).strip() or LLMLINGUA_MODEL
    try:
        from llmlingua import PromptCompressor

        _compressor = PromptCompressor(model_name=model, use_llmlingua2=True, device_map="cpu")
    except Exception as exc:  # missing torch/model/network: fall back once, fast
        _compressor_failed = True
        _log_error(f"llmlingua unavailable ({model}): {exc!r}; using deterministic fallback")
        return None
    return _compressor


def _default_llm_compress(
    text: str,
    rate: float,
    force_tokens: list[str],
    *,
    model: str | None = None,
) -> str | None:
    compressor = _get_compressor()
    if compressor is None:
        return None
    try:
        result = compressor.compress_prompt(
            text,
            rate=rate,
            force_tokens=list(force_tokens),
            force_reserve_digit=True,
            drop_consecutive=True,
        )
    except Exception as exc:
        _log_error(f"llmlingua compression failed: {exc!r}")
        return None
    if isinstance(result, dict):
        compressed = result.get("compressed_prompt")
        return compressed if isinstance(compressed, str) and compressed else None
    return result if isinstance(result, str) and result else None


def _resolve_llm(llm: Any) -> Callable[..., str | None] | None:
    if llm == "off":
        return None
    if callable(llm):
        return llm
    if not _env_bool("APPEND_COMPACT_LLM", True):
        return None
    return _default_llm_compress


def _normalize_prose(text: str) -> str:
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if blank:
                continue
            out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out).strip("\n")


# ---------------------------------------------------------------------------
# Cross-source capture dedupe
# ---------------------------------------------------------------------------

_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:user|assistant|system|human|tool|claude)\s*:\s*", re.IGNORECASE
)
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.IGNORECASE | re.DOTALL)


def _dedupe_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_DEDUPE", True)


def _capture_key(text: str, *, strip_role: bool = False) -> str:
    """Durable exact-identity key; role labels are stripped only for PreCompact.

    Raw prompts are hashed as typed.  ``_normalize_prose`` preserves line
    breaks and indentation, so a code block with changed indentation cannot
    collapse onto another capture.
    """
    body = _ROLE_PREFIX_RE.sub("", text, count=1) if strip_role else text
    return hashlib.sha256(_normalize_prose(body).encode("utf-8")).hexdigest()[:32]


def _similarity_text(text: str) -> str:
    """Comparison-only normalization: reminders and role labels removed."""
    body = _SYSTEM_REMINDER_RE.sub(" ", text)
    body = _ROLE_PREFIX_RE.sub("", body, count=1)
    body = unicodedata.normalize("NFKC", body).casefold()
    return re.sub(r"\s+", " ", body).strip()


def _word_shingles(tokens: list[str]) -> set[int]:
    size = DEDUPE_SHINGLE_WORDS
    return {
        zlib.crc32(" ".join(tokens[index:index + size]).encode("utf-8"))
        for index in range(max(0, len(tokens) - size + 1))
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _capture_similarity(a: str, b: str) -> dict[str, float | bool]:
    """Token/shingle containment and Jaccard for two normalized texts."""
    tokens_a = re.findall(r"\w+", a)
    tokens_b = re.findall(r"\w+", b)
    set_a, set_b = set(tokens_a), set(tokens_b)
    shingle_a, shingle_b = _word_shingles(tokens_a), _word_shingles(tokens_b)
    shared_tokens, shared_shingles = len(set_a & set_b), len(shingle_a & shingle_b)
    length_ratio = _ratio(len(a), len(b))
    evidence = (
        min(len(tokens_a), len(tokens_b)) >= DEDUPE_MIN_TOKENS
        and min(len(shingle_a), len(shingle_b)) >= DEDUPE_MIN_SHINGLES
    )
    return {
        "evidence": evidence,
        "token_containment_a": _ratio(shared_tokens, len(set_a)),
        "token_containment_b": _ratio(shared_tokens, len(set_b)),
        "token_jaccard": _ratio(shared_tokens, len(set_a | set_b)),
        "shingle_containment_a": _ratio(shared_shingles, len(shingle_a)),
        "shingle_containment_b": _ratio(shared_shingles, len(shingle_b)),
        "shingle_jaccard": _ratio(shared_shingles, len(shingle_a | shingle_b)),
        "substring_ab": bool(a) and a in b,
        "substring_ba": bool(b) and b in a,
        "length_ratio": length_ratio,
    }


def _record_recent_capture(
    state: dict[str, Any],
    text: str,
    source: str,
    role: str | None,
    session: str | None,
    now: float | None = None,
) -> None:
    """Remember a user capture for fuzzy matching; bounded by age and count."""
    if role != "user" or source not in ("precompact", "user_prompt"):
        return
    normalized = _similarity_text(text)
    if not normalized:
        return
    now = time.time() if now is None else now
    window = _env_int("APPEND_COMPACT_DEDUPE_WINDOW", DEFAULT_DEDUPE_WINDOW_SECONDS)
    tokens = re.findall(r"\w+", normalized)
    entry = {
        "capture_key": _capture_key(text, strip_role=source == "precompact"),
        "source": source,
        "role": role,
        "session": session,
        "at": now,
        "text": normalized[:PENDING_MAX_CHARS],
        "tokens": len(tokens),
        "shingles": len(_word_shingles(tokens)),
    }
    recent = state.setdefault("recent_captures", [])
    kept: list[dict[str, Any]] = []
    for item in recent:
        if not isinstance(item, dict):
            continue
        at = item.get("at")
        if not isinstance(at, (int, float)):
            continue
        if now - float(at) > window:
            continue
        kept.append(item)
    kept.append(entry)
    state["recent_captures"] = kept[-RECENT_CAPTURE_LIMIT:]


def _record_prompt_seen(state: dict[str, Any], session: str | None, now: float) -> None:
    if session is None:
        return
    seen = state.setdefault("prompt_seen", {})
    seen[session] = now
    _cap_prompt_seen(state)


def _age(entry: dict[str, Any], now: float) -> float | None:
    at = entry.get("created_at", entry.get("at"))
    if not isinstance(at, (int, float)):
        return None
    return now - float(at)


def _find_duplicate_capture(
    state: dict[str, Any],
    text: str,
    source: str,
    role: str | None,
    session: str | None,
    now: float | None = None,
) -> bool:
    """Return True when this capture is already covered by another source.

    The durable key is checked first; a bounded similarity gate catches
    injected wrappers.  Rules are directional so a prompt that merely overlaps
    an old capture, or adds instructions to one, is never dropped.
    """
    if not _dedupe_enabled() or role not in ("user", "assistant", "tool"):
        return False
    if source not in ("precompact", "user_prompt"):
        return False
    now = time.time() if now is None else now
    strip_role = source == "precompact"
    key = _capture_key(text, strip_role=strip_role)
    input_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    sources = ("precompact", "user_prompt") if source == "precompact" else ("precompact",)
    window = _env_int("APPEND_COMPACT_DEDUPE_WINDOW", DEFAULT_DEDUPE_WINDOW_SECONDS)
    prompt_window = _env_int("APPEND_COMPACT_CAPTURE_DEDUPE_SECONDS", DEFAULT_CAPTURE_DEDUPE_SECONDS)
    since: float | None = None
    max_age: float | None = None
    if source == "user_prompt":
        floor = state.get("prompt_seen", {}).get(session)
        if isinstance(floor, (int, float)):
            since = float(floor)
        max_age = float(prompt_window)

    match = _find_captured_entry(
        state,
        input_sha,
        session,
        capture_key=key,
        sources=sources,
        role=role,
        since=since,
        max_age=max_age,
        now=now,
    )
    if match is not None:
        _log_job(
            "dedupe",
            source=source,
            role=role,
            session=session,
            reason="capture_key" if match.get("capture_key") == key else "input_sha",
            matched_source=match.get("source"),
            matched_role=match.get("role"),
            matched_n=match.get("n"),
            capture_key=key[:12],
        )
        return True

    current = _similarity_text(text)
    if not current:
        return False
    counterpart = "user_prompt" if source == "precompact" else "precompact"
    for entry in reversed(state.get("recent_captures", [])):
        if not isinstance(entry, dict):
            continue
        if entry.get("session") != session or entry.get("role") != "user":
            continue
        if entry.get("source") != counterpart:
            continue
        candidate = entry.get("text")
        at = entry.get("at")
        if not isinstance(candidate, str) or not isinstance(at, (int, float)):
            continue
        age = now - float(at)
        if age < -5 or age > window:
            continue
        if source == "user_prompt":
            if since is not None and float(at) < since:
                continue
            if age > prompt_window:
                continue
        sim = _capture_similarity(current, candidate)
        if not sim["evidence"]:
            continue
        if source == "precompact":
            contained = bool(sim["substring_ba"]) or (
                sim["shingle_containment_b"] >= DEDUPE_CONTAINMENT
                and sim["token_containment_b"] >= DEDUPE_TOKEN_CONTAINMENT
            )
        else:
            contained = bool(sim["substring_ab"]) or (
                sim["shingle_containment_a"] >= DEDUPE_CONTAINMENT
                and sim["token_containment_a"] >= DEDUPE_TOKEN_CONTAINMENT
            )
        # A prompt is never dropped merely for overlapping an old capture: it
        # may add instructions.  Only containment in the prompt direction is
        # safe; the symmetric Jaccard rule is PreCompact-only.
        near_equal = (
            source == "precompact"
            and sim["shingle_jaccard"] >= DEDUPE_JACCARD
            and sim["token_jaccard"] >= DEDUPE_JACCARD
            and DEDUPE_LENGTH_RATIO <= sim["length_ratio"] <= 1 / DEDUPE_LENGTH_RATIO
        )
        if contained or near_equal:
            _log_job(
                "dedupe",
                source=source,
                role=role,
                session=session,
                reason="substring" if (sim["substring_ba"] if source == "precompact" else sim["substring_ab"]) else (
                    "shingle" if contained else "jaccard"
                ),
                matched_source=entry.get("source"),
                matched_role=entry.get("role"),
                matched_age=round(age, 3),
                shingle_containment=round(float(sim["shingle_containment_b"] if source == "precompact" else sim["shingle_containment_a"]), 4),
                token_containment=round(float(sim["token_containment_b"] if source == "precompact" else sim["token_containment_a"]), 4),
                shingle_jaccard=round(float(sim["shingle_jaccard"]), 4),
                token_jaccard=round(float(sim["token_jaccard"]), 4),
                capture_key=key[:12],
            )
            return True
    return False


def _looks_like_prose(text: str) -> bool:
    if not re.search(r"[A-Za-z]{3,}", text):
        return False
    meaningful = sum(1 for char in text if char.isalnum() or char.isspace())
    if meaningful / max(len(text), 1) < 0.62:
        return False
    if text.count(";") + text.count("{") + text.count("}") > len(text) * 0.02:
        return False
    return True


def _mask_entities(text: str) -> tuple[str, list[str], dict[str, str]]:
    mapping: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        original = match.group(0)
        placeholder = f"<<M{len(mapping)}>>"
        mapping[placeholder] = original
        return placeholder

    masked = _active_maskable_re().sub(replace, text)
    return masked, list(mapping), mapping


def _restore_entities(masked: str, mapping: dict[str, str]) -> str:
    restored = masked
    for placeholder, original in mapping.items():
        restored = restored.replace(placeholder, original)
    return restored


_SENTENCE_BREAK_RE = re.compile(r"[.!?]+(?=\s+|$)")
_ABBREVIATIONS = {
    "e.g", "i.e", "etc", "vs", "dr", "mr", "ms", "mrs", "prof", "inc", "ltd",
    "no", "fig", "approx", "cf", "al",
}


def _split_sentences(text: str) -> list[str]:
    """Split prose at sentence ends, skipping common abbreviations and initials."""
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK_RE.finditer(text):
        end = match.end()
        if end >= len(text):
            break
        before = text[max(0, match.start() - 24): match.start()]
        word = re.sub(r"^[^\w]+", "", before).rsplit(None, 1)[-1].lower() if before.strip() else ""
        word = word.strip(".")
        next_char = text[end:].lstrip()[:1]
        if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
            continue
        if re.fullmatch(r"(?:[a-z]\.)+[a-z]", word):
            continue
        # Lowercase sentence starts are normal in technical prose
        # ("scheduleRetryBackoff ..."), so only digits and closing
        # punctuation suppress a boundary.
        if not next_char or next_char.isdigit() or next_char in ",;:)":
            continue
        parts.append(text[start:end])
        start = end
    parts.append(text[start:])
    return [part.strip() for part in parts if part.strip()]


def _sentence_protect_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_SENTENCE_PROTECT", True)


def _rate_control_enabled() -> bool:
    return _env_bool("APPEND_COMPACT_RATE_CONTROL", True)


_LOG_LINE_RE = re.compile(
    r"(?m)^\s*(?:20\d{2}-\d{2}-\d{2}[T ]\d{2}:\d{2}|\[?\d{2}:\d{2}:\d{2}"
    r"|(?:INFO|WARN|WARNING|DEBUG|TRACE|ERROR)\b)"
)
_INSTRUCTION_RE = re.compile(
    r"\b(?:must|should|ensure|make sure|do not|don't|never|always|please|"
    r"required|need to|remember to|run|install|use|set|add|remove|update|"
    r"implement|fix|avoid)\b",
    re.IGNORECASE,
)


def _prose_rate(text: str) -> float:
    """Keep more of instruction-like prose, less of repetitive log-like prose.

    Masked literals and pinned facts survive at any rate; the rate only controls
    how aggressively connective prose is pruned.  Values are deliberately close
    to the 0.4 default: a large swing changes meaning, not just length.
    """
    if not _rate_control_enabled():
        return DEFAULT_RATE
    log_lines = len(_LOG_LINE_RE.findall(text))
    digits = sum(1 for char in text if char.isdigit())
    if log_lines >= 2 or (text and digits / len(text) > 0.18):
        return 0.25
    if len(_INSTRUCTION_RE.findall(text)) >= 2:
        return 0.55
    return DEFAULT_RATE


def compress_prose(
    text: str,
    rate: float | None = None,
    llm: Any = "auto",
) -> tuple[str, str]:
    """Compress natural-language prose only.

    Returns ``(text, mode)`` where mode is one of ``llmlingua``, ``fallback``,
    ``fallback-entity-mismatch``, ``fallback-not-smaller``, ``normalized``,
    ``error-verbatim`` or ``empty``.  Code, JSON, inline code, URLs, paths and
    identifiers are masked before the model sees the text and restored after;
    any mask lost by the compressor forces a safe fallback.
    """
    if rate is None:
        rate = _prose_rate(text)
    protect = _sentence_protect_enabled()
    stripped = text.strip()
    if not stripped:
        return text, "empty"
    if _ERROR_RE.search(stripped):
        return text, "error-verbatim"

    normalized = _normalize_prose(text)
    if len(stripped) < MIN_PROSE_CHARS or not _looks_like_prose(stripped):
        return normalized, "normalized"

    compressor = _resolve_llm(llm)
    if compressor is None:
        return normalized, "fallback"

    # Only the built-in LLMLingua path is memoized: a caller-supplied callable
    # may have side effects or vary, and "off" never reaches here.
    cache_path: Path | None = None
    if _cache_enabled() and compressor is _default_llm_compress:
        try:
            model = os.environ.get("APPEND_COMPACT_LLM_MODEL", LLMLINGUA_MODEL).strip() or LLMLINGUA_MODEL
            key = _compress_cache_key(text, rate, model, "sentence" if protect else "")
            cache_path = default_cache_dir() / "compressed" / key[:2] / f"{key}.json"
            cached = _read_cached_compression(cache_path)
        except (OSError, RuntimeError, ValueError) as exc:
            # A broken cache directory must degrade to a normal compression run.
            _log_error(f"compression cache unavailable: {exc!r}")
            cache_path, cached = None, None
        if cached is not None:
            return cached

    masked, placeholders, mapping = _mask_entities(text)
    try:
        if protect:
            sentences = _split_sentences(masked)
            if len(sentences) >= 3:
                first, last = sentences[0], sentences[-1]
                middle = " ".join(sentences[1:-1])
                # A short middle still gets its own compression: the point of
                # this path is the verbatim boundaries, not the middle's ratio.
                if len(middle) >= 40:
                    middle_placeholders = [p for p in placeholders if p in middle]
                    compressed_middle = compressor(middle, rate, middle_placeholders)
                    if compressed_middle and all(
                        p in compressed_middle for p in middle_placeholders
                    ):
                        compressed = f"{first} {compressed_middle.strip()} {last}"
                    else:
                        compressed = None
                else:
                    compressed = compressor(masked, rate, placeholders)
            else:
                compressed = compressor(masked, rate, placeholders)
        else:
            compressed = compressor(masked, rate, placeholders)
    except Exception as exc:
        _log_error(f"prose compressor raised: {exc!r}")
        compressed = None

    if not compressed or not all(placeholder in compressed for placeholder in placeholders):
        return normalized, "fallback"
    restored = _restore_entities(compressed, mapping)
    if set(extract_entities(restored)) != set(extract_entities(text)):
        return normalized, "fallback-entity-mismatch"
    if len(restored) >= len(normalized):
        return normalized, "fallback-not-smaller"
    result = restored.strip()
    if cache_path is not None:
        # Only successful model output is cached; deterministic fallbacks are
        # cheap and caching them would freeze a transient model failure.
        _write_cached_compression(cache_path, result, "llmlingua")
    return result, "llmlingua"


def _compress_prose_region(region: str, stats: dict[str, int], llm: Any) -> str:
    if not region.strip():
        return region
    lead = region[: len(region) - len(region.lstrip())]
    trail = region[len(region.rstrip()):]
    core = region.strip()

    compact = minify_json_text(core)
    if compact is not None:
        stats["json_blocks"] += 1
        return lead + compact + trail
    if core[:1] in {"{", "["}:
        # Looks like standalone JSON but does not parse: never touch it.
        return region

    pieces: list[str] = []
    for paragraph in PARA_SPLIT_RE.split(core):
        if not paragraph.strip():
            continue
        compressed, mode = compress_prose(paragraph, llm=llm)
        if mode == "llmlingua":
            stats["llm_blocks"] += 1
        elif mode == "error-verbatim":
            stats["error_blocks"] += 1
        elif mode.startswith("fallback"):
            stats["fallback_blocks"] += 1
        pieces.append(compressed)
    return lead + "\n\n".join(pieces) + trail


def compress_text(text: str, llm: Any = "auto") -> tuple[str, dict[str, int]]:
    """Compress a turn: fenced code/JSON first, then prose paragraphs."""
    stats = {
        "code_blocks": 0,
        "json_blocks": 0,
        "code_skeleton_blocks": 0,
        "code_block_ids": [],
        "llm_blocks": 0,
        "fallback_blocks": 0,
        "error_blocks": 0,
    }
    parts: list[str] = []
    position = 0
    for match in FENCE_RE.finditer(text):
        if match.start() > position:
            parts.append(_compress_prose_region(text[position:match.start()], stats, llm))
        parts.append(_compress_fence(match.group(1), match.group(2), stats))
        position = match.end()
    if position < len(text):
        parts.append(_compress_prose_region(text[position:], stats, llm))
    result = "".join(parts)
    stats["bytes_in"] = len(text)
    stats["bytes_out"] = len(result)
    return result, stats


# ---------------------------------------------------------------------------
# Append-only L1 chunks and L2 consolidation
# ---------------------------------------------------------------------------

def _ledger_lines(entities: list[str]) -> list[str]:
    lines = ["**Entity Ledger:**"]
    lines.extend([f"- {entry}" for entry in entities] or ["- (none)"])
    return lines


def _chunk_block(
    number: int,
    entities: list[str],
    text: str,
    source: str,
    raw_id: str | None = None,
) -> str:
    lines = [f"### Summary Chunk #{number}"]
    if source:
        lines.append(f"Source: {source}")
    if raw_id:
        # A pointer, not the text: the id is enough to recover the full turn.
        lines.append(f"Raw: {raw_id[:12]}")
    lines.extend(_ledger_lines(entities))
    lines.append("**Turn:**")
    lines.append(text.strip())
    return "\n".join(lines)


def _merge_entities(*lists: list[str]) -> list[str]:
    return sorted({entry for entries in lists for entry in entries})


def _digest(text: str, limit: int) -> str:
    body = text.strip()
    if len(body) <= limit:
        return body
    head = body[:limit]
    newline = head.rfind("\n")
    if newline > limit // 2:
        head = head[:newline]
    head = head.rstrip()
    if head.count("```") % 2 == 1:
        head += "\n```"
    return head + "\n…"


def _chunk_range_label(numbers: list[int]) -> str:
    """Render chunk numbers as exact ranges.

    Sessions interleave chunk numbers, so a master may cover [1, 3, 5, 7, 8, 9].
    Printing "#1–#9" would falsely imply the other sessions' chunks are included.
    """
    parts: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        parts.append(f"#{start}–#{previous}" if start != previous else f"#{start}")
        start = previous = number
    parts.append(f"#{start}–#{previous}" if start != previous else f"#{start}")
    return ", ".join(parts)


def _chunk_code_ids(chunk: dict[str, Any]) -> list[str]:
    """Raw ids of skeletonized blocks in a chunk.

    New chunks carry them in ``stats["code_block_ids"]``.  Fall back to the
    placeholder text so chunks written before that field still get pointers
    when a new master is built.  Existing masters are never rewritten.
    """
    found: set[str] = set()
    stats = chunk.get("stats")
    if isinstance(stats, dict):
        for value in stats.get("code_block_ids") or []:
            if isinstance(value, str) and _RAW_ID_RE.fullmatch(value):
                found.add(value)
    text = chunk.get("compressed_text")
    if isinstance(text, str):
        found.update(_SKELETON_RAW_ID_RE.findall(text))
    return sorted(found)


def _chunk_pointer_lines(chunk: dict[str, Any]) -> list[str]:
    """Recovery pointers placed before an L2 digest, so a cut cannot hide them."""
    lines: list[str] = []
    raw_id = chunk.get("raw_id")
    if isinstance(raw_id, str) and _RAW_ID_RE.fullmatch(raw_id):
        lines.append(f"Raw: {raw_id[:12]}")
    code_ids = _chunk_code_ids(chunk)
    if code_ids:
        lines.append("Code: " + ", ".join(block_id[:12] for block_id in code_ids))
    return lines


def _render_master_block(
    number: int,
    batch: list[dict[str, Any]],
) -> tuple[str, list[int], list[str]]:
    covered = [int(chunk["n"]) for chunk in batch]
    entities = _merge_entities(*[chunk.get("entities", []) for chunk in batch])
    lines = [
        f"### Consolidated Master Summary (L2) #{number}",
        f"Covers Summary Chunks {_chunk_range_label(covered)}",
    ]
    lines.extend(_ledger_lines(entities))
    lines.append("**Chunk Digests:**")
    for chunk in batch:
        lines.append(f"#### Chunk #{chunk['n']}")
        lines.extend(_chunk_pointer_lines(chunk))
        lines.append(_digest(chunk.get("compressed_text", ""), _l2_digest_chars()))
    return "\n".join(lines), covered, entities


def _build_master(
    state: dict[str, Any],
    batch: list[dict[str, Any]],
    session: str | None = None,
) -> dict[str, Any]:
    number = len(state.get("l2_masters", [])) + 1
    block, covered, entities = _render_master_block(number, batch)
    return {
        "n": number,
        "created": _now(),
        "covered": covered,
        "entities": entities,
        "session": session,
        "block": block,
    }


def consolidate_if_needed(
    state: dict[str, Any],
    session: str | None = None,
) -> list[dict[str, Any]]:
    """Consolidate active L1 chunks into immutable L2 masters at the threshold.

    Consolidation is per session when ``session`` is given: six chunks from one
    context form a master; chunks from another context are never mixed in.
    """
    created: list[dict[str, Any]] = []
    while len(active_l1(state, session)) >= L1_CONSOLIDATE_AT:
        batch = active_l1(state, session)[:L1_CONSOLIDATE_AT]
        master = _build_master(state, batch, session=session)
        state.setdefault("l2_masters", []).append(master)
        created.append(master)
    return created


def append_turn(
    state: dict[str, Any],
    text: str,
    source: str = "user_prompt",
    llm: Any = "auto",
    session: str | None = None,
    role: str | None = None,
    capture_key: str | None = None,
) -> dict[str, Any] | None:
    """Compress one turn and append it as an immutable L1 chunk."""
    if not text or not text.strip():
        return None
    compressed, stats = compress_text(text, llm=llm)
    if not compressed.strip():
        return None

    raw_id = archive_raw(text)
    number = max((int(chunk.get("n", 0)) for chunk in state.get("l1_chunks", [])), default=0) + 1
    entities = _merge_entities(extract_entities(text), extract_entities(compressed))
    key = capture_key or _capture_key(text, strip_role=source == "precompact")
    captured_at = time.time()
    chunk = {
        "n": number,
        "created": _now(),
        "created_at": captured_at,
        "source": source,
        "role": role,
        "session": session,
        "input_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:32],
        "capture_key": key,
        "entities": entities,
        "compressed_text": compressed,
        "block": _chunk_block(number, entities, compressed, source, raw_id),
        "raw_id": raw_id,
        "stats": stats,
    }
    state.setdefault("l1_chunks", []).append(chunk)
    _record_recent_capture(state, text, source, role, session, captured_at)
    _record_goals_for_turn(state, text, session, number, role)
    chunk["handoff"] = _handoff_from_turn(text, number, role)
    if _journal_enabled():
        tags = _harvest_tags(text, number, state, session)
        if tags:
            chunk["tags"] = tags

    totals = state.setdefault("stats", {})
    totals["turns"] = int(totals.get("turns", 0)) + 1
    totals["bytes_in"] = int(totals.get("bytes_in", 0)) + stats["bytes_in"]
    totals["bytes_out"] = int(totals.get("bytes_out", 0)) + stats["bytes_out"]
    for key in ("llm_blocks", "fallback_blocks", "error_blocks", "code_skeleton_blocks"):
        totals[key] = int(totals.get(key, 0)) + stats.get(key, 0)

    # Keep every existing chunk dict byte-identical; the master records which
    # chunks it covers.  (Consolidation never rewrites a past chunk.)
    consolidate_if_needed(state, session)
    return chunk


# ---------------------------------------------------------------------------
# Background compaction queue
# ---------------------------------------------------------------------------

def _pending_for(state: dict[str, Any], session: str | None = None) -> list[dict[str, Any]]:
    return [job for job in state.get("pending", []) if _belongs(job, session) and job.get("text")]


def pending_count(state: dict[str, Any], session: str | None = None) -> int:
    return len(_pending_for(state, session))


def _find_captured_entry(
    state: dict[str, Any],
    input_sha: str | None,
    session: str | None,
    *,
    capture_key: str | None = None,
    sources: tuple[str, ...] | None = None,
    role: str | None = None,
    since: float | None = None,
    max_age: float | None = None,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Find a pending job or committed chunk that already covers this capture."""
    allowed = set(sources or ("precompact",))
    now = time.time() if now is None else now
    for collection in ("pending", "l1_chunks"):
        for item in state.get(collection, []):
            if not isinstance(item, dict) or not _belongs(item, session):
                continue
            if item.get("source") not in allowed:
                continue
            if role is not None and item.get("role") not in (role, None):
                continue
            if since is not None or max_age is not None:
                at = item.get("created_at")
                if not isinstance(at, (int, float)):
                    continue
                at = float(at)
                if at > now + 5:
                    continue
                if since is not None and at < since:
                    continue
                if max_age is not None and now - at > max_age:
                    continue
            if capture_key and item.get("capture_key") == capture_key:
                return item
            if input_sha and item.get("input_sha") == input_sha:
                return item
    return None


def _already_captured(state: dict[str, Any], input_sha: str, session: str | None) -> bool:
    """Legacy exact PreCompact dedupe used by direct callers and the kill switch."""
    return _find_captured_entry(state, input_sha, session, sources=("precompact",)) is not None


def enqueue_turn(
    state: dict[str, Any],
    text: str,
    source: str = "user_prompt",
    session: str | None = None,
    dedupe: bool = False,
    role: str | None = None,
    capture_key: str | None = None,
) -> dict[str, Any] | None:
    """Queue a raw turn for a background worker; returns the (saved) job."""
    if not text or not text.strip():
        return None
    input_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    if dedupe and _already_captured(state, input_sha, session):
        return None
    key = capture_key or _capture_key(text, strip_role=source == "precompact")
    raw_id = archive_raw(text)
    job = {
        "id": uuid.uuid4().hex,
        "session": session,
        "source": source,
        "role": role,
        "created": _now(),
        "created_at": time.time(),
        "status": "pending",
        "input_sha": input_sha,
        "capture_key": key,
        "raw_id": raw_id,
        "entities": extract_entities(text),
        "text": text[:PENDING_MAX_CHARS],
    }
    state.setdefault("pending", []).append(job)
    _record_recent_capture(state, text, source, role, session, float(job["created_at"]))
    _log_job(
        "enqueue",
        job_id=job["id"],
        session=session,
        source=source,
        role=role,
        bytes=len(text),
        capture_key=key[:12],
    )
    return job



def queue_or_append(
    state: dict[str, Any],
    text: str,
    source: str,
    session: str | None,
    background: bool,
    role: str | None = None,
) -> dict[str, Any] | None:
    """Enqueue for background compression, or compress inline as a fallback.

    The inline fallback uses the deterministic pipeline only: when the queue is
    full, blocking a hook on model work would make a busy swarm worse, not
    better.  With background disabled the old synchronous behaviour is kept.
    """
    now = time.time()
    dedupe = source == "precompact"
    if _dedupe_enabled() and role is not None:
        duplicate = _find_duplicate_capture(state, text, source, role, session, now)
    else:
        duplicate = bool(
            dedupe
            and _already_captured(
                state, hashlib.sha256(text.encode("utf-8")).hexdigest()[:32], session
            )
        )
    if source == "user_prompt" and _dedupe_enabled() and role is not None:
        # Advance the floor even when this prompt was absorbed: a deliberate
        # identical retry must be captured as a new turn.
        _record_prompt_seen(state, session, now)
    if duplicate:
        return None

    if background and _env_bool("APPEND_COMPACT_LLM", True):
        limit = _env_int("APPEND_COMPACT_MAX_PENDING", DEFAULT_MAX_PENDING)
        total_limit = _env_int("APPEND_COMPACT_MAX_PENDING_TOTAL", DEFAULT_MAX_PENDING_TOTAL)
        if pending_count(state, session) < limit and pending_count(state) < total_limit:
            return enqueue_turn(state, text, source=source, session=session, role=role)
        _log_job(
            "inline",
            session=session,
            source=source,
            role=role,
            bytes=len(text),
            reason="queue_full",
        )
        return append_turn(state, text, source=source, llm="off", session=session, role=role)
    return append_turn(state, text, source=source, session=session, role=role)


def _requeue_stale(state: dict[str, Any], now: float) -> bool:
    timeout = _env_int("APPEND_COMPACT_CLAIM_TIMEOUT", DEFAULT_CLAIM_TIMEOUT_SECONDS)
    changed = False
    for job in state.get("pending", []):
        if job.get("status") != "working":
            continue
        claimed = job.get("claimed_at")
        if not isinstance(claimed, (int, float)) or now - float(claimed) > timeout:
            job["status"] = "pending"
            job.pop("worker", None)
            job.pop("claimed_at", None)
            changed = True
    return changed


def claim_job(state: dict[str, Any], worker: str, now: float | None = None) -> tuple[dict[str, Any] | None, bool]:
    """Claim the oldest job. Returns (job, state_changed).

    PreCompact captures go first: after a compaction the next injection is
    imminent, whereas prompt turns have until the next compaction.
    """
    now = time.time() if now is None else now
    changed = _requeue_stale(state, now)
    jobs = [job for job in state.get("pending", []) if job.get("status") == "pending" and job.get("text")]
    jobs.sort(key=lambda job: (
        0 if job.get("source") == "precompact" else 1,
        float(job.get("created_at") or 0.0),
        str(job.get("id")),
    ))
    if not jobs:
        return None, changed
    job = jobs[0]
    job["status"] = "working"
    job["worker"] = worker
    job["claimed_at"] = now
    return dict(job), True


def commit_job(
    state: dict[str, Any],
    job_id: str,
    compressed: str,
    stats: dict[str, int],
    worker: str | None = None,
) -> dict[str, Any] | None:
    """Turn a claimed job into an immutable chunk.

    Re-reads happen in the caller under ``state_lock``.  A job that vanished
    (materialised deterministically during a drain) or was claimed by someone
    else is dropped without duplicating the turn.
    """
    entry = next((job for job in state.get("pending", []) if job.get("id") == job_id), None)
    if entry is None:
        return None
    if worker is not None and entry.get("worker") != worker:
        return None
    if not compressed.strip():
        return None
    session = entry.get("session")
    number = max((int(chunk.get("n", 0)) for chunk in state.get("l1_chunks", [])), default=0) + 1
    entities = _merge_entities(extract_entities(entry.get("text", "")), extract_entities(compressed))
    source = entry.get("source", "user_prompt")
    chunk = {
        "n": number,
        "created": _now(),
        "created_at": entry.get("created_at") if isinstance(entry.get("created_at"), (int, float)) else time.time(),
        "source": source,
        "role": entry.get("role"),
        "session": session,
        "input_sha": entry.get("input_sha")
        or hashlib.sha256(str(entry.get("text", "")).encode("utf-8")).hexdigest()[:32],
        "capture_key": entry.get("capture_key")
        or _capture_key(str(entry.get("text", "")), strip_role=source == "precompact"),
        "entities": entities,
        "compressed_text": compressed,
        "block": _chunk_block(number, entities, compressed, source, entry.get("raw_id")),
        "raw_id": entry.get("raw_id"),
        "stats": stats,
    }
    state.setdefault("l1_chunks", []).append(chunk)
    state["pending"].remove(entry)
    _record_goals_for_turn(state, str(entry.get("text", "")), session, number, entry.get("role"))
    raw_text = str(entry.get("text", ""))
    chunk["handoff"] = _handoff_from_turn(raw_text, number, entry.get("role"))
    if _journal_enabled():
        tags = _harvest_tags(raw_text, number, state, session)
        if tags:
            chunk["tags"] = tags

    totals = state.setdefault("stats", {})
    totals["turns"] = int(totals.get("turns", 0)) + 1
    totals["bytes_in"] = int(totals.get("bytes_in", 0)) + int(stats.get("bytes_in", 0))
    totals["bytes_out"] = int(totals.get("bytes_out", 0)) + int(stats.get("bytes_out", 0))
    for key in ("llm_blocks", "fallback_blocks", "error_blocks", "code_skeleton_blocks"):
        totals[key] = int(totals.get(key, 0)) + int(stats.get(key, 0))
    consolidate_if_needed(state, session)
    return chunk


def materialize_pending(state: dict[str, Any], session: str | None = None) -> int:
    """Deterministically commit queued turns (no model work, fast).

    Used only when an injection deadline leaves work unfinished: a complete but
    deterministic snapshot beats a missing recent turn.
    """
    jobs = [job for job in state.get("pending", []) if job.get("text") and _belongs(job, session)]
    committed = 0
    for job in jobs:
        compressed, stats = compress_text(job["text"], llm="off")
        if commit_job(state, str(job.get("id")), compressed, stats, worker=None) is not None:
            committed += 1
    return committed


# ---------------------------------------------------------------------------
# Detached worker pool
# ---------------------------------------------------------------------------

def _workers_dir() -> Path:
    return default_cache_dir() / "workers"


def _worker_idle_seconds() -> int:
    return _env_int("APPEND_COMPACT_WORKER_IDLE", DEFAULT_WORKER_IDLE_SECONDS)


def _acquire_worker_slot():
    """Take one of the machine-wide worker slots, or return None."""
    if fcntl is None:
        return None
    slots = _env_int("APPEND_COMPACT_WORKERS", DEFAULT_WORKERS)
    root = _workers_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        for index in range(slots):
            handle = (root / f"slot-{index}.lock").open("a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                handle.close()
                continue
            return handle
    except OSError as exc:
        _log_error(f"worker slots unavailable: {exc!r}")
    return None


def spawn_worker(state_path: Path) -> bool:
    """Start one detached worker if a spawn is not already in flight."""
    if fcntl is None or os.environ.get("APPEND_COMPACT_WORKER"):
        return False
    if not _env_bool("APPEND_COMPACT_BACKGROUND", True):
        return False
    root = _workers_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        gate = (root / "spawn.lock").open("a+")
        try:
            try:
                fcntl.flock(gate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                return False
            log = (root / "worker.log").open("a", encoding="utf-8")
            try:
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", "--state", str(state_path)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=log,
                    close_fds=True,
                    start_new_session=(os.name == "posix"),
                )
            finally:
                log.close()
            return True
        finally:
            gate.close()  # closing releases the flock
    except OSError as exc:
        _log_error(f"cannot spawn compaction worker: {exc!r}")
        return False


def _heartbeat(worker: str, jobs: int, started: float, status: str) -> None:
    path = _workers_dir() / f"{os.getpid()}.json"
    try:
        _atomic_write_json(path, {
            "worker": worker,
            "pid": os.getpid(),
            "started": started,
            "jobs": jobs,
            "status": status,
            "at": _now(),
        })
    except OSError:
        pass


def drain_session(state_path: Path, session: str, wait_seconds: int | None = None) -> int:
    """Give workers a bounded window, then materialise what is left.

    Called only right before an armed injection, so the extra latency is paid
    at compaction boundaries rather than on every prompt.  Returns the number
    of turns that had to be committed deterministically.
    """
    wait = _env_int("APPEND_COMPACT_DRAIN_WAIT", DEFAULT_DRAIN_WAIT_SECONDS) if wait_seconds is None else wait_seconds
    spawn_worker(state_path)
    deadline = time.monotonic() + max(0, wait)
    while True:
        with state_lock(state_path):
            state = load_state(state_path)
            remaining = pending_count(state, session)
        if remaining == 0 or time.monotonic() >= deadline:
            break
        time.sleep(0.1)
    if remaining == 0:
        return 0
    with state_lock(state_path):
        state = load_state(state_path)
        remaining = pending_count(state, session)
        if remaining:
            materialize_pending(state, session)
        save_state(state, state_path)
    return remaining


def worker_main(state_path: Path) -> int:
    """Compress queued turns until the queue is empty or the worker ages out."""
    os.environ["APPEND_COMPACT_WORKER"] = "1"
    slot = _acquire_worker_slot()
    if slot is None:
        return 0
    try:
        with contextlib.suppress(OSError, AttributeError):
            os.nice(_env_int("APPEND_COMPACT_WORKER_NICE", 10))
        # Torch reads these at import time; dividing the cores keeps two
        # parallel workers from each spawning a full-size OpenMP pool.
        slots = _env_int("APPEND_COMPACT_WORKERS", DEFAULT_WORKERS)
        threads = max(1, (os.cpu_count() or slots) // max(1, slots))
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        worker = f"{os.getpid()}"
        started = time.time()
        deadline = started + _env_int("APPEND_COMPACT_WORKER_MAX_SECONDS", DEFAULT_WORKER_MAX_SECONDS)
        last_job = time.time()
        jobs_done = 0
        _heartbeat(worker, jobs_done, started, "idle")
        while time.time() < deadline:
            job = None
            with state_lock(state_path):
                state = load_state(state_path)
                job, changed = claim_job(state, worker)
                if changed:
                    save_state(state, state_path)
            if job is None:
                if time.time() - last_job >= _worker_idle_seconds():
                    break
                time.sleep(0.5)
                continue
            _heartbeat(worker, jobs_done, started, "working")
            job_id = str(job.get("id"))
            job_source = job.get("source")
            job_session = job.get("session")
            job_started = time.time()
            try:
                compressed, stats = compress_text(job["text"], llm="auto")
                chunk = None
                with state_lock(state_path):
                    state = load_state(state_path)
                    chunk = commit_job(state, job_id, compressed, stats, worker=worker)
                    save_state(state, state_path)
                duration = round(time.time() - job_started, 3)
                if chunk is None:
                    _log_job(
                        "dropped",
                        job_id=job_id,
                        worker=worker,
                        source=job_source,
                        session=job_session,
                        duration=duration,
                    )
                else:
                    jobs_done += 1
                    _log_job(
                        "commit",
                        job_id=job_id,
                        worker=worker,
                        source=job_source,
                        session=job_session,
                        duration=duration,
                        stats=stats,
                    )
            except Exception as exc:
                _log_error(f"worker job {job_id} failed: {exc!r}")
                _log_job(
                    "error",
                    job_id=job_id,
                    worker=worker,
                    source=job_source,
                    session=job_session,
                    duration=round(time.time() - job_started, 3),
                    error=type(exc).__name__,
                )
            finally:
                last_job = time.time()
                _heartbeat(worker, jobs_done, started, "idle")
        return 0
    finally:
        with contextlib.suppress(OSError):
            (_workers_dir() / f"{os.getpid()}.json").unlink()
        slot.close()


# ---------------------------------------------------------------------------
# Context injection
# ---------------------------------------------------------------------------

def _openrouter_extra_headers() -> dict[str, str]:
    """Parse OPENROUTER_EXTRA_HEADERS (JSON object or ``Key: Value`` list).

    Claude Code does not read this variable itself.  It is emitted in the
    injection payload so a gateway/proxy consuming the payload can forward
    attribution headers alongside the ephemeral ``cache_control`` markers.
    """
    raw = os.environ.get("OPENROUTER_EXTRA_HEADERS", "").strip()
    if not raw:
        return {}
    parsed: Any = None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return {str(key): str(value) for key, value in parsed.items()}
    headers: dict[str, str] = {}
    for part in raw.split(","):
        if ":" in part:
            key, value = part.split(":", 1)
            if key.strip():
                headers[key.strip()] = value.strip()
    return headers


def _truncate_at_line(text: str, budget: int) -> str:
    """Trim to at most ``budget`` characters without cutting a line in half."""
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    head = text[:budget]
    newline = head.rfind("\n")
    if newline > budget // 2:
        head = head[:newline]
    else:
        space = head.rfind(" ")
        if space > budget // 2:
            head = head[:space]
    return head.rstrip()


def _clamp_inject_chars(max_chars: int) -> int:
    """Never request more than Claude Code can deliver in one hook string."""
    ceiling = HOOK_OUTPUT_CAP - 200
    if max_chars <= 0:
        return ceiling
    return min(max_chars, ceiling)


def _global_entities(state: dict[str, Any], session: str | None = None) -> list[str]:
    return _merge_entities(
        *[chunk.get("entities", []) for chunk in state.get("l1_chunks", []) if _belongs(chunk, session)],
        *[master.get("entities", []) for master in state.get("l2_masters", []) if _belongs(master, session)],
    )


def _global_ledger_block(state: dict[str, Any], limit: int, session: str | None = None) -> str:
    """Deterministic union of every pinned entity, kept at the stable head.

    Masters get dropped from the injection when the cap is tight, so this
    ledger is what keeps old file paths and symbol names alive.  Entries are
    grouped by kind (``fact:``, ``path:``, ``symbol:``) so a long session's
    facts fit in the same budget as one-entry-per-line would allow.
    """
    entities = _global_entities(state, session)
    if not entities or limit <= 0:
        return ""
    header = "**Entity Ledger (pinned):**"
    groups: dict[str, list[str]] = {}
    for entry in entities:
        kind, sep, value = entry.partition(": ")
        if not sep:
            kind, value = "misc", entry
        groups.setdefault(kind, []).append(value)
    lines = [header]
    used = len(header) + 1
    truncated = False
    for kind in sorted(groups):
        current = f"- {kind}: "
        emitted = 0
        for value in groups[kind]:
            piece = value if emitted == 0 else f"; {value}"
            if used + len(current) + len(piece) + 1 > limit:
                truncated = True
                break
            current += piece
            emitted += 1
        if emitted:
            lines.append(current)
            used += len(current) + 1
        if truncated:
            lines.append(f"- … {len(entities)} entities pinned, list truncated")
            break
    return "\n".join(lines)


def _pack_recent(blocks: list[str], budget: int) -> tuple[str, int]:
    """Keep the newest whole blocks that fit, dropping oldest first.

    Returns ``(text, dropped_count)``.  When the newest block alone is over
    budget, keep its tail rather than its head: the tail holds the most recent
    lines of that turn.
    """
    if budget <= 0 or not blocks:
        return "", len(blocks)
    marker = "[… earlier part of this block truncated …]\n"
    kept: list[str] = []
    total = 0
    dropped = 0
    for block in reversed(blocks):
        cost = len(block) + (2 if kept else 0)
        if total + cost <= budget:
            kept.append(block)
            total += cost
            continue
        if not kept:
            room = max(0, budget - len(marker))
            tail = block[-room:] if room else ""
            if room:
                newline = tail.find("\n")
                if 0 <= newline < len(tail) // 2:
                    tail = tail[newline + 1:]
            kept.append((marker + tail)[:budget])
            dropped = len(blocks) - 1
        else:
            dropped = len(blocks) - len(kept)
        break
    kept.reverse()
    return "\n\n".join(kept), dropped


def _l3_index_header() -> str:
    return (
        "**L3 Index (older context; zstd --ultra -22 files under "
        f"{_l3_base_display()}/; scan: ugrep -z PATTERN FILE "
        "(zstdgrep also works), read: zstd -dc FILE):**"
    )

# Tokens that carry topic in a prompt but not across unrelated prompts.  Paths
# and symbols are deliberately kept: two turns about the same module should
# look similar even when their prose differs.
_DRIFT_TOKEN_RE = re.compile(r"[\w.:/-]{3,}", re.UNICODE)
_DRIFT_STOPWORDS = frozenset(
    """
    the and for with from into over under this that these those then than
    there their they them your you our ours was were been being have has had
    will would should could can may might must not yes now also just only
    more most other some such very about after before again further once
    here when where why how all any both each few own same too use used using
    make made get got add added run ran fix fixed check checked do does did
    done new old first last next previous current file files code test tests
    function functions line lines text data context prompt prompts token
    tokens compact compactor compaction plugin default enabled enable disable
    please want need needs like work works working implementation implement
    implemented change changes changed update updated updating
    """.split()
)


def _drift_signature(text: str) -> set[str]:
    tokens = {match.lower() for match in _DRIFT_TOKEN_RE.findall(text)}
    return {token for token in tokens if token not in _DRIFT_STOPWORDS}


def _drift_decision(
    state: dict[str, Any],
    session: str | None,
    prompt: str | None,
) -> tuple[bool, float, int]:
    """Whether the current prompt looks unrelated to recent user prompts.

    Returns ``(active, best_overlap, compared)``.  Short prompts are treated as
    follow-ups, not drift; fewer than two substantive prior prompts cannot
    establish a topic.  This is a local, deterministic statistic: no model and
    no extra state.
    """
    if not _env_bool("APPEND_COMPACT_L3_DRIFT", True) or not prompt:
        return False, 1.0, 0
    window = _env_int("APPEND_COMPACT_L3_DRIFT_WINDOW", DEFAULT_L3_DRIFT_WINDOW)
    min_tokens = _env_int(
        "APPEND_COMPACT_L3_DRIFT_MIN_TOKENS", DEFAULT_L3_DRIFT_MIN_TOKENS
    )
    threshold = _env_float(
        "APPEND_COMPACT_L3_DRIFT_THRESHOLD", DEFAULT_L3_DRIFT_THRESHOLD
    )
    current = _drift_signature(prompt)
    if len(current) < min_tokens:
        return False, 1.0, 0

    prior: list[set[str]] = []
    for chunk in state.get("l1_chunks", []):
        if not _belongs(chunk, session):
            continue
        if chunk.get("role") != "user" and chunk.get("source") != "user_prompt":
            continue
        signature = _drift_signature(str(chunk.get("compressed_text", "")))
        if len(signature) >= min_tokens:
            prior.append(signature)
    prior = prior[-window:]
    if len(prior) < 2:
        return False, 1.0, len(prior)

    best = 0.0
    for signature in prior:
        overlap = len(current & signature) / min(len(current), len(signature))
        best = max(best, overlap)
    return best < threshold, best, len(prior)


def _l3_keep_recent(drift: bool) -> tuple[int, int]:
    if drift:
        masters = _env_int(
            "APPEND_COMPACT_L3_DRIFT_KEEP_RECENT_MASTERS",
            DEFAULT_L3_DRIFT_KEEP_RECENT_MASTERS,
        )
        chunks = _env_int(
            "APPEND_COMPACT_L3_DRIFT_KEEP_RECENT_CHUNKS",
            DEFAULT_L3_DRIFT_KEEP_RECENT_CHUNKS,
        )
    else:
        masters = _env_int(
            "APPEND_COMPACT_L3_KEEP_RECENT_MASTERS",
            DEFAULT_L3_KEEP_RECENT_MASTERS,
        )
        chunks = _env_int(
            "APPEND_COMPACT_L3_KEEP_RECENT_CHUNKS",
            DEFAULT_L3_KEEP_RECENT_CHUNKS,
        )
    return masters, chunks


def _l3_clean_text(text: str) -> str:
    """Prose for summaries: drop headings, pointers and ledger lines."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("**Turn:**"):
            continue
        if line.startswith(("Source:", "Raw:", "Code:", "Files:", "Covers ")):
            continue
        if line.startswith(("- path:", "- symbol:", "- fact:")):
            continue
        lines.append(line)
    return " ".join(" ".join(lines).split())


def _l3_summary(text: str, limit: int) -> str:
    cleaned = _l3_clean_text(text)
    if not cleaned:
        return ""
    summary = _digest(cleaned, limit).replace("\n", " ")
    return summary.rstrip()


def _l3_signal(entities: list[Any], limit: int = DEFAULT_L3_ENTITY_CHARS) -> str:
    """Compact high-signal entities for an index line (paths, symbols, facts)."""
    paths: list[str] = []
    symbols: list[str] = []
    facts: list[str] = []
    for entry in entities:
        if not isinstance(entry, str):
            continue
        if entry.startswith("path: "):
            paths.append(entry[6:].strip())
        elif entry.startswith("symbol: "):
            symbols.append(entry[8:].strip())
        elif entry.startswith("fact: "):
            facts.append(entry[6:].strip())
    chosen: list[str] = []
    for value in paths[:2] + symbols[:2] + facts[:2]:
        value = " ".join(value.split())
        if len(value) > 36:
            value = value[:35].rstrip() + "…"
        if value and value not in chosen:
            chosen.append(value)
    text = ", ".join(chosen)
    if len(text) > limit:
        text = text[: limit - 1].rstrip(", ") + "…"
    return text


_L3_KEYWORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{2,}", re.UNICODE)
_L3_KEYWORD_STOPWORDS = _DRIFT_STOPWORDS | frozenset(
    """
    none true false self cls return import from def class method var let const
    if else elif while try except finally with as in is not and or pass break
    continue yield lambda async await print len str int float bool list dict
    set tuple object value values key keys item items index result results
    name names path paths file files line lines start end get add
    read write open close init main args kwargs payload response request
    status output input text chunk chunks master masters block blocks
    content message messages user assistant tool module modules
    function functions test tests error errors code
    """.split()
)


def _l3_keyword_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _L3_KEYWORD_RE.findall(_l3_clean_text(text)):
        lowered = token.lower()
        if lowered in _L3_KEYWORD_STOPWORDS:
            continue
        if "." in lowered and lowered.split(".", 1)[0] in _L3_KEYWORD_STOPWORDS:
            continue
        tokens.append(lowered)
    return tokens


def _l3_keyword_corpus(sources: list[str]) -> tuple[dict[str, int], int]:
    """Document frequency over the session's candidate blocks."""
    df: dict[str, int] = {}
    for text in sources:
        for token in set(_l3_keyword_tokens(text)):
            df[token] = df.get(token, 0) + 1
    return df, len(sources)


def _l3_keywords(
    text: str,
    df: dict[str, int],
    n_docs: int,
    count: int = DEFAULT_L3_KEYWORDS,
    exclude: set[str] | None = None,
) -> str:
    """Top TF-IDF tokens: what the rest of this block is about.

    Entities alone describe *where* a block points; keywords describe *what*
    it says, including content that never appears in a path or symbol.
    """
    tf: dict[str, int] = {}
    for token in _l3_keyword_tokens(text):
        tf[token] = tf.get(token, 0) + 1
    if not tf:
        return ""
    blocked = exclude or set()

    def weight(token: str) -> float:
        return tf[token] * math.log((n_docs + 1) / (df.get(token, 1) + 1.0))

    ranked = sorted(tf, key=lambda token: (-weight(token), -len(token), token))
    chosen: list[str] = []
    for token in ranked:
        if token in blocked:
            continue
        if any(token in kept or kept in token for kept in chosen):
            continue
        chosen.append(token)
        if len(chosen) >= max(0, count):
            break
    return ", ".join(chosen)


_L3_NUMBER_RE = re.compile(_NUMERIC_LITERAL_PATTERN)


def _l3_numbers(text: str, limit: int = 2) -> list[str]:
    """Exact numeric literals (versions, decimals, units) for an index line.

    The fact ledger deliberately skips bare numbers, so without this an old
    block's distinctive value (``0.25``, ``250ms``) disappears when L3 demotes
    the block to one line.  Only literals ``_NUMERIC_LITERAL_PATTERN`` accepts
    are used -- no bare integers from line counts or skeleton markers.
    """
    found: list[str] = []
    for match in _L3_NUMBER_RE.finditer(text):
        value = match.group(0)
        if value and value not in found:
            found.append(value)
            if len(found) >= max(0, limit):
                break
    return found


def _l3_master_source(master: dict[str, Any]) -> str:
    block = str(master.get("block", ""))
    return block.split("**Chunk Digests:**", 1)[-1] or block


def _l3_chunk_source(chunk: dict[str, Any]) -> str:
    return str(chunk.get("compressed_text", "")) or str(chunk.get("block", ""))


def _l3_item(
    kind: str,
    number: int,
    label: str,
    block: str,
    summary_source: str,
    entities: list[Any] | None = None,
    df: dict[str, int] | None = None,
    n_docs: int = 0,
) -> dict[str, Any]:
    block_id = l3_id_for(block)
    entity_signal = _l3_signal(entities or [])
    excluded = set(_l3_keyword_tokens(entity_signal))
    numeric = [value for value in _l3_numbers(summary_source) if value not in excluded]
    excluded.update(_l3_keyword_tokens(" ".join(numeric)))
    keyword_signal = _l3_keywords(
        summary_source, df or {}, n_docs, DEFAULT_L3_KEYWORDS, excluded
    )
    number_signal = ", ".join(numeric)
    signals = [part for part in (entity_signal, number_signal, keyword_signal) if part]
    summary = " | ".join(signals) or _l3_summary(
        summary_source, DEFAULT_L3_SUMMARY_CHARS
    )
    suffix = f" {summary}" if summary else ""
    line = (
        f"- {kind} #{number} [{label}]{suffix} | "
        f"{_l3_file_link(block_id)}"
    )
    return {
        "kind": kind,
        "n": number,
        "block": block,
        "id": block_id,
        "line": line,
    }


def _l3_master_item(
    master: dict[str, Any],
    df: dict[str, int] | None = None,
    n_docs: int = 0,
) -> dict[str, Any]:
    block = str(master.get("block", ""))
    covered = [int(n) for n in master.get("covered") or [] if isinstance(n, int)]
    label = f"chunks {_chunk_range_label(covered)}"
    return _l3_item(
        "L2",
        int(master.get("n", 0)),
        label,
        block,
        _l3_master_source(master),
        master.get("entities") or [],
        df,
        n_docs,
    )


def _l3_chunk_item(
    chunk: dict[str, Any],
    df: dict[str, int] | None = None,
    n_docs: int = 0,
) -> dict[str, Any]:
    block = str(chunk.get("block", ""))
    label = str(chunk.get("source") or "turn")
    return _l3_item(
        "L1",
        int(chunk.get("n", 0)),
        label,
        block,
        _l3_chunk_source(chunk),
        chunk.get("entities") or [],
        df,
        n_docs,
    )


def _pack_full_candidates(
    candidates: list[dict[str, Any]],
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Keep newest full candidates that fit; return the rest newest-first."""
    if budget <= 0 or not candidates:
        return [], list(reversed(candidates)), 0
    kept: list[dict[str, Any]] = []
    total = 0
    for position in range(len(candidates) - 1, -1, -1):
        item = candidates[position]
        cost = len(str(item.get("block", ""))) + (2 if kept else 0)
        if total + cost > budget:
            return kept[::-1], list(reversed(candidates[: position + 1])), total
        kept.append(item)
        total += cost
    return kept[::-1], [], total


def _pack_l3_index(
    entries: list[dict[str, Any]],
    budget: int,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Write selected L3 files and render index lines that fit ``budget``."""
    if budget <= 0 or not entries:
        return "", [], entries
    header = _l3_index_header()
    header_cost = len(header) + 1
    if header_cost >= budget:
        return "", [], entries
    used = header_cost
    lines = [header]
    written: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for entry in entries:
        cost = len(entry["line"]) + 1
        if used + cost > budget:
            dropped.append(entry)
            continue
        if archive_l3(entry["block"], entry["id"]) is None:
            dropped.append(entry)
            continue
        lines.append(entry["line"])
        used += cost
        written.append(entry)
    return ("\n".join(lines) if written else ""), written, dropped


def injection_payload(
    state: dict[str, Any],
    max_chars: int | None = None,
    event_name: str = "UserPromptSubmit",
    session: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any] | None:
    """Bounded payload: pinned ledger head, newest L2 masters, newest active L1.

    ``session`` isolates parallel Claude contexts: a context receives only the
    chunks and masters captured under its own session id, never a peer's.

    With L3 enabled (the default), older masters/chunks become one-line index
    entries with a verified file link instead of full blocks.  The index always
    follows the unchanged ledger and newest full masters, so this is a change
    to the *new* injection only; existing conversation messages are never
    rewritten.
    """
    max_chars = _clamp_inject_chars(
        max_chars or _env_int("APPEND_COMPACT_MAX_INJECT_CHARS", DEFAULT_MAX_INJECT_CHARS)
    )
    masters = [
        master for master in state.get("l2_masters", [])
        if master.get("block") and _belongs(master, session)
    ]
    active_chunks = [
        chunk for chunk in active_l1(state, session) if chunk.get("block")
    ]
    if not masters and not active_chunks:
        return None

    note = "\n\n[append-compactor: older context omitted to fit the 10k hook cap]"
    l3_enabled = _l3_enabled()
    if not l3_enabled:
        target = max_chars
        drift, drift_score, drift_compared = False, 1.0, 0
        keep_masters, keep_chunks = 0, 0
        recent_masters, old_masters = [], list(masters)
        recent_chunks, old_chunks = [], list(active_chunks)
        l3_present = False
    else:
        target = min(
            max_chars,
            _env_int("APPEND_COMPACT_L3_TARGET_CHARS", DEFAULT_L3_TARGET_CHARS),
        )
        target = max(1, target)
        drift, drift_score, drift_compared = _drift_decision(state, session, prompt)
        keep_masters, keep_chunks = _l3_keep_recent(drift)
        recent_masters = masters[-keep_masters:] if keep_masters else []
        old_masters = masters[:-keep_masters] if keep_masters else list(masters)
        recent_chunks = active_chunks[-keep_chunks:] if keep_chunks else []
        old_chunks = active_chunks[:-keep_chunks] if keep_chunks else list(active_chunks)
        l3_present = bool(old_masters or old_chunks)

    present = {
        "L1": bool(active_chunks),
        "L2": bool(masters),
        "L3": l3_present,
    }
    goal_sections = _render_goal_anchors(state, session, present, drift=drift)
    anchor_reserve = sum(len(text) for _, text in goal_sections)
    if goal_sections:
        anchor_reserve += 2 * max(0, len(goal_sections) - 1)
    # Pack the stable head (ledger + L2 masters) first.  For tail/after-head
    # placements the anchor lives after the head, so the no-anchor budget keeps
    # the cached head byte-identical.  For beginning/hybrid the anchor *is* part
    # of the head, so reserve its bytes before packing or it would be dropped
    # under a tight budget instead of the volatile tail.
    placement = _goal_placement()
    base_content_cap = max(1, target - len(note))
    head_reserve = 0
    if placement in {"beginning", "hybrid"} and goal_sections:
        # Only the leading anchor lives in the head; a hybrid reminder is
        # volatile and is charged to content_cap below.
        head_reserve = len(goal_sections[0][1]) + 2
    head_cap = max(1, base_content_cap - head_reserve)
    content_cap = max(1, base_content_cap - anchor_reserve)
    ledger = _global_ledger_block(
        state, min(LEDGER_BUDGET_CHARS, head_cap // 4), session
    )
    ledger_len = len(ledger) + 2 if ledger else 0
    # The handoff is additive: it never replaces the ledger, the masters or the
    # active chunks, and it lives in the volatile half of the block so the
    # cached head stays byte-identical from turn to turn.  It is rendered
    # *after* the rest is packed -- see below -- because its whole value is
    # carrying facts the packed view no longer shows.
    handoff_text = ""
    journal_note = ""
    handoff_meta: dict[str, Any] = {
        "enabled": _handoff_enabled(), "sections": 0, "items": 0, "chars": 0,
    }
    # Reserve room up front, but render only after packing: packing always
    # fills the budget, so a section that waits for leftovers would never fit.
    handoff_reserve = 0
    if _handoff_enabled() and target >= 1_200:
        handoff_reserve = min(_handoff_cap(), max(200, target // 8))
        content_cap = max(1, content_cap - handoff_reserve)

    if not l3_enabled:
        master_blocks = [str(master["block"]) for master in masters]
        active_blocks = [str(chunk["block"]) for chunk in active_chunks]
        budget = head_cap - ledger_len - len(note)

        # Live turns outrank historical masters.  Pack once at the full budget
        # to freeze the head, then repack only the volatile active text when an
        # anchor needs room.
        active_budget = (
            budget if not master_blocks else min(budget, max(1_500, budget // 2))
        )
        active_text_full, dropped_chunks_full = _pack_recent(
            active_blocks, active_budget
        )
        remaining = budget - (len(active_text_full) + 2 if active_text_full else 0)
        master_text, dropped_masters = _pack_recent(master_blocks, remaining)
        if anchor_reserve:
            volatile_budget = max(
                0,
                content_cap - ledger_len
                - (len(master_text) + 2 if master_text else 0),
            )
            active_text, dropped_chunks = _pack_recent(active_blocks, volatile_budget)
        else:
            active_text, dropped_chunks = active_text_full, dropped_chunks_full
        index_text = ""
        l3_meta: dict[str, Any] = {"enabled": False}
    else:
        # One IDF corpus per session: keywords describe what each block says,
        # while entities describe where it points.  Both are needed for an
        # agent to judge whether a full block is worth reading.
        corpus_sources = (
            [_l3_master_source(master) for master in masters]
            + [_l3_chunk_source(chunk) for chunk in active_chunks]
        )
        df, n_docs = _l3_keyword_corpus(corpus_sources)

        master_items = [
            _l3_master_item(master, df, n_docs) for master in recent_masters
        ]
        chunk_items = [
            _l3_chunk_item(chunk, df, n_docs) for chunk in recent_chunks
        ]

        # Reserve index space for all old material first.  A full block is a
        # convenience; an index line is what keeps the block discoverable.
        old_chunk_entries = [
            _l3_chunk_item(chunk, df, n_docs) for chunk in reversed(old_chunks)
        ]
        old_master_entries = [
            _l3_master_item(master, df, n_docs) for master in reversed(old_masters)
        ]
        old_entries = old_chunk_entries + old_master_entries

        index_cap = _env_int(
            "APPEND_COMPACT_L3_INDEX_BUDGET_CHARS", DEFAULT_L3_INDEX_BUDGET_CHARS
        )
        remaining = max(0, head_cap - ledger_len - len(note))
        header_cost = len(_l3_index_header()) + 1
        old_cost = header_cost + sum(len(entry["line"]) + 1 for entry in old_entries)
        index_reserve = min(index_cap, old_cost) if old_entries else 0
        full_budget = max(0, remaining - index_reserve)
        if master_items and chunk_items:
            active_budget = min(full_budget, max(1_500, full_budget // 2))
            master_budget = max(0, full_budget - active_budget)
        elif master_items:
            master_budget, active_budget = full_budget, 0
        else:
            master_budget, active_budget = 0, full_budget

        kept_masters, demoted_masters, used_masters = _pack_full_candidates(
            master_items, master_budget
        )
        kept_chunks, demoted_chunks, used_chunks = _pack_full_candidates(
            chunk_items, active_budget
        )

        if anchor_reserve:
            # Keep the masters (head) exactly as packed without an anchor, but
            # let the anchor consume the volatile tail.  Repacking can demote
            # more active chunks; those are indexed below.
            actual_chunk_budget = max(
                0, content_cap - ledger_len - used_masters - 6
            )
            kept_chunks, demoted_chunks, used_chunks = _pack_full_candidates(
                chunk_items, actual_chunk_budget
            )

        # Newest old material first: active chunks outrank covered masters.
        index_entries: list[dict[str, Any]] = []
        index_entries.extend(demoted_chunks)
        index_entries.extend(old_chunk_entries)
        index_entries.extend(demoted_masters)
        index_entries.extend(old_master_entries)

        index_available = max(
            0, content_cap - ledger_len - used_masters - used_chunks - 6
        )
        index_text, index_written, index_dropped = _pack_l3_index(
            index_entries, min(index_cap, index_available)
        )

        master_text = "\n\n".join(str(item["block"]) for item in kept_masters)
        active_text = "\n\n".join(str(item["block"]) for item in kept_chunks)
        dropped_masters = sum(
            1 for entry in index_dropped if entry.get("kind") == "L2"
        )
        dropped_chunks = sum(
            1 for entry in index_dropped if entry.get("kind") == "L1"
        )
        l3_meta = {
            "enabled": True,
            "target_chars": target,
            "drift": drift,
            "drift_score": round(drift_score, 3),
            "drift_prompts": drift_compared,
            "full_masters": len(kept_masters),
            "index_masters": sum(
                1 for entry in index_written if entry.get("kind") == "L2"
            ),
            "full_chunks": len(kept_chunks),
            "index_chunks": sum(
                1 for entry in index_written if entry.get("kind") == "L1"
            ),
            "total_masters": len(masters),
            "total_chunks": len(active_chunks),
            "represented_masters": len(kept_masters)
            + sum(1 for entry in index_written if entry.get("kind") == "L2"),
            "represented_chunks": len(kept_chunks)
            + sum(1 for entry in index_written if entry.get("kind") == "L1"),
            "dropped_masters": dropped_masters,
            "dropped_chunks": dropped_chunks,
            "omitted_masters": dropped_masters,
            "omitted_chunks": dropped_chunks,
        }

    cap = target if l3_enabled else max_chars

    def assemble(sections: list[tuple[str, str]]) -> tuple[str, str, str]:
        """Return (text, head_text, volatile_text) for the current sections.

        ``head_text`` is the byte-stable prefix: the part that stays identical
        across turns unless the goal is revised (beginning placements) or a new
        master/ledger entry arrives.  Only ``volatile_text`` should change turn
        to turn, so it belongs after the anchor in the cached prefix.
        """
        placement = _goal_placement()
        main = sections[0][1] if sections else ""
        reminder = ""
        if len(sections) > 1 and sections[1][0] == "reminder":
            reminder = sections[1][1]
        extra_note = note if (dropped_chunks or dropped_masters) else ""
        if placement == "per_level":
            per_level = {level: text for level, text in sections}
            head_parts = [ledger, master_text]
            volatile_parts = [
                handoff_text, per_level.get("L2", ""), index_text,
                per_level.get("L3", ""), active_text, per_level.get("L1", ""),
                extra_note,
            ]
        elif placement == "beginning":
            head_parts = [main, ledger, master_text]
            volatile_parts = [handoff_text, index_text, active_text, extra_note]
        elif placement == "after_head":
            head_parts = [ledger, master_text]
            volatile_parts = [main, handoff_text, index_text, active_text, extra_note]
        elif placement == "hybrid":
            head_parts = [main, ledger, master_text]
            volatile_parts = [handoff_text, index_text, active_text, reminder, extra_note]
        else:  # tail
            head_parts = [ledger, master_text]
            # The anchor stays last: tail placement exists so the instruction is
            # the last thing read.
            volatile_parts = [handoff_text, index_text, active_text, main, extra_note]
        if journal_note and journal_note not in volatile_parts:
            volatile_parts = volatile_parts + [journal_note]
        head = "\n\n".join(part for part in head_parts if part)
        volatile = "\n\n".join(part for part in volatile_parts if part)
        assembled = "\n\n".join(
            part for part in head_parts + volatile_parts if part
        ).strip()
        return assembled, head, volatile

    text, head_text, volatile_text = assemble(goal_sections)
    # Second pass: now that the packed view is known, keep only handoff facts
    # it lost.  Anything still visible in the ledger, masters, index lines or
    # live chunks is dropped, so the section cannot be a second copy of text
    # the model is already reading.  It gets whatever budget is left, and
    # nothing if that is too little to say something useful.
    if _handoff_enabled() and target >= 1_200:
        spare = min(handoff_reserve or _handoff_cap(), cap - len(text) - 4)
        if spare >= 120:
            handoff_text, handoff_meta = _render_handoff(
                state, session, ledger, spare, rest_text=text
            )
            if handoff_text:
                text, head_text, volatile_text = assemble(goal_sections)
    # The journal prompt must be present from the first turn, when there is
    # nothing to show yet -- otherwise the model is never asked to emit tags
    # and the section can never fill.
    if _journal_enabled() and JOURNAL_INSTRUCTION not in text:
        if len(text) + len(JOURNAL_INSTRUCTION) + 2 <= cap:
            journal_note = JOURNAL_INSTRUCTION
            text, head_text, volatile_text = assemble(goal_sections)
    if len(text) > cap and handoff_text:
        # The handoff yields first: it is the newest section and the least
        # established, while the anchor and the ledger are what the session has
        # already been relying on.
        overflow = len(text) - cap
        handoff_text, handoff_meta = _render_handoff(
            state, session, ledger, max(0, len(handoff_text) - overflow - 20)
        )
        text, head_text, volatile_text = assemble(goal_sections)
    if len(text) > cap:
        # Shrink the anchor next, never the other way around: the anchor is
        # the one thing this feature exists to keep.
        overflow = len(text) - cap
        reduced = _render_goal_anchors(
            state, session, present,
            max_total=max(0, anchor_reserve - overflow - 20),
            drift=drift,
        )
        text, head_text, volatile_text = assemble(reduced)
        goal_sections = reduced
    if len(text) > cap:
        # Pathological cap (e.g. a 300-char test budget): drop the anchor and
        # handoff, and fall back to the original truncation behaviour.
        goal_sections = []
        handoff_text = ""
        text, head_text, volatile_text = assemble([])
        text = _truncate_at_line(text, cap)

    blocks: list[dict[str, Any]] = []
    if head_text:
        blocks.append({
            "type": "text",
            "text": head_text,
            "cache_control": {"type": "ephemeral"},
        })
    if volatile_text:
        blocks.append({
            "type": "text",
            "text": volatile_text,
            "cache_control": {"type": "ephemeral"},
        })
    if not blocks:
        blocks.append({
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        })

    payload: dict[str, Any] = {
        # Requested flat shape (useful for custom routers/tests).
        "additional_context": text,
        # Native Claude Code hook shape (UserPromptSubmit or SessionStart).
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": text,
        },
        # Cache-aware consumers can use these blocks verbatim.
        "content_blocks": blocks,
        "metadata": {
            "compactor": {
                "plugin": PLUGIN_NAME,
                "version": VERSION,
                "l2_masters": len(masters),
                "active_l1": len(active_chunks),
                "pending": pending_count(state, session),
                "dropped_l2": dropped_masters,
                "dropped_l1": dropped_chunks,
                "pinned_entities": len(_global_entities(state, session)),
                "max_inject_chars": max_chars,
                "l3": l3_meta,
                "goals": _goal_metadata(state, session, goal_sections),
                "handoff": handoff_meta,
            }
        },
    }
    headers = _openrouter_extra_headers()
    if headers:
        payload["metadata"]["openrouter_extra_headers"] = headers
    return payload


# ---------------------------------------------------------------------------
# Frozen shared block (multi-agent prefix reuse)
# ---------------------------------------------------------------------------

def shared_block_text(
    state: dict[str, Any],
    max_chars: int | None = None,
    session: str | None = None,
) -> str | None:
    """Render the ledger and L2 masters as one byte-stable block, or ``None``.

    The block contains no timestamps, counters or session identifiers, so two
    agents that render it from the same state get identical bytes.  Active L1
    chunks are deliberately excluded: they are per-agent recency, and sharing
    them would put different bytes at different offsets.
    """
    masters = [
        master["block"] for master in state.get("l2_masters", [])
        if master.get("block") and _belongs(master, session)
    ]
    if not masters:
        return None
    limit = max_chars or _env_int("APPEND_COMPACT_SHARED_CHARS", DEFAULT_MAX_INJECT_CHARS)
    limit = max(limit, len(SHARED_BLOCK_BANNER) + 200)
    ledger = _global_ledger_block(state, min(LEDGER_BUDGET_CHARS, limit // 4), session)
    # Opt-in: a shared goal anchor is still session-scoped, and it changes only
    # when the goal version changes, so the frozen bytes stay byte-stable
    # between deliberate goal revisions.
    goal_anchor = _render_shared_goal_anchor(
        state, session, min(_goal_total_cap(), max(80, limit // 3))
    )
    budget = (
        limit
        - len(SHARED_BLOCK_BANNER)
        - 2
        - (len(ledger) + 2 if ledger else 0)
        - (len(goal_anchor) + 2 if goal_anchor else 0)
        - len(SHARED_BLOCK_NOTE)
    )
    master_text, dropped = _pack_recent(masters, budget)
    pieces = [part for part in (SHARED_BLOCK_BANNER, ledger, master_text, goal_anchor) if part]
    text = "\n\n".join(pieces).strip()
    if dropped:
        text += SHARED_BLOCK_NOTE
    if len(text) > limit:
        text = _truncate_at_line(text, limit)
    return text


def emit_shared_block(
    state: dict[str, Any],
    max_chars: int | None = None,
    cache_dir: Path | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Write the frozen block to ``<cache_dir>/shared/<sha256>.block``.

    Content-addressed naming makes accidental drift visible: the same state
    always yields the same path, and any change yields a new one.  An existing
    file is never mutated in place, so a fan-out that already handed out a path
    keeps reading the same bytes.
    """
    text = shared_block_text(state, max_chars=max_chars, session=session)
    if text is None:
        return {
            "emitted": False,
            "reason": "no L2 masters yet; nothing shared to freeze",
            "shared_path": None,
            "sha256": None,
            "chars": 0,
            "l2_masters": 0,
            "pinned_entities": len(_global_entities(state, session)),
        }
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = (cache_dir or default_cache_dir()) / "shared" / f"{digest}.block"
    needs_write = True
    with contextlib.suppress(OSError):
        needs_write = path.read_text(encoding="utf-8") != text
    if needs_write:
        _atomic_write_text(path, text)
    return {
        "emitted": True,
        "shared_path": str(path),
        "sha256": digest,
        "chars": len(text),
        "l2_masters": sum(
            1 for master in state.get("l2_masters", []) if _belongs(master, session)
        ),
        "pinned_entities": len(_global_entities(state, session)),
    }


# ---------------------------------------------------------------------------
# Hook event handling
# ---------------------------------------------------------------------------

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
TOOL_KEEP_RE = re.compile(
    r"(?i)\b(?:error|exception|traceback|failed|failure|panic|fatal|assert\w*|"
    r"expected|actual|warning)\b|[\w./\\-]+:\d+(?::\d+)?"
)


def condense_tool_output(text: str, max_chars: int = TOOL_RESULT_CHARS) -> str:
    """Shrink a tool result without losing error lines or file:line anchors.

    Strips ANSI escapes, collapses runs of identical lines, keeps the head and
    tail, and retains middle lines that look like diagnostics.  This is the
    only place tool output enters the summary: it is lossy, so it must stay
    deterministic and bounded.
    """
    if not text:
        return ""
    cleaned = ANSI_ESCAPE_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [line.rstrip() for line in cleaned.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""

    collapsed: list[str] = []
    index = 0
    while index < len(lines):
        end = index
        while end + 1 < len(lines) and lines[end + 1] == lines[index]:
            end += 1
        count = end - index + 1
        line = lines[index]
        collapsed.append(f"{line}  (×{count})" if count >= 3 and line.strip() else line)
        index = end + 1

    if len(collapsed) <= 80:
        return _truncate_at_line("\n".join(collapsed), max_chars)
    head = collapsed[:24]
    tail = collapsed[-16:]
    middle = collapsed[24:-16]
    kept = [line for line in middle if TOOL_KEEP_RE.search(line)][:40]
    out = head + [f"… {len(middle)} lines condensed …"] + kept + tail
    return _truncate_at_line("\n".join(out), max_chars)


def _read_transcript_turns(path: str | None, max_turns: int, max_chars: int) -> list[tuple[str, str]]:
    if not path:
        return []
    transcript = Path(path).expanduser()
    if not transcript.is_file():
        return []
    turns: list[tuple[str, str]] = []
    try:
        with transcript.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
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
                content = message.get("content")
                text = ""
                tool_parts: list[str] = []
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text_parts: list[str] = []
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        kind = block.get("type")
                        if kind == "text" and isinstance(block.get("text"), str):
                            text_parts.append(block["text"])
                        elif kind == "tool_result":
                            payload = block.get("content")
                            raw = ""
                            if isinstance(payload, str):
                                raw = payload
                            elif isinstance(payload, list):
                                raw = "\n".join(
                                    item.get("text", "") if isinstance(item, dict) else str(item)
                                    for item in payload
                                )
                            condensed = condense_tool_output(raw)
                            if condensed:
                                tool_parts.append(condensed)
                    text = "\n".join(text_parts)
                if not tool_parts:
                    # toolUseResult duplicates tool_result; use it only when the
                    # structured block was absent, since it can carry extra data.
                    result = record.get("toolUseResult")
                    raw = ""
                    if isinstance(result, str):
                        raw = result
                    elif isinstance(result, dict):
                        try:
                            raw = json.dumps(result, ensure_ascii=False)
                        except (TypeError, ValueError):
                            raw = ""
                    condensed = condense_tool_output(raw)
                    if condensed:
                        tool_parts.append(condensed)
                text = text.strip()
                if text:
                    turns.append((role, text))
                for tool_text in tool_parts:
                    turns.append(("tool", tool_text))
    except OSError as exc:
        _log_error(f"cannot read transcript {transcript}: {exc!r}")
        return []

    turns = turns[-max_turns:]
    kept: list[tuple[str, str]] = []
    total = 0
    for role, text in reversed(turns):
        if kept and total + len(text) > max_chars:
            break
        if not kept and len(text) > max_chars:
            text = text[:max_chars]
        kept.append((role, text))
        total += len(text)
    kept.reverse()

    # Tool output is supplementary: cap its total, preferring the newest.
    tool_total = 0
    limited: list[tuple[str, str]] = []
    for role, text in reversed(kept):
        if role == "tool":
            if tool_total + len(text) > PRECOMPACT_TOOL_CHARS:
                continue
            tool_total += len(text)
        limited.append((role, text))
    limited.reverse()
    return limited


PRECOMPACT_INSTRUCTION_CAP = 4_000


def _preservation_instructions(state: dict[str, Any], session: str | None = None) -> str:
    """Plain text appended to Claude Code's compaction instructions.

    Claude Code ignores hookSpecificOutput on PreCompact; plain stdout from a
    successful hook is appended to the compaction prompt.  Keep this a str and
    well under the 10,000-character hook-output ceiling.  Queued turns are
    included so an entity captured seconds ago is never lost just because its
    background compression has not finished.
    """
    masters = [master for master in state.get("l2_masters", []) if _belongs(master, session)]
    entities = _merge_entities(
        *[chunk.get("entities", []) for chunk in active_l1(state, session)],
        *[master.get("entities", []) for master in masters[-1:]],
        *[job.get("entities", []) for job in _pending_for(state, session)],
    )[:ENTITY_HEADER_LIMIT]
    prefix = "Entity Ledger (preserve these file paths and exported symbol names verbatim"
    if entities:
        ledger = "; ".join(entities)
        text = prefix + "; do not rename, shorten, translate or alias them): " + ledger
    else:
        text = prefix + "): none captured."
    if len(text) > PRECOMPACT_INSTRUCTION_CAP:
        head = text[: PRECOMPACT_INSTRUCTION_CAP - 2]
        cut = head.rfind("; ")
        if cut > PRECOMPACT_INSTRUCTION_CAP // 2:
            head = head[:cut]
        text = head.rstrip("; ") + "; …"
    if _journal_enabled():
        text = text + "\n\n" + JOURNAL_INSTRUCTION
    return text


def _record_injection_stats(event: str, session: str, output: dict[str, Any] | None) -> None:
    """Append one counts-only line per injection; never content, never raises."""
    if not output:
        return
    try:
        meta = (output.get("metadata") or {}).get("compactor") or {}
        text = (output.get("hookSpecificOutput") or {}).get("additionalContext") or ""
        record = {
            "ts": _now(),
            "plugin": PLUGIN_NAME,
            "version": VERSION,
            "event": event,
            "session": session,
            "chars": len(text),
            "budget": meta.get("max_inject_chars"),
            "l2_masters": meta.get("l2_masters"),
            "active_l1": meta.get("active_l1"),
            "dropped_l1": meta.get("dropped_l1"),
            "dropped_l2": meta.get("dropped_l2"),
            "pinned_entities": meta.get("pinned_entities"),
            "l3": meta.get("l3"),
            "llm_loaded": _compressor is not None,
            "llm_failed": bool(_compressor_failed),
        }
        path = default_cache_dir() / "stats.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 50 * 1024 * 1024:
            path.replace(path.with_name(path.name + ".1"))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        return


COMPACT_BOUNDARY_TAIL_BYTES = 8 * 1024 * 1024


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _compacted_since(transcript_path: Any, since: Any) -> bool | None:
    """True/False when the transcript shows whether a compaction finished after
    ``since``; None when that cannot be determined."""
    since_at = _parse_utc(since)
    if not transcript_path or since_at is None:
        return None
    path = Path(str(transcript_path)).expanduser()
    try:
        with path.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - COMPACT_BOUNDARY_TAIL_BYTES))
            tail = fh.read()
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if b'"compact_boundary"' not in line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("subtype") != "compact_boundary":
            continue
        at = _parse_utc(record.get("timestamp"))
        if at is not None and at >= since_at:
            return True
    return False


def handle_prompt_event(
    state: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    background: bool = False,
) -> dict[str, Any] | None:
    session = str(payload.get("session_id") or "default")
    prompt = args.prompt if args.prompt is not None else payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    arm = state.get("armed_sessions", {}).get(session)
    armed = bool(arm)
    if armed and not args.force_inject:
        arm_at = arm.get("at") if isinstance(arm, dict) else None
        if _compacted_since(payload.get("transcript_path"), arm_at) is False:
            # PreCompact armed the session but no compaction followed, so the
            # full context is still in place.
            state.get("armed_sessions", {}).pop(session, None)
            armed = False
    output = None
    if args.force_inject or armed:
        # Build context from history *before* appending the current prompt, so
        # the prompt is never duplicated back to the model.  Only this
        # session's chunks are eligible.
        output = injection_payload(state, session=session, prompt=prompt)
        _record_injection_stats("UserPromptSubmit", session, output)

    queue_or_append(state, prompt, "user_prompt", session, background, role="user")

    if armed:
        state.get("armed_sessions", {}).pop(session, None)
    state["last_event"] = {"event": "UserPromptSubmit", "session_id": session, "at": _now()}
    return output if (args.inject or args.force_inject) else None


def handle_sessionstart_event(
    state: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    background: bool = False,
) -> dict[str, Any] | None:
    """Inject armed context at SessionStart(matcher=compact).

    This is the documented post-compaction injection point and removes the
    one-prompt delay.  UserPromptSubmit stays as a fallback for versions where
    SessionStart does not fire; whichever runs first consumes the arm.
    """
    session = str(payload.get("session_id") or "default")
    armed = bool(state.get("armed_sessions", {}).get(session))
    output = None
    if args.force_inject or armed:
        output = injection_payload(state, event_name="SessionStart", session=session)
        _record_injection_stats("SessionStart", session, output)
    if armed:
        state.get("armed_sessions", {}).pop(session, None)
    state["last_event"] = {
        "event": "SessionStart",
        "session_id": session,
        "source": str(payload.get("source") or payload.get("matcher") or ""),
        "at": _now(),
    }
    return output if (args.inject or args.force_inject) else None


def handle_precompact_event(
    state: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    background: bool = False,
) -> str:
    """Capture the transcript, arm the session, return plain-text instructions.

    PreCompact cannot inject context and does not honour customInstructions;
    Claude Code appends the successful hook's raw stdout to its compaction
    instructions, so the return value here must stay a plain ``str``.
    """
    session = str(payload.get("session_id") or "default")
    max_turns = _env_int("APPEND_COMPACT_PRECOMPACT_TURNS", DEFAULT_PRECOMPACT_TURNS)
    max_chars = _env_int("APPEND_COMPACT_PRECOMPACT_CHARS", DEFAULT_PRECOMPACT_CHARS)
    turns = _read_transcript_turns(payload.get("transcript_path"), max_turns, max_chars)
    for role, text in turns:
        queue_or_append(
            state,
            f"{role.capitalize()}:\n{text}",
            "precompact",
            session,
            background,
            role=role,
        )

    state.setdefault("armed_sessions", {})[session] = {
        "armed": True,
        "at": _now(),
        "trigger": payload.get("trigger", ""),
        "captured_turns": len(turns),
    }
    state["last_event"] = {"event": "PreCompact", "session_id": session, "at": _now()}
    return _preservation_instructions(state, session)


# ---------------------------------------------------------------------------
# Self-test controls
# ---------------------------------------------------------------------------

def self_test(verbose: bool = False) -> bool:
    """Known-positive/known-negative controls for the measuring pipeline."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    python_source = (
        '"""Module docstring."""\n'
        "# a comment\n"
        "import os\n\n\n"
        "def add(a, b):\n"
        '    """Add two numbers."""\n'
        "    # inline comment\n"
        "    return a + b\n"
    )
    mini = minify_python_source(python_source)
    check(
        "known-positive: python-minifier shrinks and strips docstrings",
        mini is not None and len(mini) < len(python_source) and "docstring" not in mini,
    )
    check("known-negative: invalid Python is left alone", minify_python_source("def broken(:\n") is None)

    entities = extract_entities("see src/app.ts and export function buildThing() {}")
    check(
        "known-positive: entity pinning finds path and symbol",
        "path: src/app.ts" in entities and "symbol: buildThing" in entities,
    )
    check("known-negative: plain prose yields no entities", extract_entities("hello world") == [])

    check("known-positive: JSON compaction", minify_json_text('{"a": 1}') == '{"a":1}')
    check("known-negative: non-JSON untouched", minify_json_text("not json") is None)

    probe = new_state()
    for index in range(5):
        append_turn(probe, f"turn {index} touches src/module_{index}.py", llm="off")
    check("boundary: no L2 before six chunks", len(probe["l2_masters"]) == 0, str(len(probe["l2_masters"])))
    append_turn(probe, "turn 5 touches src/module_5.py", llm="off")
    check(
        "known-positive: L2 consolidation at six chunks",
        len(probe["l2_masters"]) == 1 and probe["l2_masters"][0]["covered"] == [1, 2, 3, 4, 5, 6],
    )
    check("known-negative: active L1 is empty after consolidation", active_l1(probe) == [])

    js = "// comment\nconsole.log('debug')\nconst keep = 1;\n"
    stripped = strip_line_level(js, "javascript")
    check(
        "known-positive: line-level deletion",
        "console.log" not in stripped and "// comment" not in stripped and "const keep = 1;" in stripped,
    )
    check("known-negative: meaningful code is retained", 'print("sole child")' in strip_line_level(
        'if (flag) {\n  print("sole child");\n}\n', "javascript"
    ))

    bounded = injection_payload(probe, max_chars=10_000)
    check(
        "known-positive: injection clamped under the 10k hook cap",
        bounded is not None and len(bounded["additional_context"]) <= HOOK_OUTPUT_CAP - 200,
        str(len(bounded["additional_context"]) if bounded else 0),
    )

    deep = new_state()
    for index in range(40):
        append_turn(deep, f"turn {index} touches src/deep/module_{index}.py " + "detail " * 80, llm="off")
    packed = injection_payload(deep, max_chars=3_000)
    check(
        "known-positive: cap keeps the newest turns, drops the oldest master",
        packed is not None
        and len(packed["additional_context"]) <= 3_000
        and "Summary Chunk #40" in packed["additional_context"]
        and "Consolidated Master Summary (L2) #1" not in packed["additional_context"]
        and "**Entity Ledger (pinned):**" in packed["additional_context"],
    )

    noisy = "\n".join(["line ok"] * 30 + [f"src/app.py:{i}: failure in build" for i in range(5)])
    condensed = condense_tool_output(noisy)
    check(
        "known-positive: tool condenser collapses repeats, keeps errors",
        "(×30)" in condensed and "src/app.py:0: failure" in condensed and len(condensed) < len(noisy),
    )
    check("known-negative: short clean tool output unchanged", condense_tool_output("ok\n") == "ok")

    queue_probe = new_state()
    queued = enqueue_turn(queue_probe, "queued src/queued.py buildThing", session="A")
    claimed, _ = claim_job(queue_probe, "selftest")
    second_claim, _ = claim_job(queue_probe, "selftest-2")
    queued_text, queued_stats = compress_text(claimed["text"], llm="off") if claimed else ("", {})
    committed = commit_job(
        queue_probe, str(queued.get("id")), queued_text, queued_stats, worker="selftest"
    ) if claimed else None
    check(
        "known-positive: queue claim/commit yields exactly one chunk",
        committed is not None
        and pending_count(queue_probe) == 0
        and len(queue_probe["l1_chunks"]) == 1,
    )
    check("known-negative: a claimed job is not claimable twice", second_claim is None)

    scoped = new_state()
    append_turn(scoped, "alpha context src/alpha.py", llm="off", session="A")
    append_turn(scoped, "beta context src/beta.py", llm="off", session="B")
    injection_a = injection_payload(scoped, session="A")
    injection_b = injection_payload(scoped, session="B")
    check(
        "known-positive: parallel sessions are isolated",
        injection_a is not None
        and injection_b is not None
        and "src/alpha.py" in injection_a["additional_context"]
        and "src/beta.py" not in injection_a["additional_context"]
        and "src/beta.py" in injection_b["additional_context"],
    )

    drained = new_state()
    enqueue_turn(drained, "drain src/drained.py", session="A")
    drained_count = materialize_pending(drained, "A")
    check(
        "known-positive: drain materialises deterministically",
        drained_count == 1
        and pending_count(drained) == 0
        and drained["l1_chunks"][0]["stats"]["llm_blocks"] == 0,
    )

    previous_goals = os.environ.get("APPEND_COMPACT_GOALS")
    try:
        os.environ["APPEND_COMPACT_GOALS"] = "1"
        goal_probe = new_state()
        append_turn(
            goal_probe,
            "Goal: keep the public API stable.\nConstraints:\n- Must not rename exports",
            llm="off",
        )
        for index in range(6):
            append_turn(goal_probe, f"goal probe turn {index} src/goal_{index}.py", llm="off")
        goal_payload = injection_payload(goal_probe, max_chars=5_000)
        goal_text = goal_payload["additional_context"] if goal_payload else ""
        check(
            "known-positive: explicit goal is anchored verbatim at the beginning",
            'Goal (verbatim): "keep the public API stable."' in goal_text
            and "Must not rename exports" in goal_text
            and goal_text.startswith("[GOAL ANCHOR"),
        )
        check(
            "known-positive: beginning anchor leads the stable head",
            bool(goal_payload)
            and goal_payload["content_blocks"][0]["text"].startswith("[GOAL ANCHOR"),
        )
        plain_probe = new_state()
        append_turn(plain_probe, "Please continue with the next step.", llm="off")
        check(
            "known-negative: no goal marker, no anchored goal",
            _session_goal_store(plain_probe, None, create=False) is None,
        )
    finally:
        if previous_goals is None:
            os.environ.pop("APPEND_COMPACT_GOALS", None)
        else:
            os.environ["APPEND_COMPACT_GOALS"] = previous_goals

    shared_first = shared_block_text(probe)
    check(
        "known-positive: shared block is byte-stable",
        shared_first is not None
        and shared_first == shared_block_text(probe)
        and "Consolidated Master Summary" in shared_first,
    )
    check("known-negative: no L2 masters, no shared block", shared_block_text(new_state()) is None)

    memo_prose = (
        "We should carefully inspect src/app.ts before changing buildThing, because "
        "the initialization sequence repeats several expensive checks on every request.\n\n"
        "The team agreed to document each step so future readers can reproduce it."
    )
    with tempfile.TemporaryDirectory(prefix="append-compactor-selftest-") as memo_dir:
        memo_calls: list[str] = []
        original_compress = globals()["_default_llm_compress"]
        original_cache_dir = os.environ.get("APPEND_COMPACT_CACHE_DIR")

        def fake_compress(
            text: str,
            rate: float,
            force_tokens: list[str],
            *,
            model: str | None = None,
        ) -> str:
            memo_calls.append(text)
            return re.sub(r"\b(carefully|several|expensive|future) ", "", text)

        try:
            globals()["_default_llm_compress"] = fake_compress
            os.environ["APPEND_COMPACT_CACHE_DIR"] = memo_dir
            compress_prose(memo_prose, llm="auto")
            compress_prose(memo_prose, llm="auto")
            check(
                "known-positive: memo serves a repeated paragraph once",
                len(memo_calls) == 1,
                str(len(memo_calls)),
            )
            compress_prose(memo_prose + " A different closing sentence.", llm="auto")
            check(
                "known-negative: different text misses the memo",
                len(memo_calls) == 2,
                str(len(memo_calls)),
            )
        finally:
            globals()["_default_llm_compress"] = original_compress
            if original_cache_dir is None:
                os.environ.pop("APPEND_COMPACT_CACHE_DIR", None)
            else:
                os.environ["APPEND_COMPACT_CACHE_DIR"] = original_cache_dir

    handoff_turn = _handoff_from_turn(
        "Turn 4: please set the export worker to 8 shards and keep "
        "src/export/csv_writer.py on line 64.",
        4,
        "user",
    )
    values_flat = " ".join(item for item, _ in handoff_turn.get("critical_exact_values", []))
    files_flat = " ".join(item for item, _ in handoff_turn.get("files_symbols_references", []))
    check(
        "known-positive: handoff copies an exact value and a path",
        "8 shards" in values_flat and "src/export/csv_writer.py" in files_flat,
    )
    check(
        "known-negative: tool output is not a handoff fact",
        not any(_handoff_from_turn("pytest -q -> 214 passed, 3 skipped in 8.4s", 9, "tool").values()),
    )
    check(
        "known-positive: a contiguous copy grounds",
        _grounded("port 8443", "the metrics listener is on port 8443."),
    )
    check(
        "known-negative: an invented fact does not ground",
        not _grounded("MERGE_STALL_TIMEOUT_S 30.0", "we should add a merge stall timeout."),
    )
    tags = _harvest_tags(
        "The listener is on port 8443.\n"
        "※v port 8443\n"
        "※v MERGE_STALL_TIMEOUT_S 30.0\n"
        "※t 1 apply the correction\n"
        "※d 1",
        3,
    )
    check(
        "known-positive: a grounded journal tag is kept",
        any(tag.get("kind") == "v" and tag.get("body") == "port 8443" for tag in tags),
    )
    check(
        "known-negative: an invented journal tag is dropped",
        not any(tag.get("body") == "MERGE_STALL_TIMEOUT_S 30.0" for tag in tags),
    )
    check(
        "known-positive: ※d closes the matching task",
        _journal_open_tasks([{"tags": tags}]) == [],
    )
    check(
        "known-positive: an unclosed task stays open",
        [
            task["id"]
            for task in _journal_open_tasks(
                [{"tags": [tag for tag in tags if tag.get("kind") == "t"]}]
            )
        ]
        == ["1"],
    )
    handoff_probe = new_state()
    append_turn(
        handoff_probe,
        "please use port 8443 for metrics; see src/metrics/server.py",
        llm="off",
        role="user",
    )
    handoff_payload = injection_payload(handoff_probe, max_chars=5_000)
    check(
        "known-negative: no handoff when the view already shows everything",
        handoff_payload is not None
        and "Session handoff (key facts)" not in handoff_payload["additional_context"]
        and "**Entity Ledger (pinned):**" in handoff_payload["additional_context"],
    )
    demoted = new_state()
    for index in range(20):
        append_turn(
            demoted,
            (
                f"Turn {index}: the exporter retries with backoff and the pool "
                f"sits at {index % 7 + 2} shards while we tune the worker. "
            ) * 3,
            llm="off",
            role="assistant",
        )
    demoted_view = injection_payload(demoted, max_chars=2_000)
    check(
        "known-positive: the handoff restores facts the packed view dropped",
        demoted_view is not None
        and "Session handoff (key facts)" in demoted_view["additional_context"]
        and demoted_view["metadata"]["compactor"]["handoff"]["items"] > 0,
    )

    if verbose:
        for name, ok, detail in checks:
            status = "PASS" if ok else "FAIL"
            suffix = f" ({detail})" if detail else ""
            print(f"[{status}] {name}{suffix}")
        failures = [name for name, ok, _ in checks if not ok]
        if failures:
            print(f"{len(failures)} self-test control(s) failed", file=sys.stderr)
    return all(ok for _, ok, _ in checks)


# ---------------------------------------------------------------------------
# CLI / hook entrypoint
# ---------------------------------------------------------------------------

def _read_stdin_payload(timeout: float | None = None) -> dict[str, Any]:
    """Read hook JSON from stdin.

    ``timeout`` is used only when the event is not explicit: an interactive or
    idle non-TTY stdin must not hang a manual ``--inject``/``--prompt`` call.
    Hook invocations pass an explicit event and rely on the closed stdin pipe.
    """
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    if timeout is not None:
        try:
            import select

            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if not ready:
                return {}
        except (OSError, ValueError):
            return {}
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    raw = raw.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="append-compact.py",
        description="Multilevel append-only compactor for Claude Code.",
    )
    parser.add_argument(
        "--event",
        choices=["auto", "prompt", "precompact", "sessionstart"],
        default="auto",
    )
    parser.add_argument("--inject", action="store_true", help="emit hook injection JSON when armed")
    parser.add_argument("--force-inject", action="store_true", help="emit injection JSON regardless of arming")
    parser.add_argument("--prompt", help="prompt text (otherwise read from hook stdin)")
    parser.add_argument("--state", help=f"state file (default: {DEFAULT_STATE_PATH})")
    parser.add_argument("--reset", action="store_true", help="replace the state with an empty one")
    parser.add_argument("--status", action="store_true", help="print state status as JSON")
    parser.add_argument(
        "--recall",
        metavar="ID",
        help="print the raw turn archived under ID (12+ hex prefix accepted)",
    )
    parser.add_argument(
        "--recall-l3",
        metavar="ID",
        dest="recall_l3",
        help="print the full chunk/master block behind an L3 index id (12+ hex prefix accepted)",
    )
    parser.add_argument(
        "--emit-shared-block",
        action="store_true",
        help="freeze ledger + L2 masters into a byte-stable file for multi-agent prefix reuse",
    )
    parser.add_argument("--shared-max-chars", type=int, help="cap for the emitted shared block")
    parser.add_argument("--scope", help="session id to report or share (default: all)")
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="compress synchronously instead of queueing for a worker",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-llm", action="store_true", help="disable LLMLingua for this invocation")
    parser.add_argument("--self-test", action="store_true", help="run known-positive/negative controls")
    parser.add_argument("--version", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.version:
        print(f"{PLUGIN_NAME} {VERSION}")
        return 0
    if args.self_test:
        return 0 if self_test(verbose=True) else 1
    if args.no_llm:
        os.environ["APPEND_COMPACT_LLM"] = "0"

    try:
        state_path = Path(args.state).expanduser() if args.state else default_state_path()
    except Exception as exc:  # e.g. an unexpandable ~user in APPEND_COMPACT_STATE
        _log_error(f"cannot resolve state path: {exc!r}")
        return 0

    if args.recall:
        raw_id = args.recall.strip().lower()
        content = recall_raw(raw_id)
        if content is None:
            print(f"{PLUGIN_NAME}: no verified raw turn for {raw_id!r}", file=sys.stderr)
            return 1
        if args.scope:
            with state_lock(state_path):
                state = load_state(state_path)
            in_scope = any(
                str(chunk.get("raw_id") or "").startswith(raw_id)
                and _belongs(chunk, args.scope)
                for chunk in state.get("l1_chunks", [])
            )
            if not in_scope:
                print(
                    f"{PLUGIN_NAME}: raw turn {raw_id} is not in session {args.scope!r}",
                    file=sys.stderr,
                )
                return 1
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    if args.recall_l3:
        content = recall_l3(args.recall_l3)
        if content is None:
            print(
                f"{PLUGIN_NAME}: L3 id {args.recall_l3!r} is missing, malformed, or fails hash verification",
                file=sys.stderr,
            )
            return 1
        if args.scope:
            with state_lock(state_path):
                state = load_state(state_path)
            full_id = l3_id_for(content)
            in_scope = any(
                l3_id_for(str(item.get("block") or "")) == full_id
                and _belongs(item, args.scope)
                for collection in ("l1_chunks", "l2_masters")
                for item in state.get(collection, [])
            )
            if not in_scope:
                print(
                    f"{PLUGIN_NAME}: L3 block {args.recall_l3!r} is not in session {args.scope!r}",
                    file=sys.stderr,
                )
                return 1
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    if args.worker:
        try:
            return worker_main(state_path)
        except Exception as exc:  # never leave a traceback for the host
            _log_error(f"worker failure: {exc!r}")
            return 0
    if args.reset:
        with state_lock(state_path):
            save_state(new_state(), state_path)
        print(json.dumps({"reset": True, "state": str(state_path)}))
        return 0
    if args.status:
        with state_lock(state_path):
            state = load_state(state_path)
        print(json.dumps(status_of(state, state_path, args.scope), indent=2, ensure_ascii=False))
        return 0
    if args.emit_shared_block:
        with state_lock(state_path):
            state = load_state(state_path)
        try:
            result = emit_shared_block(state, max_chars=args.shared_max_chars, session=args.scope)
        except OSError as exc:
            print(json.dumps({"emitted": False, "reason": f"write failed: {exc}"}, ensure_ascii=False))
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0

    payload: dict[str, Any] = {}
    event = args.event
    if event == "auto":
        if args.prompt is not None:
            event = "prompt"
        else:
            payload = _read_stdin_payload(timeout=0.25)
            hook_name = payload.get("hook_event_name", "")
            if hook_name == "PreCompact":
                event = "precompact"
            elif hook_name == "UserPromptSubmit":
                event = "prompt"
            elif hook_name == "SessionStart":
                event = "sessionstart"
            else:
                event = "none"
    elif event in ("precompact", "sessionstart"):
        payload = _read_stdin_payload()
    elif event == "prompt" and args.prompt is None:
        payload = _read_stdin_payload()

    # Without fcntl there is no cross-process lock, so a worker pool cannot be
    # safe; keep the synchronous path on those platforms.
    args.background = (
        _env_bool("APPEND_COMPACT_BACKGROUND", True)
        and not args.no_background
        and fcntl is not None
    )
    session = str(payload.get("session_id") or "default")

    # An armed injection is imminent.  Give background workers one short,
    # bounded window, then commit whatever is left deterministically so the
    # injected context is complete and the hook stays far inside its deadline.
    if args.background and event in ("prompt", "sessionstart"):
        try:
            with state_lock(state_path):
                probe = load_state(state_path)
            armed = bool(probe.get("armed_sessions", {}).get(session))
            if (armed or args.force_inject) and pending_count(probe, session) > 0:
                drain_session(state_path, session)
        except Exception as exc:
            _log_error(f"background drain failed: {exc!r}")

    output: Any = None
    try:
        with state_lock(state_path):
            state = load_state(state_path)
            if event == "precompact":
                output = handle_precompact_event(state, payload, args, background=args.background)
            elif event == "sessionstart":
                output = handle_sessionstart_event(state, payload, args, background=args.background)
            elif event == "prompt":
                output = handle_prompt_event(state, payload, args, background=args.background)
            elif args.inject or args.force_inject:
                output = injection_payload(state, session=args.scope)
            else:
                output = status_of(state, state_path, args.scope)
            save_state(state, state_path)
    except Exception as exc:  # hooks must never break the host session
        _log_error(f"hook failure: {exc!r}")
        return 0

    if args.background and event in ("precompact", "prompt", "sessionstart") and state.get("pending"):
        spawn_worker(state_path)

    if isinstance(output, str):
        # PreCompact: raw stdout is appended to the compaction instructions.
        sys.stdout.write(output if output.endswith("\n") else output + "\n")
    elif output is not None:
        sys.stdout.write(json.dumps(output, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
