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

State lives in ``~/.claude/append_compactor_state.json`` (override with
``--state`` or ``APPEND_COMPACT_STATE``).

Run ``append-compact.py --self-test`` to verify the known-positive and
known-negative controls used by the compression pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

try:  # POSIX only; the plugin targets Claude Code on macOS/Linux.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]


VERSION = "1.3.0"
SCHEMA_VERSION = 1
PLUGIN_NAME = "claude-append-compactor"

DEFAULT_STATE_PATH = "~/.claude/append_compactor_state.json"
L1_CONSOLIDATE_AT = 6
DEFAULT_RATE = 0.4
LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# Claude Code caps each hook output string (additionalContext, systemMessage,
# plain stdout) at 10,000 characters and replaces anything longer with a file
# path plus a 2,000-character preview; there is no setting to raise it.  Keep
# the whole payload comfortably under that wall.
HOOK_OUTPUT_CAP = 10_000
DEFAULT_MAX_INJECT_CHARS = 9_000
LEDGER_BUDGET_CHARS = 2_000
TOOL_RESULT_CHARS = 1_200
PRECOMPACT_TOOL_CHARS = 6_000

DEFAULT_PRECOMPACT_TURNS = 24
DEFAULT_PRECOMPACT_CHARS = 24_000
MIN_PROSE_CHARS = 160
L2_DIGEST_CHARS = 480
ENTITY_HEADER_LIMIT = 200


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


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def new_state() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "created": _now(),
        "l1_chunks": [],
        "l2_masters": [],
        "armed_sessions": {},
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
    for key in ("l1_chunks", "l2_masters"):
        if not isinstance(data.get(key), list):
            data[key] = []
    if not isinstance(data.get("armed_sessions"), dict):
        data["armed_sessions"] = {}
    if not isinstance(data.get("stats"), dict):
        data["stats"] = base["stats"]
    return data


def save_state(state: dict[str, Any], path: Path | None = None) -> None:
    path = Path(path) if path is not None else default_state_path()
    _atomic_write_json(path, state)


def active_l1(state: dict[str, Any]) -> list[dict[str, Any]]:
    """L1 chunks not yet absorbed into an L2 master."""
    covered: set[int] = set()
    for master in state.get("l2_masters", []):
        covered.update(int(n) for n in master.get("covered", []) if isinstance(n, int))
    return [chunk for chunk in state.get("l1_chunks", []) if int(chunk.get("n", 0)) not in covered]


def status_of(state: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    active = active_l1(state)
    stats = state.get("stats", {})
    return {
        "plugin": PLUGIN_NAME,
        "version": VERSION,
        "state_file": str(path) if path else str(default_state_path()),
        "schema": state.get("schema"),
        "created": state.get("created"),
        "l1_chunks": len(state.get("l1_chunks", [])),
        "active_l1": [c.get("n") for c in active],
        "l2_masters": len(state.get("l2_masters", [])),
        "armed_sessions": sorted(state.get("armed_sessions", {})),
        "stats": stats,
        "last_event": state.get("last_event"),
    }


# ---------------------------------------------------------------------------
# Entity and path pinning
# ---------------------------------------------------------------------------

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


def extract_entities(text: str) -> list[str]:
    """Deterministic ledger entries: paths first, then symbols, each sorted."""
    entries = [f"path: {p}" for p in sorted(extract_paths(text))]
    entries.extend(f"symbol: {s}" for s in sorted(extract_symbols(text)))
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
    r"(?im)(?:^|\b)(?:"
    r"traceback \(most recent call last\)|fatal error|panic:|syntaxerror|typeerror|"
    r"referenceerror|valueerror|modulenotfounderror|importerror|failed to|npm err!|"
    r"error ts\d+|\berror\b|\bexception\b|segmentation fault|core dumped"
    r")"
)

_FILE_PATH_PATTERN = (
    r"(?:[A-Za-z]:[\\/]|~[\\/]|\.{1,2}[\\/]|[\\/])?(?:[\w@.+-]+[\\/])+[\w@.+-]+\.(?:"
    + CODE_EXTENSIONS
    + r")"
    r"|\b[\w@.+-]+\.(?:" + CODE_EXTENSIONS + r")\b"
)

_MASKABLE_RE = re.compile(
    r"`[^`\n]+`"
    r"|https?://[^\s<>\"')\]]+"
    r"|" + _FILE_PATH_PATTERN +
    r"|\b[A-Za-z_$][\w$]*(?:_[A-Za-z0-9_$]+)+\b"
    r"|\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+\b"
    r"|\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9]*\b"
    r"|\b[A-Z][a-z0-9]*[A-Z][A-Za-z0-9]*\b"
)


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


def _compress_fence(info: str, body: str, stats: dict[str, int]) -> str:
    lang = _language(info)
    if lang in {"json", "jsonc"}:
        compact = minify_json_text(body)
        if compact is not None:
            stats["json_blocks"] += 1
            return _render_fence(info, compact)
        # Invalid JSON must be left byte-for-byte alone.
        return _render_fence(info, body.rstrip("\n"))
    if lang in {"python", "py", "python3"}:
        mini = minify_python_source(body)
        if mini is not None:
            stats["code_blocks"] += 1
            return _render_fence(info, mini)
    stripped = strip_line_level(body, lang)
    if stripped != body.rstrip("\n"):
        stats["code_blocks"] += 1
    return _render_fence(info, stripped)


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

    masked = _MASKABLE_RE.sub(replace, text)
    return masked, list(mapping), mapping


def _restore_entities(masked: str, mapping: dict[str, str]) -> str:
    restored = masked
    for placeholder, original in mapping.items():
        restored = restored.replace(placeholder, original)
    return restored


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
    rate = DEFAULT_RATE if rate is None else rate
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

    masked, placeholders, mapping = _mask_entities(text)
    try:
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
    return restored.strip(), "llmlingua"


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


def _chunk_block(number: int, entities: list[str], text: str, source: str) -> str:
    lines = [f"### Summary Chunk #{number}"]
    if source:
        lines.append(f"Source: {source}")
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


def _build_master(state: dict[str, Any], batch: list[dict[str, Any]]) -> dict[str, Any]:
    number = len(state.get("l2_masters", [])) + 1
    covered = [int(chunk["n"]) for chunk in batch]
    entities = _merge_entities(*[chunk.get("entities", []) for chunk in batch])
    lines = [
        f"### Consolidated Master Summary (L2) #{number}",
        f"Covers Summary Chunks #{covered[0]}–#{covered[-1]}",
    ]
    lines.extend(_ledger_lines(entities))
    lines.append("**Chunk Digests:**")
    for chunk in batch:
        lines.append(f"#### Chunk #{chunk['n']}")
        lines.append(_digest(chunk.get("compressed_text", ""), L2_DIGEST_CHARS))
    return {
        "n": number,
        "created": _now(),
        "covered": covered,
        "entities": entities,
        "block": "\n".join(lines),
    }


def consolidate_if_needed(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Consolidate active L1 chunks into immutable L2 masters at the threshold."""
    created: list[dict[str, Any]] = []
    while len(active_l1(state)) >= L1_CONSOLIDATE_AT:
        batch = active_l1(state)[:L1_CONSOLIDATE_AT]
        master = _build_master(state, batch)
        state.setdefault("l2_masters", []).append(master)
        created.append(master)
    return created


def append_turn(
    state: dict[str, Any],
    text: str,
    source: str = "user_prompt",
    llm: Any = "auto",
) -> dict[str, Any] | None:
    """Compress one turn and append it as an immutable L1 chunk."""
    if not text or not text.strip():
        return None
    compressed, stats = compress_text(text, llm=llm)
    if not compressed.strip():
        return None

    number = len(state.get("l1_chunks", [])) + 1
    entities = _merge_entities(extract_entities(text), extract_entities(compressed))
    chunk = {
        "n": number,
        "created": _now(),
        "source": source,
        "entities": entities,
        "compressed_text": compressed,
        "block": _chunk_block(number, entities, compressed, source),
        "stats": stats,
    }
    state.setdefault("l1_chunks", []).append(chunk)

    totals = state.setdefault("stats", {})
    totals["turns"] = int(totals.get("turns", 0)) + 1
    totals["bytes_in"] = int(totals.get("bytes_in", 0)) + stats["bytes_in"]
    totals["bytes_out"] = int(totals.get("bytes_out", 0)) + stats["bytes_out"]
    for key in ("llm_blocks", "fallback_blocks", "error_blocks"):
        totals[key] = int(totals.get(key, 0)) + stats.get(key, 0)

    # Keep every existing chunk dict byte-identical; the master records which
    # chunks it covers.  (Consolidation never rewrites a past chunk.)
    consolidate_if_needed(state)
    return chunk


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


def _global_entities(state: dict[str, Any]) -> list[str]:
    return _merge_entities(
        *[chunk.get("entities", []) for chunk in state.get("l1_chunks", [])],
        *[master.get("entities", []) for master in state.get("l2_masters", [])],
    )


def _global_ledger_block(state: dict[str, Any], limit: int) -> str:
    """Deterministic union of every pinned entity, kept at the stable head.

    Masters get dropped from the injection when the cap is tight, so this
    ledger is what keeps old file paths and symbol names alive.
    """
    entities = _global_entities(state)
    if not entities or limit <= 0:
        return ""
    header = "**Entity Ledger (pinned):**"
    lines = [header]
    used = len(header) + 1
    for entry in entities:
        line = f"- {entry}"
        if used + len(line) + 1 > limit:
            lines.append(f"- … {len(entities)} entities pinned, list truncated")
            break
        lines.append(line)
        used += len(line) + 1
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


def injection_payload(
    state: dict[str, Any],
    max_chars: int | None = None,
    event_name: str = "UserPromptSubmit",
) -> dict[str, Any] | None:
    """Bounded payload: pinned ledger head, newest L2 masters, newest active L1."""
    max_chars = _clamp_inject_chars(
        max_chars or _env_int("APPEND_COMPACT_MAX_INJECT_CHARS", DEFAULT_MAX_INJECT_CHARS)
    )
    masters = [master["block"] for master in state.get("l2_masters", []) if master.get("block")]
    active = [chunk["block"] for chunk in active_l1(state) if chunk.get("block")]
    if not masters and not active:
        return None

    note = "\n\n[append-compactor: older context omitted to fit the 10k hook cap]"
    ledger = _global_ledger_block(state, min(LEDGER_BUDGET_CHARS, max_chars // 4))
    budget = max_chars - (len(ledger) + 2 if ledger else 0) - len(note)

    # Live turns outrank historical masters.
    active_budget = budget if not masters else min(budget, max(1_500, budget // 2))
    active_text, dropped_active = _pack_recent(active, active_budget)
    remaining = budget - (len(active_text) + 2 if active_text else 0)
    master_text, dropped_masters = _pack_recent(masters, remaining)

    pieces = [part for part in (ledger, master_text, active_text) if part]
    text = "\n\n".join(pieces).strip()
    if dropped_active or dropped_masters:
        text += note
    if len(text) > max_chars:  # final hard guard; also keeps us under 10k
        text = _truncate_at_line(text, max_chars)

    blocks: list[dict[str, Any]] = []
    head_text = "\n\n".join(part for part in (ledger, master_text) if part)
    if head_text:
        blocks.append({
            "type": "text",
            "text": head_text,
            "cache_control": {"type": "ephemeral"},
        })
    if active_text:
        blocks.append({
            "type": "text",
            "text": active_text,
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
                "l2_masters": len(state.get("l2_masters", [])),
                "active_l1": len(active),
                "dropped_l2": dropped_masters,
                "dropped_l1": dropped_active,
                "pinned_entities": len(_global_entities(state)),
                "max_inject_chars": max_chars,
            }
        },
    }
    headers = _openrouter_extra_headers()
    if headers:
        payload["metadata"]["openrouter_extra_headers"] = headers
    return payload


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


def _preservation_instructions(state: dict[str, Any]) -> str:
    """Plain text appended to Claude Code's compaction instructions.

    Claude Code ignores hookSpecificOutput on PreCompact; plain stdout from a
    successful hook is appended to the compaction prompt.  Keep this a str and
    well under the 10,000-character hook-output ceiling.
    """
    entities = _merge_entities(
        *[chunk.get("entities", []) for chunk in active_l1(state)],
        *[master.get("entities", []) for master in state.get("l2_masters", [])[-1:]],
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
    return text


def handle_prompt_event(
    state: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    session = str(payload.get("session_id") or "default")
    prompt = args.prompt if args.prompt is not None else payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    armed = bool(state.get("armed_sessions", {}).get(session))
    output = None
    if args.force_inject or armed:
        # Build context from history *before* appending the current prompt, so
        # the prompt is never duplicated back to the model.
        output = injection_payload(state)

    append_turn(state, prompt, source="user_prompt")

    if armed:
        state.get("armed_sessions", {}).pop(session, None)
    state["last_event"] = {"event": "UserPromptSubmit", "session_id": session, "at": _now()}
    return output if (args.inject or args.force_inject) else None


def handle_sessionstart_event(
    state: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
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
        output = injection_payload(state, event_name="SessionStart")
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
        append_turn(state, f"{role.capitalize()}:\n{text}", source="precompact")

    state.setdefault("armed_sessions", {})[session] = {
        "armed": True,
        "at": _now(),
        "trigger": payload.get("trigger", ""),
        "captured_turns": len(turns),
    }
    state["last_event"] = {"event": "PreCompact", "session_id": session, "at": _now()}
    return _preservation_instructions(state)


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

    if args.reset:
        with state_lock(state_path):
            save_state(new_state(), state_path)
        print(json.dumps({"reset": True, "state": str(state_path)}))
        return 0
    if args.status:
        state = load_state(state_path)
        print(json.dumps(status_of(state, state_path), indent=2, ensure_ascii=False))
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

    output: Any = None
    try:
        with state_lock(state_path):
            state = load_state(state_path)
            if event == "precompact":
                output = handle_precompact_event(state, payload, args)
            elif event == "sessionstart":
                output = handle_sessionstart_event(state, payload, args)
            elif event == "prompt":
                output = handle_prompt_event(state, payload, args)
            elif args.inject or args.force_inject:
                output = injection_payload(state)
            else:
                output = status_of(state, state_path)
            save_state(state, state_path)
    except Exception as exc:  # hooks must never break the host session
        _log_error(f"hook failure: {exc!r}")
        return 0

    if isinstance(output, str):
        # PreCompact: raw stdout is appended to the compaction instructions.
        sys.stdout.write(output if output.endswith("\n") else output + "\n")
    elif output is not None:
        sys.stdout.write(json.dumps(output, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
