#!/usr/bin/env python3
"""Post-process a template-compaction results JSON into a readable analysis.

Why this exists: the harness itself refuses to write a summary when a control
fails, which is right, but it also means a run that fails only a soft
sub-criterion costs hours to redo. This reads the saved JSON and derives a
second view -- scores normalised against the model's own full-context ceiling --
so a weak local model's numbers can still be read honestly.

    python3 summarize_benchmark.py results_v7.json [--out summary_v7.md]

The normalised view answers "how much of what this model *could* recall did the
arm actually deliver". A raw 0.50 constraint recall is meaningless on a model
that only reaches 0.55 with the entire transcript in front of it; 0.91 of its
own ceiling is the honest statement.

Controls (per the standing rule for scripts that report numbers):
  known-positive -- a synthetic arm with goal 0.50 against a chosen ceiling of
                    0.75 must normalise to exactly 0.666..., to 9 digits.
  known-negative -- an arm with no metrics must render as "--", never as a
                    fabricated 0.00, and a zero ceiling must not divide.
If either control fails this exits 3 instead of printing numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

METRICS = (
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
)

# Short column labels; decision_recall and decision_correct must not collide.
LABELS = {
    "goal_recall": "goal",
    "constraint_recall": "constr",
    "success_recall": "success",
    "decision_recall": "decision_recall",
    "correction_recall": "correction",
    "artifact_recall": "artifact",
    "error_recall": "error",
    "values_recall": "values",
    "fact_recall": "fact",
    "open_loop_recall": "open-loop",
    "next_step_recall": "next-step",
    "decision_correct": "choice",
    "task_success": "task",
}


def ceilings(report: dict) -> dict[str, float]:
    """Mean of the full-context control per metric = this model's ceiling."""
    checks = ((report.get("controls") or {}).get("full_context") or {}).get("checks") or []
    per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
    for check in checks:
        detail = check.get("detail") or {}
        scored = detail.get("scored")
        if isinstance(scored, dict):
            source = scored
        else:  # older results only kept these two
            source = {k: detail.get(k) for k in ("goal_recall", "constraint_recall")}
        for metric in METRICS:
            value = source.get(metric)
            if isinstance(value, (int, float)):
                per_metric[metric].append(float(value))
        # score_recall does not emit the decision metric; the control records
        # the choice itself, so derive the ceiling from it rather than leaving
        # the column blank.
        chosen, expected = detail.get("chosen"), detail.get("expected")
        if chosen is not None and expected is not None:
            per_metric["decision_correct"].append(1.0 if chosen == expected else 0.0)
    return {
        metric: sum(values) / len(values)
        for metric, values in per_metric.items()
        if values
    }


def paired_from_rows(report: dict, baseline: str) -> dict:
    """Paired differences of every arm against ``baseline``, computed from rows.

    The results file precomputes differences against current_goals only, which
    cannot answer "is the minimal template as good as the full one" -- that is a
    comparison between two arms, neither of which is the baseline.
    """
    rows = report.get("rows") or []
    cells: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        arm = row.get("arm")
        key = (row.get("scenario"), row.get("depth"))
        if not arm or key[0] is None or key[1] is None:
            continue
        cells.setdefault(key, {})[arm] = row
    diffs: dict[str, dict[str, list[float]]] = {}
    for arm in sorted({row.get("arm") for row in rows if row.get("arm")}):
        if arm == baseline:
            continue
        for cell in cells.values():
            if arm not in cell or baseline not in cell:
                continue
            for metric in METRICS:
                a, b = cell[arm].get(metric), cell[baseline].get(metric)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    diffs.setdefault(arm, {}).setdefault(metric, []).append(float(a) - float(b))
    out: dict[str, dict] = {}
    for arm, per_metric in diffs.items():
        metrics = {}
        for metric, values in per_metric.items():
            metrics[metric] = {
                "mean_diff": sum(values) / len(values),
                "wins": sum(1 for v in values if v > 0),
                "losses": sum(1 for v in values if v < 0),
                "ties": sum(1 for v in values if v == 0),
                "n": len(values),
            }
        out[arm] = {"baseline": baseline, "metrics": metrics}
    return out


def normalise(value: float, ceiling: float | None) -> float | None:
    if ceiling is None or ceiling <= 0:
        return None
    return value / ceiling


def build(report: dict, baseline: str = "current_goals") -> list[str]:
    lines: list[str] = []
    ceiling = ceilings(report)
    settings = report.get("settings") or {}
    controls = report.get("controls") or {}
    aggregates = report.get("aggregates") or {}
    by_depth = report.get("aggregates_by_depth") or {}

    lines.append("# Template compaction -- analysis")
    lines.append("")
    lines.append(f"Source: generated {report.get('generated_at')}")
    n_per_cell = settings.get("n") or max(
        (stats.get("n", 0) for stats in aggregates.values()), default=0
    )
    lines.append(
        f"Model: {settings.get('model')} ({settings.get('provider')}), "
        f"arms {settings.get('arms')}, depths {settings.get('depths')}, "
        f"n={n_per_cell} rows per arm (one sample per scenario/depth cell), "
        f"seed {settings.get('seed')}"
    )
    # The validation policy decides what counts as a usable template; without it
    # a reader cannot tell a loose run from a strict one.
    if "template_coverage_min" in settings:
        policy = settings["template_coverage_min"]
        detail = (
            "loose: templates accepted without a decision/correction entry"
            if (isinstance(policy, (int, float)) and policy < 0)
            else "strict: decision and correction must be mentioned"
        )
        lines.append(f"Template validation: coverage_min={policy} ({detail})")
    else:
        lines.append(
            "Template validation: **not recorded in this results file** -- check the "
            "run command before comparing it to a strict-policy run."
        )
    lines.append("")
    lines.append("## Controls")
    lines.append("")
    lines.append(f"- All controls passed: **{controls.get('all_passed')}**")
    failures = controls.get("failures") or []
    if failures:
        for failure in failures:
            lines.append(f"- FAILED: {failure.get('name')}")
    no_ctx = controls.get("no_context") or {}
    if no_ctx:
        lines.append(
            f"- No-context decision accuracy: {no_ctx.get('accuracy', 0):.2f} "
            f"(chance {no_ctx.get('chance_accuracy', 0):.2f})"
        )
    lines.append("")
    lines.append("## Model ceiling (full transcript, all facts available)")
    lines.append("")
    if not ceiling:
        lines.append("_No full-context control data in this results file._")
    else:
        lines.append("| metric | ceiling |")
        lines.append("|---|--:|")
        for metric in METRICS:
            if metric in ceiling:
                lines.append(f"| {metric} | {ceiling[metric]:.3f} |")
        lines.append("")
        lines.append(
            "Any arm score is capped in practice by this ceiling: a model that "
            "recalls 0.55 of the constraints with the whole transcript in front "
            "of it cannot be expected to exceed that from a compacted view."
        )
    lines.append("")
    lines.append("## Raw arm means")
    lines.append("")
    lines.append("| arm | n | " + " | ".join(LABELS[m] for m in METRICS) + " |")
    lines.append("|---|--:|" + "--:|" * len(METRICS))
    for arm, stats in aggregates.items():
        cells = []
        for metric in METRICS:
            value = stats.get(metric)
            cells.append("--" if value is None else f"{value:.2f}")
        lines.append(f"| {arm} | {stats.get('n', 0)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Share of the model's own ceiling")
    lines.append("")
    if not ceiling:
        lines.append("_Skipped: no ceiling available._")
    else:
        lines.append("| arm | " + " | ".join(LABELS[m] for m in METRICS) + " |")
        lines.append("|---|" + "--:|" * len(METRICS))
        for arm, stats in aggregates.items():
            cells = []
            for metric in METRICS:
                value = stats.get(metric)
                ratio = normalise(value, ceiling.get(metric)) if value is not None else None
                cells.append("--" if ratio is None else f"{ratio * 100:.0f}%")
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "100% = delivered as much as the model delivers when handed the entire "
            "session. Over 100% is possible and informative: a compacted view can "
            "beat a long transcript by removing the distraction tail."
        )
    lines.append("")
    lines.append("## Per depth")
    lines.append("")
    depths = settings.get("depths") or []
    lines.append("| arm | depth | n | goal | constr | correction | artifact | fact | open-loop | choice |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for arm in aggregates:
        for depth in depths:
            stats = by_depth.get(f"{arm}@{depth}")
            if not stats:
                continue
            lines.append(
                f"| {arm} | {depth} | {stats.get('n', 0)} | "
                f"{stats.get('goal_recall', 0):.2f} | {stats.get('constraint_recall', 0):.2f} | "
                f"{stats.get('correction_recall', 0):.2f} | {stats.get('artifact_recall', 0):.2f} | "
                f"{stats.get('fact_recall', 0):.2f} | {stats.get('open_loop_recall', 0):.2f} | "
                f"{stats.get('decision_correct', 0):.2f} |"
            )
    lines.append("")
    lines.append(f"## Paired differences vs {baseline}")
    lines.append("")
    # Prefer recomputing from the rows when a baseline is named: the stored
    # block only ever compares against current_goals.
    paired = paired_from_rows(report, baseline) or (report.get("paired_differences") or {})
    if not paired:
        lines.append("_No paired data._")
    for arm, data in paired.items():
        metrics = data.get("metrics") or {}
        if not metrics:
            continue
        lines.append(f"**{arm}**")
        lines.append("")
        lines.append("| metric | mean diff | wins | losses | ties | n |")
        lines.append("|---|--:|--:|--:|--:|--:|")
        for metric in METRICS:
            value = metrics.get(metric)
            if not value:
                continue
            lines.append(
                f"| {metric} | {value['mean_diff']:+.3f} | {value['wins']} | "
                f"{value['losses']} | {value['ties']} | {value['n']} |"
            )
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append(
        "- Synthetic sessions with a single model and a small n. These are paired, "
        "same-session comparisons, not significance tests."
    )
    reduction = settings.get("runtime_reduction") or ""
    if isinstance(reduction, dict):
        reduction = (
            f"{reduction.get('scenarios_run', '?')} of "
            f"{reduction.get('scenarios_planned', '?')} planned scenarios and "
            f"{reduction.get('depths_run', '?')} of {reduction.get('depths_planned', '?')} "
            "planned depths"
        )
    if reduction:
        lines.append(f"- Scope: {reduction}")
    lines.append(
        "- Both arms and ceiling come from the same model, so the ceiling-normalized "
        "view corrects for model weakness, not for scorer bias."
    )
    return lines


def _controls_pass() -> bool:
    """Known-positive and known-negative for the normalisation itself."""
    ok = True
    ceiling = {"goal_recall": 0.75}
    got = normalise(0.50, ceiling["goal_recall"])
    want = 0.50 / 0.75
    if got is None or abs(got - want) > 5e-10:
        print(f"[control FAIL] normalise 0.50 against 0.75 gave {got}, want {want}", file=sys.stderr)
        ok = False
    if normalise(0.5, 0.0) is not None:
        print("[control FAIL] zero ceiling divided instead of returning None", file=sys.stderr)
        ok = False
    if normalise(0.5, None) is not None:
        print("[control FAIL] missing ceiling produced a number", file=sys.stderr)
        ok = False
    # known-negative: an arm with no metrics must render as "--", never as a
    # fabricated 0.00 -- a missing measurement and a measured zero are different
    # claims and must not look the same.
    fake = {"aggregates": {"ghost": {"n": 0}}, "controls": {}, "settings": {"depths": [24]}}
    text = "\n".join(build(fake))
    for line in text.splitlines():
        if line.startswith("| ghost |"):
            if "0.00" in line:
                print(f"[control FAIL] missing metrics rendered as zero: {line}", file=sys.stderr)
                ok = False
            elif "--" not in line:
                print(f"[control FAIL] missing metrics rendered as something else: {line}", file=sys.stderr)
                ok = False
            break
    else:
        print("[control FAIL] an arm with no metrics vanished from the table", file=sys.stderr)
        ok = False
    # known-positive: a real arm row does appear with its value.
    fake["aggregates"] = {"real": {"n": 4, "goal_recall": 0.5}}
    text = "\n".join(build(fake))
    if "| real | 4 | 0.50 " not in text:
        print("[control FAIL] a populated arm row was not rendered with its value", file=sys.stderr)
        ok = False
    # Pairing from rows: known-positive (a hand-computable difference) and
    # known-negative (a cell missing one arm must not be counted at all).
    rows = [
        {"scenario": "s1", "depth": 24, "arm": "base", "fact_recall": 0.5},
        {"scenario": "s1", "depth": 24, "arm": "treat", "fact_recall": 0.75},
        {"scenario": "s2", "depth": 24, "arm": "base", "fact_recall": 0.5},
        {"scenario": "s2", "depth": 96, "arm": "treat", "fact_recall": 0.5},
    ]
    paired = paired_from_rows({"rows": rows}, "base")["treat"]["metrics"]["fact_recall"]
    if paired["mean_diff"] != 0.25 or paired["wins"] != 1 or paired["n"] != 1:
        print(f"[control FAIL] pairing gave {paired}, want diff 0.25 over n=1", file=sys.stderr)
        ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--baseline",
        default="current_goals",
        help="arm the paired differences are computed against",
    )
    args = parser.parse_args(argv)

    if not _controls_pass():
        print("CONTROL FAILED: refusing to summarize", file=sys.stderr)
        return 3

    report = json.loads(args.results.read_text(encoding="utf-8"))
    lines = build(report, baseline=args.baseline)
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
