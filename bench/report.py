"""ANSI-colored rendering of scored results."""
from __future__ import annotations

import sys

from .scorer import FunctionScore, LineTag


# Colors chosen to mirror the video's legend:
#   gray   — correct (matched)
#   orange — expected but missing
#   yellow — true hallucinated garbage / mangled
#   magenta - indentation issue. 
#   blue   — extra correct lines past the primary 20
COLOR = {
    LineTag.MATCHED: "\x1b[37m",        # white/gray
    LineTag.MISSING: "\x1b[38;5;208m",  # 256-color orange
    LineTag.HALLUCINATED: "\x1b[33m",   # yellow
    LineTag.BONUS: "\x1b[36m",          # cyan (blue-ish)
    LineTag.REINDENTED: "\x1b[38;5;117m",  # light blue — content right, spacing differs
}
RESET = "\x1b[0m"
BOLD = "\x1b[1m"


def _colorize(enabled: bool | None, color: str, text: str) -> str:
    if not enabled:
        return text
    return f"{color}{text}{RESET}"


def render_function(score: FunctionScore, color: bool | None = None) -> str:
    if color is None:
        color = sys.stdout.isatty()

    if score.error:
        # Distinguish from a real recall miss — the model never actually answered.
        status = "ERROR"
        status_color = "\x1b[35m"  # magenta — visually distinct from PASS green / FAIL red
        header = (
            f"\n=== {score.name}  "
            f"[{_colorize(color, status_color, status)}]  "
            f"{score.error}"
        )
        return header

    status = "PASS" if score.passed else "FAIL"
    status_color = "\x1b[32m" if score.passed else "\x1b[31m"
    # updated for docs vs code
    header = (
        f"\n=== {score.name}  "
        f"[{_colorize(color, status_color, status)}] ===\n"
        f"  Overall : {score.primary_matched}/{score.primary_total} "
        f"({'PASS' if score.passed else 'FAIL'})\n"
        f"  Doc     : {score.doc_matched}/{score.doc_total} "
        f"({'PASS' if score.doc_passed else 'FAIL'})\n"
        f"  Code    : {score.code_matched}/{score.code_total} "
        f"({'PASS' if score.code_passed else 'FAIL'})\n"
        f"  Indent  : {score.indentation_violations}\n"
        f"  Halluc  : {score.hallucinated}\n"
        f"  Bonus   : {score.bonus_matched}\n"
        f"  Tags    : {', '.join(score.analysis_tags)}"
    )

    legend = (
        "  Legend: "
        f"{_colorize(color, COLOR[LineTag.MATCHED], '■')} Match  "
        f"{_colorize(color, COLOR[LineTag.MISSING], '■')} Missing  "
        f"{_colorize(color, COLOR[LineTag.REINDENTED], '■')} Indent  "
        f"{_colorize(color, COLOR[LineTag.HALLUCINATED], '■')} Halluc  "
        f"{_colorize(color, COLOR[LineTag.BONUS], '■')} Bonus"
    )
    
    out = [
        header,
        legend,
        "  -- model output --",
    ]

    for r in score.predicted_tagged:
        out.append("  " + _colorize(color, COLOR[r.tag], r.text))

    missing = [r for r in score.expected_tagged if r.tag == LineTag.MISSING]
    if missing:
        out.append("  -- missing expected lines --")
        for r in missing:
            out.append("  " + _colorize(color, COLOR[r.tag], r.text))

    reindented = [r for r in score.expected_tagged if r.tag == LineTag.REINDENTED]
    if reindented:
        out.append("  -- reproduced, but re-indented (not hallucinations) --")
        for r in reindented:
            out.append("  " + _colorize(color, COLOR[r.tag], r.text))
    return "\n".join(out)

def _percentage(value: int, total: int) -> float:
    return value / total * 100 if total else 0.0

def render_summary(scores: list[FunctionScore], color: bool | None = None) -> str:
    if color is None:
        color = sys.stdout.isatty()
    errored = [s for s in scores if s.error]
    real = [s for s in scores if not s.error]
    passed = sum(1 for s in real if s.passed)
    total_matched = sum(s.primary_matched for s in real)
    total_possible = sum(s.primary_total for s in real)
    total_halluc = sum(s.hallucinated for s in real)

    total_indent = sum(s.indentation_violations for s in real)
    affected_by_reindent = sum(1 for s in real if s.indentation_violations > 0)
    total_bonus = sum(s.bonus_matched for s in real)
    # docs and codes
    total_doc_matched = sum(s.doc_matched for s in real)
    total_doc_total = sum(s.doc_total for s in real)

    total_code_matched = sum(s.code_matched for s in real)
    total_code_total = sum(s.code_total for s in real)

    lines = [
        "",
        _colorize(color, BOLD, "=== SUMMARY ==="),
        f"  Pass:                  {passed}/{len(real)}"
        + (f"  ({len(errored)} errored)" if errored else ""),
        (
            f"  Primary lines matched: {total_matched}/{total_possible}"
            f"  ({_percentage(total_matched, total_possible):.1f}%)"
        ),
        (
            f"  Doc recall:            {total_doc_matched}/{total_doc_total}"
            f"  ({_percentage(total_doc_matched, total_doc_total):.1f}%)"
        ),
        (
            f"  Code recall:           {total_code_matched}/{total_code_total}"
            f"  ({_percentage(total_code_matched, total_code_total):.1f}%)"
        ),
        
        f"  Hallucinated lines:    {total_halluc}",
        f"  Bonus (extra correct): {total_bonus}",
        (
            f"  Indent violations:     {total_indent}/{total_possible}"
            f"  ({_percentage(total_indent, total_possible):.1f}%) (content correct, indentation differs; not hallucinations)"
        ),
    ]

    if total_indent:
        lines.append(
            f"    ↳ {affected_by_reindent} function(s) affected. "
            "These remain misses under strict scoring."
        )
        lines.append(
            "      Re-run with --relax-indent to score indentation-insensitively."
        )
    

    # Per-function one-liner
    lines.append("")
    lines.append("  per-function:")
    for s in scores:
        if s.error:
            mark = _colorize(color, "\x1b[35m", "!")
            lines.append(
                f"    {mark} {s.name:<40} "
                f"overall={s.primary_matched:>2}/{s.primary_total:<2}  "
                f"doc={s.doc_matched:>2}/{s.doc_total:<2}  "
                f"code={s.code_matched:>2}/{s.code_total:<2}  "
                f"halluc={s.hallucinated:>2}  "
                f"indent  : {s.indentation_violations}"
            )

        else:
            mark = _colorize(color, "\x1b[32m", "✓") if s.passed else _colorize(color, "\x1b[31m", "✗")
            lines.append(
                f"    {mark} {s.name:<40} "
                f"overall={s.primary_matched:>2}/{s.primary_total:<2}  "
                f"doc={s.doc_matched:>2}/{s.doc_total:<2}  "
                f"code={s.code_matched:>2}/{s.code_total:<2}  "
                f"indent={s.indentation_violations:>2}  "
                f"halluc={s.hallucinated:>2}"
            )
    return "\n".join(lines)
