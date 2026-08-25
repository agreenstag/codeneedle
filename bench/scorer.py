"""Align model output against ground-truth lines and classify each line."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
import math


PASS_THRESHOLD = 8  # video's threshold: ≥8 of 20 expected lines matched = pass
PASS_RATIO = PASS_THRESHOLD / 20 # per category of doc or code what constitutes a pass , scaled to category size


class LineTag(str, Enum):
    MATCHED = "matched"            # gray — expected (primary) line reproduced
    MISSING = "missing"            # orange — expected (primary) line not produced
    HALLUCINATED = "hallucinated"  # yellow — produced but not in expected window
    BONUS = "bonus"                # blue — produced, correct, past the primary 20
    REINDENTED = "reindented"      # cyan — right content, different leading whitespace

@dataclass
class LineResult:
    tag: LineTag
    text: str


@dataclass
class FunctionScore:
    name: str

    primary_matched: int
    primary_total: int

    doc_matched: int    # docs score split from code
    doc_total: int
    doc_passed: bool

    code_matched: int   # code accuracy separated
    code_total: int
    code_passed: bool

    indentation_violations: int # recall can be good, indentation can be off. 
    hallucinated: int
    bonus_matched: int

    passed: bool
    analysis_tags: list[str]    # Indicator on test state
    template_correction_applied: bool   # Fix for the model chat template causing a score bug.  


    expected_tagged: list[LineResult]   # expected primary side (matched/missing/reindented)
    predicted_tagged: list[LineResult]  # model output side (matched/halluc/bonus/reindented)

    error: str | None = None    # request errored or returned no usable content; renderers should show ERROR instead of FAIL so it isn't confused with a real recall miss
    # A line the model reproduced correctly but indented differently is not a
    # hallucination — it is a formatting difference. Counting it as one both
    # mislabels it and penalizes it twice (the expected line also goes
    # MISSING). Tracked separately so the headline number means what it says.
    reindented: int = 0
    spacing_deviation: bool = False      # True when any line differs only in whitespace

def _split_doc_and_code(lines: list[str]) -> tuple[list[str], list[str]]:
    if not lines:
        return [], []

    first = lines[0].lstrip()

    if not (
        first.startswith('"""')
        or first.startswith("'''")
    ):
        return [], lines

    quote = '"""' if first.startswith('"""') else "'''"

    # Single-line docstring:
    #     """Execute a CGI script."""
    if first.count(quote) >= 2:
        return [lines[0]], lines[1:]

    closing_idx = None

    for idx, line in enumerate(lines[1:], start=1):
        if quote in line:
            closing_idx = idx
            break

    if closing_idx is None:
        return lines, []

    return (
        lines[:closing_idx + 1],
        lines[closing_idx + 1:]
    )

def _count_indentation_violations(
    expected: list[str],
    predicted: list[str],
) -> int:
    violations = 0

    for exp, pred in zip(expected, predicted):
        if (
            exp.lstrip() == pred.lstrip()
            and exp != pred
        ):
            violations += 1

    return violations

def _analysis_tags(
    *,
    primary_matched: int,
    primary_total: int,
    doc_matched: int,
    doc_total: int,
    code_matched: int,
    code_total: int,
    indentation_violations: int,
    hallucinated: int,
    template_correction_applied: bool,
) -> list[str]:
    tags: list[str] = []

    if template_correction_applied:
        tags.append("template_correction_applied")

    if primary_matched == primary_total:
        tags.append("perfect_verbatim_recall")

    if doc_total > 0:
        if doc_matched == 0:
            tags.append("total_docstring_loss")
        elif doc_matched >= math.ceil(doc_total * 0.8):
            tags.append("strong_docstring_recall")
        else:
            tags.append("partial_docstring_loss")

    if code_total > 0:
        if code_matched == 0:
            tags.append("critical_code_recall_failure")
        elif code_matched >= math.ceil(code_total * 0.8):
            tags.append("strong_code_recall")
        else:
            tags.append("partial_code_recall")

    if indentation_violations >= primary_total // 2:
        tags.append("severe_indentation_drift")
    elif indentation_violations > 0:
        tags.append("indentation_drift")

    if hallucinated >= max(5, primary_total // 2):
        tags.append("high_hallucination_rate")
    elif hallucinated > 0:
        tags.append("hallucinations_present")


   
    return tags

def _is_leading_docstring_indent_loss(
    *,
    primary_matched: int,
    primary_total: int,
    indentation_violations: int,
    reindented_exp: set[int],
    expected_display: list[str],
    pred_display: list[str],
) -> bool:
    if primary_matched != primary_total - 1:
        return False

    if indentation_violations != 1:
        return False

    if 0 not in reindented_exp:
        return False

    if not expected_display or not pred_display:
        return False

    expected_first = expected_display[0]
    predicted_first = pred_display[0]

    if not expected_first.lstrip().startswith(('"""', "'''")):
        return False

    return (
        expected_first.startswith((" ", "\t"))
        and not predicted_first.startswith((" ", "\t"))
        and expected_first.lstrip() == predicted_first.lstrip()
    )


def score(
    name: str,
    primary: list[str],
    bonus: list[str],
    predicted_text: str,
    relax_indent: bool = False,
) -> FunctionScore:
    """Score a single function's predicted output against expected lines.

    `relax_indent=True` normalizes both sides with `.strip()` instead of
    `.rstrip()` only — i.e. leading whitespace is ignored when matching. Use
    this for models like Gemma that emit semantically-correct code but
    normalize indentation, where strict verbatim matching would unfairly
    penalize content the model actually got right. Default is strict.
    """
    predicted = _clean_output(predicted_text)
    norm = _norm_relaxed if relax_indent else _norm

    exp_primary = [norm(l) for l in primary]
    exp_bonus = [norm(l) for l in bonus]
    exp_full = exp_primary + exp_bonus
    pred = [norm(l) for l in predicted]

    # trim trailing blank lines on prediction (common model artifact)
    while pred and pred[-1] == "":
        pred.pop()

    # scores only on scoring lines. 
    primary_pred_limit = min(len(pred), len(exp_primary))

    sm = SequenceMatcher(a=exp_full, b=pred, autojunk=False)

    matched_exp = [False] * len(exp_full)
    # -2 = indentation mismatch
    # -1 = hallucinated
    #  0 = primary match
    #  1 = bonus match

    pred_kind = [-1] * len(pred)

    for block in sm.get_matching_blocks():
        if block.size == 0:
            continue
        for i in range(block.size):
            ei = block.a + i
            pi = block.b + i
            matched_exp[ei] = True
            pred_kind[pi] = 0 if ei < len(exp_primary) else 1

    # Whitespace-only differences are not hallucinations. Under strict scoring
    # a re-indented line fails to align, so it lands in the unmatched bucket on
    # BOTH sides — the expected line reads MISSING and the emitted line reads
    # HALLUCINATED, for what is a single formatting difference. Pair those two
    # back up and label them for what they are.
    #
    # Only meaningful in strict mode: with relax_indent the lines already
    # aligned, so nothing is left over to pair.
    reindented_exp: set[int] = set()

    if not relax_indent:
        unmatched_exp: dict[str, list[int]] = {}

        for i, is_matched in enumerate(matched_exp[:len(exp_primary)]):
            if not is_matched:
                key = exp_primary[i].strip()

                if key:
                    unmatched_exp.setdefault(key, []).append(i)

        for pi, kind in enumerate(pred_kind[:primary_pred_limit]):
            if kind != -1:
                continue

            key = pred[pi].strip()
            candidates = unmatched_exp.get(key)

            if not candidates:
                continue

            ei = candidates.pop(0)
            reindented_exp.add(ei)
            pred_kind[pi] = 2

    primary_matched = sum(1 for i in range(len(exp_primary)) if matched_exp[i])
    doc_expected, code_expected = _split_doc_and_code(primary)

    # added section for counting docs and code totals. 
    doc_total = len(doc_expected)
    code_total = len(code_expected)

    assert doc_total + code_total == len(primary), (
        f"{name}: doc/code split mismatch "
        f"{doc_total}+{code_total}!={len(primary)}"
    )

    # Overall scoring remains strict and uses matched_exp only.
    # Doc and code recall measure whether the content was recalled, so they
    # include exact matches and lines reproduced with different indentation.
    recalled_exp = {
        i
        for i in range(len(exp_primary))
        if matched_exp[i] or i in reindented_exp
    }

    doc_matched = sum(
        1
        for i in range(doc_total)
        if i in recalled_exp
    )

    code_matched = sum(
        1
        for i in range(doc_total, doc_total + code_total)
        if i in recalled_exp
    )

    bonus_matched = sum(
        1 for i in range(len(exp_primary), len(exp_full)) if matched_exp[i]
    )

    hallucinated = sum(1 for k in pred_kind if k == -1)

    # Blank lines shouldn't count as hallucinations (models often insert them).
    hallucinated -= sum(
        1 for i, k in enumerate(pred_kind) if k == -1 and pred[i].strip() == ""
    )

    # Display the ORIGINAL lines (with their actual indentation), not the
    # normalized form used for matching. Otherwise indent-relaxed scoring
    # would render every line lstripped, hiding the model's real output.
    expected_display = [l.rstrip() for l in primary]
    pred_display = [l.rstrip() for l in _clean_output(predicted_text)]
    while pred_display and pred_display[-1] == "":
        pred_display.pop()
    if len(pred_display) != len(pred):
        # Defensive: alignment of pred_display to pred should match because
        # both started from the same _clean_output and stripped trailing blanks.
        pred_display = pred_display[: len(pred)] + [""] * max(0, len(pred) - len(pred_display))


    # Count indentation violations as opposed to code mismatch or hallucinations
    indentation_violations = _count_indentation_violations(
        expected_display,
        pred_display[:primary_pred_limit],
    )

    # Set the correct display type pred, handle indentations
    for i, (exp, pred_line) in enumerate(
        zip(expected_display, pred_display)
    ):
        if (
            exp.lstrip() == pred_line.lstrip()
            and exp != pred_line
        ):
            if i < len(pred_kind) and pred_kind[i] == -1:
                pred_kind[i] = -2

    for i, pred_line in enumerate(pred_display[:primary_pred_limit]):
        if pred_kind[i] != -1:
            continue
        for exp_line in expected_display:
            if pred_line and pred_line in exp_line:
                pred_kind[i] = -2   # alignment issue
                break
    
    # Blank lines shouldn't count as hallucinations (models often insert them).
    hallucinated = sum(1 for i, kind in enumerate(pred_kind[:primary_pred_limit])
        if kind == -1 and pred[i].strip()
    )
        
    # Fix for chat model template always indenting first line, which introduces a false positive failure. 
    template_correction_applied = _is_leading_docstring_indent_loss(
        primary_matched=primary_matched,
        primary_total=len(exp_primary),
        indentation_violations=indentation_violations,
        reindented_exp=reindented_exp,
        expected_display=expected_display,
        pred_display=pred_display,
    )

    template_indent_correction = 0

    if template_correction_applied:
        primary_matched += 1
        indentation_violations -= 1
        template_indent_correction = 1
        matched_exp[0] = True

        if pred_kind and pred_kind[0] == -2:
            pred_kind[0] = 0

    reindented = sum(1 for kind in pred_kind[:primary_pred_limit]
        if kind in (2, -2)
    )

    doc_passed = (
        doc_total == 0
        or doc_matched >= math.ceil(doc_total * PASS_RATIO)
    )

    code_passed = (
        code_total == 0
        or code_matched >= math.ceil(code_total * PASS_RATIO)
    )

    expected_tagged = []
    for i in range(len(exp_primary)):
        if matched_exp[i]:
            tag = LineTag.MATCHED
        elif i in reindented_exp:
            tag = LineTag.REINDENTED
        else:
            tag = LineTag.MISSING
        expected_tagged.append(LineResult(tag, expected_display[i]))

    kind_to_tag = {
        0: LineTag.MATCHED,
        1: LineTag.BONUS,
        2: LineTag.REINDENTED,
        -1: LineTag.HALLUCINATED,
        -2: LineTag.REINDENTED,
    }
    predicted_tagged = [
        LineResult(kind_to_tag[pred_kind[i]], pred_display[i]) for i in range(len(pred))
    ]

    analysis_tags = _analysis_tags(
        primary_matched=primary_matched,
        primary_total=len(exp_primary),

        doc_matched=doc_matched,
        doc_total=doc_total,

        code_matched=code_matched,
        code_total=code_total,

        indentation_violations=indentation_violations,
        hallucinated=hallucinated,
        template_correction_applied=template_correction_applied,
    )

    return FunctionScore(
        name=name,

        primary_matched=primary_matched,
        primary_total=len(exp_primary),

        # updated to handle docs and code scores
        doc_matched=doc_matched,
        doc_total=doc_total,
        doc_passed=doc_passed,

        code_matched=code_matched,
        code_total=code_total,
        code_passed=code_passed,

        indentation_violations=indentation_violations,
        hallucinated=hallucinated,
        bonus_matched=bonus_matched,
        # Strict scoring still means verbatim: a re-indented line is NOT a
        # match, so pass/fail verdicts and matched counts are unchanged and
        # remain comparable with earlier runs. Use --relax-indent to score
        # these as matches.
        passed=primary_matched >= PASS_THRESHOLD,
        template_correction_applied=template_correction_applied,

        analysis_tags=analysis_tags,

        expected_tagged=expected_tagged,
        predicted_tagged=predicted_tagged,
        reindented=reindented,
        spacing_deviation=reindented > 0,
    )


def _norm(s: str) -> str:
    # Indent scoring indicates now where a problem is, if code is remembered but the indent is wrong. 
    return s.rstrip()


def _norm_relaxed(s: str) -> str:
    # Used when scoring indent-blind. Strips both leading and trailing whitespace.
    # Internal whitespace is preserved so things like `a    b` stay distinct from `a b`.
    return s.strip()


def _clean_output(text: str) -> list[str]:
    """Strip markdown fences and surrounding blank lines. Tolerant of prefix commentary."""
    lines = text.splitlines()

    # If the model wrapped output in a fenced code block, extract the fence contents.
    fence_idxs = [i for i, l in enumerate(lines) if l.lstrip().startswith("```")]
    if len(fence_idxs) >= 2:
        lines = lines[fence_idxs[0] + 1 : fence_idxs[-1]]
    else:
        # Drop any stray fence markers
        lines = [l for l in lines if not l.lstrip().startswith("```")]

    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines
