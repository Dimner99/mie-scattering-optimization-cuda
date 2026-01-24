from __future__ import annotations

import re
from dataclasses import dataclass

from sympy.parsing.mathematica import parse_mathematica


@dataclass(frozen=True)
class GradientExpressions:
    """Container for parsed gradient expressions."""
    term_tm: object | None
    term_te: object | None
    dq_tm_deps_re_n: object | None
    dq_tm_deps_im_n: object | None
    dq_tm_dr_n: object | None
    dq_te_deps_re_n: object | None
    dq_te_deps_im_n: object | None
    dq_te_dr_n: object | None


_ASSIGN_RE = re.compile(r"([A-Za-z0-9_]+)\s*=\s*(.*?)\s*;", re.DOTALL)


def parse_wolfram_gradients(text: str) -> GradientExpressions:

    # Handle line continuations
    normalized = text.replace("\\\n", "\n")
    
    matches = _ASSIGN_RE.findall(normalized)
    if not matches:
        raise ValueError("No assignments found. Expected lines like: name = expr;")

    parsed = {}
    for name, expr in matches:
        if name in {"term_tm", "term_te"}:
            parsed[name] = None
            continue
        parsed[name] = parse_mathematica(expr)

    return GradientExpressions(
        term_tm=parsed.get("term_tm"),
        term_te=parsed.get("term_te"),
        dq_tm_deps_re_n=parsed.get("dq_tm_deps_re_n"),
        dq_tm_deps_im_n=parsed.get("dq_tm_deps_im_n"),
        dq_tm_dr_n=parsed.get("dq_tm_dr_n"),
        dq_te_deps_re_n=parsed.get("dq_te_deps_re_n"),
        dq_te_deps_im_n=parsed.get("dq_te_deps_im_n"),
        dq_te_dr_n=parsed.get("dq_te_dr_n"),
    )


def parse_wolfram_gradients_file(path: str) -> GradientExpressions:
    with open(path, "r", encoding="utf-8") as f:
        return parse_wolfram_gradients(f.read())

