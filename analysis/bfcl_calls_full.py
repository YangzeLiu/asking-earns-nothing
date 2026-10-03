#!/usr/bin/env python3
"""Full call strings (name AND arguments) for one turn.

`bfcl_calls.turn_calls` returns function NAMES only. That is all the sealed Phase 5 /
PASS-audit / p1d metrics ever needed, and it stays untouched. But a name cannot tell you
whether `echo` wrote to a file or to the terminal, and it cannot be executed. This module
mirrors the three parsers in `bfcl_calls` and returns something you can hand to BFCL's own
executor.

Self-check (`python3 bfcl_calls_full.py`): for every model, category and turn in the local
rollouts, the sequence of names recovered here must equal `bfcl_calls.turn_calls` exactly.
If it ever does not, this module is wrong, not the sealed one.
"""
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bfcl_calls as B  # noqa: E402


def _render(name, args):
    """name + {k: v} -> 'name(k=<repr>, ...)', a string BFCL's eval() can run."""
    if not isinstance(args, dict):
        return f"{name}()"
    parts = []
    for k, v in args.items():
        if not isinstance(k, str) or not k.isidentifier():
            continue
        parts.append(f"{k}={v!r}")
    return f"{name}({', '.join(parts)})"


def _from_dict(d):
    """The native-FC rollout shape: {'ls': '{\"a\": true}'} or {'ls': {'a': True}}."""
    out = []
    for name, raw in d.items():
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
            except Exception:
                args = {}
        elif isinstance(raw, dict):
            args = raw
        else:
            args = {}
        out.append((name, _render(name, args)))
    return out


def _from_toolcall_tags(step):
    out = []
    for blob in B._TOOLCALL_RE.findall(step):
        blob = blob.strip().rstrip(",")
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        for o in (obj if isinstance(obj, list) else [obj]):
            if isinstance(o, dict) and isinstance(o.get("name"), str):
                out.append((o["name"], _render(o["name"], o.get("arguments") or o.get("parameters"))))
    return out


def _from_ast_groups(step):
    out = []
    cleaned = B._gemma_clean(step)
    for g in B._split_bracket_groups(cleaned):
        try:
            parsed = ast.parse(g.strip().strip("'"), mode="eval")
        except Exception:
            continue
        body = parsed.body
        elems = body.elts if isinstance(body, (ast.List, ast.Tuple)) else [body]
        for e in elems:
            if not isinstance(e, ast.Call):
                continue
            f = e.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name:
                out.append((name, ast.unparse(e)))
    return out


def _from_string(step):
    # same precedence as bfcl_calls._string_call_names: tool_call tags win outright
    if B._TOOLCALL_RE.search(step):
        blobs = B._toolcall_tag_names(step)
        full = _from_toolcall_tags(step)
        if len(full) == len(blobs):
            return full
        # a blob that only yielded a name by regex: keep the name, no arguments
        return [(n, f"{n}()") for n in blobs]
    return _from_ast_groups(step)


def raw_full_calls(turn):
    out = []
    for step in (turn if isinstance(turn, list) else [turn]):
        if isinstance(step, str):
            out.extend(_from_string(step))
        elif isinstance(step, list):
            for c in step:
                if isinstance(c, dict):
                    out.extend(_from_dict(c))
                elif isinstance(c, str):
                    out.extend(_from_string(c))
        elif isinstance(step, dict):
            out.extend(_from_dict(step))
    return out


def full_turn_calls(turn, allowed=None):
    """[(name, 'name(args)')] for the validated calls of one turn, in order."""
    calls = raw_full_calls(turn)
    if allowed is None:
        return calls
    return [(n, s) for n, s in calls if n in allowed]


# --------------------------------------------------------------------- self-check
def _selfcheck():
    from p1d_nonact_rate import MODELS
    from p1f_matched_control import load_results
    cats = ["base", "miss_func", "miss_param"]
    bad = tot = turns = 0
    for m in MODELS:
        for cat in cats:
            rows = load_results(m, cat)
            if not rows:
                continue
            for iid, seq in rows.items():
                for turn in seq:
                    turns += 1
                    a = B.turn_calls(turn, None)
                    b = [n for n, _ in raw_full_calls(turn)]
                    tot += len(a)
                    if a != b:
                        bad += 1
                        if bad <= 5:
                            print(f"  MISMATCH {m} {iid}\n    sealed={a}\n    full  ={b}")
    print(f"turns checked: {turns}, calls: {tot}, mismatching turns: {bad}")
    return bad == 0


if __name__ == "__main__":
    sys.exit(0 if _selfcheck() else 1)
