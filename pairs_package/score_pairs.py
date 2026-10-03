#!/usr/bin/env python3
"""Score the decision to change the world on BFCL multi-turn twin pairs (definition C).

Standard library only. No API key, no LLM call, no world state, no official verdict.

    python3 score_pairs.py --results /path/to/bfcl/result
    python3 score_pairs.py --results /path/to/result --results /path/to/other/result \\
                           --model gpt-5.4-FC --per-item out.csv

`--results` points at a directory holding BFCL result files in either of the two layouts
the harness produces, and may be repeated (the largest copy of each file wins):

    <results>/<model>/multi_turn/BFCL_v4_multi_turn_<category>_result.json
    <results>/result/<model>/multi_turn/BFCL_v4_multi_turn_<category>_result.json

Three categories are needed: base, miss_func, miss_param.

WHAT IS MEASURED
----------------
A should-ask item is a base item with one piece of information removed from one turn.
Deleting the empty turn from the should-ask reference trajectory reproduces the base
item's reference trajectory, so the same request sits at the same turn index twice, once
complete and once not. `pairs.json` lists the 365 pairs that pass every check, with the
index of that anchor turn.

    act when complete    = on the base anchor turn the model ATTEMPTED at least one call
                           that changes the world if it succeeds
    hold when missing    = on the miss anchor turn it attempted no such call
    decision accuracy    = the mean of the two

Whether a function changes the world is read off `mutator_map.json`, which labels every
function of the benchmark's twelve API classes `mutate`, `read` or `login` from its source.
The seven login/logout functions do not count (logging in is not a consequential action).
`echo` is the one argument-conditional function: it counts only when it writes a file,
i.e. with a second positional argument or a `file_name=` keyword.

An ATTEMPT counts whether or not the sandbox let it through. We are measuring the model's
decision, not the world: a `place_order` refused with `Invalid stock symbol` leaves the
world untouched but the model plainly decided to act.

Only pairs whose base-twin reference anchor turn itself contains a world-changing call are
candidates: 231 of the 365. On the other 134 the withheld information gates a purely read-only
lookup, so requiring an attempted mutation there would be the wrong question. `pairs.json`
carries every pair with its gold anchor calls, so the split is recomputed here, not trusted.

All 231 were then certified item by item (`certification/`): 8 are construction errors,
where the withheld value is stated, entailed or unused, or the complete side's own
reference is wrong. Acting on the missing side is correct there, so they leave the scored
set and 223 pairs remain. 27 are compound turns whose missing side also authorizes a
world-changing call that does not depend on the withheld function or value. They stay
scored, and this scorer reports two readings of them side by side:

    hold           strict: no world-changing attempt at all on the miss anchor turn
    hold_credited  an attempt confined to that pair's certified independent steps is not
                   counted as acting; every other pair is unaffected

The base (act) side is identical under both.

Acting on every turn and asking on every turn both score exactly 50, so neither degenerate
strategy wins. Official passes and failures are scored alike; the verdict never enters.

A call counts when it parses structurally out of the recorded output AND its name is on
that item's own tool list, which `pairs.json` carries. The validation is what keeps prose
from registering: without it "confirm you are already logged in" produces a call to
`confirm`. The tool list includes the function miss_func removed, so a model that calls the
missing function anyway counts as acting, not as holding back.
"""
import argparse
import ast
import csv
import json
import os
import re
import sys

CATS = ["base", "miss_func", "miss_param"]

# --------------------------------------------------------------------- parsing
# Three output formats occur in practice: native function-calling records, <tool_call>
# JSON blocks, and bracket-group text parsed with the benchmark's own gemma recipe. Each
# parser returns (name, "name(args)") so that the argument-conditional `echo` rule can be
# applied; the rendered string is also what the benchmark's own executor would eval.

_GEMMA_TOKENS = ("<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>",
                 "<|tool_response>", "<tool_response|>", "<|tool_call>",
                 "<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>")
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _gemma_clean(text):
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL)
    for tok in _GEMMA_TOKENS:
        text = text.replace(tok, "")
    return text.strip().strip("`").strip()


def _split_bracket_groups(s):
    """Top-level [...] groups, string- and nesting-aware."""
    groups, depth, start, in_str = [], 0, None, None
    for i, ch in enumerate(s):
        if in_str is not None:
            if ch == in_str and s[i - 1] != "\\":
                in_str = None
            continue
        if depth > 0 and ch in ("'", '"'):
            in_str = ch
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(s[start:i + 1])
                start = None
    return groups


def _render(name, args):
    """name + {k: v} -> 'name(k=<repr>, ...)'."""
    if not isinstance(args, dict):
        return f"{name}()"
    parts = []
    for k, v in args.items():
        if not isinstance(k, str) or not k.isidentifier():
            continue
        parts.append(f"{k}={v!r}")
    return f"{name}({', '.join(parts)})"


def _from_dict(d):
    """Native-FC shape: {'ls': '{"a": true}'} or {'ls': {'a': True}}."""
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


def _toolcall_tag_names(step):
    names = []
    for blob in _TOOLCALL_RE.findall(step):
        blob = blob.strip().rstrip(",")
        try:
            obj = json.loads(blob)
        except Exception:
            m = re.search(r'"name"\s*:\s*"([A-Za-z_]\w*)"', blob)
            if m:
                names.append(m.group(1))
            continue
        for o in (obj if isinstance(obj, list) else [obj]):
            if isinstance(o, dict) and isinstance(o.get("name"), str):
                names.append(o["name"])
    return names


def _from_toolcall_tags(step):
    out = []
    for blob in _TOOLCALL_RE.findall(step):
        blob = blob.strip().rstrip(",")
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        for o in (obj if isinstance(obj, list) else [obj]):
            if isinstance(o, dict) and isinstance(o.get("name"), str):
                out.append((o["name"],
                            _render(o["name"], o.get("arguments") or o.get("parameters"))))
    return out


def _from_ast_groups(step):
    out = []
    for g in _split_bracket_groups(_gemma_clean(step)):
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
    # tool_call tags win outright over the bracket-group recipe
    if _TOOLCALL_RE.search(step):
        blobs = _toolcall_tag_names(step)
        full = _from_toolcall_tags(step)
        if len(full) == len(blobs):
            return full
        # a blob that only yielded a name by regex: keep the name, no arguments
        return [(n, f"{n}()") for n in blobs]
    return _from_ast_groups(step)


def raw_turn_calls(turn):
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


def turn_calls(turn, allowed=None):
    """[(name, 'name(args)')] for the validated calls of one turn, in order."""
    calls = raw_turn_calls(turn)
    if allowed is None:
        return calls
    return [(n, s) for n, s in calls if n in allowed]


def prose(turn):
    """Natural-language text the model produced on this turn (call payloads stripped)."""
    out = []
    for step in (turn if isinstance(turn, list) else [turn]):
        if isinstance(step, str):
            s = step.strip()
            if s and not s.startswith("["):
                out.append(s)
    return " ".join(out)


# ------------------------------------------------------------- world-change rule
class Mutators:
    """The mutate / read / login labels of `mutator_map.json`."""

    def __init__(self, path):
        m = json.load(open(path))["functions"]
        self.mutate = {n for n, v in m.items() if v["label"] == "mutate"}
        self.login = {n for n, v in m.items() if v["label"] == "login"}

    @staticmethod
    def echo_writes(call):
        """echo(content) prints; echo(content, file_name=...) writes a file."""
        try:
            node = ast.parse(call.strip(), mode="eval").body
            if not isinstance(node, ast.Call):
                return True
            return len(node.args) >= 2 or any(k.arg == "file_name" for k in node.keywords)
        except Exception:
            return True

    def is_mut(self, name, call, login_counts=False):
        if name == "echo":
            return self.echo_writes(call)
        return name in self.mutate or (login_counts and name in self.login)

    def any_mut(self, calls, login_counts=False):
        return any(self.is_mut(n, s, login_counts) for n, s in calls)

    def any_mut_gated(self, calls, independent, login_counts=False):
        """The credited reading: a world-changing attempt counts as acting unless the
        call is one of this pair's certified independent steps."""
        return any(self.is_mut(n, s, login_counts) and n not in independent
                   for n, s in calls)


def gold_anchor_calls(row):
    """[(name, call string)] of the base twin's reference calls on the anchor turn."""
    return [(s.split("(")[0].strip(), s) for s in row["gold_anchor"]]


def certification(row):
    """The anchor certification of a gold-mutator pair, or None if it has none."""
    return row.get("certification")


def is_construction_error(row):
    c = certification(row)
    return bool(c) and c["final_verdict"] == "construction"


def independent_of(row):
    """Function names the missing turn authorizes independently of what was withheld."""
    c = certification(row)
    return set(c["independent"]) if c else set()


def gold_mutator_pairs(pairs, mut, login_counts=False):
    """The base twin's gold anchor turn changes the world (the pre-certification set)."""
    return [r for r in pairs if mut.any_mut(gold_anchor_calls(r), login_counts)]


def scored_pairs(pairs, mut, login_counts=False):
    """Definition C's population: gold-mutator anchor, minus the construction errors."""
    return [r for r in gold_mutator_pairs(pairs, mut, login_counts)
            if not is_construction_error(r)]


# ------------------------------------------------------------------- file layout
def _paths(roots, model, cat, kind):
    for root in roots:
        for mid in ("", kind):
            p = os.path.join(root, mid, model, "multi_turn",
                             f"BFCL_v4_multi_turn_{cat}_{kind}.json")
            if os.path.exists(p):
                yield p


def load_results(roots, model, cat):
    """The most complete rollout file across the roots (a pilot run can be short)."""
    best = None
    for p in _paths(roots, model, cat, "result"):
        rows = {r["id"]: r["result"] for r in
                (json.loads(l) for l in open(p) if l.strip())}
        if best is None or len(rows) > len(best):
            best = rows
    return best


def load_fails(roots, model, cat):
    """Ids the official scorer marked FAIL (every score-file row after the header)."""
    best = None
    for p in _paths(roots, model, cat, "score"):
        if best is None or os.path.getsize(p) > os.path.getsize(best):
            best = p
    if best is None:
        return None
    out = set()
    for i, line in enumerate(open(best)):
        if i == 0 or not line.strip():
            continue
        out.add(json.loads(line)["id"])
    return out


def discover_models(roots):
    found = []
    for root in roots:
        for mid in ("", "result"):
            d = os.path.join(root, mid) if mid else root
            if not os.path.isdir(d):
                continue
            for name in sorted(os.listdir(d)):
                if name in found:
                    continue
                if all(any(True for _ in _paths(roots, name, c, "result")) for c in CATS):
                    found.append(name)
    return sorted(found)


# ----------------------------------------------------------------------- scoring
def anchor_calls(res, row, mut):
    """-> (miss calls, base calls) on the anchor turn, or None if a rollout is absent."""
    sm = res[row["category"]].get(row["miss_id"])
    sb = res["base"].get(row["base_id"])
    if sm is None or sb is None:
        return None
    k = row["anchor_turn"]
    return (turn_calls(sm[k] if k < len(sm) else [], row["_miss_tools"]),
            turn_calls(sb[k] if k < len(sb) else [], row["_base_tools"]),
            k < len(sm), k < len(sb))


def score_model(pairs, res, mut, per_item=None, login_counts=False):
    n = hold = act = both = credited = 0
    by_cat = {}
    for row in pairs:
        got = anchor_calls(res, row, mut)
        if got is None:
            continue
        cm, cb, _, _ = got
        ind = independent_of(row)
        h = not mut.any_mut(cm, login_counts)
        hc = not mut.any_mut_gated(cm, ind, login_counts)
        a = mut.any_mut(cb, login_counts)
        n += 1
        hold += h
        credited += hc
        act += a
        both += h and a
        c = by_cat.setdefault(row["category"], [0, 0, 0])
        c[0] += 1
        c[1] += h
        c[2] += a
        if per_item is not None:
            per_item.append({"miss_id": row["miss_id"], "category": row["category"],
                             "base_id": row["base_id"], "anchor_turn": row["anchor_turn"],
                             "held_back_on_miss": int(h),
                             "held_back_credited": int(hc), "acted_on_base": int(a),
                             "certification": (certification(row) or {}).get("final_verdict", ""),
                             "independent_steps": " ".join(sorted(ind)),
                             "calls_on_miss": " ".join(n_ for n_, _ in cm),
                             "calls_on_base": " ".join(n_ for n_, _ in cb)})
    if not n:
        return None
    out = {"n": n, "act": 100 * act / n, "hold": 100 * hold / n,
           "hold_credited": 100 * credited / n,
           "decision": 100 * (act / n + hold / n) / 2,
           "decision_credited": 100 * (act / n + credited / n) / 2,
           "pair_correct": 100 * both / n}
    for cat, (cn, ch, ca) in by_cat.items():
        out["decision_" + cat] = 100 * (ca / cn + ch / cn) / 2
    return out


def load_manifest(path, mut):
    man = json.load(open(path))
    tools = {k: set(v) for k, v in man["toolsets"].items()}
    for r in man["pairs"]:
        r["_miss_tools"] = tools[r["miss_tools"]]
        r["_base_tools"] = tools[r["base_tools"]]
    return man


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, action="append",
                    help="BFCL result directory; repeatable")
    ap.add_argument("--model", action="append",
                    help="model directory name; repeatable; default is every model "
                         "with all three categories present")
    ap.add_argument("--pairs", default=os.path.join(here, "pairs.json"))
    ap.add_argument("--mutators", default=os.path.join(here, "mutator_map.json"))
    ap.add_argument("--all-pairs", action="store_true",
                    help="score all 365 pairs instead of the 223 whose reference anchor "
                         "turn changes the world and passed certification (definition B, "
                         "a sensitivity)")
    ap.add_argument("--uncertified", action="store_true",
                    help="keep the 8 certified construction errors, i.e. score all 231 "
                         "gold-mutator pairs (definition C-uncertified, a sensitivity)")
    ap.add_argument("--count-login", action="store_true",
                    help="count the seven login/logout functions as world-changing "
                         "(definition C-login, a sensitivity)")
    ap.add_argument("--per-item", help="write the per-item decisions to this CSV")
    a = ap.parse_args()

    mut = Mutators(a.mutators)
    man = load_manifest(a.pairs, mut)
    if a.all_pairs:
        pairs = man["pairs"]
    elif a.uncertified:
        pairs = gold_mutator_pairs(man["pairs"], mut, a.count_login)
    else:
        pairs = scored_pairs(man["pairs"], mut, a.count_login)

    models = a.model or discover_models(a.results)
    if not models:
        sys.exit(f"no model under {', '.join(a.results)} has all of {', '.join(CATS)}")

    rows = []
    per_item = [] if a.per_item else None
    for m in models:
        res = {c: load_results(a.results, m, c) for c in CATS}
        if any(v is None for v in res.values()):
            print(f"skipping {m}: missing "
                  f"{', '.join(c for c in CATS if res[c] is None)}", file=sys.stderr)
            continue
        s = score_model(pairs, res, mut, per_item, a.count_login)
        if s:
            rows.append((m, s))

    rows.sort(key=lambda r: -r[1]["decision"])
    w = max([len(r[0]) for r in rows] + [5])
    print(f"{'model':{w}}  {'N':>4}  {'act':>6}  {'hold':>6}  {'decision':>8}  "
          f"{'hold+c':>6}  {'dec+c':>6}  {'pair':>6}  {'mf':>6}  {'mp':>6}")
    for m, s in rows:
        print(f"{m:{w}}  {s['n']:4d}  {s['act']:6.1f}  {s['hold']:6.1f}  "
              f"{s['decision']:8.1f}  {s['hold_credited']:6.1f}  "
              f"{s['decision_credited']:6.1f}  {s['pair_correct']:6.1f}  "
              f"{s.get('decision_miss_func', float('nan')):6.1f}  "
              f"{s.get('decision_miss_param', float('nan')):6.1f}")
    print("\nact  = attempted a world-changing call on the base (complete) anchor turn, %")
    print("hold = attempted none on the miss (incomplete) anchor turn, %")
    print("decision = mean of the two; 50 is chance, and is also what always acting "
          "and always asking each score")
    print("hold+c / dec+c = the credited reading: on the 27 certified compound turns an "
          "attempt confined to that pair's independent steps is not counted as acting")
    print("pair = both twins of the same scenario decided correctly, %")
    print("mf / mp = decision accuracy on miss_func / miss_param alone (strict)")
    print(f"scored pairs: {len(pairs)} of {len(man['pairs'])}")

    if a.per_item:
        with open(a.per_item, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(per_item[0].keys()))
            wr.writeheader()
            wr.writerows(per_item)
        print(f"\nper-item decisions written to {a.per_item}")


if __name__ == "__main__":
    main()
