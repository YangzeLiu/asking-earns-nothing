#!/usr/bin/env python3
"""P1-y: two further questions on the certified definition-C twin pairs (p2c_cert.py).

The pair set is built exactly as in `p1v_c_companions.py`: `build_pairs()` ->
`content_check()` -> keep the pairs whose base-twin GOLD anchor turn contains a
world-changing call (`any_mut`, login excluded, `echo` decided per call). That is the
231-pair population Table 1 is computed on.

QUESTION 1 — "a second question loses the item"
-----------------------------------------------
Take, per model, the should-ask items on which the model HELD BACK on the anchor turn k
(no world-changing attempt there) and which the OFFICIAL scorer nevertheless FAILS. Where
does the official scorer actually fail? The score row carries the failing turn index j:

  * `multi_turn:empty_turn_model_response`  -> "... for turn {j}" in the error message
  * `multi_turn:execution_response_mismatch`-> "... for turn {j}" in the error message
  * `multi_turn:instance_state_mismatch`    -> NO turn in the message; the checker attaches
        the per-turn `execution_result` list it had accumulated when it bailed out, so
        j = len(error["execution_result"]) - 1 (one entry is appended per turn, including
        the empty-gold anchor turn, before the state check runs)
  * `multi_turn:force_terminated` / `multi_turn:inference_error` -> raised by the runner
        before the per-turn loop, no turn index exists at all

j is then classified against the anchor k:

  (a) j > k, the model's turn j has NO validated call and its prose is non-empty: the model
      spoke instead of calling on a later turn. This is the "second question loses the item"
      channel. Sub-counted: prose on turn j contains "?" or the turn-k reply already did,
      i.e. the item plausibly dies on a repeated clarification request.
  (b) j > k and turn j does have validated calls: a later execution failure, unrelated.
  (c) j == k: should be impossible, the checker `continue`s on empty-gold turns.
  (d) j < k: the item was already lost before the anchor.
  (e) j not recoverable (force_terminated / inference_error).

QUESTION 2 — placeholder / sentinel arguments
---------------------------------------------
Among the world-changing ATTEMPTS on the miss-side anchor turn (the calls that make an item
count as "acting when the information was removed"), how many carry an obviously invented
argument: a string constant that is empty or contains one of "?", "unknown", "placeholder",
"TBD", "N/A", "xxx", "<", ">", "your_", "missing", "example" (case-insensitive except for
the bracket and "?" literals). Arguments are read off the rendered call string from
`full_turn_calls` and parsed with `ast`; every string constant reachable from the call's
arguments is inspected, including strings nested in lists and dicts. The base-side anchor
attempts are measured the same way as a control.

That literal token list is very noisy: any English argument with a question mark in it,
and any long markdown body containing "<" or "?", trips it, and several such arguments are
copied verbatim out of the user's own prompt. A STRICT column is therefore reported
alongside: a string constant that is empty, or is short (<= 60 chars) and either is
bracketed end to end (`<...>`, `[...]`, `{...}`), or consists only of "?"/"-"/"_"
characters, or contains one of "placeholder", "unknown", "tbd", "n/a", "xxx", "your_",
"_here", "to be determined", "to_be_", "fill_in". The strict column is the one to quote.

No LLM anywhere. Output: analysis/p1y_second_question.md
"""
import ast
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import allowed_names  # noqa: E402
from bfcl_calls_full import full_turn_calls  # noqa: E402
from p1d_nonact_rate import PAPER_MODELS as MODELS, find_file  # noqa: E402
from p1f_matched_control import build_pairs, load_results, ASK  # noqa: E402
from p1h_pair_checks import content_check, gold_calls  # noqa: E402
from p1i_holdback_content import prose  # noqa: E402
from p1s_attempt_decision import any_mut, is_mut  # noqa: E402
from p2c_cert import load_final  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TURN_RE = re.compile(r"for turn (\d+)")
SENTINELS = ["?", "unknown", "placeholder", "tbd", "n/a", "xxx", "<", ">",
             "your_", "missing", "example"]
STRICT = ["placeholder", "unknown", "tbd", "n/a", "xxx", "your_", "_here",
          "to be determined", "to_be_", "fill_in"]


def gold_at(gb, pairs, iid):
    bid, k, _ = pairs[iid]
    g = gb.get(bid) or []
    return [(s.split("(")[0].strip(), s) for s in (g[k] if k < len(g) else [])]


def fail_turns(model, cat):
    """-> {failed_item_id: (error_type, failing_turn_index or None)}."""
    p = find_file(model, cat, "score")
    if not p:
        return None
    out = {}
    for i, line in enumerate(open(p)):
        if i == 0 or not line.strip():
            continue
        r = json.loads(line)
        e = r.get("error")
        if not isinstance(e, dict):
            out[r["id"]] = ("unparsed_error", None)
            continue
        et = e.get("error_type", "?")
        j = None
        m = TURN_RE.search(str(e.get("error_message", "")))
        if m:
            j = int(m.group(1))
        elif isinstance(e.get("execution_result"), list):
            j = len(e["execution_result"]) - 1
        out[r["id"]] = (et, j)
    return out


def str_constants(call):
    """Every string constant reachable from the arguments of a rendered call string."""
    try:
        node = ast.parse(call.strip(), mode="eval").body
    except Exception:
        return None
    if not isinstance(node, ast.Call):
        return None
    vals = []
    for sub in list(node.args) + [k.value for k in node.keywords]:
        for n in ast.walk(sub):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                vals.append(n.value)
    return vals


def is_sentinel(call):
    """(loose token hit, strict placeholder hit, the value that tripped strict)."""
    vals = str_constants(call)
    loose, strict, why = False, False, None
    for v in vals or []:
        s = v.strip()
        low = s.lower()
        if s == "" or any(t in low for t in SENTINELS):
            loose = True
        if s == "":
            strict, why = True, why or "<empty string>"
            continue
        if len(s) > 60:
            continue
        bracketed = (s[0], s[-1]) in (("<", ">"), ("[", "]"), ("{", "}"))
        punct_only = set(s) <= set("?-_ ")
        if bracketed or punct_only or any(t in low for t in STRICT):
            strict, why = True, why or s
    return loose, strict, why


def main():
    pairs, _ = build_pairs()
    ok, _, _ = content_check(pairs)
    pairs = {i: pairs[i] for i in ok}
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    gb = gold_calls("base")
    construction, _, _ = load_final()
    C = [i for i in pairs if any_mut(gold_at(gb, pairs, i)) and i not in construction]
    N = len(C)

    L = [f"# P1-y — second-question losses and placeholder arguments ({N} pairs, def. C)",
         "", "Generated by `analysis/p1y_second_question.py`. No judge, no LLM. Pair set "
         "identical to `p1v_c_companions.py`.", ""]

    q1, q2 = {}, {}
    etypes = collections.Counter()
    for m in MODELS:
        res = {c: load_results(m, c) for c in ASK + ["base"]}
        if any(v is None for v in res.values()):
            continue
        ft = {c: fail_turns(m, c) for c in ASK}
        if any(v is None for v in ft.values()):
            continue
        cnt = collections.Counter()
        ex_a = []
        att_m, att_b = [], []
        sen_m, sen_b = [], []
        for iid in C:
            bid, k, cat = pairs[iid]
            sm, sb = res[cat].get(iid), res["base"].get(bid)
            if sm is None or sb is None:
                continue
            cm = full_turn_calls(sm[k] if k < len(sm) else [],
                                 allow[cat].get(iid, set()))
            cb = full_turn_calls(sb[k] if k < len(sb) else [],
                                 allow["base"].get(bid, set()))
            # ---- Q2: sentinel arguments on world-changing attempts
            for calls, att, sen in ((cm, att_m, sen_m), (cb, att_b, sen_b)):
                for n, s in calls:
                    if not is_mut(n, s):
                        continue
                    loose, strict, why = is_sentinel(s)
                    att.append(s)
                    if loose or strict:
                        sen.append((s, loose, strict, why))
            # ---- Q1: held back on the anchor AND officially failed
            if any_mut(cm):
                continue
            if iid not in ft[cat]:
                continue
            cnt["held_fail"] += 1
            et, j = ft[cat][iid]
            etypes[et] += 1
            if j is None:
                cnt["e"] += 1
                continue
            if j < k:
                cnt["d"] += 1
            elif j == k:
                cnt["c"] += 1
            else:
                tj = sm[j] if j < len(sm) else []
                cj = full_turn_calls(tj, allow[cat].get(iid, set()))
                pj = prose(tj)
                if not cj and pj:
                    cnt["a"] += 1
                    if et == "multi_turn:empty_turn_model_response":
                        cnt["a_bfcl"] += 1
                    pk = prose(sm[k] if k < len(sm) else [])
                    if "?" in pj or "?" in pk:
                        cnt["aq"] += 1
                    if len(ex_a) < 3:
                        ex_a.append((iid, k, j, pj[:120].replace("\n", " ")))
                elif cj:
                    cnt["b"] += 1
                else:
                    cnt["a_silent"] += 1
        q1[m] = (cnt, ex_a)
        q2[m] = (att_m, sen_m, att_b, sen_b)

    # ------------------------------------------------------------------ Q1 table
    L += ["## Q1 — where the official scorer actually fails, on held-back items", "",
          "Held back = no world-changing attempt on the anchor turn k. Failed = official "
          "FAIL. (a) = the failing turn j>k has no validated call but does have prose; "
          "(a?) = of those, turn j or turn k contains a question mark; (aE) = of those, BFCL's own error type is `empty_turn_model_response`, i.e. the scorer also saw no decodable call on turn j; (a-silent) = j>k "
          "with neither call nor prose; (b) = j>k with calls; (c) = j==k; (d) = j<k; "
          "(e) = no turn index in the score row (force_terminated / inference_error).", "",
          "| model | held back & failed | (a) | (a?) | (aE) | (a-silent) | (b) | (c) | (d) | (e) "
          f"| (a) as % of all {N} |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    pool = collections.Counter()
    for m, (c, _) in q1.items():
        pool.update(c)
        L.append(f"| {m} | {c['held_fail']} | {c['a']} | {c['aq']} | {c['a_bfcl']} | {c['a_silent']} | "
                 f"{c['b']} | {c['c']} | {c['d']} | {c['e']} | {100*c['a']/N:.1f}% |")
    hf = pool["held_fail"] or 1
    L.append(f"| **pooled** | {pool['held_fail']} | {pool['a']} | {pool['aq']} | "
             f"{pool['a_bfcl']} | {pool['a_silent']} | {pool['b']} | {pool['c']} | {pool['d']} | {pool['e']} "
             f"| {100*pool['a']/(N*len(q1)):.1f}% |")
    L += ["", "Pooled shares of the held-back-and-failed items: "
          f"(a) {100*pool['a']/hf:.1f}%, (a?) {100*pool['aq']/hf:.1f}%, "
          f"(a-silent) {100*pool['a_silent']/hf:.1f}%, (b) {100*pool['b']/hf:.1f}%, "
          f"(c) {100*pool['c']/hf:.1f}%, (d) {100*pool['d']/hf:.1f}%, "
          f"(e) {100*pool['e']/hf:.1f}%.",
          "", "Error types seen on these rows: " +
          ", ".join(f"`{k}` {v}" for k, v in etypes.most_common()), ""]

    L += ["### Example (a) turns", ""]
    for m, (_, ex) in q1.items():
        for iid, k, j, p in ex:
            L.append(f"- `{m}` {iid} anchor k={k}, failing turn j={j}: {p}")
    L.append("")

    # ------------------------------------------------------------------ Q2 table
    L += ["## Q2 — placeholder / sentinel arguments on anchor-turn world-changing attempts",
          "", "Loose tokens: " + ", ".join(f"`{t}`" for t in SENTINELS) +
          ", plus the empty string — very noisy, see the docstring. Strict = empty string, "
          "or a short (<=60 char) value that is bracketed end to end, is punctuation only, "
          "or contains " + ", ".join(f"`{t}`" for t in STRICT) + ". Attempts are counted "
          "per CALL, not per item.", "",
          "| model | miss attempts | miss loose | % | miss strict | % | base attempts | "
          "base loose | % | base strict | % |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()

    def cells(att, sen):
        lo = sum(1 for _, l, _, _ in sen if l)
        st = sum(1 for _, _, s, _ in sen if s)
        n = len(att)
        return n, lo, st, (100*lo/n if n else 0.0), (100*st/n if n else 0.0)

    for m, (am, sm_, ab, sb_) in q2.items():
        n1, l1, s1, p1, r1 = cells(am, sm_)
        n2, l2, s2, p2, r2 = cells(ab, sb_)
        for k_, v_ in zip("n1 l1 s1 n2 l2 s2".split(), (n1, l1, s1, n2, l2, s2)):
            tot[k_] += v_
        L.append(f"| {m} | {n1} | {l1} | {p1:.1f} | {s1} | {r1:.1f} | {n2} | {l2} | "
                 f"{p2:.1f} | {s2} | {r2:.1f} |")
    L.append(f"| **pooled** | {tot['n1']} | {tot['l1']} | "
             f"{100*tot['l1']/max(tot['n1'],1):.1f} | {tot['s1']} | "
             f"{100*tot['s1']/max(tot['n1'],1):.1f} | {tot['n2']} | {tot['l2']} | "
             f"{100*tot['l2']/max(tot['n2'],1):.1f} | {tot['s2']} | "
             f"{100*tot['s2']/max(tot['n2'],1):.1f} |")
    for side, idx, lab in ((0, 1, "miss"), (1, 3, "base")):
        L += ["", f"### Strict-flagged {lab}-side calls (up to 3 per model)", ""]
        for m, tup in q2.items():
            shown = 0
            for s, _, strict, why in tup[idx]:
                if strict and shown < 3:
                    shown += 1
                    L.append(f"- `{m}`: `{s[:100]}`  (tripped by `{why}`)")
    L.append("")

    txt = "\n".join(L)
    open(os.path.join(HERE, "p1y_second_question.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
