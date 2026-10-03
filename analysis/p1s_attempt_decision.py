#!/usr/bin/env python3
"""P1-s: does the model DECIDE to change the world on the under-specified turn?

WHY (2026-09-16)
----------------
An "executed" definition decides holding back by replaying the
model's calls and diffing BFCL's world state. A mutating call that left the world
unchanged therefore reads as restraint: gpt-5.4's `place_order(symbol="PROM")` is refused
with `Invalid stock symbol`, the state does not move, and the model is credited for
holding back although it plainly decided to act. Across the seven models 262 mutating
calls on the information-removed turn left the world unchanged this way, rejected or
no-ops (`startEngine` 56, `cp` 21, `post_tweet` 21, `mv` 18...), inside 173 of the 700
anchor turns that carried a mutation attempt at all.

We measure the DECISION, so an attempt counts whether or not it landed. Whether a
function changes state on success is read off the source once, in `mutator_map.json`
(built and then checked by the authors); `echo` is the only
argument-conditional function in the suite and is decided per call. The seven
login/logout functions do write state_checker-visible attributes; they are carved out
because logging in is not a consequential action, and the login-counted variant is
reported below as a sensitivity.

FOUR DEFINITIONS, and why only two of them are admissible
---------------------------------------------------------
Writing `act` for the base twin's anchor turn and `hold` for the miss twin's:

  A  act = any call            hold = no mutating call     REJECTED, see below
  B  act = attempted a mutator hold = no mutating call      symmetric, all 365
  C  act = attempted a mutator hold = no mutating call      symmetric, gold-mutator items only
  D  act = any call            hold = no call at all        symmetric, all 365 (published)

A is not admissible. Its two sides ask different questions, so the strategy "emit one
read-only call on every turn" scores act=100, hold=100, accuracy 100 without ever
deciding anything, and the chance-50 property the metric is built on is gone. The
degenerate-strategy table at the end is computed, not hand-written.

B is admissible but mis-specified: 132 of the 365 base twins have a gold anchor turn
that is purely read-only (`get_zipcode_based_on_city` 39x, `estimate_distance` 20x,
`get_stock_info` 14x, `cd` 12x), and on those items requiring the model to attempt a
mutation is simply wrong. C restricts to the pairs where the withheld information
actually gates a world-changing call, which is the population the question is about.

ANCHOR CERTIFICATION (2026-09-16 night, p2c_cert.py)
----------------------------------------------------
All 231 definition-C pairs were certified by hand by the authors against a fixed set of
questions, reading the item alone and no model output (p2c_cert.py). 8 pairs are
construction errors (the withheld value is stated, entailed, or unused; or the complete
side's reference is wrong) and leave the scored set: N = 223. 27 pairs are compound turns
whose missing side also authorizes a world-changing call that does not depend on the
withheld function/value; they stay in the strict column, and the credited column does not
count an attempt that stays inside those independent steps as acting.

Output: analysis/p1s_attempt_decision.md
"""
import ast
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import allowed_names  # noqa: E402
from bfcl_calls_full import full_turn_calls  # noqa: E402
from p1d_nonact_rate import PAPER_MODELS as MODELS, SEED, B  # noqa: E402
from p1f_matched_control import build_pairs, load_results, ASK  # noqa: E402
from p1h_pair_checks import content_check, gold_calls  # noqa: E402
from p2c_cert import load_final, gated  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = json.load(open(os.path.join(HERE, "mutator_map.json")))["functions"]
MUT = {n for n, v in MAP.items() if v["label"] == "mutate"}
LOGIN = {n for n, v in MAP.items() if v["label"] == "login"}


def echo_writes(s):
    """echo(content) prints; echo(content, file_name=...) writes a file."""
    try:
        node = ast.parse(s.strip(), mode="eval").body
        if not isinstance(node, ast.Call):
            return True
        return len(node.args) >= 2 or any(k.arg == "file_name" for k in node.keywords)
    except Exception:
        return True


def is_mut(name, call, login_counts=False):
    """`name` is the AST-derived name full_turn_calls already validated."""
    if name == "echo":
        return echo_writes(call)
    return name in MUT or (login_counts and name in LOGIN)


def any_mut(calls, login_counts=False):
    return any(is_mut(n, s, login_counts) for n, s in calls)


def boot_ci(v, twin, seed=SEED):
    """Cluster bootstrap on the base twin, as in p1g_paired_decision.py (one fresh
    stream per model, so a model's CI does not depend on the order models are listed;
    p1g draws all models from one stream, so its CI bounds can differ by 0.1)."""
    rng = np.random.default_rng(seed)
    groups = [np.where(twin == t)[0] for t in sorted(set(twin.tolist()))]
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        out[b] = v[np.concatenate([groups[p] for p in pick])].mean() * 100
    return np.percentile(out, [2.5, 97.5])


def main():
    pairs, _ = build_pairs()
    ok, _, _ = content_check(pairs)
    pairs = {i: pairs[i] for i in ok}
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    gb = gold_calls("base")

    def gold_calls_at(iid):
        bid, k, _ = pairs[iid]
        g = gb.get(bid) or []
        return [(s.split("(")[0].strip(), s) for s in (g[k] if k < len(g) else [])]

    gold_mut = {iid: any_mut(gold_calls_at(iid)) for iid in pairs}
    gold_mut_login = {iid: any_mut(gold_calls_at(iid), True) for iid in pairs}
    n_gm = sum(gold_mut.values())

    # anchor certification (p2c_cert.py): construction errors leave the scored set,
    # compound turns keep an independent-step list for the credited reading
    construction, compound, _ = load_final()
    cert_ok = {i for i in pairs if gold_mut[i] and i not in construction}
    n_cert = len(cert_ok)
    n_cert3 = len(cert_ok - set(compound))

    rec, ids_ref, trunc = {}, None, {}
    for m in MODELS:
        res = {c: load_results(m, c) for c in ASK + ["base"]}
        if any(v is None for v in res.values()):
            continue
        ids, rows, tb, tm = [], [], 0, 0
        for iid, (bid, k, cat) in pairs.items():
            sb, sm = res["base"].get(bid), res[cat].get(iid)
            if sb is None or sm is None:
                continue
            # a rollout that stopped before turn k has no anchor turn to score; it is
            # kept (scored as no call) to match p1g, and counted here for the footnote
            reached = (k < len(sb), k < len(sm))
            tb += not reached[0]
            tm += not reached[1]
            cb = full_turn_calls(sb[k] if reached[0] else [], allow["base"].get(bid, set()))
            cm = full_turn_calls(sm[k] if reached[1] else [], allow[cat].get(iid, set()))
            ids.append(iid)
            rows.append(dict(
                any_b=len(cb) > 0, mut_b=any_mut(cb), mutL_b=any_mut(cb, True),
                any_m=len(cm) > 0, mut_m=any_mut(cm), mutL_m=any_mut(cm, True),
                mutG_m=any(is_mut(n, s) and gated(n, iid, compound) for n, s in cm),
                reached=all(reached), cat=cat))
        # every model must see the same pairs in the same order, or the shared
        # resample below would silently misalign the columns
        if ids_ref is None:
            ids_ref = ids
        assert ids == ids_ref, f"{m}: pair set differs from the first model"
        rec[m] = rows
        trunc[m] = (tb, tm)

    # (description, act key, hold key (negated), row filter)
    DEFS = {
        "C": (f"PRIMARY: symmetric, mutator on both sides, the {n_gm} gold-mutator pairs "
              f"minus the {len(construction)} certified construction errors = {n_cert} "
              "pairs (login excluded)", "mut_b", "mut_m", lambda i, r: i in cert_ok),
        "C-credited": (f"same {n_cert} pairs; on the {len(compound)} certified compound "
                       "turns an attempt confined to the independent steps is not counted "
                       "as acting (p2a_cert/compound_split.json)", "mut_b", "mutG_m",
                       lambda i, r: i in cert_ok),
        "C-uncertified": (f"sensitivity: all {n_gm} gold-mutator pairs before the anchor "
                          "certification (the table of the pre-certification draft)",
                          "mut_b", "mut_m", lambda i, r: gold_mut[i]),
        "C-no-compound": (f"sensitivity: C minus the {len(compound)} compound turns as well "
                          f"({n_cert3} pairs)", "mut_b", "mut_m",
                          lambda i, r: i in cert_ok and i not in compound),
        "B": ("symmetric, mutator on both sides, all 365", "mut_b", "mut_m",
              lambda i, r: True),
        "D": ("symmetric, any call on both sides, all 365 (the pre-2026-09-16 primary)",
              "any_b", "any_m", lambda i, r: True),
        "C-login": (f"sensitivity: C with the seven login/logout functions counted as "
                    f"mutating ({sum(1 for i in pairs if gold_mut_login[i] and i not in construction)} pairs)",
                    "mutL_b", "mutL_m", lambda i, r: gold_mut_login[i] and i not in construction),
        "C-reached": ("sensitivity: C on pairs where both rollouts reached the anchor "
                      "turn (per-model N)", "mut_b", "mut_m",
                      lambda i, r: i in cert_ok and r["reached"]),
    }
    out = ["# P1-s decision to change the world on the under-specified turn", "",
           f"An attempt counts whether or not the sandbox let it through. "
           f"{len(pairs)} content-verified pairs; {n_gm} of them have a base-twin gold "
           f"anchor turn that contains a world-changing call, {len(pairs)-n_gm} are "
           f"purely read-only there. Login/logout functions are not counted as "
           f"world-changing; the login-counted variant is reported as a "
           f"sensitivity. Anchor certification (p2c_cert.py): {len(construction)} "
           f"construction errors removed, {len(compound)} compound turns kept with an "
           f"independent-step list; primary N = {n_cert}.", "",
           "Definition A (act = any call, hold = no mutating call) is **rejected**: its "
           "two sides ask different questions, so \"emit one read-only call every turn\" "
           "scores 100 and the chance-50 property is lost.", ""]

    for key, (desc, ak, hk, sub) in DEFS.items():
        out += [f"## Definition {key} — {desc}", "",
                "| model | act when complete | hold when removed | decision acc [95% CI] "
                "| both twins right | N |", "|---|---|---|---|---|---|"]
        store = {}
        for m, rows in rec.items():
            sel = [j for j, r in enumerate(rows) if sub(ids_ref[j], r)]
            a = np.array([rows[j][ak] for j in sel], float)
            h = np.array([not rows[j][hk] for j in sel], float)
            acc = (a + h) / 2
            twin = np.array([pairs[ids_ref[j]][0] for j in sel])
            lo, hi = boot_ci(acc, twin)
            store[m] = (acc, a * h, sel, 100*acc.mean(), lo, hi, 100*a.mean(), 100*h.mean())
        for m in sorted(store, key=lambda x: -store[x][3]):
            _, both, sel, v, lo, hi, a, h = store[m]
            out.append(f"| {m} | {a:.1f} | {h:.1f} | {v:.1f} [{lo:.1f}, {hi:.1f}] | "
                       f"{100*both.mean():.1f} | {len(sel)} |")
        names = list(store)
        sp = max(store[m][3] for m in names) - min(store[m][3] for m in names)
        line = f"Spread {sp:.1f} points."
        if len({len(store[m][2]) for m in names}) == 1:
            # rank-1 frequency needs one shared resample, so only when N is shared
            sel0 = store[names[0]][2]
            twin = np.array([pairs[ids_ref[j]][0] for j in sel0])
            groups = [np.where(twin == t)[0] for t in sorted(set(twin.tolist()))]
            Mx = np.vstack([store[m][0] for m in names])
            rng = np.random.default_rng(SEED)
            first = np.zeros(len(names))
            for b in range(B):
                pick = rng.integers(0, len(groups), size=len(groups))
                s = np.concatenate([groups[p] for p in pick])
                first[Mx[:, s].mean(1).argmax()] += 1
            first /= B
            line += " Rank-1 frequency: " + ", ".join(
                f"{n} {100*f:.1f}%" for n, f in
                sorted(zip(names, first), key=lambda x: -x[1]) if f)
        out += ["", line, ""]

    out += ["## Rollouts that never reached the anchor turn", "",
            "Kept and scored as no call, as in p1g. Pairs (base twin, miss twin) per model:",
            ""] + [f"- {m}: {tb} / {tm}" for m, (tb, tm) in trunc.items()] + [""]

    # degenerate strategies, computed from the predicates rather than typed in:
    # each strategy fixes (any call?, mutating call?) on every turn of both twins
    strategies = {"always attempt a mutating call": (True, True),
                  "never call anything": (False, False),
                  "one read-only call on every turn": (True, False)}
    defs = {"A (rejected)": ("any", "mut"), "B": ("mut", "mut"),
            "C": ("mut", "mut"), "D": ("any", "any")}
    out += ["## Degenerate strategies (information-blind, so the same on both twins)", "",
            "| strategy | " + " | ".join(defs) + " |", "|---|" + "---|" * len(defs)]
    for sname, (anyc, mutc) in strategies.items():
        cells = []
        for ak, hk in defs.values():
            act = {"any": anyc, "mut": mutc}[ak]
            hold = not {"any": anyc, "mut": mutc}[hk]
            v = 50 * (act + hold)
            cells.append(f"**{v}**" if v != 50 else f"{v}")
        out.append(f"| {sname} | " + " | ".join(cells) + " |")
    out.append("")
    txt = "\n".join(out)
    open(os.path.join(HERE, "p1s_attempt_decision.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
