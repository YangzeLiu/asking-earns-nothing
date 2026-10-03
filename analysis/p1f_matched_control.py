#!/usr/bin/env python3
"""P1-f: matched fully-specified control for the judge-free non-action rate.

The non-act rate of `p1d_nonact_rate.py` says how often a model issues no validated tool
call on the under-specified turn. It does not say whether the model is holding back
*because* information is missing or is simply passive. Every should-ask item is a
perturbation of a base item with the same scenario index: removing the single empty turn
from the should-ask gold trajectory reproduces the base gold trajectory exactly, so the
base turn at the anchor index is the same request with the information present.

This script pairs them and reports non-act(miss) - non-act(base_twin), per model, PASS and
FAIL alike. No LLM judge is involved anywhere; the call extractor is the sealed one used by
Phase 5, the PASS audit and p1d.

Output: analysis/p1f_matched_control.md
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import tag, turn_calls, allowed_names  # noqa: E402
from p1d_nonact_rate import find_file, anchor_turns, MODELS, EXCSET, PA_DIR, SEED, B, ROOTS  # noqa: E402

ASK = ["miss_func", "miss_param"]


def gold_shape(cat):
    out = {}
    for l in open(os.path.join(PA_DIR, f"BFCL_v4_multi_turn_{cat}.json")):
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = [len(t) for t in r["ground_truth"]]
    return out


def build_pairs():
    """-> {ask_id: (base_id, anchor_turn)} for items whose base twin aligns exactly."""
    gb = gold_shape("base")
    pairs, drop = {}, collections.Counter()
    for cat in ASK:
        gm = gold_shape(cat)
        anchors = anchor_turns(cat)
        for iid, shape in gm.items():
            if iid in EXCSET[cat]:
                drop["bad key"] += 1
                continue
            a = anchors.get(iid, [])
            if len(a) != 1:
                drop["multi-anchor"] += 1
                continue
            p = a[0]
            bid = "multi_turn_base_" + iid.rsplit("_", 1)[1]
            if bid not in gb:
                drop["no base twin"] += 1
                continue
            if shape[:p] + shape[p + 1:] != gb[bid]:
                drop["shape differs"] += 1
                continue
            pairs[iid] = (bid, p, cat)
    return pairs, drop


def calls_on_turn(results, allowed, iid, k):
    seq = results.get(iid)
    if seq is None:
        return None
    turn = seq[k] if k < len(seq) else []
    return turn_calls(turn, allowed.get(iid, set()))


def load_results(model, cat):
    """Prefer the most complete rollout file: the venv tree still holds 50-item base pilots
    for the three models whose full base runs live in rollouts_extra, and find_file returns
    the venv copy first."""
    best = None
    for loc in ROOTS:
        q = os.path.join(loc, "result", model, "multi_turn",
                         f"BFCL_v4_multi_turn_{cat}_result.json")
        if not os.path.exists(q):
            continue
        rows = {r["id"]: r["result"] for r in (json.loads(l) for l in open(q) if l.strip())}
        if best is None or len(rows) > len(best):
            best = rows
    return best


def measure(model, pairs):
    res = {c: load_results(model, c) for c in ASK + ["base"]}
    if any(v is None for v in res.values()):
        return None
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    rows = []
    for iid, (bid, k, cat) in pairs.items():
        cm = calls_on_turn(res[cat], allow[cat], iid, k)
        cb = calls_on_turn(res["base"], allow["base"], bid, k)
        if cm is None or cb is None:
            continue
        rows.append((iid, cat,
                     len(cm) == 0, any(tag(x) == "mutate" for x in cm),
                     len(cb) == 0, any(tag(x) == "mutate" for x in cb)))
    return rows


def ci(diff, rng):
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    m = diff[idx].mean(1) * 100
    return np.percentile(m, [2.5, 97.5])


def main():
    pairs, drop = build_pairs()
    out = ["# P1-f matched fully-specified control (no judge)", "",
           f"Pairs usable: {len(pairs)}  (dropped: {dict(drop)})", "",
           "non-act = no validated tool call on that turn; miss = under-specified turn,",
           "base = same scenario, same turn index, information present. PASS and FAIL alike.", "",
           "| model | N | non-act miss % | non-act base % | difference [95% CI] | mutate miss % | mutate base % |",
           "|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(SEED)
    for model in MODELS:
        rows = measure(model, pairs)
        if not rows:
            out.append(f"| {model} | — | no rollouts | | | | |")
            continue
        nm = np.array([r[2] for r in rows], float)
        nb = np.array([r[4] for r in rows], float)
        mm = np.array([r[3] for r in rows], float)
        mb = np.array([r[5] for r in rows], float)
        lo, hi = ci(nm - nb, rng)
        out.append(f"| {model} | {len(rows)} | {100*nm.mean():.1f} | {100*nb.mean():.1f} | "
                   f"{100*(nm-nb).mean():+.1f} [{lo:.1f}, {hi:.1f}] | {100*mm.mean():.1f} | {100*mb.mean():.1f} |")
    txt = "\n".join(out)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1f_matched_control.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
