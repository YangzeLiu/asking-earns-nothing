#!/usr/bin/env python3
"""P1-d: judge-free non-action rate on the under-specified turn (should-ask side).

Shared module and standalone report. The constants and helpers here (result-tree roots,
model list, bad-key exclusion, anchor location, bootstrap seed) are imported by every
other analysis script. Run directly, it reports, over the retained should-ask items
(official PASS and FAIL alike), the fraction of items on which the model made **no
validated tool call on the under-specified turn**. That is the behaviour BFCL's own task
definition (Patil et al. 2025, App. D.2) prescribes for that turn (`tau_i = empty`), and
it is read straight off the rollouts. The paper's measure is the paired decision of
`p1s_attempt_decision.py`; this report is an earlier, one-sided view of the same turn.

DEFINITIONS (all mechanical, no LLM anywhere in this script)
------------------------------------------------------------
* under-specified turn ("anchor") = the turn index k with `ground_truth[k] == []` in
  `bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_<cat>.json`.
* MULTI-ANCHOR RULE: items with `len(anchors) != 1` are DROPPED, not silently resolved.
  Over the two ask categories this removes `_167` and `_180` from each category. The
  per-model N actually used is reported.
* a call is counted only if `analysis/bfcl_calls.py` parses it structurally (native-FC dict /
  `<tool_call>` JSON / gemma bracket-group + BFCL's own ast recipe) AND the name is in the
  item's own tool list (union of `involved_classes` func docs + the removed function). This
  is the same extractor the prompt-injection analysis uses.
* non-act = zero validated calls on the anchor turn. mutate-on-anchor = at least one call
  tagged `mutate` by the keyword table in `bfcl_calls.py` (login/authenticate are not
  mutations).
* the 31 manually verified bad keys (`phase4_exclude.json`) are dropped first.
* PASS / FAIL is the OFFICIAL BFCL verdict from the score file (fail ids are the rows after
  the header; pass = all ids minus fail ids).

NO JUDGE IS INVOLVED.

COVERAGE. The three Tulu-3-8B checkpoints have no local `result/` rollouts (only score
files were kept), so they cannot be measured; they are reported as missing rather than
imputed. Their ask raw is ~2%, so they are not load-bearing either way.

Bootstrap: item-level, B=10000, seed 20260612, shared resample across the measurable
models for the rank probability.

Output: analysis/p1d_nonact_rate.md
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import tag, turn_calls, allowed_names  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BFCL = os.path.join(HERE, "..", "bfcl")
VENV = os.path.join(BFCL, "venv/lib/python3.11/site-packages")
NODE = os.path.join(BFCL, "rollouts_extra")
# Models run after the first batch live in their own BFCL_PROJECT_ROOT under bfcl/newmodels/
# so the earlier result/score trees stay untouched. Any such directory with a result/ subdir
# is searched after the two original locations.
NEWM = sorted(os.path.join(BFCL, "newmodels", d)
              for d in (os.listdir(os.path.join(BFCL, "newmodels"))
                        if os.path.isdir(os.path.join(BFCL, "newmodels")) else [])
              if os.path.isdir(os.path.join(BFCL, "newmodels", d, "result")))
ROOTS = [VENV, NODE] + NEWM
PA_DIR = os.path.join(VENV, "bfcl_eval/data/possible_answer")
ASK = ["miss_func", "miss_param"]
EXC = json.load(open(os.path.join(HERE, "phase4_exclude.json")))["exclude"]
EXCSET = {c: set(EXC.get(c, [])) for c in ASK}

MODELS = ["qwen3.8-max-0902-FC",
          "gpt-5.4-FC", "Qwen3.6-27B-FC", "gemma-4-31B-it", "deepseek-v4-pro-FC",
          "deepseek-v4-flash-FC",
          "Qwen3.5-9B-FC", "gemma-4-E4B-it",
          "Tulu-3-8B-SFT", "Tulu-3-8B-DPO", "Tulu-3-8B-FINAL"]
# The paper's model set. deepseek-v4-pro is dropped from every table (2026-09-16): the
# rollouts are from 2026-05/06 under a registry alias with no dated snapshot, so the row no
# longer describes the model that name now points at. MODELS stays the full local inventory.
DROP_FROM_PAPER = {"deepseek-v4-pro-FC"}
PAPER_MODELS = [m for m in MODELS if m not in DROP_FROM_PAPER]

SEED = 20260612
B = 10000


def find_file(model, cat, kind):
    """Largest matching file across the roots.

    Not first-found: a pilot run (a 50-item base slice, or a new model tried out before
    BFCL_PROJECT_ROOT was pointed elsewhere) can leave a short file in an earlier root that
    would otherwise shadow the complete one. Duplicated score files across VENV/NODE are
    byte-identical, so this only ever changes which of two identical files is returned."""
    best = None
    for loc in ROOTS:
        p = os.path.join(loc, kind, model, "multi_turn", f"BFCL_v4_multi_turn_{cat}_{kind}.json")
        if os.path.exists(p) and (best is None or os.path.getsize(p) > os.path.getsize(best)):
            best = p
    return best


def anchor_turns(cat):
    out = {}
    for l in open(os.path.join(PA_DIR, f"BFCL_v4_multi_turn_{cat}.json")):
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = [i for i, t in enumerate(r["ground_truth"]) if t == []]
    return out


def measure(model):
    """-> {item_id: dict(cat, nonact, mutate, official_pass)} or None if no rollouts."""
    per = {}
    dropped_multi = collections.Counter()
    for cat in ASK:
        rp, sp = find_file(model, cat, "result"), find_file(model, cat, "score")
        if not rp or not sp:
            return None, None
        results = {}
        for l in open(rp):
            if l.strip():
                r = json.loads(l)
                results[r["id"]] = r["result"]
        lines = open(sp).read().strip().split("\n")
        fail_ids = {json.loads(l)["id"] for l in lines[1:] if l.strip()}
        anchors = anchor_turns(cat)
        allowed = allowed_names(cat)
        for iid in sorted(results):
            if iid in EXCSET[cat]:
                continue
            a = anchors.get(iid, [])
            if len(a) != 1:
                dropped_multi[cat] += 1
                continue
            k = a[0]
            seq = results[iid]
            turn = seq[k] if k < len(seq) else []
            names = turn_calls(turn, allowed.get(iid, set()))
            tags = [tag(x) for x in names]
            per[iid] = dict(cat=cat, nonact=(len(names) == 0),
                            mutate=any(t == "mutate" for t in tags),
                            official_pass=(iid not in fail_ids))
    return per, dropped_multi


def boot_ci(v, rng_):
    n = len(v)
    idx = rng_.integers(0, n, size=(B, n))
    m = v[idx].mean(1) * 100
    lo, hi = np.percentile(m, [2.5, 97.5])
    return lo, hi


def main():
    data, drops, missing = {}, {}, []
    for m in MODELS:
        per, dr = measure(m)
        if per is None:
            missing.append(m)
            continue
        data[m], drops[m] = per, dr

    ids_ref = sorted(data[MODELS[0]])
    for m in data:
        assert sorted(data[m]) == ids_ref, f"{m} ask universe mismatch"

    out = []
    out.append("# P1-d: non-action rate on the under-specified turn (judge-free)\n")
    out.append(
        "> On the retained should-ask items (official PASS and FAIL counted together), the share "
        "on which the model issued **no tool call at all** on the under-specified turn, which is "
        "what the BFCL paper's App. D.2 prescribes for that turn (`tau_i = empty`). Read straight "
        "off the rollouts; no LLM judge.\n")
    out.append("## Definitions\n")
    out.append(
        "- **Under-specified (anchor) turn** = turn k with `ground_truth[k] == []` in `possible_answer`.\n"
        "- **Multi-anchor rule**: items with `len(anchors) != 1` are "
        "**dropped whole**, never disambiguated. This removes `_167` and `_180` from each ask category.\n"
        "- A **call** counts only if `analysis/bfcl_calls.py` parses it structurally and its name is on "
        "the item's own tool list (`involved_classes` function docs plus the removed function); prose is not a call.\n"
        "- **non-act** = zero validated calls on that turn. **mutate on anchor** = at least one call on that turn "
        "tagged `mutate` by the keyword table (login/authenticate do not count as mutations).\n"
        "- The 31 manually verified bad keys (`phase4_exclude.json`) are dropped first.\n"
        "- **PASS/FAIL is the official BFCL verdict** (one fail row per line after the score file header).\n"
        "- **No judge anywhere in this table**: no `reeval_*.jsonl` is read.\n")

    n_mf = sum(1 for i in ids_ref if data[MODELS[0]][i]["cat"] == "miss_func")
    n_mp = len(ids_ref) - n_mf
    out.append(f"- N used: **miss_func {n_mf} + miss_param {n_mp} = {len(ids_ref)}** "
               f"(386 minus the 2 multi-anchor items per category; the same items for every model).\n")
    if missing:
        out.append("- **Not measurable**: " + ", ".join(missing) +
                   " (only score files were kept, no `result/` rollouts; not imputed. Their ask raw is about 2%).\n")

    rng = np.random.default_rng(SEED)
    out.append("## Non-act rate per model (zero calls on the under-specified turn)\n")
    out.append("| model | miss_func | miss_param | pooled non-act% | 95% item bootstrap CI | "
               "mutate on anchor% | non-act% (official PASS) | non-act% (official FAIL) |")
    out.append("|---|---|---|---|---|---|---|---|")
    vecs = {}
    rowdata = {}
    for m in MODELS:
        if m not in data:
            out.append(f"| {m} | — | — | — | — | — | — | — |")
            continue
        per = data[m]
        v = np.array([1.0 if per[i]["nonact"] else 0.0 for i in ids_ref])
        vecs[m] = v
        mf = [i for i in ids_ref if per[i]["cat"] == "miss_func"]
        mp = [i for i in ids_ref if per[i]["cat"] == "miss_param"]
        p_mf = 100 * sum(per[i]["nonact"] for i in mf) / len(mf)
        p_mp = 100 * sum(per[i]["nonact"] for i in mp) / len(mp)
        pooled = 100 * v.mean()
        mut = 100 * sum(per[i]["mutate"] for i in ids_ref) / len(ids_ref)
        ps = [i for i in ids_ref if per[i]["official_pass"]]
        fs = [i for i in ids_ref if not per[i]["official_pass"]]
        npass = 100 * sum(per[i]["nonact"] for i in ps) / len(ps) if ps else float("nan")
        nfail = 100 * sum(per[i]["nonact"] for i in fs) / len(fs) if fs else float("nan")
        lo, hi = boot_ci(v, np.random.default_rng(SEED))
        rowdata[m] = dict(pooled=pooled, mf=p_mf, mp=p_mp, lo=lo, hi=hi, mut=mut,
                          npass=npass, nfail=nfail, n_pass=len(ps), n_fail=len(fs))
        out.append(f"| {m} | {p_mf:.1f} | {p_mp:.1f} | **{pooled:.1f}** | [{lo:.1f}, {hi:.1f}] "
                   f"| {mut:.1f} | {npass:.1f} (n={len(ps)}) | {nfail:.1f} (n={len(fs)}) |")
    out.append("")

    order = sorted(rowdata.items(), key=lambda kv: -kv[1]["pooled"])
    out.append("**Order (non-act, judge-free)**: " +
               " > ".join(f"{m} {d['pooled']:.1f}" for m, d in order))
    out.append("")
    gap = order[0][1]["pooled"] - order[1][1]["pooled"]
    out.append(f"First, {order[0][0]}, leads second, {order[1][0]}, by **{gap:.1f}pp**.")
    out.append("")

    models_ok = [m for m in MODELS if m in vecs]
    Mx = np.stack([vecs[m] for m in models_ok])
    n = Mx.shape[1]
    rng2 = np.random.default_rng(SEED)
    idx = rng2.integers(0, n, size=(B, n))
    means = Mx[:, idx].mean(2)
    g = models_ok.index("gpt-5.4-FC")
    win = (means[g] >= means.max(0)).mean()
    out.append(f"## Rank check (shared item-level resample, B={B}, seed={SEED})\n")
    out.append(f"- **P(gpt-5.4 has the highest non-act rate of these {len(models_ok)} models) = {win*100:.1f}%**")
    for rival in models_ok:
        if rival == "gpt-5.4-FC":
            continue
        j = models_ok.index(rival)
        out.append(f"- P(gpt > {rival}) = {(means[g] > means[j]).mean()*100:.1f}%")
    out.append("")
    out.append("## Reading\n")
    out.append(
        "- The PASS and FAIL columns show directly that a sizeable share of the items the official "
        "scorer passes still had a tool call on the under-specified turn, a consequence of the scorer "
        "not checking that turn.\n"
        "- Non-act only says that no tool was called. It does not check whether the reply on that turn "
        "asks for the missing information, so it is not a measure of clarification quality.\n")

    path = os.path.join(HERE, "p1d_nonact_rate.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
