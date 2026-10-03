#!/usr/bin/env python3
"""P1-n: how often does a model reach the complete-information outcome WITHOUT the
information? ("X": a lower bound on successful workarounds.)

Reading five Qwen3.8-Max trajectories by hand turned up an item where `mv` was withheld
and the model performed the move as `cp` -> verify byte-identical -> `rm`. It did the job.
Counting that as "failed to hold back" is not obviously fair, so we measure how often it
happens instead of arguing about it.

METHOD (executed, no judge, no keyword table)
---------------------------------------------
For each content-verified pair, with BFCL's own executor and BFCL's own notion of state
(`state_checker`: public attributes of the involved instances):

    base twin's GOLD turns 0..k-1   -> G_before ;  + GOLD turn k  -> G_after
    model's own miss turns 0..k-1   -> M_before ;  + model turn k -> M_after

Restrict to items where M_before == G_before, i.e. the model arrives at the anchor in the
state the gold trajectory arrives in; otherwise the comparison is contaminated by earlier
divergence. On that subset:

    M_after == G_after   -> reached the complete-information outcome  ("workaround")
    M_after == M_before  -> no state change                           ("held back")
    otherwise            -> changed the world some other way          ("diverged")

M_after == G_after is a LOWER bound on a successful workaround: a model may reach the
user's goal by a different but equally acceptable state, which this counts as diverged.

Output: analysis/p1n_workaround.md
"""
import json
import os
import re
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import allowed_names  # noqa: E402
from bfcl_calls_full import full_turn_calls  # noqa: E402
from p1d_nonact_rate import PAPER_MODELS as MODELS, VENV, PA_DIR  # noqa: E402
from p1f_matched_control import build_pairs, load_results, ASK  # noqa: E402
from p1h_pair_checks import content_check  # noqa: E402

sys.path.insert(0, VENV)
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as U  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: E402
    execute_multi_turn_func_call as EX,
)

DATA = os.path.join(VENV, "bfcl_eval/data")
HERE = os.path.dirname(os.path.abspath(__file__))


def entries(cat):
    p = os.path.join(DATA, f"BFCL_v4_multi_turn_{cat}.json")
    return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}


def gold(cat):
    p = os.path.join(PA_DIR, f"BFCL_v4_multi_turn_{cat}.json")
    return {json.loads(l)["id"]: json.loads(l)["ground_truth"] for l in open(p) if l.strip()}


def fresh(tag, entry, eid):
    for cls in entry["involved_classes"]:
        U.__dict__.pop(re.sub(r"[-./:]", "_", f"{tag}_{eid}_{cls}_instance"), None)
    EX([], entry["initial_config"], entry["involved_classes"], tag, eid)


def step(tag, entry, eid, calls):
    return EX(list(calls), entry["initial_config"], entry["involved_classes"], tag, eid)


def snap(tag, entry, eid):
    out = {}
    for cls in entry["involved_classes"]:
        name = re.sub(r"[-./:]", "_", f"{tag}_{eid}_{cls}_instance")
        inst = U.__dict__.get(name)
        if inst is not None:
            out[cls] = {k: deepcopy(v) for k, v in vars(inst).items() if not k.startswith("_")}
    return out


def main():
    pairs, _ = build_pairs()
    ok, _, _ = content_check(pairs)
    pairs = {i: pairs[i] for i in ok}
    ent = {c: entries(c) for c in ASK + ["base"]}
    gld = gold("base")
    allow = {c: allowed_names(c) for c in ASK}

    # the target state for every pair, computed once from the base twin's gold
    target = {}
    for iid, (bid, k, cat) in pairs.items():
        be = ent["base"][bid]
        eid = f"{bid}__{k}"
        fresh("p1n_gold", be, eid)
        for j in range(k):
            step("p1n_gold", be, eid, gld[bid][j])
        gb = snap("p1n_gold", be, eid)
        step("p1n_gold", be, eid, gld[bid][k])
        target[iid] = (gb, snap("p1n_gold", be, eid))
    print(f"targets built for {len(target)} pairs", flush=True)

    lines = ["# P1-n reaching the complete-information outcome without the information", "",
             f"{len(pairs)} content-verified pairs. Aligned = the model arrives at the anchor in",
             "the same world state the base twin's gold trajectory arrives in. Percentages are of",
             "the aligned subset. `workaround` is a lower bound (see the module docstring).", "",
             "| model | aligned N | held back | workaround | diverged | workaround % of aligned |",
             "|---|---|---|---|---|---|"]
    detail = {}
    for m in MODELS:
        res = {c: load_results(m, c) for c in ASK}
        if any(v is None for v in res.values()):
            continue
        n_al = held = work = div = 0
        hits = []
        for iid, (bid, k, cat) in pairs.items():
            seq = res[cat].get(iid)
            if seq is None:
                continue
            e = ent[cat][iid]
            eid = f"{m}__{iid}"
            try:
                fresh("p1n_m", e, eid)
                for j in range(k):
                    step("p1n_m", e, eid, [s for _, s in full_turn_calls(seq[j], allow[cat].get(iid, set()))])
                mb = snap("p1n_m", e, eid)
                step("p1n_m", e, eid, [s for _, s in full_turn_calls(seq[k] if k < len(seq) else [], allow[cat].get(iid, set()))])
                ma = snap("p1n_m", e, eid)
            except Exception:
                continue
            gb, ga = target[iid]
            if mb != gb:
                continue                      # diverged before the anchor; not comparable
            n_al += 1
            if ma == mb:
                held += 1
            elif ma == ga:
                work += 1
                hits.append(iid)
            else:
                div += 1
        detail[m] = {"aligned": n_al, "held": held, "workaround": work, "diverged": div,
                     "ids": hits}
        pct = 100 * work / n_al if n_al else 0
        lines.append(f"| {m} | {n_al} | {held} | {work} | {div} | {pct:.1f} |")
        print(f"{m}: aligned={n_al} held={held} workaround={work} diverged={div}", flush=True)

    # how many items are workaroundable by ANY of the measured models
    union = set()
    for v in detail.values():
        union |= set(v["ids"])
    lines += ["", f"Items on which at least one measured model reached the complete-information",
              f"state without the information: **{len(union)} of {len(pairs)}** "
              f"({100*len(union)/len(pairs):.1f}%).", ""]
    json.dump(detail, open(os.path.join(HERE, "p1n_workaround.json"), "w"))
    txt = "\n".join(lines)
    open(os.path.join(HERE, "p1n_workaround.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
