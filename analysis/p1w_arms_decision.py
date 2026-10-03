#!/usr/bin/env python3
"""P1-w: the paired decision accuracy (definition C) under the Phase 5 prompt arms.

`p1t_phase5_attempt.py` reports, per arm, how often the model attempted a world-changing
call on the under-specified turn. This script asks the two-sided question
the main table asks: on the certified gold-mutator pairs (p2c_cert.py), what is the decision accuracy under
the neutral prompt and under the injected prompt, for the arms that generated both
twins? gemma-4-31B-it ran all four categories under pro-caution and pro-action, so both
sides exist; gpt-5.4 ran only the two should-ask categories under pro-action, so only
its hold side moves and its act side is carried over from the neutral base run.

Output: analysis/p1w_arms_decision.md
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import allowed_names  # noqa: E402
from bfcl_calls_full import full_turn_calls  # noqa: E402
from p1d_nonact_rate import SEED, B  # noqa: E402
from p1f_matched_control import build_pairs, load_results, ASK  # noqa: E402
from p1h_pair_checks import content_check, gold_calls  # noqa: E402
from p1s_attempt_decision import any_mut, boot_ci  # noqa: E402
from p2c_cert import load_final  # noqa: E402
from phase5_mechanical import NEUTRAL, VENV_SP, P5, load_results as load_arm, mcnemar_exact  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, model, variant_dir, neutral_dir, has base)
ARMS = [
    ("pro-caution", "gemma-4-31B-it", os.path.join(P5, "procaution"), NEUTRAL, True),
    ("pro-action", "gemma-4-31B-it", os.path.join(P5, "proaction_gemma"), NEUTRAL, True),
    ("pro-action", "gpt-5.4-FC", os.path.join(P5, "proaction_gpt"), VENV_SP, True),
]


def arm_file(d, model, cat):
    p = os.path.join(d, "result", model, "multi_turn", f"BFCL_v4_multi_turn_{cat}_result.json")
    return load_arm(p) if os.path.exists(p) else None


def main():
    pairs, _ = build_pairs()
    ok, _, _ = content_check(pairs)
    pairs = {i: pairs[i] for i in ok}
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    gb = gold_calls("base")

    def gold_mut(iid):
        bid, k, _ = pairs[iid]
        g = gb.get(bid) or []
        return any_mut([(s.split("(")[0].strip(), s) for s in (g[k] if k < len(g) else [])])

    construction, _, _ = load_final()
    C = [i for i in pairs if gold_mut(i) and i not in construction]
    L = ["# P1-w decision accuracy (definition C) under the Phase 5 arms", "",
         f"{len(C)} certified gold-mutator pairs (construction errors out), restricted per arm to the pairs whose items the arm "
         "generated. act = attempted a world-changing call on the base anchor, hold = no such "
         "attempt on the miss anchor. McNemar exact on paired flips, CI by cluster bootstrap "
         "over base twins (B=10000, seed 20260612).", "",
         "| arm | model | N | act neu->var | hold neu->var | decision acc neu->var | "
         "act flips v/n, p | hold flips v/n, p |", "|---|---|---|---|---|---|---|---|"]
    for label, model, vdir, ndir, has_base in ARMS:
        var = {c: arm_file(vdir, model, c) for c in ASK + (["base"] if has_base else [])}
        if any(v is None for v in var.values()):
            L.append(f"| {label} | {model} | (arm not present: {vdir}) | | | | | |")
            continue
        neu = {c: load_arm(os.path.join(ndir, "result", model, "multi_turn",
                                        f"BFCL_v4_multi_turn_{c}_result.json"))
               for c in ASK + ["base"]}
        rows, keys = [], []
        for iid in C:
            bid, k, cat = pairs[iid]
            if iid not in var[cat] or iid not in neu[cat] or bid not in neu["base"]:
                continue
            if has_base and bid not in var["base"]:
                continue

            def at(seq, allowed):
                return any_mut(full_turn_calls(seq[k] if k < len(seq) else [], allowed))
            act_n = at(neu["base"][bid], allow["base"].get(bid, set()))
            act_v = at(var["base"][bid], allow["base"].get(bid, set())) if has_base else act_n
            hold_n = not at(neu[cat][iid], allow[cat].get(iid, set()))
            hold_v = not at(var[cat][iid], allow[cat].get(iid, set()))
            rows.append((bid, act_n, act_v, hold_n, hold_v))
            keys.append((bid, k))
        n = len(rows)
        twin = np.array([r[0] for r in rows])
        an = np.array([r[1] for r in rows], float)
        av = np.array([r[2] for r in rows], float)
        hn = np.array([r[3] for r in rows], float)
        hv = np.array([r[4] for r in rows], float)
        accn, accv = (an + hn) / 2, (av + hv) / 2
        lo_n, hi_n = boot_ci(accn, twin)
        lo_v, hi_v = boot_ci(accv, twin)
        ab = int(((av == 1) & (an == 0)).sum())
        ac = int(((an == 1) & (av == 0)).sum())
        hb = int(((hv == 1) & (hn == 0)).sum())
        hc = int(((hn == 1) & (hv == 0)).sum())
        act_cell = (f"{ab} / {ac}, p={mcnemar_exact(ab, ac):.3g}" if has_base
                    else "(no base arm; act carried over)")
        L.append(f"| {label} | {model} | {n} | {100*an.mean():.1f}->{100*av.mean():.1f} "
                 f"| {100*hn.mean():.1f}->{100*hv.mean():.1f} "
                 f"| {100*accn.mean():.1f} [{lo_n:.1f}, {hi_n:.1f}] -> "
                 f"{100*accv.mean():.1f} [{lo_v:.1f}, {hi_v:.1f}] "
                 f"| {act_cell} | {hb} / {hc}, p={mcnemar_exact(hb, hc):.3g} |")
        # paired CI on the accuracy difference
        rng = np.random.default_rng(SEED)
        groups = [np.where(twin == t)[0] for t in sorted(set(twin.tolist()))]
        d = accv - accn
        out = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, len(groups), size=len(groups))
            out[b] = d[np.concatenate([groups[p] for p in pick])].mean() * 100
        lo, hi = np.percentile(out, [2.5, 97.5])
        L.append(f"|  |  |  |  |  | change {100*d.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] |  |  |")
        if has_base:
            # act flips counted once per distinct base anchor turn (pair weighting check)
            seen = {}
            for (bid, a_n, a_v, _h1, _h2), key in zip(rows, keys):
                seen[key] = (a_n, a_v)
            ab1 = sum(1 for a_n, a_v in seen.values() if a_v and not a_n)
            ac1 = sum(1 for a_n, a_v in seen.values() if a_n and not a_v)
            an1 = 100 * np.mean([a for a, _ in seen.values()])
            av1 = 100 * np.mean([a for _, a in seen.values()])
            L.append(f"|  |  | {len(seen)} anchors |  act per distinct anchor {an1:.1f}->{av1:.1f} |  |  "
                     f"| {ab1} / {ac1}, p={mcnemar_exact(ab1, ac1):.3g} |  |")
    txt = "\n".join(L)
    open(os.path.join(HERE, "p1w_arms_decision.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
