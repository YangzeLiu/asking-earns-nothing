#!/usr/bin/env python3
"""P1-g: paired action-decision accuracy, fully mechanical.

Every should-ask item is a perturbation of a base item with the same scenario index: drop
the single empty turn from the should-ask gold trajectory and you get the base gold
trajectory exactly. So the same request appears twice at the same turn index, once with the
needed information present (base) and once with it removed (miss_func / miss_param).

That gives a binary decision task with a ground truth fixed by the benchmark's own design:
on the information-complete turn the model should act, on the information-missing turn it
should not. We measure it with no LLM judge, no world state, no official verdict, and with
official PASS and FAIL treated alike:

    act-when-complete   = 1 - non-act(base turn p)
    refrain-when-missing = non-act(miss turn p)
    decision accuracy   = mean of the two (balanced accuracy of the act/not-act decision)

Always acting scores 50, always asking scores 50, so neither degenerate strategy wins.

Output: analysis/p1g_paired_decision.md
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1d_nonact_rate import PAPER_MODELS as MODELS, SEED, B, find_file  # noqa: E402
from p1f_matched_control import build_pairs, measure  # noqa: E402
from p1h_pair_checks import content_check  # noqa: E402


def fail_ids(model, cat):
    """Item ids the official scorer marked FAIL (every score-file row after the header)."""
    p = find_file(model, cat, "score")
    if not p:
        return None
    out = set()
    for i, line in enumerate(open(p)):
        if i == 0 or not line.strip():
            continue
        out.add(json.loads(line)["id"])
    return out


def official_on(model, ids_by_cat):
    """Official accuracy restricted to exactly the paired items."""
    ok = tot = 0
    for cat, ids in ids_by_cat.items():
        f = fail_ids(model, cat)
        if f is None:
            return None
        tot += len(ids)
        ok += sum(1 for i in ids if i not in f)
    return 100.0 * ok / tot if tot else None


def main():
    pairs, drop = build_pairs()
    ok_ids, bad_gold, bad_pre = content_check(pairs)
    pairs = {i: pairs[i] for i in ok_ids}          # content-verified pairs only
    miss_ids = {}
    base_ids = set()
    for iid, (bid, k, cat) in pairs.items():
        miss_ids.setdefault(cat, set()).add(iid)
        base_ids.add(bid)
    rng = np.random.default_rng(SEED)
    out = ["# P1-g paired action-decision accuracy (no judge, no world state)", "",
           f"{len(pairs)} content-verified paired items "
           f"(shape-matched {len(pairs) + len(bad_gold) + len(bad_pre)}, "
           f"dropped at pairing: {dict(drop)}, "
           f"dropped at content check: gold {len(bad_gold)}, pre-anchor text {len(bad_pre)}). "
           "PASS and FAIL alike.", "",
           "| model | N | act when complete % | refrain when missing % | decision acc % [95% CI] | "
           "pair-correct % | official raw, miss items % | official raw, base twins % |",
           "|---|---|---|---|---|---|---|---|"]
    rows_all = {}
    for m in MODELS:
        rows = measure(m, pairs)
        if not rows:
            out.append(f"| {m} | — | no rollouts | | | |")
            continue
        refrain = np.array([r[2] for r in rows], float)      # non-act on the miss turn
        act = np.array([not r[4] for r in rows], float)      # acted on the base turn
        acc = (refrain + act) / 2
        # Cluster bootstrap on the base twin: the 365 pairs share only 188 twins (177 of
        # them serve both a miss_func and a miss_param item), so resampling rows
        # independently would treat correlated rows as independent evidence.
        twin = [pairs[r[0]][0] for r in rows]
        order = sorted(set(twin))
        gidx = {t: np.where(np.array(twin) == t)[0] for t in order}
        groups = [gidx[t] for t in order]
        means = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, len(groups), size=len(groups))
            means[b] = acc[np.concatenate([groups[p] for p in pick])].mean() * 100
        lo, hi = np.percentile(means, [2.5, 97.5])
        rawm = official_on(m, miss_ids)
        rawb = official_on(m, {"base": base_ids})
        pc = 100 * np.mean(refrain * act)
        rows_all[m] = acc
        ids_all = [r[0] for r in rows]
        fm = "—" if rawm is None else f"{rawm:.1f}"
        fb = "—" if rawb is None else f"{rawb:.1f}"
        out.append(f"| {m} | {len(rows)} | {100*act.mean():.1f} | {100*refrain.mean():.1f} | "
                   f"{100*acc.mean():.1f} [{lo:.1f}, {hi:.1f}] | {pc:.1f} | {fm} | {fb} |")
    # rank-1 frequency under a shared item resample
    names = [m for m in MODELS if m in rows_all]
    if names:
        M = np.vstack([rows_all[m] for m in names])
        twin = [pairs[i][0] for i in ids_all]
        order = sorted(set(twin))
        groups = [np.where(np.array(twin) == t)[0] for t in order]
        first = np.zeros(len(names))
        for b in range(B):
            pick = rng.integers(0, len(groups), size=len(groups))
            sel = np.concatenate([groups[p] for p in pick])
            first[M[:, sel].mean(1).argmax()] += 1
        first /= B
        out += ["", "Rank-1 frequency under a shared base-twin cluster resample:"]
        for n, f in sorted(zip(names, first), key=lambda x: -x[1]):
            out.append(f"- {n}: {100*f:.1f}%")
    txt = "\n".join(out)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p1g_paired_decision.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
