#!/usr/bin/env python3
"""P1-h: the robustness checks the paired decision measure has to survive.

`p1f_matched_control.py` pairs a should-ask item with its base twin on gold SHAPE
(the per-turn call counts match once the empty anchor turn is deleted). A reader is
entitled to ask for more than that, so this script runs five further checks, all from
the stored rollouts, no judge anywhere:

  1. CONTENT-level pairing. The gold call strings themselves must match turn for turn
     once the anchor is deleted, and the user text of every turn before the anchor must
     be character-identical. Pairs that fail are reported, not silently kept.
  2. HISTORY divergence. Even with identical user text, the model's own calls before the
     anchor can differ between the twins. Decision accuracy is recomputed on the subset
     where the model's pre-anchor validated call sequence is identical in both.
  3. PER-CATEGORY split. "Hold back" means report a missing function in miss_func and
     ask for a value in miss_param; the two are reported separately.
  4. STATE-CHANGING definition. Hold-back under "no validated call at all" and under
     "no state-changing call" (a read-only lookup then a question counts as holding back).
  5. PAIR-correct rate. Both twins right on the same scenario. Always-act and always-ask
     score 0 on this one instead of 50.

Output: analysis/p1h_pair_checks.md
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import tag, turn_calls, allowed_names  # noqa: E402
from p1d_nonact_rate import PAPER_MODELS as MODELS, PA_DIR, SEED, B, VENV  # noqa: E402
from p1f_matched_control import build_pairs, load_results, ASK  # noqa: E402

DATA = os.path.join(VENV, "bfcl_eval", "data")
HERE = os.path.dirname(os.path.abspath(__file__))


def gold_calls(cat):
    out = {}
    for l in open(os.path.join(PA_DIR, f"BFCL_v4_multi_turn_{cat}.json")):
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = r["ground_truth"]
    return out


def questions(cat):
    out = {}
    for l in open(os.path.join(DATA, f"BFCL_v4_multi_turn_{cat}.json")):
        if l.strip():
            r = json.loads(l)
            out[r["id"]] = [[m.get("content", "") for m in t] for t in r["question"]]
    return out


def content_check(pairs):
    gb, qb = gold_calls("base"), questions("base")
    gm = {c: gold_calls(c) for c in ASK}
    qm = {c: questions(c) for c in ASK}
    ok, bad_gold, bad_pre = [], [], []
    for iid, (bid, k, cat) in pairs.items():
        g = gm[cat][iid]
        if g[:k] + g[k + 1:] != gb[bid]:
            bad_gold.append(iid)
            continue
        q = qm[cat][iid]
        if q[:k] != qb[bid][:k]:
            bad_pre.append(iid)
            continue
        ok.append(iid)
    return ok, bad_gold, bad_pre


def decision_rows(model, pairs, keep=None):
    """-> list of (iid, cat, act_base, hold_miss, hold_miss_mut, same_history)."""
    res = {c: load_results(model, c) for c in ASK + ["base"]}
    if any(v is None for v in res.values()):
        return None
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    rows = []
    for iid, (bid, k, cat) in pairs.items():
        if keep is not None and iid not in keep:
            continue
        sm, sb = res[cat].get(iid), res["base"].get(bid)
        if sm is None or sb is None:
            continue
        cm = turn_calls(sm[k] if k < len(sm) else [], allow[cat].get(iid, set()))
        cb = turn_calls(sb[k] if k < len(sb) else [], allow["base"].get(bid, set()))
        hm = [turn_calls(sm[j], allow[cat].get(iid, set())) for j in range(min(k, len(sm)))]
        hb = [turn_calls(sb[j], allow["base"].get(bid, set())) for j in range(min(k, len(sb)))]
        rows.append((iid, cat,
                     len(cb) > 0,                                   # acted when complete
                     len(cm) == 0,                                  # held back when missing
                     not any(tag(x) == "mutate" for x in cm),       # held back, state-changing def
                     hm == hb))                                     # identical pre-anchor history
    return rows


def boot(vals, rng):
    n = len(vals)
    idx = rng.integers(0, n, size=(B, n))
    m = vals[idx].mean(1) * 100
    return np.percentile(m, [2.5, 97.5])


def acc(rows, hold_idx=3):
    a = np.array([r[2] for r in rows], float)
    h = np.array([r[hold_idx] for r in rows], float)
    return 100 * a.mean(), 100 * h.mean(), 50 * (a.mean() + h.mean())


def main():
    pairs, drop = build_pairs()
    ok, bad_gold, bad_pre = content_check(pairs)
    okset = set(ok)
    out = ["# P1-h robustness checks on the paired decision measure", "",
           f"Shape-matched pairs from p1f: **{len(pairs)}**  (dropped: {dict(drop)})", "",
           "## 1. Content-level pairing", "",
           f"- gold call strings identical turn for turn after deleting the anchor, AND",
           f"  every user turn before the anchor character-identical: **{len(ok)}/{len(pairs)}**",
           f"- gold strings differ: {len(bad_gold)} {bad_gold[:10]}",
           f"- pre-anchor user text differs: {len(bad_pre)} {bad_pre[:10]}", ""]

    rng = np.random.default_rng(SEED)
    out += ["## 2-5. Per model, on the content-verified pairs", "",
            "| model | N | act complete | hold missing | decision acc [95% CI] | "
            "hold (state-changing def) | decision acc (sc) | pair-correct | "
            "same-history N | decision acc (same history) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    percat = ["", "### Per category (content-verified pairs, any-call definition)", "",
              "| model | miss_func N | act | hold | dec | miss_param N | act | hold | dec |",
              "|---|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        rows = decision_rows(model, pairs, keep=okset)
        if not rows:
            continue
        a, h, d = acc(rows)
        av = np.array([r[2] for r in rows], float)
        hv = np.array([r[3] for r in rows], float)
        lo, hi = boot((av + hv) / 2, rng)
        a2, h2, d2 = acc(rows, hold_idx=4)
        pc = 100 * np.mean([r[2] and r[3] for r in rows])
        sh = [r for r in rows if r[5]]
        sa, shh, sd = acc(sh) if sh else (0, 0, 0)
        out.append(f"| {model} | {len(rows)} | {a:.1f} | {h:.1f} | {d:.1f} [{lo:.1f}, {hi:.1f}] | "
                   f"{h2:.1f} | {d2:.1f} | {pc:.1f} | {len(sh)} | {sd:.1f} |")
        cells = []
        for cat in ASK:
            sub = [r for r in rows if r[1] == cat]
            ca, ch, cd = acc(sub) if sub else (0, 0, 0)
            cells.append(f"{len(sub)} | {ca:.1f} | {ch:.1f} | {cd:.1f}")
        percat.append(f"| {model} | " + " | ".join(cells) + " |")
    txt = "\n".join(out + percat)
    open(os.path.join(HERE, "p1h_pair_checks.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
