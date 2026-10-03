#!/usr/bin/env python3
"""Phase 5 mechanical metrics: the prompt-injection arms (no LLM judge, paired).

Tests whether injecting a system-prompt nudge moves agents' ACTION CHOICE on the
under-specified (anchor) turn, on the SAME items, vs the neutral rollout.

  pro-caution (gemma-4-31B-it): expect anchor-turn mutate-rate to DROP (less guessing /
      inventing the removed value; more hold-and-ask).
  pro-action  (gpt-5.4-FC; also Qwen3.5-9B-FC, whose handler ignored the injection):
      expect anchor-turn act/mutate-rate to RISE.

should-ask cats (miss_func, miss_param): the ONE oracle=[] turn is the anchor — a required
  input was deliberately removed. We count what the model did THERE.
should-act cats (base, long_context): no single anchor; we report total-call volume and
  empty-turn (defer) incidence as a coarse activity proxy.

Metrics are mechanical (function-name -> observe/mutate via the keyword table in
`bfcl_calls.py`). Reported as paired McNemar on the anchor-turn mutate decision. The paper's
attempt rates use `mutator_map.json` instead and come from `p1t_phase5_attempt.py`; the
official-raw table at the end of this report is the one the paper cites from here.
"""
import json, re, os, glob, collections, sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import tag, turn_calls, allowed_names   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BFCL = os.path.join(HERE, "..", "bfcl")
NEUTRAL = os.path.join(BFCL, "rollouts_extra")
PA_DIR = os.path.join(BFCL, "venv/lib/python3.11/site-packages/bfcl_eval/data/possible_answer")
P5 = os.path.join(HERE, "..", "bfcl", "phase5_results")

def load_results(path):
    d = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        d[r["id"]] = r["result"]
    return d


def anchor_turns(pa_path):
    """id -> list of oracle==[] turn indices. (file is JSONL, one record per line)"""
    out = {}
    for l in open(pa_path):
        if not l.strip():
            continue
        r = json.loads(l)
        gt = r["ground_truth"]
        out[r["id"]] = [i for i, t in enumerate(gt) if t == []]
    return out


def mcnemar_exact(b, c):
    """two-sided exact binomial on discordant pairs (b = variant-only, c = neutral-only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    return min(1.0, p)


def anchor_metric(model, cat, variant_dir, neutral_dir=None):
    pa = anchor_turns(os.path.join(PA_DIR, f"BFCL_v4_multi_turn_{cat}.json"))
    neu = load_results(os.path.join(neutral_dir or NEUTRAL, "result", model, "multi_turn",
                                    f"BFCL_v4_multi_turn_{cat}_result.json"))
    var = load_results(os.path.join(variant_dir, "result", model, "multi_turn",
                                    f"BFCL_v4_multi_turn_{cat}_result.json"))
    allowed = allowed_names(cat)
    ids = [i for i in neu if i in var and i in pa and len(pa[i]) == 1]
    rows = []
    for i in ids:
        k = pa[i][0]
        for res, who in ((neu, "n"), (var, "v")):
            seq = res[i]
            turn = seq[k] if k < len(seq) else []
            names = turn_calls(turn, allowed.get(i, set()))
            tags = [tag(n) for n in names]
            rows.append((i, who, bool(names), any(t == "mutate" for t in tags)))
    by = {(i, w): (acted, mut) for i, w, acted, mut in rows}
    n = len(ids)
    act_n = sum(by[(i, "n")][0] for i in ids)
    act_v = sum(by[(i, "v")][0] for i in ids)
    mut_n = sum(by[(i, "n")][1] for i in ids)
    mut_v = sum(by[(i, "v")][1] for i in ids)
    # paired mutate-decision flips
    b = sum(1 for i in ids if by[(i, "v")][1] and not by[(i, "n")][1])   # variant mutated, neutral didn't
    c = sum(1 for i in ids if by[(i, "n")][1] and not by[(i, "v")][1])   # neutral mutated, variant didn't
    # paired acted-decision flips (any call at all, not just mutate)
    ab = sum(1 for i in ids if by[(i, "v")][0] and not by[(i, "n")][0])  # variant acted, neutral didn't
    ac = sum(1 for i in ids if by[(i, "n")][0] and not by[(i, "v")][0])  # neutral acted, variant didn't
    return dict(n=n, act_n=act_n, act_v=act_v, mut_n=mut_n, mut_v=mut_v,
                b=b, c=c, p=mcnemar_exact(b, c),
                ab=ab, ac=ac, p_act=mcnemar_exact(ab, ac))


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return 0 if not n else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def act_volume(model, cat, variant_dir):
    """should-act coarse proxy: MEDIAN calls/item (mean/empty-turn corrupted by thrashers,
    some items loop to 100s of turns) + thrash rate (items with >20 turns)."""
    neu = load_results(os.path.join(NEUTRAL, "result", model, "multi_turn",
                                    f"BFCL_v4_multi_turn_{cat}_result.json"))
    var = load_results(os.path.join(variant_dir, "result", model, "multi_turn",
                                    f"BFCL_v4_multi_turn_{cat}_result.json"))
    ids = [i for i in neu if i in var]
    allowed = allowed_names(cat)

    def stats(res):
        per = [sum(len(turn_calls(t, allowed.get(i, set()))) for t in res[i]) for i in ids]
        thrash = sum(1 for i in ids if len(res[i]) > 20)
        return _median(per), thrash

    mn, tn = stats(neu)
    mv, tv = stats(var)
    return dict(n=len(ids), med_n=mn, med_v=mv, thrash_n=tn, thrash_v=tv)


def load_score_passfail(path):
    """BFCL score.json = header line + one line per FAIL. -> (n_total, set_of_fail_ids)."""
    lines = [l for l in open(path).read().strip().split("\n") if l.strip()]
    header = json.loads(lines[0])
    fails = {json.loads(l)["id"] for l in lines[1:]}
    return header, fails


def raw_paired(model, cat, variant_dir, neutral_dir):
    """Official BFCL PASS/FAIL, paired per item, neutral vs variant.

    The gemma-4-31B-it pro-caution arm has official score files as well, so the score-side
    effect is measured on a second model. `b` = variant-only
    pass (neutral FAIL -> variant PASS), `c` = neutral-only pass (the reverse); exact
    two-sided McNemar on the discordant pairs, the same test used for the behaviour flips.
    """
    vp = os.path.join(variant_dir, "score", model, "multi_turn", f"BFCL_v4_multi_turn_{cat}_score.json")
    np_ = os.path.join(neutral_dir, "score", model, "multi_turn", f"BFCL_v4_multi_turn_{cat}_score.json")
    if not (os.path.exists(vp) and os.path.exists(np_)):
        return None
    hv, fv = load_score_passfail(vp)
    hn, fn = load_score_passfail(np_)
    total = min(hv["total_count"], hn["total_count"])
    ids = [f"multi_turn_{cat}_{i}" for i in range(total)]
    b = sum(1 for i in ids if i in fn and i not in fv)      # fail -> pass
    c = sum(1 for i in ids if i not in fn and i in fv)      # pass -> fail
    return dict(n=total, acc_n=hn["accuracy"], acc_v=hv["accuracy"],
                cor_n=hn["correct_count"], cor_v=hv["correct_count"],
                b=b, c=c, p=mcnemar_exact(b, c))


VENV_SP = os.path.join(BFCL, "venv/lib/python3.11/site-packages")
# (label, model, variant_dir, neutral_dir-or-None, has_act_cats)
CONFIG = [
    ("pro-caution (expect mutate↓)", "gemma-4-31B-it", os.path.join(P5, "procaution"), None, True),
    ("pro-action (expect act/mutate↑)", "Qwen3.5-9B-FC", os.path.join(P5, "proaction"), None, True),
    ("pro-action (expect act/mutate↑)", "gpt-5.4-FC", os.path.join(P5, "proaction_gpt"), VENV_SP, False),
]
ASK = ["miss_func", "miss_param"]
ACT = ["base", "long_context"]

print("# Phase 5: prompt-injection arms, mechanical metrics (paired, no judge)\n")
print("> Calls are extracted with `analysis/bfcl_calls.py`: each of the three result formats (native")
print("> function-calling records, `<tool_call>` JSON blocks, gemma bracket groups parsed with BFCL's own")
print("> ast recipe) is parsed structurally, and every name is validated against the item's own tool list,")
print("> so prose never registers as a call. `mutate` below is the keyword tag of `bfcl_calls.py` over all")
print("> single-anchor items of a category. The paper's attempt rates use `mutator_map.json`, drop the bad")
print("> keys and the construction errors, and come from `p1t_phase5_attempt.py`.\n")
print("- The Qwen3.5-9B-FC pro-action arm received no injection (dose 0): `QwenFCHandler` does not read")
print("  `BFCL_EXTRA_SYSTEM_PROMPT`, and the first turn of its result file carries no injected system or")
print("  developer message. Its rows are kept for the record and are not used in the paper.")
print("- Selectivity: should-act activity (median calls per item, thrash count) barely moves under injection.\n")
print("## should-ask anchor turn (the turn whose gold trajectory is empty)\n")
print("The injection targets this turn: act and fill in a value (mutate), or hold back and ask. Paired McNemar on the direction of the flips.\n")
print("| nudge | model | cat | N | acted% neu→var | mutate% neu→var | flips v-only / n-only | McNemar p | acted flips v-only/n-only | McNemar p(acted) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for label, model, vdir, ndir, _ in CONFIG:
    for cat in ASK:
        vp = os.path.join(vdir, "result", model, "multi_turn", f"BFCL_v4_multi_turn_{cat}_result.json")
        if not os.path.exists(vp):
            print(f"| {label} | {model} | {cat} | — | (variant not run) | | | | | |")
            continue
        r = anchor_metric(model, cat, vdir, ndir)
        print(f"| {label} | {model} | {cat} | {r['n']} "
              f"| {100*r['act_n']/r['n']:.1f}→{100*r['act_v']/r['n']:.1f} "
              f"| {100*r['mut_n']/r['n']:.1f}→{100*r['mut_v']/r['n']:.1f} "
              f"| {r['b']} / {r['c']} | {r['p']:.3g} "
              f"| {r['ab']} / {r['ac']} | {r['p_act']:.3g} |")

print("\n## should-act activity (no single anchor turn; the median is robust to runaway items)\n")
print("Some items run away to hundreds of turns, which distorts the mean and the empty-turn count, so this reports median calls per item and the thrash count (items with more than 20 turns).\n")
print("| nudge | model | cat | N | median calls/item neu→var | thrash items (>20 turns) neu→var |")
print("|---|---|---|---|---|---|")
for label, model, vdir, ndir, has_act in CONFIG:
    if not has_act:
        continue                       # gpt pro-action runs ask cats only
    for cat in ACT:
        r = act_volume(model, cat, vdir)
        print(f"| {label} | {model} | {cat} | {r['n']} "
              f"| {r['med_n']:.1f}→{r['med_v']:.1f} "
              f"| {r['thrash_n']}→{r['thrash_v']} |")

# ---------------------------------------------------------------------------
# Official raw score, paired per item.
# The gemma-4-31B-it pro-caution arm was scored by the official BFCL scorer at the same
# time as its rollouts (bfcl/phase5_results/procaution/score/), so the score-side result is
# available on a SECOND model and in the OPPOSITE direction.
# ---------------------------------------------------------------------------
print("\n## Official raw score, paired per item\n")
print("Each item's official PASS/FAIL is read straight from the score file (one fail row per line after the header) and paired by id.")
print("`fail→pass` and `pass→fail` count the flips in each direction; McNemar is the exact binomial test on the discordant pairs.")
print("The denominator is 200 (the official one, bad keys included), so these rows cannot be subtracted from the behaviour rows above (denominator 198).\n")
print("| nudge | model | cat | N | raw% neu→var | Δpp | fail→pass | pass→fail | McNemar p |")
print("|---|---|---|---|---|---|---|---|---|")
for label, model, vdir, ndir, has_act in CONFIG:
    # raw_paired pairs ids 0..N-1, so it needs complete categories; the gpt-5.4 base arm
    # covers only the 153 base twins and is paired by id in p1t_phase5_attempt.py instead
    for cat in (ASK + ACT if has_act else ASK):
        r = raw_paired(model, cat, vdir, ndir or NEUTRAL)
        if r is None:
            continue
        print(f"| {label} | {model} | {cat} | {r['n']} "
              f"| {100*r['acc_n']:.1f}→{100*r['acc_v']:.1f} | {100*(r['acc_v']-r['acc_n']):+.1f} "
              f"| {r['b']} | {r['c']} | {r['p']:.3g} |")
print("\n- pro-caution on gemma-4-31B-it is the second arm with official scores, in the opposite direction")
print("  to pro-action on gpt-5.4: it lowers the mutate rate on the under-specified turn and gains no official")
print("  raw score. The behavioural effect sits on the under-specified turn of the should-ask categories, while")
print("  the significant raw drops fall mainly on the two should-act categories.")
print("- The Qwen3.5-9B-FC arm has dose 0 (the handler ignores the injection); its raw rows are kept for the record only.")
