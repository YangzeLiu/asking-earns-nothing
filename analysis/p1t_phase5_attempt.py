#!/usr/bin/env python3
"""P1-t: Phase 5 anchor-turn effect under the ATTEMPTED definition of a state change.

The paper uses one definition of "acting" everywhere: the model attempted a call that
changes the world on success (`mutator_map.json`, echo decided per call, the seven
login/logout functions excluded), whether or not the sandbox let it through. This
recomputes the prompt-injection arms under that definition, so the injection analysis
uses the same predicate as the main table.

Differences from `phase5_mechanical.py`: (1) the predicate is the attempted
world-changing call of `mutator_map.json`, not the keyword tag; (2) the 31 manually
verified bad keys in `phase4_exclude.json` and the certified construction errors are
dropped, as the main table drops them (phase5_mechanical keeps all 198 single-anchor
items per category).

Output: analysis/p1t_phase5_attempt.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import allowed_names  # noqa: E402
from bfcl_calls_full import full_turn_calls  # noqa: E402
from phase5_mechanical import (  # noqa: E402
    VENV_SP, load_score_passfail,
    CONFIG, ASK, NEUTRAL, PA_DIR, P5, anchor_turns, load_results, mcnemar_exact, raw_paired,
)
from p1d_nonact_rate import EXCSET  # noqa: E402
from p2c_cert import load_final  # noqa: E402

# the certified construction errors of the anchor certification (p2c_cert.py): on those
# items the withheld value is stated, entailed or unused, so acting there is correct and
# they are dropped from the under-specified-turn attempt rate as from Table 1
CONSTRUCTION = load_final()[0]
from p1s_attempt_decision import any_mut  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# arms not in phase5_mechanical.py's CONFIG; (label, model, variant_dir, neutral_dir, _)
EXTRA = [
    ("pro-action (expect act/mutate↑)", "gemma-4-31B-it", os.path.join(P5, "proaction_gemma"), None, True),
    # gpt-5.4 base arm: only the 153 base twins of the 231 pairs were generated,
    # so the official raw is paired over the ids present in the variant result file.
    ("pro-action (expect act/mutate↑)", "gpt-5.4-FC", os.path.join(P5, "proaction_gpt"), VENV_SP, True),
]
ALL_CATS = ["base", "long_context", "miss_func", "miss_param"]


def raw_paired_present(model, cat, variant_dir, neutral_dir):
    """raw_paired, but over the ids present in the variant result file (partial arms)."""
    import json
    rp = os.path.join(variant_dir, "result", model, "multi_turn", f"BFCL_v4_multi_turn_{cat}_result.json")
    vp = os.path.join(variant_dir, "score", model, "multi_turn", f"BFCL_v4_multi_turn_{cat}_score.json")
    np_ = os.path.join(neutral_dir, "score", model, "multi_turn", f"BFCL_v4_multi_turn_{cat}_score.json")
    if not (os.path.exists(rp) and os.path.exists(vp) and os.path.exists(np_)):
        return None
    ids = [json.loads(l)["id"] for l in open(rp)]
    _, fv = load_score_passfail(vp)
    _, fn = load_score_passfail(np_)
    b = sum(1 for i in ids if i in fn and i not in fv)
    c = sum(1 for i in ids if i not in fn and i in fv)
    n = len(ids)
    return dict(n=n, acc_n=sum(1 for i in ids if i not in fn) / n, acc_v=sum(1 for i in ids if i not in fv) / n,
                b=b, c=c, p=mcnemar_exact(b, c))


def arm(model, cat, variant_dir, neutral_dir, allowed, pa):
    neu = load_results(os.path.join(neutral_dir or NEUTRAL, "result", model, "multi_turn",
                                    f"BFCL_v4_multi_turn_{cat}_result.json"))
    var = load_results(os.path.join(variant_dir, "result", model, "multi_turn",
                                    f"BFCL_v4_multi_turn_{cat}_result.json"))
    ids = sorted(i for i in neu if i in var and i in pa and len(pa[i]) == 1
                 and i not in EXCSET[cat] and i not in CONSTRUCTION)
    by = {}
    for i in ids:
        k = pa[i][0]
        al = allowed.get(i, set())
        for res, who in ((neu, "n"), (var, "v")):
            seq = res[i]
            by[(i, who)] = any_mut(full_turn_calls(seq[k] if k < len(seq) else [], al))
    n = len(ids)
    b = sum(1 for i in ids if by[(i, "v")] and not by[(i, "n")])
    c = sum(1 for i in ids if by[(i, "n")] and not by[(i, "v")])
    return dict(n=n, mut_n=sum(by[(i, "n")] for i in ids),
                mut_v=sum(by[(i, "v")] for i in ids), b=b, c=c, p=mcnemar_exact(b, c))


def main():
    L = ["# P1-t Phase 5 anchor turn under the attempted definition of a state change", "",
         "Same arms and McNemar as `phase5_mechanical.md`; \"acting\" = attempted a call",
         "that changes the world on success (`mutator_map.json`), landed or not; the 31",
         "hand-verified bad keys are dropped as in the main table. The Qwen3.5-9B-FC",
         "pro-action arm received no injection (dose 0) and is listed for the record only.", "",
         "| nudge | model | cat | N | attempted-mutation% neu->var | flips v-only / n-only "
         "| McNemar p |", "|---|---|---|---|---|---|---|"]
    for cat in ASK:
        allowed = allowed_names(cat)
        pa = anchor_turns(os.path.join(PA_DIR, f"BFCL_v4_multi_turn_{cat}.json"))
        for label, model, vdir, ndir, _ in CONFIG + EXTRA:
            vp = os.path.join(vdir, "result", model, "multi_turn",
                              f"BFCL_v4_multi_turn_{cat}_result.json")
            if not os.path.exists(vp):
                continue
            r = arm(model, cat, vdir, ndir, allowed, pa)
            L.append(f"| {label} | {model} | {cat} | {r['n']} "
                     f"| {100*r['mut_n']/r['n']:.1f}->{100*r['mut_v']/r['n']:.1f} "
                     f"| {r['b']} / {r['c']} | {r['p']:.3g} |")
    L += ["", "## Official raw for the arms not in phase5_mechanical.md", "",
          "| nudge | model | cat | N | raw% neu->var | flips fail->pass / pass->fail | McNemar p |",
          "|---|---|---|---|---|---|---|"]
    for label, model, vdir, ndir, _ in EXTRA:
        for cat in ALL_CATS:
            r = raw_paired_present(model, cat, vdir, ndir or NEUTRAL)
            if r is None:
                continue
            L.append(f"| {label} | {model} | {cat} | {r['n']} | {100*r['acc_n']:.1f}->{100*r['acc_v']:.1f} "
                     f"| {r['b']} / {r['c']} | {r['p']:.3g} |")
    txt = "\n".join(L)
    open(os.path.join(HERE, "p1t_phase5_attempt.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
