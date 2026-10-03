#!/usr/bin/env python3
"""Build the release package for the paired action-decision measure (definition C).

The package is what the Reproducibility Statement promises: the 365 twin pairs as a
machine-readable manifest, a turn-level scorer that runs on any BFCL output directory
with the standard library alone, a verification script that recomputes every published
number from the frozen inputs and prints one PASS/FAIL line each, the 31 bad keys, the
held-back-turn classifier's prompt and per-turn outputs, and SHA-256 hashes.

WHAT THE PACKAGE NOW SCORES (2026-09-16)
----------------------------------------
The published definition used to be "any validated call on the anchor turn", over all 365
pairs. The paper's primary metric is now definition C:

    act when complete   on the base twin's anchor turn the model ATTEMPTED at least one
                        call that changes the world if it succeeds
    hold when missing   on the miss twin's anchor turn it attempted no such call
    decision accuracy   the mean of the two, so chance is exactly 50 on both sides

Whether a function changes the world is read off `analysis/mutator_map.json`, which labels
every function of the twelve API classes from its source; the file is shipped inside the
package so the predicate travels with it. The seven login/logout functions do not count,
and `echo` counts only when it writes a file (a second positional argument
or `file_name=`). An attempt counts whether or not the sandbox let it through: we measure
the decision, not the world.

Only the 231 pairs whose base-twin gold anchor turn itself contains a world-changing call
are candidates (117 miss_func + 114 miss_param). The other 134 are excluded because the
withheld information there gates a read-only lookup (132) or a bare login (2); they stay in
the manifest with their reason so the split is auditable.

ANCHOR CERTIFICATION (2026-09-16 night, p2c_cert.py)
----------------------------------------------------
All 231 candidates were then certified by hand: the authors ruled on every pair against a
fixed set of questions, reading the item alone and no model output. 8 are construction errors and
leave the scored set, so the primary N is 223 (116 miss_func + 107 miss_param, 151 distinct
base twins); 27 are compound turns that also authorize a world-changing call independent of
what was withheld, and they stay scored with an `independent` list, which defines the
credited reading of the hold side. Every one of the 231 carries its certification in the
manifest, and `certification/` ships the questions, the ruling and reason on every
pair, and the compound split. The paper's credited column additionally credits the miss_func
executor-replay workaround (p1z); that needs BFCL's executor and is NOT reproduced here.

Calls come out of the rollouts
with the same structural parse and name whitelist as `analysis/bfcl_calls_full.py`, and the
generator asserts that the shipped standalone scorer agrees with it on every anchor turn of
every paper model.

Number sources: `analysis/p1s_attempt_decision.md` (definitions C, C-credited and
C-uncertified, the C-login and C-reached sensitivities, spreads, rank-1 frequencies,
truncation counts) and
`analysis/p1v_c_companions.md` (official raw, by category, same-history subset, held-back
content, verdict-vs-attempt, base non-attempts, bad-keys-restored).

Nothing here reads or copies a credential. The classifier prompt is shipped as text and
the package never calls an API.

    python3 analysis/make_release_pairs.py            # -> release_pairs/

`release_pairs/` is a build product and is gitignored; this generator is the source.
"""
import csv
import collections
import hashlib
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "release_pairs")
SRC = os.path.join(HERE, "release_src")
sys.path.insert(0, HERE)

import p1d_nonact_rate as P1D  # noqa: E402
from bfcl_calls import allowed_names  # noqa: E402
from bfcl_calls_full import full_turn_calls  # noqa: E402
from p1d_nonact_rate import PAPER_MODELS as MODELS, SEED, B, anchor_turns  # noqa: E402
from p1f_matched_control import ASK, build_pairs, gold_shape, load_results  # noqa: E402
from p1g_paired_decision import fail_ids  # noqa: E402
from p1h_pair_checks import content_check, gold_calls  # noqa: E402
from p1i_holdback_content import prose  # noqa: E402
from p1s_attempt_decision import LOGIN, any_mut, is_mut  # noqa: E402
from p2c_cert import gated, load_final  # noqa: E402

CONSTRUCTION_REASON = "anchor certification: construction error"
# fields of p2a_anchor_cert_final.jsonl that travel into the manifest, in this order
CERT_FIELDS = ["withheld", "final_verdict", "final_reason", "revised_after_discussion",
               "independent", "gated"]

DISPLAY = {
    "gpt-5.4-FC": "gpt-5.4",
    "gemma-4-31B-it": "gemma-4-31B-it",
    "gemma-4-E4B-it": "gemma-4-E4B-it",
    "Qwen3.5-9B-FC": "Qwen3.5-9B",
    "Qwen3.6-27B-FC": "Qwen3.6-27B",
    "deepseek-v4-flash-FC": "deepseek-v4-flash",
    "qwen3.8-max-0902-FC": "Qwen3.8-Max",
}


def load_standalone():
    """Import the shipped scorer so the generator can check it against the reference."""
    spec = importlib.util.spec_from_file_location("release_score_pairs",
                                                  os.path.join(SRC, "score_pairs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pairing_drops():
    """Ids dropped before the content check, with the reason (build_pairs keeps counts)."""
    gb = gold_shape("base")
    rows = []
    for cat in ASK:
        gm = gold_shape(cat)
        anchors = anchor_turns(cat)
        for iid, shape in gm.items():
            if iid in P1D.EXCSET[cat]:
                rows.append({"id": iid, "stage": "pairing",
                             "reason": "on the manually verified bad-key list"})
                continue
            a = anchors.get(iid, [])
            if len(a) != 1:
                rows.append({"id": iid, "stage": "pairing",
                             "reason": "more than one turn has an empty reference "
                                       "trajectory, so the anchor would need judgment"})
                continue
            p = a[0]
            bid = "multi_turn_base_" + iid.rsplit("_", 1)[1]
            if bid not in gb:
                rows.append({"id": iid, "stage": "pairing", "reason": "no base twin"})
                continue
            if shape[:p] + shape[p + 1:] != gb[bid]:
                rows.append({"id": iid, "stage": "pairing",
                             "reason": "per-turn call counts differ from the twin"})
    return rows


def gold_anchor_of(gb, pairs, iid):
    bid, k, _ = pairs[iid]
    g = gb.get(bid) or []
    return list(g[k]) if k < len(g) else []


def exclusion_reason(names):
    if not names:
        return "gold anchor turn has no calls"
    if all(n in LOGIN for n in names):
        return "gold anchor turn is login only"
    if any(n in LOGIN for n in names):
        return "gold anchor turn is read-only apart from a login"
    return "gold anchor turn is read-only"


def restored_pairs():
    """The pairs that come back if the 31 bad keys are not excluded (definition C)."""
    saved = {c: set(P1D.EXCSET[c]) for c in ASK}
    for c in ASK:
        P1D.EXCSET[c].clear()
    p2, _ = build_pairs()
    ok2, _, _ = content_check(p2)
    for c in ASK:
        P1D.EXCSET[c].update(saved[c])
    return {i: p2[i] for i in ok2}


CERT_QUESTIONS = """\
Anchor certification: the questions answered by hand on every candidate pair, reading the
initial world state, the dialogue up to the anchor turn, the anchor turn in both versions,
the scripted follow-up on the incomplete side, the tool list with the withheld function
marked, and the reference calls. No model output was in view.

Q1  Is the withheld value stated in, or logically entailed by, the dialogue? A value that is
    merely discoverable in the environment, or guessable, does not count. (yes -> construction)
Q2  Is the withheld value or function actually needed by the reference call on that turn?
    (no -> construction)
Q3  Does the incomplete turn also authorize a world-changing step that does not depend on
    what was withheld, given the user's authorization and the simulator's preconditions?
    (yes -> compound; the independent steps are listed in compound_split.json)
Q4  Are the complete side's information and reference themselves adequate?
    (no -> construction)

Verdicts: valid, construction (leaves the scored set), compound (stays scored; the credited
reading does not count an attempt confined to the independent steps). adjudication.csv gives
the ruling and reason for every pair; revised=1 marks a first-pass ruling revised on a second
reading, with the final reason recorded.
"""


def build_manifest():
    pairs, drop = build_pairs()
    ok_ids, bad_gold, bad_pre = content_check(pairs)
    kept = {i: pairs[i] for i in ok_ids}
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    gb = gold_calls("base")
    construction, compound, cert_rows = load_final()

    # de-duplicate the tool lists: hundreds of items share a handful of class unions
    toolsets = {}

    def tkey(names):
        s = tuple(sorted(names))
        h = hashlib.sha256("\x00".join(s).encode()).hexdigest()[:12]
        toolsets.setdefault(h, list(s))
        return h

    def row_for(iid, bid, k, cat, mapping):
        gold = gold_anchor_of(gb, mapping, iid)
        names = [s.split("(")[0].strip() for s in gold]
        gm = any_mut([(n, s) for n, s in zip(names, gold)])
        cert = cert_rows.get(iid)
        bad = iid in construction
        row = {
            "miss_id": iid, "category": cat, "base_id": bid, "anchor_turn": k,
            "miss_tools": tkey(allow[cat].get(iid, set())),
            "base_tools": tkey(allow["base"].get(bid, set())),
            "gold_anchor": gold,
            "scored": bool(gm and not bad),
            "exclusion_reason": (CONSTRUCTION_REASON if bad else
                                 None if gm else exclusion_reason(names)),
            "certification": {f: cert[f] for f in CERT_FIELDS} if cert else None,
        }
        assert gm or cert is None, f"{iid}: certified but not a gold-mutator pair"
        return row

    rows = [row_for(i, *kept[i], kept) for i in sorted(kept)]
    certified = [r for r in rows if r["certification"]]
    assert len(certified) == len(cert_rows), \
        f"{len(cert_rows)} certified pairs but {len(certified)} in the manifest"
    rest = restored_pairs()
    extra = [row_for(i, *rest[i], rest) for i in sorted(rest)
             if i not in kept and any_mut([(s.split("(")[0].strip(), s)
                                           for s in gold_anchor_of(gb, rest, i)])]

    scored = [r for r in rows if r["scored"]]
    excl = [r for r in rows if not r["scored"]]
    # the read-only exclusions, i.e. everything dropped before the certification
    ro = [r for r in excl if r["exclusion_reason"] != CONSTRUCTION_REASON]
    freq = collections.Counter(n for r in ro for n in
                               (s.split("(")[0].strip() for s in r["gold_anchor"]))
    counts = {
        "should_ask_items": 400,
        "dropped_at_pairing": dict(drop),
        "shape_matched": len(kept) + len(bad_gold) + len(bad_pre),
        "dropped_at_content_check": {"gold strings differ": len(bad_gold),
                                     "pre-anchor user text differs": len(bad_pre)},
        "content_verified_pairs": len(rows),
        "certified_pairs": len(certified),
        "construction_errors": len(construction),
        "compound_pairs": len(compound),
        "certification_verdicts": dict(collections.Counter(
            r["certification"]["final_verdict"] for r in certified)),
        "scored_pairs": len(scored),
        "excluded_pairs": len(excl),
        "scored_miss_func": sum(1 for r in scored if r["category"] == "miss_func"),
        "scored_miss_param": sum(1 for r in scored if r["category"] == "miss_param"),
        "excluded_miss_func": sum(1 for r in excl if r["category"] == "miss_func"),
        "excluded_miss_param": sum(1 for r in excl if r["category"] == "miss_param"),
        "excluded_reasons": dict(collections.Counter(r["exclusion_reason"] for r in excl)),
        "read_only_excluded_pairs": len(ro),
        "read_only_excluded_miss_func": sum(1 for r in ro if r["category"] == "miss_func"),
        "read_only_excluded_miss_param": sum(1 for r in ro if r["category"] == "miss_param"),
        # the eight most common gold functions on the read-only excluded anchors, as a
        # mapping: two of them tie at 9, and a list would assert an order the md lacks
        "excluded_top_gold_functions": dict(
            sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:8]),
        "scored_pairs_login": sum(
            1 for r in rows if r["exclusion_reason"] != CONSTRUCTION_REASON
            and any_mut([(s.split("(")[0].strip(), s) for s in r["gold_anchor"]], True)),
        "distinct_base_twins_all": len({r["base_id"] for r in rows}),
        "distinct_base_twins_scored": len({r["base_id"] for r in scored}),
        "bad_key_restored_pairs": len(scored) + len(extra),
        "bad_key_restored_added": len(extra),
    }
    dropped = pairing_drops() + \
        [{"id": i, "stage": "content check", "reason": "gold call strings differ from the twin"}
         for i in sorted(bad_gold)] + \
        [{"id": i, "stage": "content check", "reason": "dialogue before the anchor was rewritten"}
         for i in sorted(bad_pre)]
    man = {
        "description": ("Twin pairs for the action decision on BFCL multi-turn. Each row is a "
                        "should-ask item, the base item whose reference trajectory it reproduces "
                        "once the empty anchor turn is deleted, and the index of that anchor turn. "
                        "On the base turn the model should change the world; on the miss turn the "
                        "official task definition says it should ask."),
        "definition": (
            "Definition C. act = on the base twin's anchor turn the model attempted at least "
            "one call that changes the world on success; hold = on the miss twin's anchor turn "
            "it attempted none. World-changing is read off mutator_map.json (labels mutate / "
            "read / login); the seven login/logout functions do not count, and echo counts only "
            "when it writes a file. An attempt counts whether or not the sandbox accepted it. "
            "Only the pairs whose base-twin gold anchor turn itself contains a world-changing "
            "call are candidates; the rest carry an exclusion_reason."),
        "certification": (
            "Every one of the " + str(len(certified)) + " candidate pairs was certified item "
            "by item, by hand (see certification/): the authors answered a fixed set of "
            "questions about the item alone, with no model output in front of them. "
            + str(len(construction)) + " pairs are construction errors — the withheld value "
            "is stated, entailed or unused, or the complete side's own reference is wrong — "
            "so acting on the missing side is correct there and they leave the scored set "
            "(scored=false, exclusion_reason \"" + CONSTRUCTION_REASON + "\"). "
            + str(len(compound)) + " pairs are compound turns whose missing side also "
            "authorizes a world-changing call that does not depend on what was withheld; "
            "they stay scored, and their certification lists those calls under "
            "\"independent\". Strict hold = no world-changing attempt at all; credited hold = "
            "no world-changing attempt outside that list. The base (act) side is the same "
            "under both. The paper's credited column additionally credits the miss_func "
            "executor-replay workaround, which needs BFCL's executor and is not reproduced "
            "in this package."),
        "counts": counts,
        "checks_every_pair_passed": [
            "the should-ask item has exactly one turn whose reference trajectory is empty",
            "deleting that turn reproduces the base item's per-turn call counts",
            "the reference call strings match the base item's turn for turn",
            "every user turn before the anchor is character-identical in the two items",
            "the item is not on the bad-key list",
        ],
        "mutator_map": "mutator_map.json",
        "toolsets": toolsets,
        "pairs": rows,
        "bad_key_restored_pairs": extra,
        "dropped": dropped,
    }
    return man, kept, rows, extra


# --------------------------------------------------------------------- measurement
def measure(model, rows_spec, allow, compound=None):
    """Per-pair facts under the reference implementation, keyed by miss id."""
    res = {c: load_results(model, c) for c in ASK + ["base"]}
    if any(v is None for v in res.values()):
        return None
    compound = compound or {}
    out = {}
    for row in rows_spec:
        iid, bid, k, cat = (row["miss_id"], row["base_id"], row["anchor_turn"],
                            row["category"])
        sm, sb = res[cat].get(iid), res["base"].get(bid)
        if sm is None or sb is None:
            continue
        tm_ = allow[cat].get(iid, set())
        tb_ = allow["base"].get(bid, set())
        rm, rb = k < len(sm), k < len(sb)
        turn_m, turn_b = (sm[k] if rm else []), (sb[k] if rb else [])
        cm = full_turn_calls(turn_m, tm_)
        cb = full_turn_calls(turn_b, tb_)
        hm = [[n for n, _ in full_turn_calls(sm[j], tm_)] for j in range(min(k, len(sm)))]
        hb = [[n for n, _ in full_turn_calls(sb[j], tb_)] for j in range(min(k, len(sb)))]
        out[iid] = dict(cat=cat, base=bid, turn_m=turn_m, turn_b=turn_b,
                        act=any_mut(cb), hold=not any_mut(cm),
                        holdG=not any(is_mut(n, s) and gated(n, iid, compound)
                                      for n, s in cm),
                        actL=any_mut(cb, True), holdL=not any_mut(cm, True),
                        n_b=len(cb), n_m=len(cm), reached=(rm and rb),
                        reached_b=rb, reached_m=rm, text_m=bool(prose(turn_m)),
                        same=(hb == hm and len(hb) == k and len(hm) == k))
    return out


def boot_ci(v, twin, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.array(twin)
    groups = [np.where(arr == t)[0] for t in sorted(set(twin))]
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        out[b] = v[np.concatenate([groups[p] for p in pick])].mean() * 100
    return np.percentile(out, [2.5, 97.5])


def rank1(accs, names, twin):
    arr = np.array(twin)
    groups = [np.where(arr == t)[0] for t in sorted(set(twin))]
    M = np.vstack([accs[m] for m in names])
    rng = np.random.default_rng(SEED)
    first = np.zeros(len(names))
    for _ in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        sel = np.concatenate([groups[p] for p in pick])
        first[M[:, sel].mean(1).argmax()] += 1
    # f / B first, then scale, exactly as p1s does: a frequency that lands on a .x5
    # boundary (C-login's 1195/10000) otherwise rounds the other way than the paper's
    return {n: round(100 * (f / B), 1) for n, f in zip(names, first)}


def one_table(rec, names, ids_of, base_of, a="act", h="hold"):
    tab, accs, sizes = {}, {}, {}
    for m in names:
        rows = rec[m]
        ids = ids_of(m, rows)
        av = np.array([rows[i][a] for i in ids], float)
        hv = np.array([rows[i][h] for i in ids], float)
        acc = (av + hv) / 2
        lo, hi = boot_ci(acc, [base_of[i] for i in ids])
        tab[m] = {"display": DISPLAY.get(m, m), "n": len(ids),
                  "act": round(100 * av.mean(), 1), "hold": round(100 * hv.mean(), 1),
                  "decision": round(100 * acc.mean(), 1),
                  "ci95": [round(lo, 1), round(hi, 1)],
                  "both": round(100 * float(np.mean(av * hv)), 1)}
        accs[m] = acc
        sizes[m] = ids
    # from the unrounded means, as p1s does: rounding first can move the spread 0.1
    means = [100 * accs[m].mean() for m in names]
    spread = round(max(means) - min(means), 1)
    r1 = None
    if len({len(sizes[m]) for m in names}) == 1:
        r1 = rank1(accs, names, [base_of[i] for i in sizes[names[0]]])
    return tab, spread, r1


def build_expected(rows, extra):
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    # the credited reading, rebuilt from the manifest rows rather than re-read from disk
    compound = {r["miss_id"]: {"independent": r["certification"]["independent"],
                               "gated": r["certification"]["gated"]}
                for r in rows if r["certification"]
                and r["certification"]["final_verdict"] == "compound"}
    rec, rec_extra = {}, {}
    for m in MODELS:
        r = measure(m, rows, allow, compound)
        if r is None:
            continue
        rec[m] = r
        rec_extra[m] = measure(m, extra, allow, compound) or {}
    names = [m for m in MODELS if m in rec]
    by_id = {r["miss_id"]: r for r in rows + extra}
    base_of = {r["miss_id"]: r["base_id"] for r in rows + extra}
    C = [r["miss_id"] for r in rows if r["scored"]]
    # C-uncertified: the gold-mutator set before the certification, construction errors in
    CU = [r["miss_id"] for r in rows
          if any_mut([(s.split("(")[0].strip(), s) for s in r["gold_anchor"]])]
    CL = [r["miss_id"] for r in rows
          if r["exclusion_reason"] != CONSTRUCTION_REASON
          and any_mut([(s.split("(")[0].strip(), s) for s in r["gold_anchor"]], True)]

    tabs, spread, r1s = {}, {}, {}
    tabs["C"], spread["C"], r1s["C"] = one_table(
        rec, names, lambda m, rw: [i for i in C if i in rw], base_of)
    tabs["C-credited"], spread["C-credited"], r1s["C-credited"] = one_table(
        rec, names, lambda m, rw: [i for i in C if i in rw], base_of, "act", "holdG")
    tabs["C-uncertified"], spread["C-uncertified"], r1s["C-uncertified"] = one_table(
        rec, names, lambda m, rw: [i for i in CU if i in rw], base_of)
    tabs["C-login"], spread["C-login"], r1s["C-login"] = one_table(
        rec, names, lambda m, rw: [i for i in CL if i in rw], base_of, "actL", "holdL")
    tabs["C-reached"], spread["C-reached"], r1s["C-reached"] = one_table(
        rec, names, lambda m, rw: [i for i in C if i in rw and rw[i]["reached"]], base_of)

    trunc = {m: [sum(1 for r in rec[m].values() if not r["reached_b"]),
                 sum(1 for r in rec[m].values() if not r["reached_m"])] for m in names}

    def acc_mean(rw, ids, a="act", h="hold"):
        av = np.array([rw[i][a] for i in ids], float)
        hv = np.array([rw[i][h] for i in ids], float)
        return 100 * av.mean(), 100 * hv.mean(), 50 * (av.mean() + hv.mean())

    ro = [r for r in rows if not r["scored"]
          and r["exclusion_reason"] != CONSTRUCTION_REASON]
    RO = [r["miss_id"] for r in ro]
    anchors = sorted({(by_id[i]["base_id"], by_id[i]["anchor_turn"]) for i in C})

    bycat, sameh, heldc, noatt, badk, exclat, peranchor = {}, {}, {}, {}, {}, {}, {}
    pooled_h = pooled_s = 0
    for m in names:
        rw = rec[m]
        ids = [i for i in C if i in rw]
        xids = [i for i in RO if i in rw]
        exclat[m] = {
            "n": len(xids),
            "act_base": round(100 * float(np.mean([rw[i]["act"] for i in xids])), 1),
            "act_miss": round(100 * float(np.mean([not rw[i]["hold"] for i in xids])), 1),
            "either": round(100 * float(np.mean(
                [rw[i]["act"] or not rw[i]["hold"] for i in xids])), 1)}
        seen = {}
        for i in ids:
            seen[(by_id[i]["base_id"], by_id[i]["anchor_turn"])] = rw[i]["act"]
        peranchor[m] = {
            "anchors": len(seen),
            "act_per_pair": round(100 * float(np.mean([rw[i]["act"] for i in ids])), 1),
            "act_per_anchor": round(100 * float(np.mean(list(seen.values()))), 1)}
        bycat[m] = {}
        for cat in ASK:
            sub = [i for i in ids if rw[i]["cat"] == cat]
            a_, h_, d_ = acc_mean(rw, sub)
            bycat[m][cat] = {"n": len(sub), "act": round(a_, 1), "hold": round(h_, 1),
                             "decision": round(d_, 1)}
        sub = [i for i in ids if rw[i]["same"]]
        a_, h_, d_ = acc_mean(rw, sub)
        sameh[m] = {"n": len(sub), "act": round(a_, 1), "hold": round(h_, 1),
                    "decision": round(d_, 1)}
        held = [i for i in ids if rw[i]["hold"]]
        tx = sum(1 for i in held if rw[i]["text_m"])
        heldc[m] = {"held": len(held),
                    "read_only": sum(1 for i in held if rw[i]["n_m"] > 0),
                    "text": tx, "silent": len(held) - tx}
        pooled_h += len(held)
        pooled_s += len(held) - tx
        no = [i for i in ids if not rw[i]["act"]]
        z = sum(1 for i in no if rw[i]["n_b"] == 0)
        noatt[m] = {"n": len(no), "zero_calls": z, "read_only": len(no) - z}
        d1 = acc_mean(rw, ids)[2]
        allrows = dict(rw)
        allrows.update(rec_extra[m])
        ids2 = ids + list(rec_extra[m])
        d2 = acc_mean(allrows, ids2)[2]
        badk[m] = {"decision": round(d1, 1), "restored": round(d2, 1),
                   "change": round(d2 - d1, 2)}

    # official verdict
    base_ids = {base_of[i] for i in C}
    raw_m, raw_b, dec, vva = {}, {}, {}, {}
    for m in names:
        f = {c: fail_ids(m, c) for c in ASK + ["base"]}
        if any(v is None for v in f.values()):
            continue
        raw_m[m] = 100 * sum(1 for i in C if i not in f[by_id[i]["category"]]) / len(C)
        raw_b[m] = 100 * sum(1 for b in base_ids if b not in f["base"]) / len(base_ids)
        rw = rec[m]
        ids = [i for i in C if i in rw]
        dec[m] = acc_mean(rw, ids)[2]
        p = [i for i in C if i not in f[by_id[i]["category"]]]
        q = [i for i in C if i in f[by_id[i]["category"]]]
        vva[m] = {"passes": len(p),
                  "pass_attempt": round(100 * float(np.mean([not rw[i]["hold"] for i in p])), 1),
                  "fails": len(q),
                  "fail_attempt": round(100 * float(np.mean([not rw[i]["hold"] for i in q])), 1)}
    ordn = [m for m in names if m in raw_m]
    rm = spearmanr([raw_m[n] for n in ordn], [dec[n] for n in ordn])
    rb = spearmanr([raw_b[n] for n in ordn], [dec[n] for n in ordn])

    return {
        "definition": "C",
        "models_order": names,
        "bootstrap": {"B": B, "seed": SEED,
                      "resampled_unit": "the base twin each pair shares, not the pair",
                      "streams": "one fresh stream per model for the intervals, one "
                                 "shared resample for the rank-1 frequency"},
        "counts": {k: v for k, v in [
            ("content_verified_pairs", len(rows)),
            ("certified_pairs", len(CU)),
            ("construction_errors", len(CU) - len(C)),
            ("compound_pairs", len(compound)),
            ("certification_verdicts", dict(collections.Counter(
                r["certification"]["final_verdict"] for r in rows if r["certification"]))),
            ("scored_pairs", len(C)),
            ("excluded_pairs", len(rows) - len(C)),
            ("scored_pairs_login", len(CL)),
            ("scored_miss_func", sum(1 for i in C if by_id[i]["category"] == "miss_func")),
            ("scored_miss_param", sum(1 for i in C if by_id[i]["category"] == "miss_param")),
            ("excluded_miss_func", sum(1 for r in rows
                                       if not r["scored"] and r["category"] == "miss_func")),
            ("excluded_miss_param", sum(1 for r in rows
                                        if not r["scored"] and r["category"] == "miss_param")),
            ("excluded_reasons", dict(collections.Counter(
                r["exclusion_reason"] for r in rows if not r["scored"]))),
            ("read_only_excluded_pairs", len(ro)),
            ("read_only_excluded_miss_func",
             sum(1 for r in ro if r["category"] == "miss_func")),
            ("read_only_excluded_miss_param",
             sum(1 for r in ro if r["category"] == "miss_param")),
            ("excluded_top_gold_functions", dict(sorted(
                collections.Counter(s.split("(")[0].strip() for r in ro
                                    for s in r["gold_anchor"]).items(),
                key=lambda x: (-x[1], x[0]))[:8])),
            ("distinct_base_twins_all", len({r["base_id"] for r in rows})),
            ("distinct_base_twins_scored", len(base_ids)),
            ("distinct_base_anchor_turns_scored", len(anchors)),
            ("bad_key_restored_pairs", len(C) + len(extra)),
            ("bad_key_restored_added", len(extra)),
        ]},
        "tables": tabs,
        "spread": spread,
        "rank1": {k: r1s[k] for k in ("C", "C-credited", "C-uncertified", "C-login")},
        "truncation": trunc,
        "companions": {
            "by_category": bycat,
            "excluded_attempts": exclat,
            "act_per_anchor": peranchor,
            "same_history": sameh,
            "heldback_content": heldc,
            "heldback_pooled": {"held": pooled_h, "silent": pooled_s},
            "base_no_attempt": noatt,
            "bad_keys_restored": {"pairs": len(C) + len(extra), "added": len(extra),
                                  "models": badk},
            "distinct_base_twins": len(base_ids),
            "official_raw": {m: {"miss": round(raw_m[m], 1), "base": round(raw_b[m], 1),
                                 "decision": round(dec[m], 1)} for m in ordn},
            "verdict_vs_attempt": vva,
            "spearman": {"miss_rho": round(rm.correlation, 2),
                         "miss_p": round(rm.pvalue, 2),
                         "base_rho": round(rb.correlation, 2),
                         "base_p": round(rb.pvalue, 2)},
            "ranks": {"official": sorted(ordn, key=lambda x: -raw_m[x]),
                      "decision": sorted(ordn, key=lambda x: -dec[x])},
        },
    }, rec


# --------------------------------------------------------- scorer agreement check
def check_scorer(mod, rows, rec):
    """The shipped standalone scorer must agree with bfcl_calls_full + p1s on every
    anchor turn of every paper model, for the calls, the predicate, the login variant,
    and the credited (compound-gated) reading of the hold side."""
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    mut = mod.Mutators(os.path.join(HERE, "mutator_map.json"))
    by_id = {r["miss_id"]: r for r in rows}
    n = 0
    for m, rw in rec.items():
        for iid, r in rw.items():
            row = by_id[iid]
            miss_calls = full_turn_calls(r["turn_m"], allow[r["cat"]].get(iid, set()))
            for turn, tools, ref_calls in (
                    (r["turn_m"], allow[r["cat"]].get(iid, set()), miss_calls),
                    (r["turn_b"], allow["base"].get(r["base"], set()),
                     full_turn_calls(r["turn_b"], allow["base"].get(r["base"], set())))):
                got = mod.turn_calls(turn, tools)
                assert got == ref_calls, f"{m} {iid}: {got} != {ref_calls}"
                assert mut.any_mut(got) == any_mut(ref_calls), f"{m} {iid}: mut differs"
                assert mut.any_mut(got, True) == any_mut(ref_calls, True), \
                    f"{m} {iid}: mut(login) differs"
                n += 1
            # act / hold / hold_credited, the three numbers the package publishes
            assert mut.any_mut(mod.turn_calls(
                r["turn_b"], allow["base"].get(r["base"], set()))) == r["act"], \
                f"{m} {iid}: act differs"
            assert (not mut.any_mut(mod.turn_calls(
                r["turn_m"], allow[r["cat"]].get(iid, set())))) == r["hold"], \
                f"{m} {iid}: hold differs"
            assert (not mut.any_mut_gated(
                mod.turn_calls(r["turn_m"], allow[r["cat"]].get(iid, set())),
                mod.independent_of(row))) == r["holdG"], \
                f"{m} {iid}: hold_credited differs"
            assert row["scored"] == (mut.any_mut(mod.gold_anchor_calls(row))
                                     and not mod.is_construction_error(row)), \
                f"{iid}: scored flag differs"
    return n


def sha_dir(path):
    lines = []
    for base, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(files):
            if f == "MANIFEST.sha256":
                continue
            fp = os.path.join(base, f)
            h = hashlib.sha256(open(fp, "rb").read()).hexdigest()
            lines.append(f"{h}  {os.path.relpath(fp, path)}")
    return "\n".join(sorted(lines, key=lambda l: l.split("  ", 1)[1])) + "\n"


def main():
    manifest, kept, rows, extra = build_manifest()
    c = manifest["counts"]
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "holdback"))
    os.makedirs(os.path.join(OUT, "certification"))
    json.dump(manifest, open(os.path.join(OUT, "pairs.json"), "w"), indent=1)
    print(f"pairs.json: {c['content_verified_pairs']} pairs "
          f"({c['scored_pairs']} scored, {c['excluded_pairs']} excluded: "
          f"{c['excluded_reasons']}), {len(manifest['toolsets'])} distinct tool lists, "
          f"{len(manifest['dropped'])} earlier drops listed")
    print(f"certification: {c['certified_pairs']} pairs certified, "
          f"{c['certification_verdicts']}")

    open(os.path.join(OUT, "certification", "questions.txt"), "w").write(CERT_QUESTIONS)
    _, _, cert_rows = load_final()
    with open(os.path.join(OUT, "certification", "adjudication.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["miss_id", "category", "anchor_turn", "withheld", "verdict", "reason",
                    "revised"])
        for iid in sorted(cert_rows, key=lambda i: (cert_rows[i]["category"],
                                                   int(i.rsplit("_", 1)[1]))):
            r = cert_rows[iid]
            w.writerow([iid, r["category"], r["anchor_turn"], ";".join(r["withheld"]),
                        r["final_verdict"], r["final_reason"],
                        int(r["revised_after_discussion"])])
    print(f"certification/adjudication.csv: {len(cert_rows)} rulings")
    shutil.copyfile(os.path.join(HERE, "p2a_cert", "compound_split.json"),
                    os.path.join(OUT, "certification", "compound_split.json"))

    shutil.copyfile(os.path.join(HERE, "mutator_map.json"),
                    os.path.join(OUT, "mutator_map.json"))

    expected, rec = build_expected(rows, extra)
    json.dump(expected, open(os.path.join(OUT, "expected.json"), "w"), indent=1)
    print(f"expected.json: {len(expected['tables']['C'])} models, "
          f"{len(expected['models_order'])} with rollouts")

    mod = load_standalone()
    n = check_scorer(mod, rows, rec)
    print(f"standalone scorer agrees with bfcl_calls_full + p1s on {n} anchor turns "
          f"({len(rec)} models x {len(rows)} pairs x 2 sides)")

    for src, dst in [(os.path.join(ROOT, "release", "bad_keys.json"), "bad_keys.json"),
                     (os.path.join(HERE, "p1x_holdback_says.jsonl"), "holdback/labels.jsonl"),
                     (os.path.join(HERE, "p1x_holdback_says.md"), "holdback/summary.md")]:
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(OUT, dst))
            print(f"copied {dst}")
        else:
            print(f"WARNING: {os.path.relpath(src, ROOT)} not found, {dst} not shipped")

    try:
        import p1x_holdback_says as P1X
        open(os.path.join(OUT, "holdback", "prompt.txt"), "w").write(P1X.SYS.rstrip() + "\n")
    except Exception as e:                                          # pragma: no cover
        print(f"WARNING: could not ship the classifier prompt ({e})")

    for f in ("score_pairs.py", "verify_pairs.py", "README.md"):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(OUT, f))
    for f in ("score_pairs.py", "verify_pairs.py"):
        os.chmod(os.path.join(OUT, f), 0o755)

    open(os.path.join(OUT, "MANIFEST.sha256"), "w").write(sha_dir(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
