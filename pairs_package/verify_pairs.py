#!/usr/bin/env python3
"""Recompute every published number of the paired action-decision measure and diff it.

    python3 verify_pairs.py --results /path/to/bfcl/result
    python3 verify_pairs.py --results DIR --results DIR2 --scores /path/to/bfcl/score

Needs NumPy for the bootstrap; `score_pairs.py` alone needs nothing but the standard
library. Every check prints PASS or FAIL with the expected and the recomputed value. A
check whose inputs are not present locally prints SKIP, and a SKIP is never a PASS. The
exit status is non-zero if anything failed.

Checked against `expected.json`:

  * the manifest itself: 365 content-verified pairs, the 231 certified candidates, the 8
    construction errors and 27 compound turns the certification found, the 223 scored
    under definition C and the 142 excluded with their reason, per category, distinct base
    twins, and that the shipped mutator map plus the certification reproduce the `scored`
    flag of every row;
  * definition C, per model: act when complete, hold when missing, decision accuracy and
    its cluster-bootstrap interval, both-twins-correct, N; the spread across models; the
    rank-1 frequency of each model under one shared resample;
  * definition C-credited, the same 223 pairs with the compound turns credited: on those
    an attempt confined to the pair's certified independent steps is not counted as acting;
  * the three sensitivities: C-uncertified (all 231 candidates, i.e. the numbers before the
    certification), C-login (the seven login/logout functions counted as world-changing,
    225 pairs) and C-reached (only pairs where both rollouts reached the anchor turn,
    per-model N);
  * the rollouts that never reached the anchor turn, per model, per side;
  * the companion numbers: decision accuracy by category, the identical-pre-anchor-history
    subset, what the held-back turns contain, the base anchors with no attempt, the
    bad-keys-restored sensitivity, the world-changing attempts on the read-only excluded
    pairs, and the act rate counted once per distinct base anchor turn;
  * with `--scores`, the official raw verdict on the 223 miss items and on their base
    twins, the Spearman correlation against decision accuracy, the two rankings, and the
    attempt rate among official passes and failures.

The interval resamples the base twin each pair shares, not the pair: the 223 scored pairs
rest on 151 distinct base items, most of which serve both a miss_func and a miss_param
item, so resampling rows would treat correlated rows as independent evidence. One fresh
random stream per model, so a model's interval does not depend on the order models are
listed in.
"""
import argparse
import json
import math
import os
import sys

sys.dont_write_bytecode = True      # no __pycache__ inside the shipped package

import numpy as np

import score_pairs as S  # noqa: E402

RESULTS = []


def record(status, name, expected, got):
    RESULTS.append((status, name, expected, got))


def check(name, expected, got, tol=0.05):
    if expected is None:
        return record("SKIP", name, "not published", "")
    if isinstance(expected, (list, str)):
        return record("PASS" if list(expected) == list(got) else "FAIL", name, expected, got)
    ok = abs(float(expected) - float(got)) <= tol
    record("PASS" if ok else "FAIL", name, expected, round(float(got), 3))


def skip(name, why):
    record("SKIP", name, why, "")


# ------------------------------------------------------------------- statistics
def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-300:
            d = 1e-300
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b) (Numerical Recipes continued fraction)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf2(t, df):
    """Two-sided tail of Student's t, the same quantity scipy's spearmanr reports."""
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def rankdata(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(x, y):
    rx, ry = np.array(rankdata(x)), np.array(rankdata(y))
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if n <= 2 or abs(r) >= 1.0:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return r, t_sf2(abs(t), n - 2)


def boot_ci(v, twin, seed, B):
    """Cluster bootstrap on the base twin; one fresh stream per call, as in the paper."""
    rng = np.random.default_rng(seed)
    arr = np.array(twin)
    groups = [np.where(arr == t)[0] for t in sorted(set(twin))]
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        out[b] = v[np.concatenate([groups[p] for p in pick])].mean() * 100
    return np.percentile(out, [2.5, 97.5])


def rank1(acc_by_model, names, twin, seed, B):
    arr = np.array(twin)
    groups = [np.where(arr == t)[0] for t in sorted(set(twin))]
    M = np.vstack([acc_by_model[m] for m in names])
    rng = np.random.default_rng(seed)
    first = np.zeros(len(names))
    for _ in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        sel = np.concatenate([groups[p] for p in pick])
        first[M[:, sel].mean(1).argmax()] += 1
    return {n: 100 * f / B for n, f in zip(names, first)}


# ------------------------------------------------------------------- measurement
def measure(rows_spec, res, mut):
    """Per-pair facts for one model: the two predicates on both twins, plus companions."""
    out = {}
    for row in rows_spec:
        sm = res[row["category"]].get(row["miss_id"])
        sb = res["base"].get(row["base_id"])
        if sm is None or sb is None:
            continue
        k = row["anchor_turn"]
        tm, tb = row["_miss_tools"], row["_base_tools"]
        rm, rb = k < len(sm), k < len(sb)
        cm = S.turn_calls(sm[k] if rm else [], tm)
        cb = S.turn_calls(sb[k] if rb else [], tb)
        hm = [[n for n, _ in S.turn_calls(sm[j], tm)] for j in range(min(k, len(sm)))]
        hb = [[n for n, _ in S.turn_calls(sb[j], tb)] for j in range(min(k, len(sb)))]
        out[row["miss_id"]] = dict(
            cat=row["category"], base=row["base_id"],
            act=mut.any_mut(cb), hold=not mut.any_mut(cm),
            holdG=not mut.any_mut_gated(cm, S.independent_of(row)),
            actL=mut.any_mut(cb, True), holdL=not mut.any_mut(cm, True),
            n_b=len(cb), n_m=len(cm), reached=(rm and rb), reached_b=rb, reached_m=rm,
            text_m=bool(S.prose(sm[k] if rm else [])),
            same=(hb == hm and len(hb) == k and len(hm) == k))
    return out


def acc_of(rows, ids, a="act", h="hold"):
    av = np.array([rows[i][a] for i in ids], float)
    hv = np.array([rows[i][h] for i in ids], float)
    return av, hv, (av + hv) / 2


def table_checks(tag, exp_tab, exp_spread, exp_rank1, rec, ids_of, pairs_by_id,
                 seed, B, order, a="act", h="hold"):
    accs, sizes = {}, {}
    for m in order:
        if m not in rec:
            skip(f"{tag} {m}: every number", "rollouts not present locally")
            continue
        rows = rec[m]
        ids = ids_of(m, rows)
        av, hv, acc = acc_of(rows, ids, a, h)
        e = exp_tab[m]
        check(f"{tag} {m}: N", e["n"], len(ids), tol=0)
        check(f"{tag} {m}: act when complete", e["act"], 100 * av.mean())
        check(f"{tag} {m}: hold when removed", e["hold"], 100 * hv.mean())
        check(f"{tag} {m}: decision accuracy", e["decision"], 100 * acc.mean())
        check(f"{tag} {m}: both twins right", e["both"], 100 * float(np.mean(av * hv)))
        lo, hi = boot_ci(acc, [pairs_by_id[i]["base_id"] for i in ids], seed, B)
        check(f"{tag} {m}: interval low", e["ci95"][0], lo, tol=0.1)
        check(f"{tag} {m}: interval high", e["ci95"][1], hi, tol=0.1)
        accs[m] = acc
        sizes[m] = ids
    if len(accs) == len(exp_tab):
        vals = [100 * accs[m].mean() for m in accs]
        check(f"{tag}: spread", exp_spread, max(vals) - min(vals), tol=0.1)
        if exp_rank1 is not None and len({len(sizes[m]) for m in sizes}) == 1:
            names = [m for m in order if m in accs]
            twin = [pairs_by_id[i]["base_id"] for i in sizes[names[0]]]
            got = rank1(accs, names, twin, seed, B)
            for m in names:
                check(f"{tag} {m}: rank-1 frequency", exp_rank1.get(m, 0.0), got[m], tol=0.1)
    else:
        skip(f"{tag}: spread and rank-1 frequency", "needs every model's rollouts")
    return accs


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, action="append")
    ap.add_argument("--scores", action="append", default=None,
                    help="BFCL score directory; enables the official-verdict checks")
    ap.add_argument("--pairs", default=os.path.join(here, "pairs.json"))
    ap.add_argument("--mutators", default=os.path.join(here, "mutator_map.json"))
    ap.add_argument("--expected", default=os.path.join(here, "expected.json"))
    a = ap.parse_args()

    mut = S.Mutators(a.mutators)
    man = S.load_manifest(a.pairs, mut)
    exp = json.load(open(a.expected))
    B, seed = exp["bootstrap"]["B"], exp["bootstrap"]["seed"]
    order = exp["models_order"]
    pairs = man["pairs"]
    by_id = {r["miss_id"]: r for r in pairs}

    # ------------------------------------------------------------ manifest / §1
    ec = exp["counts"]
    check("manifest: content-verified pairs", ec["content_verified_pairs"], len(pairs), tol=0)
    candidates = S.gold_mutator_pairs(pairs, mut)
    derived = {r["miss_id"] for r in S.scored_pairs(pairs, mut)}
    check("manifest: scored flag agrees with the shipped predicate",
          0, sum(1 for r in pairs if r["scored"] != (r["miss_id"] in derived)), tol=0)
    check("manifest: certified candidate pairs", ec["certified_pairs"],
          len(candidates), tol=0)
    check("manifest: every candidate carries its certification", 0,
          sum(1 for r in candidates if not S.certification(r)), tol=0)
    check("manifest: no other pair carries a certification", 0,
          sum(1 for r in pairs if S.certification(r)) - len(candidates), tol=0)
    for verdict, n in ec["certification_verdicts"].items():
        check(f"manifest: certified {verdict}", n,
              sum(1 for r in candidates
                  if S.certification(r)["final_verdict"] == verdict), tol=0)
    check("manifest: construction errors", ec["construction_errors"],
          sum(1 for r in pairs if S.is_construction_error(r)), tol=0)
    check("manifest: compound pairs", ec["compound_pairs"],
          sum(1 for r in pairs if S.independent_of(r)), tol=0)
    check("manifest: construction errors are the only certified exclusions",
          ec["construction_errors"],
          sum(1 for r in candidates if not r["scored"]), tol=0)
    check("manifest: scored pairs", ec["scored_pairs"], len(derived), tol=0)
    check("manifest: excluded pairs", ec["excluded_pairs"], len(pairs) - len(derived), tol=0)
    check("manifest: C-login pairs", ec["scored_pairs_login"],
          len(S.scored_pairs(pairs, mut, True)), tol=0)
    for cat in ("miss_func", "miss_param"):
        check(f"manifest: scored {cat}", ec["scored_" + cat],
              sum(1 for r in pairs if r["scored"] and r["category"] == cat), tol=0)
        check(f"manifest: excluded {cat}", ec["excluded_" + cat],
              sum(1 for r in pairs if not r["scored"] and r["category"] == cat), tol=0)
    for reason, n in ec["excluded_reasons"].items():
        check(f"manifest: excluded because {reason}", n,
              sum(1 for r in pairs if r.get("exclusion_reason") == reason), tol=0)
    read_only = [r for r in pairs if not r["scored"] and not S.is_construction_error(r)]
    check("manifest: read-only excluded pairs", ec["read_only_excluded_pairs"],
          len(read_only), tol=0)
    for cat in ("miss_func", "miss_param"):
        check(f"manifest: read-only excluded {cat}", ec["read_only_excluded_" + cat],
              sum(1 for r in read_only if r["category"] == cat), tol=0)
    check("manifest: distinct base twins, all pairs", ec["distinct_base_twins_all"],
          len({r["base_id"] for r in pairs}), tol=0)
    check("manifest: distinct base twins, scored", ec["distinct_base_twins_scored"],
          len({r["base_id"] for r in pairs if r["scored"]}), tol=0)
    check("manifest: distinct base anchor turns, scored",
          ec["distinct_base_anchor_turns_scored"],
          len({(r["base_id"], r["anchor_turn"]) for r in pairs if r["scored"]}), tol=0)
    freq = {}
    for r in read_only:
        for n, _ in S.gold_anchor_calls(r):
            freq[n] = freq.get(n, 0) + 1
    top = dict(sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:8])
    for name, cnt in ec["excluded_top_gold_functions"].items():
        check(f"manifest: {name} on the excluded gold anchors", cnt, top.get(name, 0), tol=0)
    check("manifest: the eight most common functions on the excluded anchors",
          sorted(ec["excluded_top_gold_functions"]), sorted(top))
    check("manifest: total pairs with the bad keys restored",
          ec["bad_key_restored_pairs"],
          len(derived) + len(man["bad_key_restored_pairs"]), tol=0)
    check("manifest: pairs added back by restoring the bad keys",
          ec["bad_key_restored_added"], len(man["bad_key_restored_pairs"]), tol=0)

    scored = [r for r in pairs if r["scored"]]
    scoredL = S.scored_pairs(pairs, mut, True)
    C = [r["miss_id"] for r in scored]
    CU = [r["miss_id"] for r in candidates]
    CL = [r["miss_id"] for r in scoredL]
    RO = [r["miss_id"] for r in read_only]

    # ------------------------------------------------------------------ rollouts
    restored = list(man["bad_key_restored_pairs"])
    for r in restored:
        r["_miss_tools"] = set(man["toolsets"][r["miss_tools"]])
        r["_base_tools"] = set(man["toolsets"][r["base_tools"]])
    by_id.update({r["miss_id"]: r for r in restored})
    rec, rec_extra = {}, {}
    for m in order:
        res = {c: S.load_results(a.results, m, c) for c in S.CATS}
        if any(v is None for v in res.values()):
            continue
        rec[m] = measure(pairs, res, mut)
        rec_extra[m] = measure(restored, res, mut)

    # ------------------------------------------------- definition C and the two others
    table_checks("C", exp["tables"]["C"], exp["spread"]["C"], exp["rank1"]["C"],
                 rec, lambda m, rows: [i for i in C if i in rows], by_id, seed, B, order)
    table_checks("C-credited", exp["tables"]["C-credited"], exp["spread"]["C-credited"],
                 exp["rank1"]["C-credited"], rec,
                 lambda m, rows: [i for i in C if i in rows], by_id, seed, B, order,
                 a="act", h="holdG")
    table_checks("C-uncertified", exp["tables"]["C-uncertified"],
                 exp["spread"]["C-uncertified"], exp["rank1"]["C-uncertified"], rec,
                 lambda m, rows: [i for i in CU if i in rows], by_id, seed, B, order)
    table_checks("C-login", exp["tables"]["C-login"], exp["spread"]["C-login"],
                 exp["rank1"]["C-login"], rec,
                 lambda m, rows: [i for i in CL if i in rows], by_id, seed, B, order,
                 a="actL", h="holdL")
    table_checks("C-reached", exp["tables"]["C-reached"], exp["spread"]["C-reached"], None,
                 rec, lambda m, rows: [i for i in C if i in rows and rows[i]["reached"]],
                 by_id, seed, B, order)

    # ------------------------------------------------- rollouts short of the anchor turn
    for m in order:
        if m not in rec:
            skip(f"truncation {m}", "rollouts not present locally")
            continue
        tb = sum(1 for i, r in rec[m].items() if not r["reached_b"])
        tm = sum(1 for i, r in rec[m].items() if not r["reached_m"])
        check(f"truncation {m}: base twins short of the anchor", exp["truncation"][m][0],
              tb, tol=0)
        check(f"truncation {m}: miss twins short of the anchor", exp["truncation"][m][1],
              tm, tol=0)

    # ------------------------------------------------------------- companion numbers
    cp = exp["companions"]
    for m in order:
        if m not in rec:
            continue
        rows = rec[m]
        ids = [i for i in C if i in rows]
        for cat in ("miss_func", "miss_param"):
            sub = [i for i in ids if rows[i]["cat"] == cat]
            av, hv, acc = acc_of(rows, sub)
            e = cp["by_category"][m][cat]
            check(f"by category {m} {cat}: N", e["n"], len(sub), tol=0)
            check(f"by category {m} {cat}: act", e["act"], 100 * av.mean())
            check(f"by category {m} {cat}: hold", e["hold"], 100 * hv.mean())
            check(f"by category {m} {cat}: decision", e["decision"], 100 * acc.mean())
        sub = [i for i in ids if rows[i]["same"]]
        av, hv, acc = acc_of(rows, sub)
        e = cp["same_history"][m]
        check(f"same pre-anchor history {m}: N", e["n"], len(sub), tol=0)
        check(f"same pre-anchor history {m}: act", e["act"], 100 * av.mean())
        check(f"same pre-anchor history {m}: hold", e["hold"], 100 * hv.mean())
        check(f"same pre-anchor history {m}: decision", e["decision"], 100 * acc.mean())
        held = [i for i in ids if rows[i]["hold"]]
        e = cp["heldback_content"][m]
        check(f"held-back turns {m}: held back", e["held"], len(held), tol=0)
        check(f"held-back turns {m}: with read-only calls", e["read_only"],
              sum(1 for i in held if rows[i]["n_m"] > 0), tol=0)
        check(f"held-back turns {m}: with text", e["text"],
              sum(1 for i in held if rows[i]["text_m"]), tol=0)
        check(f"held-back turns {m}: silent", e["silent"],
              sum(1 for i in held if not rows[i]["text_m"]), tol=0)
        xids = [i for i in RO if i in rows]
        e = cp["excluded_attempts"][m]
        check(f"read-only excluded pairs {m}: N", e["n"], len(xids), tol=0)
        check(f"read-only excluded pairs {m}: attempted on the base anchor", e["act_base"],
              100 * float(np.mean([rows[i]["act"] for i in xids])))
        check(f"read-only excluded pairs {m}: attempted on the miss anchor", e["act_miss"],
              100 * float(np.mean([not rows[i]["hold"] for i in xids])))
        check(f"read-only excluded pairs {m}: attempted on either", e["either"],
              100 * float(np.mean([rows[i]["act"] or not rows[i]["hold"] for i in xids])))
        seen = {}
        for i in ids:
            seen[(by_id[i]["base_id"], by_id[i]["anchor_turn"])] = rows[i]["act"]
        e = cp["act_per_anchor"][m]
        check(f"act per anchor {m}: distinct base anchor turns", e["anchors"],
              len(seen), tol=0)
        check(f"act per anchor {m}: act per pair", e["act_per_pair"],
              100 * float(np.mean([rows[i]["act"] for i in ids])))
        check(f"act per anchor {m}: act per anchor", e["act_per_anchor"],
              100 * float(np.mean(list(seen.values()))))
        no = [i for i in ids if not rows[i]["act"]]
        e = cp["base_no_attempt"][m]
        check(f"base anchors without an attempt {m}: N", e["n"], len(no), tol=0)
        check(f"base anchors without an attempt {m}: zero calls", e["zero_calls"],
              sum(1 for i in no if rows[i]["n_b"] == 0), tol=0)
        check(f"base anchors without an attempt {m}: read-only calls", e["read_only"],
              len(no) - sum(1 for i in no if rows[i]["n_b"] == 0), tol=0)
        e = cp["bad_keys_restored"]["models"][m]
        d1 = acc_of(rows, ids)[2].mean() * 100
        ids2 = ids + [i for i in rec_extra[m]]
        allrows = dict(rows)
        allrows.update(rec_extra[m])
        d2 = acc_of(allrows, ids2)[2].mean() * 100
        check(f"bad keys restored {m}: decision on the scored pairs", e["decision"], d1)
        check(f"bad keys restored {m}: decision with the bad keys back", e["restored"], d2)
        check(f"bad keys restored {m}: change", e["change"], d2 - d1, tol=0.005)
    if rec:
        tot_h = sum(sum(1 for i in C if i in rec[m] and rec[m][i]["hold"]) for m in rec)
        tot_s = sum(sum(1 for i in C if i in rec[m] and rec[m][i]["hold"]
                        and not rec[m][i]["text_m"]) for m in rec)
        if len(rec) == len(order):
            check("held-back turns pooled: total", cp["heldback_pooled"]["held"], tot_h, tol=0)
            check("held-back turns pooled: silent", cp["heldback_pooled"]["silent"],
                  tot_s, tol=0)
        else:
            skip("held-back turns pooled", "needs every model's rollouts")

    # ------------------------------------------------- official verdict (needs --scores)
    raw_m, raw_b, dec = {}, {}, {}
    base_ids = {r["base_id"] for r in scored}
    for m in order:
        if m not in rec:
            continue
        fails = {c: (S.load_fails(a.scores, m, c) if a.scores else None) for c in S.CATS}
        if any(v is None for v in fails.values()):
            skip(f"official verdict {m}", "score files not given (--scores)")
            continue
        rows = rec[m]
        ids = [i for i in C if i in rows]
        ok = sum(1 for i in C if i not in fails[by_id[i]["category"]])
        raw_m[m] = 100 * ok / len(C)
        raw_b[m] = 100 * sum(1 for b in base_ids if b not in fails["base"]) / len(base_ids)
        dec[m] = acc_of(rows, ids)[2].mean() * 100
        e = cp["official_raw"][m]
        check(f"official raw {m}: miss items", e["miss"], raw_m[m])
        check(f"official raw {m}: base twins", e["base"], raw_b[m])
        check(f"official raw {m}: decision accuracy", e["decision"], dec[m])
        p = [i for i in C if i not in fails[by_id[i]["category"]]]
        q = [i for i in C if i in fails[by_id[i]["category"]]]
        e = cp["verdict_vs_attempt"][m]
        check(f"verdict vs attempt {m}: passes", e["passes"], len(p), tol=0)
        check(f"verdict vs attempt {m}: attempted among passes", e["pass_attempt"],
              100 * float(np.mean([not rows[i]["hold"] for i in p])) if p else 0)
        check(f"verdict vs attempt {m}: fails", e["fails"], len(q), tol=0)
        check(f"verdict vs attempt {m}: attempted among fails", e["fail_attempt"],
              100 * float(np.mean([not rows[i]["hold"] for i in q])) if q else 0)
    if len(raw_m) == len(order):
        names = [m for m in order if m in raw_m]
        check("official raw: distinct base twins", cp["distinct_base_twins"],
              len(base_ids), tol=0)
        r1, p1 = spearman([raw_m[n] for n in names], [dec[n] for n in names])
        r2, p2 = spearman([raw_b[n] for n in names], [dec[n] for n in names])
        check("Spearman raw(miss) x decision: rho", cp["spearman"]["miss_rho"], r1, tol=0.005)
        check("Spearman raw(miss) x decision: p", cp["spearman"]["miss_p"], p1, tol=0.005)
        check("Spearman raw(base) x decision: rho", cp["spearman"]["base_rho"], r2, tol=0.005)
        check("Spearman raw(base) x decision: p", cp["spearman"]["base_p"], p2, tol=0.005)
        check("official rank (raw on miss items)", cp["ranks"]["official"],
              sorted(names, key=lambda x: -raw_m[x]))
        check("decision rank", cp["ranks"]["decision"], sorted(names, key=lambda x: -dec[x]))
    else:
        skip("Spearman and the two rankings", "needs every model's score files")

    w = max(len(r[1]) for r in RESULTS)
    for status, name, expected, got in RESULTS:
        print(f"{status:4}  {name:{w}}  expected {expected}  got {got}")
    n_fail = sum(1 for r in RESULTS if r[0] == "FAIL")
    n_skip = sum(1 for r in RESULTS if r[0] == "SKIP")
    n_pass = sum(1 for r in RESULTS if r[0] == "PASS")
    print(f"\n{n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP, {len(RESULTS)} checks total")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
