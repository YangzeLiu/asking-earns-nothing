#!/usr/bin/env python3
"""P1-v: companion numbers for the paper under definition C (p1s_attempt_decision.py).

Everything the paper cites next to Table 1 that is not in p1s itself, recomputed on the
231 gold-mutator pairs with the same predicate (attempted a world-changing call, login
excluded, echo decided per call):

  1. the 134 excluded pairs: by category, and what their gold anchor turn contains
  2. official raw on the 231 miss items and on their base twins; Spearman vs decision acc
  3. decision accuracy by category
  4. same pre-anchor history subset (validated call names identical on both twins)
  5. what the held-back miss anchor turns contain: text / silent / read-only lookups
  6. official passes vs fails: attempted a world-changing call on the anchor
  7. base anchor turns without a world-changing attempt: zero calls vs read-only calls
  8. bad keys added back: decision accuracy with the 31 excluded ids restored
  9. the 134 excluded pairs: world-changing attempts on either twin, same predicate

Output: analysis/p1v_c_companions.md
"""
import collections
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1d_nonact_rate as P1D  # noqa: E402
from bfcl_calls import allowed_names  # noqa: E402
from bfcl_calls_full import full_turn_calls  # noqa: E402
from p1d_nonact_rate import PAPER_MODELS as MODELS  # noqa: E402
from p1f_matched_control import build_pairs, load_results, ASK  # noqa: E402
from p1g_paired_decision import fail_ids  # noqa: E402
from p1h_pair_checks import content_check, gold_calls  # noqa: E402
from p1i_holdback_content import prose  # noqa: E402
from p2c_cert import load_final  # noqa: E402
from p1s_attempt_decision import any_mut, is_mut, LOGIN  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def gold_at(gb, pairs, iid):
    bid, k, _ = pairs[iid]
    g = gb.get(bid) or []
    return [(s.split("(")[0].strip(), s) for s in (g[k] if k < len(g) else [])]


def rows_for(model, pairs, allow):
    res = {c: load_results(model, c) for c in ASK + ["base"]}
    if any(v is None for v in res.values()):
        return None
    out = {}
    for iid, (bid, k, cat) in pairs.items():
        sb, sm = res["base"].get(bid), res[cat].get(iid)
        if sb is None or sm is None:
            continue
        tb = sb[k] if k < len(sb) else []
        tm = sm[k] if k < len(sm) else []
        cb = full_turn_calls(tb, allow["base"].get(bid, set()))
        cm = full_turn_calls(tm, allow[cat].get(iid, set()))
        hb = [[n for n, _ in full_turn_calls(sb[j], allow["base"].get(bid, set()))]
              for j in range(min(k, len(sb)))]
        hm = [[n for n, _ in full_turn_calls(sm[j], allow[cat].get(iid, set()))]
              for j in range(min(k, len(sm)))]
        out[iid] = dict(cat=cat, act=any_mut(cb), hold=not any_mut(cm),
                        n_b=len(cb), n_m=len(cm), text_m=bool(prose(tm)),
                        same=(hb == hm and len(hb) == k and len(hm) == k))
    return out


def acc(rows, ids):
    a = np.array([rows[i]["act"] for i in ids], float)
    h = np.array([rows[i]["hold"] for i in ids], float)
    return 100 * a.mean(), 100 * h.mean(), 50 * (a.mean() + h.mean())


def main():
    pairs, _ = build_pairs()
    ok, _, _ = content_check(pairs)
    pairs = {i: pairs[i] for i in ok}
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    gb = gold_calls("base")
    gold_mut = {i: any_mut(gold_at(gb, pairs, i)) for i in pairs}
    construction, _, _ = load_final()
    C = [i for i in pairs if gold_mut[i] and i not in construction]
    X = [i for i in pairs if not gold_mut[i]]
    L = [f"# P1-v companion numbers under definition C ({len(C)} certified pairs; "
         f"{len(construction)} construction errors out, p2c_cert.py)", ""]

    # 1. excluded pairs
    L += [f"## 1. The {len(X)} excluded pairs", ""]
    bycat = collections.Counter(pairs[i][2] for i in X)
    kind = collections.Counter()
    top = collections.Counter()
    for i in X:
        g = gold_at(gb, pairs, i)
        names = [n for n, _ in g]
        if not names:
            kind["empty gold anchor"] += 1
        elif all(n in LOGIN for n in names):
            kind["login only"] += 1
        elif any(n in LOGIN for n in names):
            kind["read + login"] += 1
        else:
            kind["read-only"] += 1
        for n in names:
            top[n] += 1
    L.append("By category: " + ", ".join(f"{c} {n}" for c, n in sorted(bycat.items())))
    L.append("Gold anchor content: " + ", ".join(f"{k} {n}" for k, n in kind.most_common()))
    L.append("Most common gold functions on those anchors: " +
             ", ".join(f"{n} {c}" for n, c in top.most_common(8)))
    L.append(f"C by category: " + ", ".join(
        f"{c} {n}" for c, n in sorted(collections.Counter(pairs[i][2] for i in C).items())))
    L.append("")

    # per-model rows
    rec = {}
    for m in MODELS:
        r = rows_for(m, pairs, allow)
        if r is not None:
            rec[m] = r

    # 2. official raw on the 231 and Spearman
    L += [f"## 2. Official raw on the {len(C)} pairs", "",
          "| model | raw on miss items | raw on base twins (deduped) | decision acc |",
          "|---|---|---|---|"]
    miss_by_cat = collections.defaultdict(set)
    base_ids = set()
    for i in C:
        miss_by_cat[pairs[i][2]].add(i)
        base_ids.add(pairs[i][0])
    raw_m, raw_b, dec = {}, {}, {}
    for m, rows in rec.items():
        f = {c: fail_ids(m, c) for c in ASK}
        fb = fail_ids(m, "base")
        if any(v is None for v in f.values()) or fb is None:
            continue
        tot = sum(len(v) for v in miss_by_cat.values())
        okc = sum(1 for c, ids in miss_by_cat.items() for i in ids if i not in f[c])
        raw_m[m] = 100 * okc / tot
        raw_b[m] = 100 * sum(1 for b in base_ids if b not in fb) / len(base_ids)
        dec[m] = acc(rows, C)[2]
        L.append(f"| {m} | {raw_m[m]:.1f} | {raw_b[m]:.1f} | {dec[m]:.1f} |")
    names = list(raw_m)
    rm = spearmanr([raw_m[n] for n in names], [dec[n] for n in names])
    rb = spearmanr([raw_b[n] for n in names], [dec[n] for n in names])
    L += ["", f"{len(base_ids)} distinct base twins. Spearman raw(miss) x decision = "
          f"{rm.correlation:+.2f} (p={rm.pvalue:.2f}); raw(base twins) x decision = "
          f"{rb.correlation:+.2f} (p={rb.pvalue:.2f}).",
          "Official rank (raw on miss items): " + ", ".join(
              f"{n} {i+1}" for i, n in enumerate(sorted(names, key=lambda x: -raw_m[x]))),
          "Decision rank: " + ", ".join(
              f"{n} {i+1}" for i, n in enumerate(sorted(names, key=lambda x: -dec[x]))), ""]

    # 3. by category
    L += ["## 3. Decision accuracy by category", "",
          "| model | miss_func N | act | hold | dec | miss_param N | act | hold | dec |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, rows in rec.items():
        cells = [m]
        for c in ASK:
            ids = [i for i in C if pairs[i][2] == c]
            a, h, d = acc(rows, ids)
            cells += [str(len(ids)), f"{a:.1f}", f"{h:.1f}", f"{d:.1f}"]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # 4. same history
    L += ["## 4. Pairs with identical validated pre-anchor call names on both twins", "",
          "| model | N | act | hold | dec |", "|---|---|---|---|---|"]
    for m, rows in rec.items():
        ids = [i for i in C if rows[i]["same"]]
        a, h, d = acc(rows, ids)
        L.append(f"| {m} | {len(ids)} | {a:.1f} | {h:.1f} | {d:.1f} |")
    L.append("")

    # 5. held-back miss anchors: content
    L += ["## 5. Held-back miss anchor turns (no world-changing attempt)", "",
          "| model | held back | with read-only calls | with text | silent |",
          "|---|---|---|---|---|"]
    tot_h = tot_s = 0
    for m, rows in rec.items():
        ids = [i for i in C if rows[i]["hold"]]
        ro = sum(1 for i in ids if rows[i]["n_m"] > 0)
        tx = sum(1 for i in ids if rows[i]["text_m"])
        tot_h += len(ids)
        tot_s += len(ids) - tx
        L.append(f"| {m} | {len(ids)} | {ro} | {tx} | {len(ids)-tx} |")
    L += ["", f"Pooled: {tot_h} held-back turns, {tot_s} silent.", ""]

    # 6. official pass / fail vs attempt on anchor
    L += [f"## 6. Official verdict vs world-changing attempt on the anchor ({len(C)} miss items)",
          "", "| model | passes | attempted % | fails | attempted % |", "|---|---|---|---|---|"]
    for m, rows in rec.items():
        f = {c: fail_ids(m, c) for c in ASK}
        if any(v is None for v in f.values()):
            continue
        p = [i for i in C if i not in f[pairs[i][2]]]
        q = [i for i in C if i in f[pairs[i][2]]]
        pa = 100 * np.mean([not rows[i]["hold"] for i in p]) if p else 0
        qa = 100 * np.mean([not rows[i]["hold"] for i in q]) if q else 0
        L.append(f"| {m} | {len(p)} | {pa:.1f} | {len(q)} | {qa:.1f} |")
    L.append("")

    # 7. base anchors without an attempt
    L += ["## 7. Base anchor turns without a world-changing attempt", "",
          "| model | no attempt | of which zero calls | of which read-only calls |",
          "|---|---|---|---|"]
    for m, rows in rec.items():
        ids = [i for i in C if not rows[i]["act"]]
        z = sum(1 for i in ids if rows[i]["n_b"] == 0)
        L.append(f"| {m} | {len(ids)} | {z} | {len(ids)-z} |")
    L.append("")

    # 8. bad keys restored
    saved = {c: set(P1D.EXCSET[c]) for c in ASK}
    for c in ASK:
        P1D.EXCSET[c].clear()
    pairs2, _ = build_pairs()
    ok2, _, _ = content_check(pairs2)
    pairs2 = {i: pairs2[i] for i in ok2}
    for c in ASK:
        P1D.EXCSET[c].update(saved[c])
    gold2 = {i: any_mut(gold_at(gb, pairs2, i)) for i in pairs2}
    C2 = [i for i in pairs2 if gold2[i] and i not in construction]
    added = [i for i in C2 if i not in set(C)]
    L += [f"## 8. Bad keys restored: {len(C2)} pairs ({len(added)} added back)", "",
          f"| model | dec ({len(C)}) | dec (restored) | change |", "|---|---|---|---|"]
    for m in rec:
        r2 = rows_for(m, pairs2, allow)
        d1 = acc(rec[m], C)[2]
        d2 = acc(r2, C2)[2]
        L.append(f"| {m} | {d1:.1f} | {d2:.1f} | {d2-d1:+.2f} |")
    L.append("")

    # 9. the excluded pairs: world-changing attempts on either twin (gold makes none)
    L += [f"## 9. World-changing attempts on the {len(X)} excluded pairs (gold anchor makes none)", "",
          "| model | N | attempted on base anchor | attempted on miss anchor | either |",
          "|---|---|---|---|---|"]
    for m, rows in rec.items():
        ids = [i for i in X if i in rows]
        a = 100 * np.mean([rows[i]["act"] for i in ids])
        h = 100 * np.mean([not rows[i]["hold"] for i in ids])
        e = 100 * np.mean([rows[i]["act"] or not rows[i]["hold"] for i in ids])
        L.append(f"| {m} | {len(ids)} | {a:.1f} | {h:.1f} | {e:.1f} |")
    L.append("")
    # 10. pair weighting: the act column counted once per distinct base anchor turn
    anchors = sorted({(pairs[i][0], pairs[i][1]) for i in C})
    L += [f"## 10. Act rate counted once per distinct base anchor turn ({len(anchors)} anchors, "
          f"{len(base_ids)} base twins, {len(C)} pairs)", "",
          "| model | act (per pair) | act (per anchor) | change |", "|---|---|---|---|"]
    for m, rows in rec.items():
        per_pair = 100 * np.mean([rows[i]["act"] for i in C])
        seen = {}
        for i in C:
            seen[(pairs[i][0], pairs[i][1])] = rows[i]["act"]
        per_anchor = 100 * np.mean(list(seen.values()))
        L.append(f"| {m} | {per_pair:.1f} | {per_anchor:.1f} | {per_anchor-per_pair:+.1f} |")
    L.append("")
    txt = "\n".join(L)
    open(os.path.join(HERE, "p1v_c_companions.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
