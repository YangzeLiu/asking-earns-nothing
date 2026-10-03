#!/usr/bin/env python3
"""P1-z: what IS the attempt on the withheld turn, and is it a successful workaround?

THE QUESTION
------------
Definition C scores a miss_func item as "did not hold back" whenever the model attempted a
world-changing call on the anchor turn. But on the miss_func side the function the item
withholds is not in the model's tool list, so such an attempt is one of two very different
things:

  (a) a call to the removed name itself — the model guessed the hidden tool; or
  (b) a call to some OTHER mutating function that is in the list.

(b) can be a genuine workaround: `mv` is withheld, the model does `cp` then `rm` and the
user's goal is reached anyway. Counting that as a wrong decision is arguable, so we measure
how often it happens instead of arguing about it.

WHAT THIS SCRIPT DOES
---------------------
On the 231 definition-C pairs (`p1s_attempt_decision.py`; base gold anchor contains a
world-changing call), split by category (117 miss_func / 114 miss_param), for each of the
seven paper models:

1. attempts on the withheld anchor turn = `hold == False` = the model attempted a
   world-changing call there (`any_mut`, `mutator_map.json`, login excluded, echo per call).
2. of those, how many call a REMOVED function. Removed set = the names listed in the item's
   `missed_function` field of `BFCL_v4_multi_turn_miss_func.json`. That is BFCL's own
   holdout mechanism: `bfcl_eval/utils.py::populate_test_cases_with_predefined_functions`
   pops exactly those docs out of `entry["function"]` and re-injects them at the turn given
   by the dict key. On all 117 miss_func pairs here the key is k+1, i.e. the doc is absent
   for every turn up to and including the anchor turn k. miss_param items have no
   `missed_function`, so the removed set is empty by construction and column (b) collapses
   to "attempt with a guessed value", which is the whole of column (a) there.
   NOTE `allowed_names(cat)` in `bfcl_calls.py` deliberately ADDS the `missed_function`
   names back into the validation list, so a call to a removed name is counted as a call
   (Appendix D). Verified: every removed name is in `allowed_names` for its item.
   NOTE ALSO that BFCL only removes the DOC, not the method: the class instance still has
   it, so a call to a removed name really executes in the sandbox (and in the rollout).
3. of the attempts that call NO removed name (i.e. other mutators only), replay with BFCL's
   own executor and BFCL's own state (`p1n_workaround.py` machinery, reused by import):
       base twin's GOLD turns 0..k-1 -> G_before ; + GOLD turn k -> G_after
       model's own miss turns 0..k-1 -> M_before ; + model turn k -> M_after
   aligned = M_before == G_before (otherwise the comparison is contaminated by earlier
   divergence and the attempt is unclassifiable). Among aligned:
       M_after == G_after  -> successful workaround
       M_after == M_before -> attempted but the world did not move (sandbox refusal, no-op)
       otherwise           -> diverged
   M_after == G_after is a LOWER bound: an equally acceptable but different end state counts
   as diverged. It is also, on a few items, too GENEROUS: on 14 of the 231 pairs the gold
   anchor's own mutating call leaves the state untouched (a rejected order, an add that was
   already there), so G_after == G_before and any attempt that fails trivially "reaches"
   G_after. Those pairs are flagged and a strict variant that refuses to credit them is
   reported alongside.
4. a variant decision accuracy on the 231 pairs in which a successful workaround is credited
   as a correct decision: hold' = hold OR workaround, acc' = (act + hold')/2. Same cluster
   bootstrap over base twins as p1s (`boot_ci`, B and SEED from p1d_nonact_rate). A
   sensitivity column also credits removed-name attempts that reach G_after.
5. every successful workaround listed with the model's calls and the base gold anchor call,
   for human spot-checking.

Run: `python3 p1z_missfunc_attempts.py` from analysis/ (same as p1n; the BFCL venv
site-packages is put on sys.path by p1n_workaround). Output: analysis/p1z_missfunc_attempts.md
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import allowed_names  # noqa: E402
from bfcl_calls_full import full_turn_calls  # noqa: E402
from p1d_nonact_rate import PAPER_MODELS as MODELS, SEED, B, VENV  # noqa: E402
from p1f_matched_control import build_pairs, load_results, ASK  # noqa: E402
from p1h_pair_checks import content_check, gold_calls  # noqa: E402
from p1s_attempt_decision import any_mut, is_mut, boot_ci  # noqa: E402
# p1n owns the executor replay; importing it also puts the BFCL venv on sys.path
from p1n_workaround import entries, gold, fresh, step, snap  # noqa: E402
from p2c_cert import load_final, gated  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(VENV, "bfcl_eval/data")


def removed_names(cat, ent_cat):
    """id -> set of function names BFCL withheld from the tool list (miss_func only)."""
    out = {}
    for iid, e in ent_cat.items():
        s = set()
        for turn_funcs in (e.get("missed_function") or {}).values():
            for f in turn_funcs:
                if isinstance(f, str):
                    s.add(f)
                elif isinstance(f, dict) and "name" in f:
                    s.add(f["name"])
        out[iid] = s
    return out


def main():
    pairs, _ = build_pairs()
    ok, _, _ = content_check(pairs)
    pairs = {i: pairs[i] for i in ok}
    allow = {c: allowed_names(c) for c in ASK + ["base"]}
    gb = gold_calls("base")

    def gold_at(iid):
        bid, k, _ = pairs[iid]
        g = gb.get(bid) or []
        return [(s.split("(")[0].strip(), s) for s in (g[k] if k < len(g) else [])]

    construction, compound, _ = load_final()
    C = [i for i in pairs if any_mut(gold_at(i)) and i not in construction]
    ent = {c: entries(c) for c in ASK + ["base"]}
    gld = gold("base")
    rem = {c: removed_names(c, ent[c]) for c in ASK}

    # holdout key sanity: the doc must still be absent on the anchor turn
    key_ok = 0
    for iid in C:
        bid, k, cat = pairs[iid]
        if cat != "miss_func":
            continue
        keys = [int(x) for x in (ent[cat][iid].get("missed_function") or {})]
        key_ok += all(x > k for x in keys)
    n_mf = sum(1 for i in C if pairs[i][2] == "miss_func")
    assert key_ok == n_mf, "a withheld doc is re-injected at or before the anchor turn"
    assert all(rem["miss_func"][i] <= allow["miss_func"][i] for i in C
               if pairs[i][2] == "miss_func"), "removed name outside allowed_names"

    # --- target states, once per pair (base twin's gold up to and including the anchor)
    target = {}
    for iid in C:
        bid, k, cat = pairs[iid]
        be = ent["base"][bid]
        eid = f"{bid}__{k}"
        fresh("p1z_gold", be, eid)
        for j in range(k):
            step("p1z_gold", be, eid, gld[bid][j])
        g_before = snap("p1z_gold", be, eid)
        step("p1z_gold", be, eid, gld[bid][k])
        target[iid] = (g_before, snap("p1z_gold", be, eid))
    # pairs where even the gold anchor call does not move the state: there "reached G_after"
    # is satisfied by doing nothing, so crediting it as a workaround would be vacuous
    noop = {iid for iid in C if target[iid][0] == target[iid][1]}
    print(f"targets built for {len(target)} pairs; {len(noop)} gold-no-op", flush=True)

    # --- per model
    rec = {}
    for m in MODELS:
        res = {c: load_results(m, c) for c in ASK + ["base"]}
        if any(v is None for v in res.values()):
            print(f"{m}: no rollouts, skipped", flush=True)
            continue
        rows = {}
        for iid in C:
            bid, k, cat = pairs[iid]
            sb, sm = res["base"].get(bid), res[cat].get(iid)
            if sb is None or sm is None:
                continue
            cb = full_turn_calls(sb[k] if k < len(sb) else [], allow["base"].get(bid, set()))
            cm = full_turn_calls(sm[k] if k < len(sm) else [], allow[cat].get(iid, set()))
            act = any_mut(cb)
            hold = not any_mut(cm)
            # compound reading: an attempt confined to the certified independent steps
            hold_c = not any(is_mut(n, s_) and gated(n, iid, compound) for n, s_ in cm)
            R = rem[cat].get(iid, set())
            hit = [n for n, _ in cm if n in R]
            hit_mut = [n for n, s in cm if n in R and is_mut(n, s)]
            row = dict(cat=cat, act=act, hold=hold, hold_c=hold_c, calls=[s for _, s in cm],
                       removed=sorted(R), hit=hit, hit_mut=hit_mut, cls=None,
                       noop=iid in noop)
            if not hold:
                # replay the model's own miss trajectory to the anchor
                e = ent[cat][iid]
                eid = f"{m}__{iid}"
                try:
                    fresh("p1z_m", e, eid)
                    for j in range(min(k, len(sm))):
                        step("p1z_m", e, eid,
                             [s for _, s in full_turn_calls(sm[j], allow[cat].get(iid, set()))])
                    mb = snap("p1z_m", e, eid)
                    step("p1z_m", e, eid, [s for _, s in cm])
                    ma = snap("p1z_m", e, eid)
                except Exception:
                    row["cls"] = "replay error"
                else:
                    g_before, g_after = target[iid]
                    if mb != g_before:
                        row["cls"] = "unaligned"
                    elif ma == g_after:
                        row["cls"] = "workaround"
                    elif ma == mb:
                        row["cls"] = "no state change"
                    else:
                        row["cls"] = "diverged"
            rows[iid] = row
        rec[m] = rows
        att = sum(1 for r in rows.values() if not r["hold"])
        print(f"{m}: {len(rows)} pairs, {att} attempts classified", flush=True)

    # ------------------------------------------------------------------ markdown
    L = ["# P1-z what the attempt on the withheld turn actually is", "",
         f"The {len(C)} definition-C pairs of `p1s_attempt_decision.py` "
         f"({n_mf} miss_func + {len(C)-n_mf} miss_param; the {len(construction)} certified "
         f"construction errors are out, p2c_cert.py), seven paper models.", "",
         "**attempt** = the model attempted a world-changing call on the withheld anchor turn "
         "k (`any_mut`, `mutator_map.json`, login excluded, `echo` only with `file_name`); "
         "this is exactly `hold == False` in p1s, and an attempt counts whether or not the "
         "sandbox let it through.", "",
         "**removed function** = the names in the item's `missed_function` field. BFCL's "
         "`utils.py::populate_test_cases_with_predefined_functions` pops exactly those docs "
         "out of the item's tool list and re-injects them at the turn given by the dict key; "
         f"on all {n_mf} miss_func pairs here that key is k+1, so the doc is absent on the "
         "anchor turn. `bfcl_calls.allowed_names` adds those names back for validation "
         "(Appendix D), so a call to a removed name is counted as a call — verified above. "
         "BFCL removes the doc only, not the method, so such a call really executes. "
         "miss_param items have no `missed_function`: the removed set is empty there and "
         "every attempt is an attempt with a guessed value.", "",
         "**workaround** (executed, no judge, `p1n_workaround.py` machinery): replay the base "
         "twin's gold turns 0..k-1 -> `G_before`, + gold turn k -> `G_after`; replay the "
         "model's own miss-twin turns 0..k-1 -> `M_before`, + its turn k -> `M_after`; state "
         "is BFCL's own `state_checker` view (public attributes of the involved instances). "
         "*aligned* = `M_before == G_before`; unaligned attempts are not classifiable. Among "
         "aligned: `M_after == G_after` = **workaround**, `M_after == M_before` = the world "
         "did not move, else diverged. `M_after == G_after` is a lower bound (a different but "
         "equally acceptable end state reads as diverged) and, on the 14 pairs whose gold "
         "anchor call does not move the state either, too generous: there any failed attempt "
         "matches `G_after`, so those hits are marked *vacuous* and a strict variant drops "
         "them.", ""]

    def table(cat, title, with_removed):
        cols = ["model", "pairs", "attempts"]
        if with_removed:
            cols += ["calls a removed name", "of which the removed call mutates",
                     "other mutators only"]
        cols += ["aligned", "workaround", "of which vacuous", "no state change", "diverged",
                 "unaligned", "workaround % of aligned"]
        out = [f"## {title}", "", "| " + " | ".join(cols) + " |",
               "|" + "---|" * len(cols)]
        for m, rows in rec.items():
            ids = [i for i in rows if rows[i]["cat"] == cat]
            att = [i for i in ids if not rows[i]["hold"]]
            hit = [i for i in att if rows[i]["hit"]]
            hitm = [i for i in hit if rows[i]["hit_mut"]]
            oth = [i for i in att if not rows[i]["hit"]]
            c = lambda tag: sum(1 for i in oth if rows[i]["cls"] == tag)  # noqa: E731
            al = c("workaround") + c("no state change") + c("diverged")
            cells = [m, str(len(ids)), str(len(att))]
            if with_removed:
                cells += [str(len(hit)), str(len(hitm)), str(len(oth))]
            vac = sum(1 for i in oth if rows[i]["cls"] == "workaround" and rows[i]["noop"])
            cells += [str(al), str(c("workaround")), str(vac), str(c("no state change")),
                      str(c("diverged")), str(c("unaligned")),
                      f"{100*c('workaround')/al:.1f}" if al else "—"]
            out.append("| " + " | ".join(cells) + " |")
        err = sum(1 for rows in rec.values() for i in rows
                  if rows[i]["cat"] == cat and rows[i]["cls"] == "replay error")
        nv = len({i for rows in rec.values() for i in rows
                  if rows[i]["cat"] == cat and rows[i]["noop"]})
        out += ["", f"Replay errors (attempt could not be executed at all): {err}. "
                    f"*vacuous* = the pair is one of the {nv} in this category whose gold "
                    f"anchor call itself leaves the state unchanged, so `G_after == G_before` "
                    f"and any failed attempt matches it.", ""]
        return out

    L += table("miss_func", f"1. miss_func ({n_mf} pairs): is the attempt the removed "
                            "function, or another route?", True)
    L += table("miss_param", f"2. miss_param ({len(C)-n_mf} pairs): every attempt is an "
                             "attempt with a guessed value", False)

    # ---- 3. variant decision accuracy
    L += ["## 3. Decision accuracy if a successful workaround counts as a correct decision",
          "",
          "`hold'` = held back OR reached the complete-information state by another route "
          "(other-mutator attempts only). `acc'` = (act + hold')/2. The last column is a "
          "sensitivity that additionally credits attempts calling a removed name that "
          "nevertheless land on `G_after`; `acc' strict` instead refuses to credit the "
          "vacuous workarounds (gold-no-op pairs). Cluster bootstrap over base twins, "
          f"B={B}, seed {SEED}, as in p1s.", "",
          "| model | N | act | hold | hold' | acc [95% CI] | acc' [95% CI] | change | "
          "acc' strict | acc' (removed-name credited too) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m, rows in rec.items():
        ids = [i for i in C if i in rows]
        a = np.array([rows[i]["act"] for i in ids], float)
        h = np.array([rows[i]["hold"] for i in ids], float)
        w = np.array([rows[i]["cls"] == "workaround" and not rows[i]["hit"] for i in ids], float)
        w2 = np.array([rows[i]["cls"] == "workaround" for i in ids], float)
        ws = np.array([rows[i]["cls"] == "workaround" and not rows[i]["hit"]
                       and not rows[i]["noop"] for i in ids], float)
        h1 = np.maximum(h, w)
        h2 = np.maximum(h, w2)
        hs = np.maximum(h, ws)
        twin = np.array([pairs[i][0] for i in ids])
        acc, acc1, acc2, accs = (a + h) / 2, (a + h1) / 2, (a + h2) / 2, (a + hs) / 2
        lo, hi = boot_ci(acc, twin)
        lo1, hi1 = boot_ci(acc1, twin)
        L.append(f"| {m} | {len(ids)} | {100*a.mean():.1f} | {100*h.mean():.1f} | "
                 f"{100*h1.mean():.1f} | {100*acc.mean():.1f} [{lo:.1f}, {hi:.1f}] | "
                 f"{100*acc1.mean():.1f} [{lo1:.1f}, {hi1:.1f}] | "
                 f"{100*(acc1-acc).mean():+.2f} | {100*accs.mean():.1f} | "
                 f"{100*acc2.mean():.1f} |")
    L.append("")

    # ---- 4. the credited column of Table 1: miss_func workarounds (strict) + compound turns
    L += ["## 4. The credited reading: miss_func workarounds (strict) and certified compound turns",
          "",
          "Two credits on the withheld side, both leaving the complete side untouched. "
          "**workaround** (miss_func only): held back OR reached the complete-information "
          "state by another route, other-mutator attempts, gold-no-op pairs excluded. "
          f"**compound** ({len(compound)} certified pairs, `p2a_cert/compound_split.json`): "
          "the attempt stays inside the world-changing steps the turn authorizes "
          "independently of the withheld function/value. `credited` = held back OR either. "
          "Cluster bootstrap as above.", "",
          "| model | N | acc [95% CI] | +compound only | +workaround only | credited [95% CI] | change | workarounds | compound credits |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, rows in rec.items():
        ids = [i for i in C if i in rows]
        a = np.array([rows[i]["act"] for i in ids], float)
        h = np.array([rows[i]["hold"] for i in ids], float)
        hc = np.array([rows[i]["hold_c"] for i in ids], float)
        wmf = np.array([rows[i]["cat"] == "miss_func" and rows[i]["cls"] == "workaround"
                        and not rows[i]["hit"] and not rows[i]["noop"] for i in ids], float)
        h_w = np.maximum(h, wmf)
        h_all = np.maximum(hc, wmf)
        twin = np.array([pairs[i][0] for i in ids])
        acc, acc_c, acc_w, acc2 = (a + h) / 2, (a + hc) / 2, (a + h_w) / 2, (a + h_all) / 2
        lo, hi = boot_ci(acc, twin)
        lo2, hi2 = boot_ci(acc2, twin)
        n_cc = int((hc - h).sum())
        L.append(f"| {m} | {len(ids)} | {100*acc.mean():.1f} [{lo:.1f}, {hi:.1f}] | "
                 f"{100*acc_c.mean():.1f} | {100*acc_w.mean():.1f} | "
                 f"{100*acc2.mean():.1f} [{lo2:.1f}, {hi2:.1f}] | {100*(acc2-acc).mean():+.2f} | "
                 f"{int(wmf.sum())} | {n_cc} |")
    # rank-1 frequency under the credited reading, one shared resample
    names = list(rec)
    ids0 = [i for i in C if i in rec[names[0]]]
    twin = np.array([pairs[i][0] for i in ids0])
    groups = [np.where(twin == t)[0] for t in sorted(set(twin.tolist()))]
    Mx = np.vstack([np.array([(rec[m][i]["act"] + max(rec[m][i]["hold_c"],
                    rec[m][i]["cat"] == "miss_func" and rec[m][i]["cls"] == "workaround"
                    and not rec[m][i]["hit"] and not rec[m][i]["noop"])) / 2 for i in ids0], float)
                   for m in names])
    rng = np.random.default_rng(SEED)
    first = np.zeros(len(names))
    for b in range(B):
        pick = rng.integers(0, len(groups), size=len(groups))
        sidx = np.concatenate([groups[p] for p in pick])
        first[Mx[:, sidx].mean(1).argmax()] += 1
    first /= B
    L += ["", "Rank-1 frequency under the credited reading: " + ", ".join(
        f"{n} {100*f:.1f}%" for n, f in sorted(zip(names, first), key=lambda x: -x[1]) if f), ""]

    # ---- 4. unclassifiable / coverage caveats
    L += ["## 5. Coverage", "",
          "| model | pairs scored | attempts | unaligned attempts | replay errors |",
          "|---|---|---|---|---|"]
    for m, rows in rec.items():
        att = [i for i in rows if not rows[i]["hold"]]
        L.append(f"| {m} | {len(rows)} | {len(att)} | "
                 f"{sum(1 for i in att if rows[i]['cls'] == 'unaligned')} | "
                 f"{sum(1 for i in att if rows[i]['cls'] == 'replay error')} |")
    L.append("")

    # ---- 5. spot-check list
    L += ["## 6. Every successful workaround, for spot-checking", "",
          "`other mutators only` rows: the attempt named no removed function and the world "
          "ended in the complete-information state. Base gold = the base twin's gold calls on "
          "the same turn.", "",
          "A row marked *vacuous* sits on a pair whose gold anchor call does not move the "
          "state either, so the match is not evidence of anything.", "",
          "| model | item | cat | removed | model calls on the withheld turn | base gold | |",
          "|---|---|---|---|---|---|---|"]
    n_hits = 0
    for m, rows in rec.items():
        for iid in C:
            r = rows.get(iid)
            if not r or r["cls"] != "workaround" or r["hit"]:
                continue
            n_hits += 1
            g = "; ".join(s for _, s in gold_at(iid))
            mc = "; ".join(r["calls"])
            L.append(f"| {m} | {iid} | {r['cat']} | {', '.join(r['removed']) or '—'} | "
                     f"`{mc}` | `{g}` | {'*vacuous*' if r['noop'] else ''} |")
    n_vac = sum(1 for m, rows in rec.items() for i in C
                if rows.get(i) and rows[i]["cls"] == "workaround" and not rows[i]["hit"]
                and rows[i]["noop"])
    L += ["", f"{n_hits} (model, item) successful workarounds, of which {n_vac} vacuous.", ""]
    extra = [(m, i) for m, rows in rec.items() for i in C
             if rows.get(i) and rows[i]["cls"] == "workaround" and rows[i]["hit"]]
    L += [f"Attempts that named a removed function and still reached `G_after` "
          f"(the sensitivity column of §3): {len(extra)}"
          + (" — " + ", ".join(f"{m}/{i}" for m, i in extra) if extra else ""), ""]

    txt = "\n".join(L)
    open(os.path.join(HERE, "p1z_missfunc_attempts.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
