#!/usr/bin/env python3
"""Phase 4 — cross-model answer-key-wrong detector (mechanical, judge-free).

Pipeline:
  (1) all-strong-fail filter: keep only items EVERY strong model failed. A genuine
      bad key fails every capable model (a model doing the right thing per the user's
      intent produces an end-state != the wrong oracle -> fails). An item any strong
      model PASSED has an achievable key -> drop.
  (2) consensus: for each survivor, extract each model's PRODUCED call sequence per turn
      (the calls themselves, not any judge's reading of them) and
      the oracle gold. Group strong models by normalized produced calls; if >=2 strong
      models agree with each other but differ from gold -> oracle is the outlier -> SUSPECT.
      If every model fails differently -> genuinely hard, NOT a bad key.
  (3) render compact md for human review (with the deepseek trace as the anchor).

Data lives in two places: frontier + Qwen3.6 in venv, the rest on rollouts_extra.
Usage: python3 phase4_badkey.py [category=base]
"""
import json, os, re, sys, collections
from clean_trace import split_gemma_step, render_call, load_jsonl_by_id, clip

# read-only navigation calls: filtered out of the consensus signal + compact render
# (they are exploration noise, not the answer; keep mutating + content/answer calls)
NAV = {"pwd", "ls", "cd", "find", "du", "whoami", "get_current_time"}

BFCL = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(BFCL, "venv/lib/python3.11/site-packages")
NODE = os.path.join(BFCL, "rollouts_extra")
LOC = {
    "deepseek-v4-pro-FC": VENV, "gpt-5.4-FC": VENV, "Qwen3.6-27B-FC": VENV,
    "gemma-4-31B-it": NODE, "Qwen3.5-9B-FC": NODE, "gemma-4-E4B-it": NODE,
}
STRONG = {
    "base":         ["deepseek-v4-pro-FC", "gpt-5.4-FC", "gemma-4-31B-it", "Qwen3.6-27B-FC"],
    "long_context": ["deepseek-v4-pro-FC", "gpt-5.4-FC", "gemma-4-31B-it", "Qwen3.6-27B-FC"],
    "miss_func":    ["deepseek-v4-pro-FC", "gpt-5.4-FC", "gemma-4-31B-it", "Qwen3.6-27B-FC"],
    "miss_param":   ["deepseek-v4-pro-FC", "gpt-5.4-FC", "gemma-4-31B-it", "Qwen3.6-27B-FC"],
}
ALL_MODELS = list(LOC.keys())


def rp(m, cat): return os.path.join(LOC[m], f"result/{m}/multi_turn/BFCL_v4_multi_turn_{cat}_result.json")
def sp(m, cat): return os.path.join(LOC[m], f"score/{m}/multi_turn/BFCL_v4_multi_turn_{cat}_score.json")


def failset(m, cat):
    p = sp(m, cat)
    if not os.path.exists(p):
        return None
    return {json.loads(l)["id"] for l in open(p) if l.strip() and '"id"' in l}


def has_result(m, cat):
    p = rp(m, cat)
    return os.path.exists(p) and sum(1 for l in open(p) if l.strip()) >= 200


def norm_call(s):
    return re.sub(r"[\s'\"]", "", str(s)).lower()


def calls_per_turn(entry):
    """list (per turn) of produced call-strings, both API-structured and gemma-string handlers."""
    out = []
    for steps in entry.get("result", []):
        steps = steps if isinstance(steps, list) else [steps]
        calls = []
        for st in steps:
            if isinstance(st, list):
                for c in st:
                    try:
                        calls.append(render_call(c))
                    except Exception:
                        calls.append(str(c))
            elif isinstance(st, str):
                _, cc = split_gemma_step(st)
                calls += cc
        out.append(calls)
    return out


def cname(c):
    return c.split("(")[0].strip().lower()


def substantive(cpt):
    """flatten per-turn calls, drop pure navigation -> the answer-bearing calls."""
    return [c for turn in cpt for c in turn if cname(c) not in NAV]


def main():
    cat = sys.argv[1] if len(sys.argv) > 1 else "base"
    strong = [m for m in STRONG[cat] if has_result(m, cat) and failset(m, cat) is not None]
    others = [m for m in ALL_MODELS if m not in strong and has_result(m, cat)]
    fs = {m: failset(m, cat) for m in strong}
    cand = sorted(set.intersection(*fs.values()), key=lambda x: int(x.split("_")[-1]))

    ds = {d["id"]: d for d in (json.loads(l) for l in open(os.path.join(VENV, f"bfcl_eval/data/BFCL_v4_multi_turn_{cat}.json")) if l.strip())}
    goldf = {d["id"]: d for d in (json.loads(l) for l in open(os.path.join(VENV, f"bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_{cat}.json")) if l.strip())}
    results = {m: load_jsonl_by_id(rp(m, cat)) for m in strong + others}
    scores = {}
    for m in strong + others:
        p = sp(m, cat)
        if os.path.exists(p):
            scores[m] = {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip() and '"id"' in l}

    L = [f"# Phase 4 bad-key candidates — multi_turn_{cat}",
         f"- strong universe (★) = {strong}",
         f"- other models = {others}",
         f"- all-strong-fail ∩ = **{len(cand)}** items (human-review pool before consensus)",
         "- ★=strong; 🔴=≥2 strong produce the SAME non-gold call-set (oracle is outlier → suspect bad key); ⚪=divergent (likely genuinely hard)\n"]

    consensus = 0
    for cid in cand:
        gold = goldf[cid].get("ground_truth") or goldf[cid].get("possible_answer") or []
        nturn = len(gold)
        gold_sub = [c for turn in gold for c in (turn if isinstance(turn, list) else [turn]) if cname(c) not in NAV]
        gold_sig = frozenset(norm_call(c) for c in gold_sub)
        sig = {}
        for m in strong:
            e = results[m].get(cid)
            sig[m] = frozenset(norm_call(c) for c in substantive(calls_per_turn(e))) if e else frozenset()
        groups = collections.defaultdict(list)
        for m, s in sig.items():
            groups[s].append(m)
        biggest_sig = max(groups, key=lambda s: len(groups[s]))
        biggest = groups[biggest_sig]
        hit = len(biggest) >= 2 and biggest_sig and biggest_sig != gold_sig
        if hit:
            consensus += 1
        tag = f"🔴 CONSENSUS {len(biggest)} strong agree ≠gold ({','.join(m.split('-FC')[0][:8] for m in biggest)})" if hit else "⚪ divergent"
        L.append(f"\n## {cid}  {tag}")
        # user task (the question text decides what's actually correct -> essential for judging a bad key)
        qs = ds[cid].get("question", [])
        utext = " ⏎ ".join(m["content"] for turn in qs for m in turn if m.get("role") == "user")
        L.append(f"- **Q:** {clip(utext, 500)}")
        L.append(f"- GOLD : {clip(' | '.join(gold_sub), 280)}")
        for m in strong + others:
            e = results[m].get(cid)
            sc = substantive(calls_per_turn(e)) if e else []
            mark = "★" if m in strong else "·"
            # BFCL's own fail reason: which turn, expected vs got
            err = scores.get(m, {}).get(cid, {}).get("error", {})
            if isinstance(err, str):
                try: err = json.loads(err)
                except Exception: err = {}
            et = err.get("error_type", "").replace("multi_turn:", "") if err else ""
            em = clip(str(err.get("error_message", "")), 150) if err else ""
            why = f"  ⟦{et}: {em}⟧" if et else ""
            L.append(f"- {mark} {m.split('-FC')[0][:13]:13}: {clip(' | '.join(sc), 230) if sc else '—'}{why}")

    L[5:5] = [f"- **CONSENSUS-flagged = {consensus}/{len(cand)}** (≥2 strong agree on a non-gold answer → likely bad keys; the rest are likely genuinely hard, → legitimate B1)\n"]
    out = os.path.join(BFCL, f"../analysis/phase4_badkey_{cat}.md")
    open(out, "w").write("\n".join(L))
    print(f"wrote {out}")
    print(f"category={cat}  strong={[m.split('-FC')[0] for m in strong]}")
    print(f"all-strong-fail = {len(cand)}   consensus-flagged = {consensus}   divergent = {len(cand)-consensus}")


if __name__ == "__main__":
    main()
