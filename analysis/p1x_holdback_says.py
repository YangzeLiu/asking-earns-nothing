#!/usr/bin/env python3
"""P1-x: what the held-back turn says, on definition C.

The paper's metric is definition C (`p1s_attempt_decision.py`): acting means the model
ATTEMPTED a world-changing call on the anchor turn (`mutator_map.json`, login excluded,
`echo` only when it writes a file), landed or not, and only the pairs whose base-twin
gold anchor turn carries such a call are used. Holding back is the negation, so
read-only lookups on the turn are allowed inside the hold cell (`p1v_c_companions.md` 5).
This file runs a single-turn judge on every held-back miss-side anchor turn of the seven
paper models and shows the judge the lookups the model made. The judge sees that turn
only, never the world state or the official verdict, and no number in Table 1 depends
on its labels. Run with an API key in the environment (see `llm_client.py`).

Labels:
  ask_targeted  asks for, or reports as unavailable, the specific missing information
  ask_other     asks the user something, but not about the missing information
  refuse        declines or says it cannot proceed, without requesting the information
  claim         proceeds in prose: answers, guesses the value, or reports the task done
  other         none of the above

Output: analysis/p1x_holdback_says.jsonl and .md
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_client as J  # noqa: E402
from bfcl_calls import allowed_names  # noqa: E402
from bfcl_calls_full import full_turn_calls  # noqa: E402
from p1d_nonact_rate import PAPER_MODELS as MODELS, VENV  # noqa: E402
from p1f_matched_control import build_pairs, load_results, ASK  # noqa: E402
from p1h_pair_checks import content_check, questions, gold_calls  # noqa: E402
from p1i_holdback_content import prose  # noqa: E402
from p1s_attempt_decision import any_mut  # noqa: E402
from p2c_cert import load_final  # noqa: E402
from p1v_c_companions import gold_at  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(VENV, "bfcl_eval", "data")
OUTP = os.path.join(HERE, "p1x_holdback_says.jsonl")

SYS = """You classify ONE reply from a tool-calling assistant on ONE turn of a dialogue.

On this turn the user asked for something but a piece of information was deliberately removed \
from the setup. You are shown the request as the model received it, the same request with the \
information present, and the model's natural-language reply. On this turn the model made no \
call that would change the world; it MAY have made read-only lookups, which are listed for you. \
Classify the natural-language reply.

Answer with ONE label and nothing else:

ask_targeted - the reply asks the user for the removed information, or names it as missing, or \
states that the needed capability/function is unavailable. A request phrased as a statement \
("I need the ticker symbol before I can place the order") counts. Asking for it among other \
things counts.
ask_other - the reply asks the user a question, but not about the removed information.
refuse - the reply declines or says it cannot proceed, without requesting the removed information.
claim - the reply proceeds anyway in prose: it supplies or assumes a value for the removed \
information, answers the request, or reports the work as done.
other - none of the above (restates the request, an empty or broken reply, unrelated text).
"""


def build_jobs():
    pairs, _ = build_pairs()
    ok, _, _ = content_check(pairs)
    pairs = {i: pairs[i] for i in ok}
    gb = gold_calls("base")
    # definition C: keep only pairs whose base-twin gold anchor turn changes the world
    construction = load_final()[0]
    pairs = {i: p for i, p in pairs.items()
             if any_mut(gold_at(gb, pairs, i)) and i not in construction}
    qb = questions("base")
    qm = {c: questions(c) for c in ASK}
    missed = {}
    for cat in ASK:
        missed[cat] = {}
        for l in open(os.path.join(DATA, f"BFCL_v4_multi_turn_{cat}.json")):
            if l.strip():
                r = json.loads(l)
                mf = r.get("missed_function") or {}
                names = []
                for turn_funcs in mf.values():
                    for f in turn_funcs:
                        names.append(f["name"] if isinstance(f, dict) else str(f))
                missed[cat][r["id"]] = names
    jobs = []
    for model in MODELS:
        res = {c: load_results(model, c) for c in ASK}
        if any(v is None for v in res.values()):
            continue
        allow = {c: allowed_names(c) for c in ASK}
        for iid, (bid, k, cat) in pairs.items():
            seq = res[cat].get(iid)
            if seq is None:
                continue
            turn = seq[k] if k < len(seq) else []
            calls = full_turn_calls(turn, allow[cat].get(iid, set()))
            if any_mut(calls):
                continue
            jobs.append(dict(
                model=model, id=iid, cat=cat, turn=k,
                user_missing=" ".join(qm[cat][iid][k]),
                user_complete=" ".join(qb[bid][k]) if k < len(qb[bid]) else "",
                removed_function=missed[cat].get(iid, []),
                lookups=[n for n, _ in calls],
                reply=prose(turn)[:4000]))
    return jobs


def one(job, client):
    extra = (f"\nA function was removed from the model's tool list on this turn: "
             f"{', '.join(job['removed_function'])}\n" if job["removed_function"] else "")
    look = ", ".join(job["lookups"]) or "none"
    user = (f"USER REQUEST AS THE MODEL SAW IT:\n{job['user_missing']}\n\n"
            f"THE SAME REQUEST WITH THE INFORMATION PRESENT (for reference only; the model "
            f"never saw this):\n{job['user_complete']}\n{extra}\n"
            f"READ-ONLY LOOKUPS THE MODEL MADE ON THIS TURN: {look}\n\n"
            f"THE MODEL'S REPLY ON THIS TURN:\n{job['reply'] or '(no text at all)'}\n\nLabel:")
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=J.JUDGE_MODEL, temperature=0,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}])
            lab = r.choices[0].message.content.strip().lower().split()[0].strip(".,'\"")
            if lab in ("ask_targeted", "ask_other", "refuse", "claim", "other"):
                return dict(job, label=lab)
        except Exception as e:
            err = str(e)[:200]
    return dict(job, label="error", error=locals().get("err", "bad label"))


def main():
    jobs = build_jobs()
    done = {}
    if os.path.exists(OUTP):
        for l in open(OUTP):
            if l.strip():
                r = json.loads(l)
                if r.get("label") not in (None, "error"):
                    done[(r["model"], r["id"])] = r
    todo = [j for j in jobs if (j["model"], j["id"]) not in done]
    print(f"{len(jobs)} held-back turns, {len(done)} already judged, {len(todo)} to go",
          flush=True)
    client = J.make_client()
    out = [done[(j["model"], j["id"])] for j in jobs if (j["model"], j["id"]) in done]
    with ThreadPoolExecutor(max_workers=16) as ex, open(OUTP, "a") as fh:
        for i, r in enumerate(ex.map(lambda j: one(j, client), todo)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            out.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
    report(out)


def report(rows):
    import collections
    LABELS = ["ask_targeted", "ask_other", "refuse", "claim", "other", "error"]
    lines = ["# P1-x what the held-back turn says under definition C (single-turn judge)", "",
             "Hold set: the miss-side anchor turn of a certified definition-C pair (223 pairs, gold",
             "anchor changes the world, construction errors out) on which the model attempted no",
             "world-changing call. Read-only",
             "lookups are allowed and are shown to the judge. One judge call per turn; the judge",
             "sees only that turn's request, the same request with the information present, the",
             "lookups, and the model's reply. No official verdict, no world state, no other turn.",
             "",
             "| model | held back | " + " | ".join(LABELS) + " | targeted share | claim share |",
             "|---|---|" + "---|" * (len(LABELS) + 2)]
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        if not sub:
            continue
        c = collections.Counter(r["label"] for r in sub)
        lines.append(f"| {model} | {len(sub)} | " + " | ".join(str(c[l]) for l in LABELS) +
                     f" | {100.0*c['ask_targeted']/len(sub):.1f}% |"
                     f" {100.0*c['claim']/len(sub):.1f}% |")
    c = collections.Counter(r["label"] for r in rows)
    n = len(rows)
    lines += ["", f"POOLED: {n} held-back turns, targeted {100.0*c['ask_targeted']/n:.1f}%, "
              f"claim {100.0*c['claim']/n:.1f}% "
              f"(ask_other {c['ask_other']}, refuse {c['refuse']}, other {c['other']}, "
              f"error {c['error']}).", "",
              "## Held-back turns split by whether the model looked things up", "",
              "| lookups on the turn | turns | " + " | ".join(LABELS) + " | targeted share |",
              "|---|---|" + "---|" * (len(LABELS) + 1)]
    for lab, sub in (("with read-only lookups", [r for r in rows if r.get("lookups")]),
                     ("no calls at all", [r for r in rows if not r.get("lookups")])):
        if not sub:
            continue
        c = collections.Counter(r["label"] for r in sub)
        lines.append(f"| {lab} | {len(sub)} | " + " | ".join(str(c[l]) for l in LABELS) +
                     f" | {100.0*c['ask_targeted']/len(sub):.1f}% |")
    txt = "\n".join(lines)
    open(os.path.join(HERE, "p1x_holdback_says.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report([json.loads(l) for l in open(OUTP) if l.strip()])
    else:
        main()
