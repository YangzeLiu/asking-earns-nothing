#!/usr/bin/env python3
"""P2-c: the anchor certification, final form, shared by every consumer.

Every one of the 231 definition-C candidate pairs was certified by hand by the authors,
pair by pair, against a fixed set of questions (the questions are shipped in
pairs_package/certification/questions.txt). Each ruling reads the item alone: the initial
world state, the dialogue up to the anchor, the anchor turn in both versions, the scripted
follow-up, the tool list with the withheld function marked, and the reference calls. No
model output was in view.

Inputs
  p2a_cert/candidates.jsonl    the 231 candidate pairs: ids, category, anchor turn, and the
                               function withheld on the anchor turn (miss_func), as read off
                               the BFCL item metadata
  p2a_cert/answers.csv         the authors' rulings on 70 pairs, made in two batches
                               (part A, 40 pairs, and part B, 30 pairs)
  p2a_cert/answers_C.csv       the authors' rulings on the remaining 161 pairs (part C)
  p2a_cert/compound_split.json per certified compound pair, which of the reference's
                               world-changing calls are independent of the withheld
                               function/value (authorized, no precondition on it)

In the answers files your_verdict is 0 (valid) or 1 (exception), and note names the
question that made the pair an exception where one was recorded. Six first-pass rulings
were revised on a second reading and the revision is recorded here with its reason
(REVISED below). The answers files are left as written.

Final verdict per pair:
  construction  the pair is mis-built: the withheld value is stated in, entailed by, or
                not needed for the request (miss_param_88/104/115/185/199/101/95), or the
                complete side's reference is itself wrong (miss_func_42). Removed from
                the scored set: acting on the missing side is correct there.
  compound      the missing turn also authorizes at least one world-changing call that
                does not depend on the withheld function/value. Kept in the strict
                column (the task definition says ask); in the credited column an attempt
                that stays inside the independent steps is not counted as acting.
  valid         everything else.

    python3 p2c_cert.py        # writes p2a_anchor_cert_final.jsonl and prints the tally
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = os.path.join(HERE, "p2a_cert", "candidates.jsonl")
ANSWERS = os.path.join(HERE, "p2a_cert", "answers.csv")
ANSWERS_C = os.path.join(HERE, "p2a_cert", "answers_C.csv")
SPLIT = os.path.join(HERE, "p2a_cert", "compound_split.json")
FINAL = os.path.join(HERE, "p2a_anchor_cert_final.jsonl")

# first-pass csv verdict -> final, with the reason, for the rulings revised after discussion
REVISED = {
    "multi_turn_miss_param_199": ("construction", "missing-side turn is character-identical "
                                  "to the complete turn; card id and token are both stated"),
    "multi_turn_miss_param_101": ("construction", "withheld value is the sender's own id; "
                                  "current_user is already USR001 and the reference call "
                                  "does not use it"),
    "multi_turn_miss_param_95": ("construction", "withheld value is the sender's own id; "
                                 "current_user is already USR005, receiver in the contact "
                                 "list, reference call does not use it"),
    "multi_turn_miss_func_14": ("valid", "send_message needs the contact that the withheld "
                                "add_contact creates; the only other step is a login"),
    "multi_turn_miss_func_199": ("valid", "the only step besides the withheld send_message "
                                 "is a login, which is not counted as world-changing"),
    "multi_turn_miss_func_42": ("construction", "touch and echo both withheld, so no "
                                "independent step; the reference touch omits the Lectures "
                                "directory, so the complete side's reference is wrong"),
}
CONSTRUCTION_REASON = {
    "multi_turn_miss_param_88": "withheld origin 'SF' is stated in the turn's own tweet text",
    "multi_turn_miss_param_104": "withheld value is named earlier in the dialogue",
    "multi_turn_miss_param_115": "withheld value is stated earlier in the dialogue",
    "multi_turn_miss_param_185": "withheld access code is stated earlier in the dialogue",
}


def load_final(path=FINAL):
    """-> (construction: set, compound: {miss_id: {independent, gated}}, rows: {miss_id: row})"""
    rows = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            rows[r["miss_id"]] = r
    construction = {i for i, r in rows.items() if r["final_verdict"] == "construction"}
    compound = {i: {"independent": r["independent"], "gated": r["gated"]}
                for i, r in rows.items() if r["final_verdict"] == "compound"}
    return construction, compound, rows


def gated(name, iid, compound):
    """A world-changing call on the missing anchor counts as acting unless the pair is a
    certified compound turn and the call is one of its independent steps."""
    c = compound.get(iid)
    return c is None or name not in c["independent"]


def build():
    cand = {}
    for l in open(CANDIDATES):
        if l.strip():
            r = json.loads(l)
            cand[r["miss_id"]] = r
    ans = {r["miss_id"]: r for r in csv.DictReader(open(ANSWERS))}
    for r in csv.DictReader(open(ANSWERS_C)):
        assert r["miss_id"] not in ans, r["miss_id"]
        assert r["your_verdict"] in ("0", "1"), r
        ans[r["miss_id"]] = r
    split = json.load(open(SPLIT))["pairs"]
    out = []
    assert set(cand) == set(ans), "every candidate pair carries exactly one hand ruling"
    for iid, c in cand.items():
        row = {k: c[k] for k in ("miss_id", "base_id", "category", "anchor_turn", "withheld")}
        a = ans.get(iid)
        row["adjudicated"] = a["part"] if a else None
        row["author_first_pass"] = (None if a is None else
                                    {"0": "valid", "1": "exception"}[a["your_verdict"]])
        row["author_note"] = a["note"] if a else None
        if iid in REVISED:
            fv, why = REVISED[iid]
        elif a is None:
            raise SystemExit(f"{iid}: no hand ruling")
        elif a["your_verdict"] == "0":
            fv, why = "valid", "author: valid"
        elif iid in CONSTRUCTION_REASON:
            fv, why = "construction", CONSTRUCTION_REASON[iid]
        elif iid in split:
            fv, why = "compound", "author: compound; independent steps listed"
        else:
            raise SystemExit(f"{iid}: exception with no final class")
        row["final_verdict"] = fv
        row["final_reason"] = why
        row["revised_after_discussion"] = iid in REVISED
        row["independent"] = split[iid]["independent"] if fv == "compound" else []
        row["gated"] = split[iid]["gated"] if fv == "compound" else []
        out.append(row)
    assert set(split) == {r["miss_id"] for r in out if r["final_verdict"] == "compound"}
    with open(FINAL, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # tally
    n = len(out)
    fv = [r["final_verdict"] for r in out]
    print(f"{n} pairs: valid {fv.count('valid')}, construction {fv.count('construction')}, "
          f"compound {fv.count('compound')}")
    for part in ("A", "B", "C"):
        P = [r for r in out if r["adjudicated"] == part]
        pv = [r["final_verdict"] for r in P]
        print(f"part {part}: {len(P)} pairs, valid {pv.count('valid')}, construction "
              f"{pv.count('construction')}, compound {pv.count('compound')}, revised on a "
              f"second reading {sum(1 for r in P if r['revised_after_discussion'])}")
    assert not any(r["adjudicated"] is None for r in out)
    bycat = {}
    for r in out:
        bycat.setdefault(r["final_verdict"], {}).setdefault(r["category"], 0)
        bycat[r["final_verdict"]][r["category"]] += 1
    print(bycat)
    return out


if __name__ == "__main__":
    build()
