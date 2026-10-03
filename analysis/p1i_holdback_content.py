#!/usr/bin/env python3
"""P1-i: what is in the turn when the model does not call a tool?

The paired decision measure of `p1g` counts a turn with no validated tool call as holding
back. The obvious objection is that silence, a refusal and a question all land in that cell,
so a model that simply stalls would look cautious. This script splits the cell mechanically,
with no judge: on every held-back anchor turn (official PASS and FAIL alike) it reads the
model's own natural-language output and records whether there is text at all and whether the
text contains a question mark. That is a lower bound on asking, not a semantic check, but it
costs nothing and it is reproducible from the released rollouts.

Output: analysis/p1i_holdback_content.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfcl_calls import turn_calls, allowed_names, raw_turn_calls  # noqa: E402
from p1d_nonact_rate import PAPER_MODELS as MODELS, VENV  # noqa: E402
from p1f_matched_control import build_pairs, load_results, ASK  # noqa: E402
from p1h_pair_checks import content_check  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def prose(turn):
    """Natural-language text the model produced on this turn (call payloads stripped)."""
    out = []
    for step in (turn if isinstance(turn, list) else [turn]):
        if isinstance(step, str):
            s = step.strip()
            if s and not s.startswith("["):
                out.append(s)
    return " ".join(out)


def main():
    pairs, _ = build_pairs()
    ok, _, _ = content_check(pairs)
    pairs = {i: pairs[i] for i in ok}
    lines = ["# P1-i what the held-back turn contains (mechanical, no judge)", "",
             f"{len(pairs)} content-verified pairs. A held-back turn is an anchor turn with no",
             "validated tool call. PASS and FAIL alike. `?` is a lower bound on asking.", "",
             "| model | held back | with text | text has `?` | silent | `?` share of held back |",
             "|---|---|---|---|---|---|"]
    for model in MODELS:
        res = {c: load_results(model, c) for c in ASK}
        if any(v is None for v in res.values()):
            continue
        allow = {c: allowed_names(c) for c in ASK}
        held = text = q = 0
        for iid, (bid, k, cat) in pairs.items():
            seq = res[cat].get(iid)
            if seq is None:
                continue
            turn = seq[k] if k < len(seq) else []
            if turn_calls(turn, allow[cat].get(iid, set())):
                continue
            held += 1
            t = prose(turn)
            if t:
                text += 1
                if "?" in t:
                    q += 1
        if held:
            lines.append(f"| {model} | {held} | {text} | {q} | {held - text} | "
                         f"{100.0 * q / held:.1f}% |")
    txt = "\n".join(lines)
    open(os.path.join(HERE, "p1i_holdback_content.md"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
