#!/usr/bin/env python3
"""Wash BFCL multi_turn_base raw logs into human-readable markdown for co-reading.

Renders ONE scenario across N models, turn-by-turn, surfacing:
  - the user's question(s) per turn
  - the GOLD minimal calls per turn
  - each model's per-step trace: thinking prose (💭), tool calls (🔧), final text (💬)
  - verdict (PASS/FAIL) + error type per model

Data sources (so it works for BOTH passed and failed entries):
  - prompt + gold : dataset files (cover ALL ids)
  - raw trace + reasoning_content : result file (cover ALL ids)
  - error + decoded : score file (only FAILED ids)
  - verdict : PASS iff id present in result and NOT in the score fail set

Usage:
  python3 clean_trace.py <id> [model1 model2 ...]      # writes traces/base_<id>.md
  python3 clean_trace.py 2                              # default 3-model compare
  python3 clean_trace.py 2 gemma-4-31B-it gemma-4-31B-it-think

Default models compare frontier vs gemma-31B think/no-think.
"""
import json, os, re, sys

ROOT = os.path.join(os.path.dirname(__file__), "venv/lib/python3.11/site-packages")
CAT = "multi_turn"
SCORE = "BFCL_v4_multi_turn_base_score.json"
RESULT = "BFCL_v4_multi_turn_base_result.json"
DATA = os.path.join(ROOT, "bfcl_eval/data/BFCL_v4_multi_turn_base.json")
GOLD = os.path.join(ROOT, "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json")
OUTDIR = os.path.join(os.path.dirname(__file__), "traces")

DEFAULT_MODELS = ["deepseek-v4-pro-FC", "gemma-4-31B-it", "gemma-4-31B-it-think"]
THINK_CAP = 1400   # thinking prose is the whole point of qualitative reading -> generous
TEXT_CAP = 700


def norm_id(x):
    return x if str(x).startswith("multi_turn_base_") else f"multi_turn_base_{x}"


def clip(s, n):
    return s if len(s) <= n else s[:n] + f" …(+{len(s)-n} chars)"


def load_jsonl_by_id(path):
    out = {}
    for l in open(path):
        if l.strip():
            d = json.loads(l)
            if "id" in d:          # score file line 0 is a summary w/o id -> skip
                out[d["id"]] = d
    return out


def load_dataset(path):
    out = {}
    for l in open(path):
        if l.strip():
            d = json.loads(l)
            out[d["id"]] = d
    return out


def render_call(d):
    """{funcname: '{"a": 1}'} -> funcname(a=1)"""
    (fn, args), = d.items()
    try:
        a = json.loads(args) if isinstance(args, str) else args
        inner = ", ".join(f"{k}={v!r}" for k, v in a.items())
    except Exception:
        inner = str(args)
    return f"{fn}({inner})"


def split_gemma_step(s):
    """gemma step string -> (thinking_or_text, [call_strings]).

    Format: 'thought\\n<prose>[call(), call()]' or 'thought\\n<prose><final text>'.
    Find top-level [...] groups (string-aware, depth>0 quote guard so prose
    apostrophes don't eat the trailing call list), strip them out; remainder is
    thinking (when calls present) or the final user-facing answer (when none).
    """
    groups, depth, start, in_str, spans = [], 0, None, None, []
    for i, ch in enumerate(s):
        if in_str is not None:
            if ch == in_str and s[i - 1] != "\\":
                in_str = None
            continue
        if depth > 0 and ch in ("'", '"'):
            in_str = ch
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(s[start:i + 1]); spans.append((start, i + 1)); start = None
    # individual calls inside each [...] group
    calls = []
    for g in groups:
        body = g[1:-1]
        # split on top-level commas between func() calls
        parts, d2, st2, q2 = [], 0, 0, None
        for j, ch in enumerate(body):
            if q2 is not None:
                if ch == q2 and body[j - 1] != "\\":
                    q2 = None
                continue
            if d2 > 0 and ch in ("'", '"'):
                q2 = ch
            elif ch in "([":
                d2 += 1
            elif ch in ")]":
                d2 -= 1
            elif ch == "," and d2 == 0:
                parts.append(body[st2:j]); st2 = j + 1
        parts.append(body[st2:])
        calls += [p.strip() for p in parts if p.strip()]
    # remainder text with call groups removed
    rem, prev = [], 0
    for a, b in spans:
        rem.append(s[prev:a]); prev = b
    rem.append(s[prev:])
    text = "".join(rem).strip()
    if text.startswith("thought"):
        text = text[len("thought"):].lstrip("\n :")
    return text, calls


def render_model_turn(model, entry, t):
    """Return list of markdown lines for one model's steps in turn t."""
    lines = []
    result = entry.get("result", [])
    if t >= len(result):
        lines.append("  _(no output this turn)_")
        return lines
    steps = result[t]
    steps = steps if isinstance(steps, list) else [steps]
    reasoning = entry.get("reasoning_content")
    rt = reasoning[t] if (reasoning and t < len(reasoning)) else None
    for si, st in enumerate(steps):
        # deepseek-style thinking (parallel reasoning_content)
        if rt and isinstance(rt, list) and si < len(rt) and rt[si]:
            lines.append(f"  - 💭 {clip(str(rt[si]).strip(), THINK_CAP)}")
        if isinstance(st, list):  # structured calls (API handlers)
            for c in st:
                lines.append(f"  - 🔧 `{render_call(c)}`")
        elif isinstance(st, str):
            text, calls = split_gemma_step(st)
            if calls:  # gemma: prose is thinking, then the call list
                if text:
                    lines.append(f"  - 💭 {clip(text, THINK_CAP)}")
                for c in calls:
                    lines.append(f"  - 🔧 `{c}`")
            else:      # final user-facing answer (no calls)
                lines.append(f"  - 💬 {clip(text or st.strip(), TEXT_CAP)}")
        else:
            lines.append(f"  - ? {clip(str(st), TEXT_CAP)}")
    if not steps:
        lines.append("  _(empty turn)_")
    return lines


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    mid = norm_id(sys.argv[1])
    models = sys.argv[2:] or DEFAULT_MODELS

    ds = load_dataset(DATA)
    gold_ds = load_dataset(GOLD)
    questions = ds[mid]["question"]
    gold = gold_ds[mid]["ground_truth"] if "ground_truth" in gold_ds[mid] else gold_ds[mid].get("possible_answer")

    # per-model: result entry, fail-or-None, verdict
    mdata = {}
    for m in models:
        res = load_jsonl_by_id(os.path.join(ROOT, "result", m, CAT, RESULT))
        fails = {i: d for i, d in load_jsonl_by_id(os.path.join(ROOT, "score", m, CAT, SCORE)).items()
                 if d.get("valid") is False}
        entry = res.get(mid)
        verdict = "PASS" if (entry is not None and mid not in fails) else ("FAIL" if mid in fails else "MISSING")
        mdata[m] = {"entry": entry, "fail": fails.get(mid), "verdict": verdict}

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"{mid.replace('multi_turn_base_', 'base_')}.md")
    L = []
    head = "  ".join(f"{m.replace('deepseek-','').replace('-it','')}={mdata[m]['verdict']}" for m in models)
    L.append(f"# {mid}  —  {head}\n")

    nturns = len(questions)
    for t in range(nturns):
        L.append(f"\n## TURN {t}")
        for msg in questions[t]:
            role = msg.get("role", "user").upper()
            L.append(f"**{role}:** {msg['content']}")
        if gold and t < len(gold):
            gcalls = " ; ".join(gold[t]) if isinstance(gold[t], list) else str(gold[t])
            L.append(f"\n**GOLD:** `{gcalls}`")
        for m in models:
            L.append(f"\n### {m}  [{mdata[m]['verdict']}]")
            if mdata[m]["entry"] is None:
                L.append("  _(no result)_"); continue
            L += render_model_turn(m, mdata[m]["entry"], t)

    # errors
    L.append("\n## ERRORS")
    for m in models:
        f = mdata[m]["fail"]
        if not f:
            L.append(f"- **{m}**: (passed)")
            continue
        err = f.get("error", {})
        err = json.loads(err) if isinstance(err, str) else err
        et = err.get("error_type", "?").replace("multi_turn:", "")
        msg = clip(str(err.get("error_message", "")), 400)
        L.append(f"- **{m}**: `{et}` — {msg}")

    open(out, "w").write("\n".join(L))
    print(f"wrote {out}  ({len(L)} lines, {nturns} turns, models={models})")


if __name__ == "__main__":
    main()
