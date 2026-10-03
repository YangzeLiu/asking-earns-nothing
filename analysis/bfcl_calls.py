#!/usr/bin/env python3
"""Shared mechanical tool-call extractor for BFCL multi-turn result files.

WHY THIS EXISTS
---------------
An earlier regex extractor ran `re.finditer(r"(\\b[a-zA-Z_]\\w*)\\s*\\(", step)`
over EVERY string in a turn, including ordinary assistant prose. Two failure modes:

  (a) FALSE POSITIVES from prose. `multi_turn_miss_param_65` (gpt-5.4-FC neutral,
      anchor turn 0) is a pure clarification question containing
      "...your Twitter username/password (or confirm you're already logged in)."
      -> "password" was extracted as a called function -> acted=1.
      `multi_turn_miss_func_69` (gpt-5.4-FC neutral, anchor turn 0) contains
      "- Destination: Rivermist (ZIP 83214)" -> "Rivermist" was extracted and,
      because "rm" is a substring of "rivermist", tagged as a MUTATING call.

  (b) FALSE NEGATIVES for the Qwen handlers. `Qwen3.5-9B-FC` / `Qwen3.6-27B-FC`
      emit `<tool_call>{"name": "mv", "arguments": {...}}</tool_call>` TEXT, which
      contains no "(" at all -> every real call was invisible to the regex, and the
      only thing the counter ever saw in those files was prose.

The replacement parses each result format structurally and then validates every
extracted name against the function list actually available to that item (union of
the item's `involved_classes` function docs, i.e. BFCL's own tool list). A token
that is not a real tool name of that item is not a call.

Three result formats occur in this repo:
  1. native FC (gpt-5.4-FC, deepseek-v4-pro-FC): step = list of {func_name: json_args}
  2. Qwen tool-call tags (Qwen3.5-9B-FC, Qwen3.6-27B-FC): step = string with
     <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  3. prompting mode (gemma-4-31B-it, gemma-4-E4B-it): step = string 'thought\\n[cd(...), mkdir(...)]'
     -> parsed with BFCL's own gemma recipe (strip gemma tokens, split top-level
     [...] groups string/nesting-aware, ast.parse each group in eval mode and keep
     ast.Call nodes) — same algorithm as
     bfcl_eval/model_handler/local_inference/gemma.py::decode_ast.

The mutate/observe keyword table is UNCHANGED; it is simply applied only to
validated function names.
"""
import ast
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_SP = os.path.join(HERE, "..", "bfcl", "venv", "lib", "python3.11", "site-packages")
DATA_DIR = os.path.join(VENV_SP, "bfcl_eval", "data")
FUNC_DOC_DIR = os.path.join(DATA_DIR, "multi_turn_func_doc")

# bfcl_eval/constants/executable_backend_config.py::MULTI_TURN_FUNC_DOC_FILE_MAPPING
FUNC_DOC_FILE = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}

# ---------------------------------------------------------------- keyword tagger
# (UNCHANGED from the sealed table; only the input names are now validated)
_MUTATE_KW = ("create", "mkdir", "touch", "write", "echo", "mv", "move", "rename", "cp", "copy",
              "rm", "remove", "delete", "rmdir", "edit", "append", "insert", "update", "set_",
              "post", "send", "message", "tweet", "retweet", "comment", "reply", "follow", "unfollow",
              "buy", "sell", "order", "place_order", "cancel", "trade", "transfer", "deposit",
              "withdraw", "pay", "charge", "book", "reserve", "purchase", "register", "add_",
              "fill", "lock", "unlock", "activate", "deactivate", "login", "logout", "authenticate",
              "start", "stop", "close", "open_", "change_", "modify", "press", "set", "add")
_OBSERVE_KW = ("cd", "ls", "pwd", "cat", "find", "du", "wc", "sort", "grep", "head", "tail",
               "get", "view", "read", "list", "show", "display", "search", "query", "check",
               "estimate", "compute", "calculate", "mean", "sum", "logarithm", "power", "divide",
               "multiply", "subtract", "absolute", "round", "balance", "info", "detail", "status",
               "history", "lookup", "fetch", "retrieve", "current", "available")
# login/authenticate are NOT consequential mutations (logging in is not a consequential action)
_NONMUT = {"login", "logout", "authenticate"}


def tag(name):
    n = (name or "").lower()
    if n in _NONMUT:
        return "observe"
    for kw in _OBSERVE_KW:
        if n == kw or n.startswith(kw):
            return "observe"
    for kw in _MUTATE_KW:
        if kw in n:
            return "mutate"
    return "observe"


# ------------------------------------------------------------ item -> tool list
_CLASS_FUNCS = {}
_ALLOWED = {}


def _class_funcs(cls):
    if cls not in _CLASS_FUNCS:
        p = os.path.join(FUNC_DOC_DIR, FUNC_DOC_FILE[cls])
        names = set()
        for line in open(p):
            if line.strip():
                names.add(json.loads(line)["name"])
        _CLASS_FUNCS[cls] = names
    return _CLASS_FUNCS[cls]


def allowed_names(cat):
    """id -> set of function names available to that item (union of involved_classes).

    Includes the deliberately removed `missed_function` (it belongs to the class),
    so an attempted call on it still counts as a call.
    """
    if cat not in _ALLOWED:
        out = {}
        p = os.path.join(DATA_DIR, f"BFCL_v4_multi_turn_{cat}.json")
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            s = set()
            for cls in r.get("involved_classes", []):
                s |= _class_funcs(cls)
            for turn_funcs in (r.get("missed_function") or {}).values():
                for f in turn_funcs:
                    if isinstance(f, str):
                        s.add(f)
                    elif isinstance(f, dict) and "name" in f:
                        s.add(f["name"])
            for f in (r.get("excluded_function") or []):
                if isinstance(f, str):
                    s.add(f)
            out[r["id"]] = s
        _ALLOWED[cat] = out
    return _ALLOWED[cat]


# ------------------------------------------------------------- text-mode parsers
_GEMMA_TOKENS = ("<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>",
                 "<|tool_response>", "<tool_response|>", "<|tool_call>",
                 "<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>")


def _gemma_clean(text):
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL)
    for tok in _GEMMA_TOKENS:
        text = text.replace(tok, "")
    return text.strip().strip("`").strip()


def _split_bracket_groups(s):
    """Top-level [...] groups, string- and nesting-aware (BFCL gemma handler recipe)."""
    groups, depth, start, in_str = [], 0, None, None
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
                groups.append(s[start:i + 1])
                start = None
    return groups


def _ast_call_names(group):
    """Names of ast.Call nodes in a '[f(...), g(...)]' group (BFCL ast_parse recipe)."""
    try:
        parsed = ast.parse(group.strip().strip("'"), mode="eval")
    except Exception:
        return []
    body = parsed.body
    elems = body.elts if isinstance(body, (ast.List, ast.Tuple)) else [body]
    names = []
    for e in elems:
        if not isinstance(e, ast.Call):
            continue
        f = e.func
        if isinstance(f, ast.Name):
            names.append(f.id)
        elif isinstance(f, ast.Attribute):
            names.append(f.attr)
    return names


_TOOLCALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _toolcall_tag_names(step):
    names = []
    for blob in _TOOLCALL_RE.findall(step):
        blob = blob.strip().rstrip(",")
        try:
            obj = json.loads(blob)
        except Exception:
            m = re.search(r'"name"\s*:\s*"([A-Za-z_]\w*)"', blob)
            if m:
                names.append(m.group(1))
            continue
        for o in (obj if isinstance(obj, list) else [obj]):
            if isinstance(o, dict) and isinstance(o.get("name"), str):
                names.append(o["name"])
    return names


def _string_call_names(step):
    names = _toolcall_tag_names(step)
    if names:
        return names
    cleaned = _gemma_clean(step)
    for g in _split_bracket_groups(cleaned):
        names.extend(_ast_call_names(g))
    return names


# ------------------------------------------------------------------- public API
def raw_turn_calls(turn):
    """All structurally-parsed candidate call names in one turn (before validation)."""
    names = []
    for step in (turn if isinstance(turn, list) else [turn]):
        if isinstance(step, str):
            names.extend(_string_call_names(step))
        elif isinstance(step, list):
            for c in step:
                if isinstance(c, dict):
                    names.extend(c.keys())
                elif isinstance(c, str):
                    names.extend(_string_call_names(c))
        elif isinstance(step, dict):
            names.extend(step.keys())
    return names


def turn_calls(turn, allowed=None):
    """Validated function names called in one turn.

    `allowed` = set of tool names available to the item (from allowed_names(cat)).
    Passing None disables validation (structural parsing only).
    """
    names = raw_turn_calls(turn)
    if allowed is None:
        return names
    return [n for n in names if n in allowed]
