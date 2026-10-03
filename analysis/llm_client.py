#!/usr/bin/env python3
"""OpenAI-compatible chat client for the one script in this repository that calls a model.

`p1x_holdback_says.py` (the single-turn classifier of the held-back turns) imports
`make_client()` and `JUDGE_MODEL` from here. No number in Table 1 depends on it.

Environment
  JUDGE_MODEL      model name sent with every request (default: deepseek-v4-flash, the
                   classifier used in the paper)
  JUDGE_BASE_URL   OpenAI-compatible endpoint (default: https://api.deepseek.com)
  JUDGE_API_KEY    API key for that endpoint (DEEPSEEK_API_KEY is accepted as a fallback)

Nothing is read from a file and no key is stored anywhere in this repository.
"""
import os

JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")


def make_client():
    """Return an `openai.OpenAI` client for JUDGE_BASE_URL, with proxy variables cleared."""
    for k in ["ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"]:
        os.environ.pop(k, None)
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("JUDGE_API_KEY") or os.environ["DEEPSEEK_API_KEY"],
                  base_url=JUDGE_BASE_URL)
