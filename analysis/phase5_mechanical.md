# Phase 5: prompt-injection arms, mechanical metrics (paired, no judge)

> Calls are extracted with `analysis/bfcl_calls.py`: each of the three result formats (native
> function-calling records, `<tool_call>` JSON blocks, gemma bracket groups parsed with BFCL's own
> ast recipe) is parsed structurally, and every name is validated against the item's own tool list,
> so prose never registers as a call. `mutate` below is the keyword tag of `bfcl_calls.py` over all
> single-anchor items of a category. The paper's attempt rates use `mutator_map.json`, drop the bad
> keys and the construction errors, and come from `p1t_phase5_attempt.py`.

- The Qwen3.5-9B-FC pro-action arm received no injection (dose 0): `QwenFCHandler` does not read
  `BFCL_EXTRA_SYSTEM_PROMPT`, and the first turn of its result file carries no injected system or
  developer message. Its rows are kept for the record and are not used in the paper.
- Selectivity: should-act activity (median calls per item, thrash count) barely moves under injection.

## should-ask anchor turn (the turn whose gold trajectory is empty)

The injection targets this turn: act and fill in a value (mutate), or hold back and ask. Paired McNemar on the direction of the flips.

| nudge | model | cat | N | acted% neu→var | mutate% neu→var | flips v-only / n-only | McNemar p | acted flips v-only/n-only | McNemar p(acted) |
|---|---|---|---|---|---|---|---|---|---|
| pro-caution (expect mutate↓) | gemma-4-31B-it | miss_func | 198 | 62.1→58.1 | 35.4→30.8 | 2 / 11 | 0.0225 | 3 / 11 | 0.0574 |
| pro-caution (expect mutate↓) | gemma-4-31B-it | miss_param | 198 | 41.4→28.8 | 26.3→15.7 | 1 / 22 | 5.72e-06 | 2 / 27 | 1.62e-06 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | miss_func | 198 | 74.7→78.8 | 37.4→40.4 | 12 / 6 | 0.238 | 15 / 7 | 0.134 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | miss_param | 198 | 59.1→59.6 | 34.3→33.3 | 7 / 9 | 0.804 | 11 / 10 | 1 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_func | 198 | 47.5→60.6 | 22.2→30.8 | 21 / 4 | 0.000911 | 28 / 2 | 8.68e-07 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_param | 198 | 27.3→51.0 | 18.2→32.8 | 31 / 2 | 1.31e-07 | 50 / 3 | 5.52e-12 |

## should-act activity (no single anchor turn; the median is robust to runaway items)

Some items run away to hundreds of turns, which distorts the mean and the empty-turn count, so this reports median calls per item and the thrash count (items with more than 20 turns).

| nudge | model | cat | N | median calls/item neu→var | thrash items (>20 turns) neu→var |
|---|---|---|---|---|---|
| pro-caution (expect mutate↓) | gemma-4-31B-it | base | 200 | 8.0→7.0 | 0→0 |
| pro-caution (expect mutate↓) | gemma-4-31B-it | long_context | 200 | 6.0→6.0 | 38→37 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | base | 200 | 7.0→7.0 | 0→0 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | long_context | 200 | 7.0→7.0 | 42→42 |

## Official raw score, paired per item

Each item's official PASS/FAIL is read straight from the score file (one fail row per line after the header) and paired by id.
`fail→pass` and `pass→fail` count the flips in each direction; McNemar is the exact binomial test on the discordant pairs.
The denominator is 200 (the official one, bad keys included), so these rows cannot be subtracted from the behaviour rows above (denominator 198).

| nudge | model | cat | N | raw% neu→var | Δpp | fail→pass | pass→fail | McNemar p |
|---|---|---|---|---|---|---|---|---|
| pro-caution (expect mutate↓) | gemma-4-31B-it | miss_func | 200 | 54.0→49.0 | -5.0 | 6 | 16 | 0.0525 |
| pro-caution (expect mutate↓) | gemma-4-31B-it | miss_param | 200 | 48.0→48.5 | +0.5 | 19 | 18 | 1 |
| pro-caution (expect mutate↓) | gemma-4-31B-it | base | 200 | 65.0→58.5 | -6.5 | 10 | 23 | 0.0351 |
| pro-caution (expect mutate↓) | gemma-4-31B-it | long_context | 200 | 39.0→33.0 | -6.0 | 2 | 14 | 0.00418 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | miss_func | 200 | 53.5→50.5 | -3.0 | 12 | 18 | 0.362 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | miss_param | 200 | 45.0→42.0 | -3.0 | 10 | 16 | 0.327 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | base | 200 | 60.0→57.5 | -2.5 | 7 | 12 | 0.359 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | long_context | 200 | 34.5→36.0 | +1.5 | 12 | 9 | 0.664 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_func | 200 | 33.5→49.5 | +16.0 | 42 | 10 | 9.06e-06 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_param | 200 | 31.5→45.0 | +13.5 | 41 | 14 | 0.000355 |

- pro-caution on gemma-4-31B-it is the second arm with official scores, in the opposite direction
  to pro-action on gpt-5.4: it lowers the mutate rate on the under-specified turn and gains no official
  raw score. The behavioural effect sits on the under-specified turn of the should-ask categories, while
  the significant raw drops fall mainly on the two should-act categories.
- The Qwen3.5-9B-FC arm has dose 0 (the handler ignores the injection); its raw rows are kept for the record only.
