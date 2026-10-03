# P1-t Phase 5 anchor turn under the attempted definition of a state change

Same arms and McNemar as `phase5_mechanical.md`; "acting" = attempted a call
that changes the world on success (`mutator_map.json`), landed or not; the 31
hand-verified bad keys are dropped as in the main table. The Qwen3.5-9B-FC
pro-action arm received no injection (dose 0) and is listed for the record only.

| nudge | model | cat | N | attempted-mutation% neu->var | flips v-only / n-only | McNemar p |
|---|---|---|---|---|---|---|
| pro-caution (expect mutate↓) | gemma-4-31B-it | miss_func | 190 | 33.7->27.4 | 1 / 13 | 0.00183 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | miss_func | 190 | 28.9->31.1 | 10 / 6 | 0.454 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_func | 190 | 18.4->22.1 | 12 / 5 | 0.143 |
| pro-action (expect act/mutate↑) | gemma-4-31B-it | miss_func | 190 | 33.7->36.3 | 10 / 5 | 0.302 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_func | 190 | 18.4->22.1 | 12 / 5 | 0.143 |
| pro-caution (expect mutate↓) | gemma-4-31B-it | miss_param | 184 | 21.7->12.0 | 1 / 19 | 4.01e-05 |
| pro-action (expect act/mutate↑) | Qwen3.5-9B-FC | miss_param | 184 | 21.2->21.2 | 4 / 4 | 1 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_param | 184 | 11.4->22.8 | 22 / 1 | 5.72e-06 |
| pro-action (expect act/mutate↑) | gemma-4-31B-it | miss_param | 184 | 21.7->23.4 | 8 / 5 | 0.581 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_param | 184 | 11.4->22.8 | 22 / 1 | 5.72e-06 |

## Official raw for the arms not in phase5_mechanical.md

| nudge | model | cat | N | raw% neu->var | flips fail->pass / pass->fail | McNemar p |
|---|---|---|---|---|---|---|
| pro-action (expect act/mutate↑) | gemma-4-31B-it | base | 200 | 65.0->64.0 | 9 / 11 | 0.824 |
| pro-action (expect act/mutate↑) | gemma-4-31B-it | long_context | 200 | 39.0->34.5 | 6 / 15 | 0.0784 |
| pro-action (expect act/mutate↑) | gemma-4-31B-it | miss_func | 200 | 54.0->54.0 | 12 / 12 | 1 |
| pro-action (expect act/mutate↑) | gemma-4-31B-it | miss_param | 200 | 48.0->48.5 | 15 / 14 | 1 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | base | 153 | 39.2->62.7 | 43 / 7 | 2.1e-07 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_func | 200 | 33.5->49.5 | 42 / 10 | 9.06e-06 |
| pro-action (expect act/mutate↑) | gpt-5.4-FC | miss_param | 200 | 31.5->45.0 | 41 / 14 | 0.000355 |
