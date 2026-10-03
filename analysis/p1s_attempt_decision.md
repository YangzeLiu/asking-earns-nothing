# P1-s decision to change the world on the under-specified turn

An attempt counts whether or not the sandbox let it through. 365 content-verified pairs; 231 of them have a base-twin gold anchor turn that contains a world-changing call, 134 are purely read-only there. Login/logout functions are not counted as world-changing; the login-counted variant is reported as a sensitivity. Anchor certification (p2c_cert.py): 8 construction errors removed, 27 compound turns kept with an independent-step list; primary N = 223.

Definition A (act = any call, hold = no mutating call) is **rejected**: its two sides ask different questions, so "emit one read-only call every turn" scores 100 and the chance-50 property is lost.

## Definition C — PRIMARY: symmetric, mutator on both sides, the 231 gold-mutator pairs minus the 8 certified construction errors = 223 pairs (login excluded)

| model | act when complete | hold when removed | decision acc [95% CI] | both twins right | N |
|---|---|---|---|---|---|
| gpt-5.4-FC | 83.4 | 78.0 | 80.7 [77.3, 84.1] | 61.4 | 223 |
| gemma-4-31B-it | 97.8 | 59.2 | 78.5 [75.1, 81.8] | 57.4 | 223 |
| qwen3.8-max-0902-FC | 89.7 | 66.8 | 78.3 [74.3, 82.2] | 57.4 | 223 |
| Qwen3.5-9B-FC | 91.5 | 63.2 | 77.4 [73.5, 81.1] | 56.5 | 223 |
| Qwen3.6-27B-FC | 97.8 | 56.5 | 77.1 [73.3, 81.0] | 55.6 | 223 |
| deepseek-v4-flash-FC | 92.4 | 59.2 | 75.8 [71.8, 79.6] | 53.4 | 223 |
| gemma-4-E4B-it | 83.4 | 52.0 | 67.7 [63.6, 71.8] | 39.0 | 223 |

Spread 13.0 points. Rank-1 frequency: gpt-5.4-FC 77.8%, qwen3.8-max-0902-FC 10.1%, gemma-4-31B-it 8.6%, Qwen3.5-9B-FC 1.8%, Qwen3.6-27B-FC 1.7%, deepseek-v4-flash-FC 0.0%

## Definition C-credited — same 223 pairs; on the 27 certified compound turns an attempt confined to the independent steps is not counted as acting (p2a_cert/compound_split.json)

| model | act when complete | hold when removed | decision acc [95% CI] | both twins right | N |
|---|---|---|---|---|---|
| gpt-5.4-FC | 83.4 | 83.0 | 83.2 [79.7, 86.5] | 66.4 | 223 |
| gemma-4-31B-it | 97.8 | 65.9 | 81.8 [78.4, 85.1] | 63.7 | 223 |
| qwen3.8-max-0902-FC | 89.7 | 71.7 | 80.7 [76.9, 84.5] | 62.3 | 223 |
| Qwen3.6-27B-FC | 97.8 | 63.7 | 80.7 [77.0, 84.2] | 62.8 | 223 |
| Qwen3.5-9B-FC | 91.5 | 70.0 | 80.7 [76.9, 84.4] | 63.2 | 223 |
| deepseek-v4-flash-FC | 92.4 | 65.9 | 79.1 [75.2, 82.9] | 60.1 | 223 |
| gemma-4-E4B-it | 83.4 | 57.8 | 70.6 [66.7, 74.6] | 44.4 | 223 |

Spread 12.6 points. Rank-1 frequency: gpt-5.4-FC 67.0%, gemma-4-31B-it 16.3%, qwen3.8-max-0902-FC 7.8%, Qwen3.6-27B-FC 4.6%, Qwen3.5-9B-FC 4.1%, deepseek-v4-flash-FC 0.2%

## Definition C-uncertified — sensitivity: all 231 gold-mutator pairs before the anchor certification (the table of the pre-certification draft)

| model | act when complete | hold when removed | decision acc [95% CI] | both twins right | N |
|---|---|---|---|---|---|
| gpt-5.4-FC | 83.1 | 76.6 | 79.9 [76.3, 83.3] | 59.7 | 231 |
| qwen3.8-max-0902-FC | 90.0 | 65.4 | 77.7 [73.9, 81.5] | 56.3 | 231 |
| gemma-4-31B-it | 97.8 | 57.6 | 77.7 [74.4, 81.1] | 55.8 | 231 |
| Qwen3.5-9B-FC | 91.8 | 61.9 | 76.8 [73.1, 80.4] | 55.4 | 231 |
| Qwen3.6-27B-FC | 97.8 | 55.4 | 76.6 [72.8, 80.4] | 54.5 | 231 |
| deepseek-v4-flash-FC | 92.6 | 57.6 | 75.1 [71.4, 78.9] | 51.9 | 231 |
| gemma-4-E4B-it | 83.5 | 50.6 | 67.1 [63.0, 71.2] | 37.7 | 231 |

Spread 12.8 points. Rank-1 frequency: gpt-5.4-FC 74.9%, qwen3.8-max-0902-FC 12.7%, gemma-4-31B-it 7.8%, Qwen3.6-27B-FC 2.3%, Qwen3.5-9B-FC 2.2%, deepseek-v4-flash-FC 0.0%

## Definition C-no-compound — sensitivity: C minus the 27 compound turns as well (196 pairs)

| model | act when complete | hold when removed | decision acc [95% CI] | both twins right | N |
|---|---|---|---|---|---|
| qwen3.8-max-0902-FC | 88.8 | 75.5 | 82.1 [77.9, 86.2] | 65.3 | 196 |
| gpt-5.4-FC | 82.1 | 82.1 | 82.1 [78.4, 85.8] | 64.3 | 196 |
| gemma-4-31B-it | 98.5 | 65.3 | 81.9 [78.4, 85.3] | 63.8 | 196 |
| Qwen3.6-27B-FC | 98.0 | 63.8 | 80.9 [77.0, 84.7] | 62.8 | 196 |
| Qwen3.5-9B-FC | 90.8 | 70.9 | 80.9 [76.8, 84.7] | 63.3 | 196 |
| deepseek-v4-flash-FC | 92.3 | 67.3 | 79.8 [75.8, 83.8] | 60.7 | 196 |
| gemma-4-E4B-it | 83.7 | 57.7 | 70.7 [66.4, 74.9] | 43.9 | 196 |

Spread 11.5 points. Rank-1 frequency: gpt-5.4-FC 34.3%, qwen3.8-max-0902-FC 34.0%, gemma-4-31B-it 18.9%, Qwen3.5-9B-FC 6.0%, Qwen3.6-27B-FC 5.9%, deepseek-v4-flash-FC 0.8%

## Definition B — symmetric, mutator on both sides, all 365

| model | act when complete | hold when removed | decision acc [95% CI] | both twins right | N |
|---|---|---|---|---|---|
| gpt-5.4-FC | 52.6 | 84.4 | 68.5 [65.6, 71.5] | 37.8 | 365 |
| gemma-4-31B-it | 61.9 | 71.5 | 66.7 [63.9, 69.6] | 35.3 | 365 |
| Qwen3.5-9B-FC | 58.1 | 74.2 | 66.2 [63.3, 69.0] | 35.1 | 365 |
| qwen3.8-max-0902-FC | 58.1 | 74.0 | 66.0 [62.9, 69.2] | 36.4 | 365 |
| Qwen3.6-27B-FC | 62.5 | 69.3 | 65.9 [62.9, 68.9] | 34.8 | 365 |
| deepseek-v4-flash-FC | 58.6 | 69.0 | 63.8 [60.7, 67.1] | 32.9 | 365 |
| gemma-4-E4B-it | 53.2 | 67.4 | 60.3 [57.5, 63.2] | 24.1 | 365 |

Spread 8.2 points. Rank-1 frequency: gpt-5.4-FC 86.9%, gemma-4-31B-it 5.7%, qwen3.8-max-0902-FC 3.9%, Qwen3.6-27B-FC 1.9%, Qwen3.5-9B-FC 1.6%

## Definition D — symmetric, any call on both sides, all 365 (the pre-2026-09-16 primary)

| model | act when complete | hold when removed | decision acc [95% CI] | both twins right | N |
|---|---|---|---|---|---|
| gpt-5.4-FC | 91.8 | 63.0 | 77.4 [74.7, 80.1] | 54.8 | 365 |
| gemma-4-31B-it | 99.5 | 48.2 | 73.8 [71.2, 76.5] | 47.9 | 365 |
| gemma-4-E4B-it | 88.8 | 41.9 | 65.3 [62.5, 68.1] | 32.3 | 365 |
| Qwen3.5-9B-FC | 97.3 | 32.6 | 64.9 [62.4, 67.5] | 30.7 | 365 |
| Qwen3.6-27B-FC | 98.1 | 30.7 | 64.4 [61.9, 66.8] | 29.9 | 365 |
| deepseek-v4-flash-FC | 99.5 | 19.5 | 59.5 [57.5, 61.5] | 19.2 | 365 |
| qwen3.8-max-0902-FC | 99.2 | 8.8 | 54.0 [52.6, 55.6] | 8.5 | 365 |

Spread 23.4 points. Rank-1 frequency: gpt-5.4-FC 99.4%, gemma-4-31B-it 0.6%

## Definition C-login — sensitivity: C with the seven login/logout functions counted as mutating (225 pairs)

| model | act when complete | hold when removed | decision acc [95% CI] | both twins right | N |
|---|---|---|---|---|---|
| gpt-5.4-FC | 84.0 | 75.6 | 79.8 [76.3, 83.2] | 59.6 | 225 |
| qwen3.8-max-0902-FC | 90.7 | 64.4 | 77.6 [73.7, 81.3] | 56.0 | 225 |
| gemma-4-31B-it | 97.8 | 57.3 | 77.6 [74.1, 81.0] | 55.6 | 225 |
| Qwen3.5-9B-FC | 91.6 | 61.8 | 76.7 [72.9, 80.4] | 55.1 | 225 |
| Qwen3.6-27B-FC | 97.8 | 54.2 | 76.0 [72.2, 79.7] | 53.3 | 225 |
| deepseek-v4-flash-FC | 93.8 | 56.0 | 74.9 [71.1, 78.6] | 51.6 | 225 |
| gemma-4-E4B-it | 83.6 | 51.1 | 67.3 [63.3, 71.3] | 38.2 | 225 |

Spread 12.4 points. Rank-1 frequency: gpt-5.4-FC 76.8%, qwen3.8-max-0902-FC 11.9%, gemma-4-31B-it 7.4%, Qwen3.5-9B-FC 2.7%, Qwen3.6-27B-FC 1.0%, deepseek-v4-flash-FC 0.1%

## Definition C-reached — sensitivity: C on pairs where both rollouts reached the anchor turn (per-model N)

| model | act when complete | hold when removed | decision acc [95% CI] | both twins right | N |
|---|---|---|---|---|---|
| gpt-5.4-FC | 83.4 | 78.0 | 80.7 [77.3, 84.1] | 61.4 | 223 |
| gemma-4-31B-it | 97.7 | 59.0 | 78.4 [75.0, 81.7] | 57.2 | 222 |
| qwen3.8-max-0902-FC | 89.7 | 66.8 | 78.3 [74.3, 82.2] | 57.4 | 223 |
| Qwen3.5-9B-FC | 91.9 | 63.1 | 77.5 [73.7, 81.1] | 56.8 | 222 |
| Qwen3.6-27B-FC | 97.8 | 56.5 | 77.1 [73.3, 81.0] | 55.6 | 223 |
| deepseek-v4-flash-FC | 92.4 | 59.2 | 75.8 [71.8, 79.6] | 53.4 | 223 |
| gemma-4-E4B-it | 88.2 | 50.2 | 69.2 [65.0, 73.3] | 41.2 | 211 |

Spread 11.5 points.

## Rollouts that never reached the anchor turn

Kept and scored as no call, as in p1g. Pairs (base twin, miss twin) per model:

- qwen3.8-max-0902-FC: 0 / 0
- gpt-5.4-FC: 0 / 0
- Qwen3.6-27B-FC: 0 / 1
- gemma-4-31B-it: 0 / 2
- deepseek-v4-flash-FC: 0 / 0
- Qwen3.5-9B-FC: 2 / 1
- gemma-4-E4B-it: 20 / 19

## Degenerate strategies (information-blind, so the same on both twins)

| strategy | A (rejected) | B | C | D |
|---|---|---|---|---|
| always attempt a mutating call | 50 | 50 | 50 | 50 |
| never call anything | 50 | 50 | 50 | 50 |
| one read-only call on every turn | **100** | 50 | 50 | 50 |

