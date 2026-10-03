# P1-x what the held-back turn says under definition C (single-turn judge)

Hold set: the miss-side anchor turn of a certified definition-C pair (223 pairs, gold
anchor changes the world, construction errors out) on which the model attempted no
world-changing call. Read-only
lookups are allowed and are shown to the judge. One judge call per turn; the judge
sees only that turn's request, the same request with the information present, the
lookups, and the model's reply. No official verdict, no world state, no other turn.

| model | held back | ask_targeted | ask_other | refuse | claim | other | error | targeted share | claim share |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-max-0902-FC | 149 | 136 | 5 | 0 | 8 | 0 | 0 | 91.3% | 5.4% |
| gpt-5.4-FC | 174 | 151 | 10 | 3 | 7 | 3 | 0 | 86.8% | 4.0% |
| Qwen3.6-27B-FC | 126 | 110 | 4 | 0 | 11 | 1 | 0 | 87.3% | 8.7% |
| gemma-4-31B-it | 132 | 118 | 0 | 2 | 7 | 5 | 0 | 89.4% | 5.3% |
| deepseek-v4-flash-FC | 132 | 119 | 4 | 0 | 8 | 1 | 0 | 90.2% | 6.1% |
| Qwen3.5-9B-FC | 141 | 117 | 6 | 0 | 6 | 12 | 0 | 83.0% | 4.3% |
| gemma-4-E4B-it | 116 | 88 | 9 | 2 | 4 | 13 | 0 | 75.9% | 3.4% |

POOLED: 970 held-back turns, targeted 86.5%, claim 5.3% (ask_other 38, refuse 7, other 35, error 0).

## Held-back turns split by whether the model looked things up

| lookups on the turn | turns | ask_targeted | ask_other | refuse | claim | other | error | targeted share |
|---|---|---|---|---|---|---|---|---|
| with read-only lookups | 496 | 428 | 18 | 3 | 37 | 10 | 0 | 86.3% |
| no calls at all | 474 | 411 | 20 | 4 | 14 | 25 | 0 | 86.7% |
