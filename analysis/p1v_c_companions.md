# P1-v companion numbers under definition C (223 certified pairs; 8 construction errors out, p2c_cert.py)

## 1. The 134 excluded pairs

By category: miss_func 65, miss_param 69
Gold anchor content: read-only 132, login only 2
Most common gold functions on those anchors: get_zipcode_based_on_city 39, estimate_distance 20, get_stock_info 14, cd 12, wc 9, get_watchlist 9, cat 8, check_tire_pressure 8
C by category: miss_func 116, miss_param 107

## 2. Official raw on the 223 pairs

| model | raw on miss items | raw on base twins (deduped) | decision acc |
|---|---|---|---|
| qwen3.8-max-0902-FC | 47.1 | 52.3 | 78.3 |
| gpt-5.4-FC | 32.7 | 39.1 | 80.7 |
| Qwen3.6-27B-FC | 58.7 | 77.5 | 77.1 |
| gemma-4-31B-it | 50.7 | 62.9 | 78.5 |
| deepseek-v4-flash-FC | 52.0 | 66.9 | 75.8 |
| Qwen3.5-9B-FC | 52.5 | 57.6 | 77.4 |
| gemma-4-E4B-it | 22.9 | 25.8 | 67.7 |

151 distinct base twins. Spearman raw(miss) x decision = -0.11 (p=0.82); raw(base twins) x decision = -0.11 (p=0.82).
Official rank (raw on miss items): Qwen3.6-27B-FC 1, Qwen3.5-9B-FC 2, deepseek-v4-flash-FC 3, gemma-4-31B-it 4, qwen3.8-max-0902-FC 5, gpt-5.4-FC 6, gemma-4-E4B-it 7
Decision rank: gpt-5.4-FC 1, gemma-4-31B-it 2, qwen3.8-max-0902-FC 3, Qwen3.5-9B-FC 4, Qwen3.6-27B-FC 5, deepseek-v4-flash-FC 6, gemma-4-E4B-it 7

## 3. Decision accuracy by category

| model | miss_func N | act | hold | dec | miss_param N | act | hold | dec |
|---|---|---|---|---|---|---|---|---|
| qwen3.8-max-0902-FC | 116 | 91.4 | 57.8 | 74.6 | 107 | 87.9 | 76.6 | 82.2 |
| gpt-5.4-FC | 116 | 84.5 | 75.0 | 79.7 | 107 | 82.2 | 81.3 | 81.8 |
| Qwen3.6-27B-FC | 116 | 96.6 | 56.9 | 76.7 | 107 | 99.1 | 56.1 | 77.6 |
| gemma-4-31B-it | 116 | 98.3 | 53.4 | 75.9 | 107 | 97.2 | 65.4 | 81.3 |
| deepseek-v4-flash-FC | 116 | 92.2 | 54.3 | 73.3 | 107 | 92.5 | 64.5 | 78.5 |
| Qwen3.5-9B-FC | 116 | 93.1 | 59.5 | 76.3 | 107 | 89.7 | 67.3 | 78.5 |
| gemma-4-E4B-it | 116 | 83.6 | 47.4 | 65.5 | 107 | 83.2 | 57.0 | 70.1 |

## 4. Pairs with identical validated pre-anchor call names on both twins

| model | N | act | hold | dec |
|---|---|---|---|---|
| qwen3.8-max-0902-FC | 128 | 92.2 | 65.6 | 78.9 |
| gpt-5.4-FC | 182 | 82.4 | 76.9 | 79.7 |
| Qwen3.6-27B-FC | 197 | 99.0 | 58.4 | 78.7 |
| gemma-4-31B-it | 212 | 97.6 | 58.5 | 78.1 |
| deepseek-v4-flash-FC | 149 | 93.3 | 57.7 | 75.5 |
| Qwen3.5-9B-FC | 193 | 92.7 | 64.2 | 78.5 |
| gemma-4-E4B-it | 196 | 87.8 | 49.5 | 68.6 |

## 5. Held-back miss anchor turns (no world-changing attempt)

| model | held back | with read-only calls | with text | silent |
|---|---|---|---|---|
| qwen3.8-max-0902-FC | 149 | 132 | 149 | 0 |
| gpt-5.4-FC | 174 | 46 | 174 | 0 |
| Qwen3.6-27B-FC | 126 | 69 | 126 | 0 |
| gemma-4-31B-it | 132 | 44 | 131 | 1 |
| deepseek-v4-flash-FC | 132 | 95 | 131 | 1 |
| Qwen3.5-9B-FC | 141 | 75 | 141 | 0 |
| gemma-4-E4B-it | 116 | 35 | 103 | 13 |

Pooled: 970 held-back turns, 15 silent.

## 6. Official verdict vs world-changing attempt on the anchor (223 miss items)

| model | passes | attempted % | fails | attempted % |
|---|---|---|---|---|
| qwen3.8-max-0902-FC | 105 | 29.5 | 118 | 36.4 |
| gpt-5.4-FC | 73 | 16.4 | 150 | 24.7 |
| Qwen3.6-27B-FC | 131 | 31.3 | 92 | 60.9 |
| gemma-4-31B-it | 113 | 29.2 | 110 | 52.7 |
| deepseek-v4-flash-FC | 116 | 27.6 | 107 | 55.1 |
| Qwen3.5-9B-FC | 117 | 26.5 | 106 | 48.1 |
| gemma-4-E4B-it | 51 | 45.1 | 172 | 48.8 |

## 7. Base anchor turns without a world-changing attempt

| model | no attempt | of which zero calls | of which read-only calls |
|---|---|---|---|
| qwen3.8-max-0902-FC | 23 | 1 | 22 |
| gpt-5.4-FC | 37 | 21 | 16 |
| Qwen3.6-27B-FC | 5 | 3 | 2 |
| gemma-4-31B-it | 5 | 2 | 3 |
| deepseek-v4-flash-FC | 17 | 1 | 16 |
| Qwen3.5-9B-FC | 19 | 8 | 11 |
| gemma-4-E4B-it | 37 | 23 | 14 |

## 8. Bad keys restored: 230 pairs (7 added back)

| model | dec (223) | dec (restored) | change |
|---|---|---|---|
| qwen3.8-max-0902-FC | 78.3 | 78.3 | +0.01 |
| gpt-5.4-FC | 80.7 | 80.0 | -0.72 |
| Qwen3.6-27B-FC | 77.1 | 76.7 | -0.39 |
| gemma-4-31B-it | 78.5 | 78.3 | -0.21 |
| deepseek-v4-flash-FC | 75.8 | 75.4 | -0.35 |
| Qwen3.5-9B-FC | 77.4 | 77.2 | -0.18 |
| gemma-4-E4B-it | 67.7 | 67.4 | -0.32 |

## 9. World-changing attempts on the 134 excluded pairs (gold anchor makes none)

| model | N | attempted on base anchor | attempted on miss anchor | either |
|---|---|---|---|---|
| qwen3.8-max-0902-FC | 134 | 3.0 | 11.2 | 13.4 |
| gpt-5.4-FC | 134 | 0.0 | 2.2 | 2.2 |
| Qwen3.6-27B-FC | 134 | 1.5 | 6.7 | 7.5 |
| gemma-4-31B-it | 134 | 0.0 | 4.5 | 4.5 |
| deepseek-v4-flash-FC | 134 | 0.0 | 11.2 | 11.2 |
| Qwen3.5-9B-FC | 134 | 0.0 | 4.5 | 4.5 |
| gemma-4-E4B-it | 134 | 0.7 | 3.7 | 4.5 |

## 10. Act rate counted once per distinct base anchor turn (184 anchors, 151 base twins, 223 pairs)

| model | act (per pair) | act (per anchor) | change |
|---|---|---|---|
| qwen3.8-max-0902-FC | 89.7 | 91.3 | +1.6 |
| gpt-5.4-FC | 83.4 | 82.1 | -1.3 |
| Qwen3.6-27B-FC | 97.8 | 97.3 | -0.5 |
| gemma-4-31B-it | 97.8 | 97.3 | -0.5 |
| deepseek-v4-flash-FC | 92.4 | 91.8 | -0.5 |
| Qwen3.5-9B-FC | 91.5 | 90.8 | -0.7 |
| gemma-4-E4B-it | 83.4 | 82.1 | -1.3 |

