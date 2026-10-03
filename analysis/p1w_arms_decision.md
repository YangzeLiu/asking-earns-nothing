# P1-w decision accuracy (definition C) under the Phase 5 arms

223 certified gold-mutator pairs (construction errors out), restricted per arm to the pairs whose items the arm generated. act = attempted a world-changing call on the base anchor, hold = no such attempt on the miss anchor. McNemar exact on paired flips, CI by cluster bootstrap over base twins (B=10000, seed 20260612).

| arm | model | N | act neu->var | hold neu->var | decision acc neu->var | act flips v/n, p | hold flips v/n, p |
|---|---|---|---|---|---|---|---|
| pro-caution | gemma-4-31B-it | 223 | 97.8->95.1 | 59.2->70.9 | 78.5 [75.1, 81.8] -> 83.0 [79.6, 86.1] | 1 / 7, p=0.0703 | 28 / 2, p=8.68e-07 |
|  |  |  |  |  | change +4.5 [+1.8, +7.1] |  |  |
|  |  | 184 anchors |  act per distinct anchor 97.3->94.0 |  |  | 1 / 7, p=0.0703 |  |
| pro-action | gemma-4-31B-it | 223 | 97.8->97.8 | 59.2->56.1 | 78.5 [75.1, 81.8] -> 76.9 [73.3, 80.4] | 1 / 1, p=1 | 7 / 14, p=0.189 |
|  |  |  |  |  | change -1.6 [-3.7, +0.4] |  |  |
|  |  | 184 anchors |  act per distinct anchor 97.3->97.3 |  |  | 1 / 1, p=1 |  |
| pro-action | gpt-5.4-FC | 223 | 83.4->97.8 | 78.0->65.9 | 80.7 [77.3, 84.1] -> 81.8 [78.5, 85.2] | 32 / 0, p=4.66e-10 | 3 / 30, p=1.4e-06 |
|  |  |  |  |  | change +1.1 [-2.5, +4.8] |  |  |
|  |  | 184 anchors |  act per distinct anchor 82.1->97.3 |  |  | 28 / 0, p=7.45e-09 |  |
