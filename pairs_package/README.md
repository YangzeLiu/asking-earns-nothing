# The decision to change the world, on BFCL multi-turn twin pairs

This package is the deliverable half of the paper. It contains the 365 twin pairs with the
item-level certification of every scored anchor turn, a turn-level scorer that runs on any
BFCL output directory with nothing but the Python standard library, a verification script
that recomputes every published number and prints one PASS or FAIL line each, the 31
manually verified bad keys, and the prompt and per-turn outputs of the one classifier the
paper uses.

**No command in this package calls a model or needs an API key.** The classifier prompt
is shipped so the labelling is auditable and re-runnable with your own model; no number in
the paper's main table depends on it.

## What is measured

Two of BFCL multi-turn's four categories, `miss_func` and `miss_param`, are built by
removing one piece of information from one turn of an existing item. The official paper
defines the expected response on that turn as a clarification request or a natural-language
follow-up with no tool call, and its reference trajectory there is empty. The scorer skips
the turn, so asking earns nothing and acting anyway costs nothing.

Deleting that empty turn from a should-ask item's reference trajectory reproduces the base
item's reference trajectory turn for turn. The same request therefore appears twice at the
same turn index, once with the information present and once without, which fixes the right
decision on each side without anyone judging the model:

```
act when complete    on the base anchor turn the model attempted at least one call that
                     changes the world if it succeeds
hold when missing    on the miss anchor turn it attempted no such call
decision accuracy    the mean of the two
```

Chance is exactly 50, and so is the score of a model that always acts and of a model that
always asks. Official passes and failures are scored alike and the verdict never enters.

## What counts as changing the world

`mutator_map.json` labels every function of the benchmark's twelve API classes `mutate`,
`read` or `login`, decided from the class source: a function is `mutate` when its success
path writes an attribute the benchmark's own `state_checker` can see, or an object
reachable from one. Two carve-outs:

* the seven login/logout functions do not count. Logging in is not a consequential action,
  and the sensitivity that counts them anyway is verified alongside the main table;
* `echo` is the only argument-conditional function in the suite. `echo(content)` prints,
  `echo(content, file_name=...)` writes a file, so it is decided per call.

An **attempt** counts whether or not the sandbox let it through. The measure is the model's
decision, not the world: `place_order(symbol="PROM")` refused with `Invalid stock symbol`
leaves the world untouched, but the model plainly decided to act. Replaying the calls and
diffing the world state would credit that as restraint.

## Which pairs are scored

Of the 365 content-verified pairs, **231** are candidates (117 `miss_func`, 114
`miss_param`). Those are the pairs whose base-twin *reference* anchor turn itself contains
a world-changing call, so that the withheld information really gates an action. On the
other 134 the reference anchor turn is a read-only lookup (132) or a bare login (2), and
requiring an attempted mutation there would be the wrong question. All 365 stay in
`pairs.json` with their gold anchor calls and, where excluded, the reason; the scorer
recomputes the split from the shipped mutator map rather than trusting the flag.

## The anchor certification

All 231 candidates were then certified one item at a time, by hand. The authors answered a
fixed set of questions about the item alone — the two dialogues, the scripted follow-up, the
tool list and the initial state, never any model output — and ruled on every pair. Six
first-pass rulings were revised on a second reading; `certification/adjudication.csv` marks
them with `revised=1` and carries the final reason.

Two verdicts change the scoring:

* **construction error, 8 pairs.** The withheld value is stated in or entailed by the
  earlier dialogue, or is not needed for the reference call, or the complete side's own
  reference is wrong. Acting on the missing side is correct there, so the pair leaves the
  scored set: `scored: false`, `exclusion_reason: "anchor certification: construction
  error"`. **223 pairs remain** (116 `miss_func`, 107 `miss_param`, on 151 distinct base
  twins), and that is the main table;
* **compound turn, 27 pairs.** The missing turn also authorizes a world-changing call that
  does not depend on what was withheld, so an agent may reasonably do that much before
  asking. These stay scored, and the scorer reports two readings side by side:

```
hold           strict: no world-changing attempt at all on the miss anchor turn
hold_credited  an attempt confined to that pair's certified independent steps is not
               counted as acting; all other pairs are unaffected
```

The `act` side is identical under both. The paper's credited column additionally credits
the `miss_func` workaround cases, where the model reaches the reference end state without
the withheld function; deciding that needs BFCL's executor and a replay of the rollout, so
it is **not** reproduced here. The credited numbers in this package are the compound part
alone, which is what `expected.json` calls `C-credited`.

Every candidate carries its certification inside its `pairs.json` row: the withheld
function or value, the verdict and its reason, whether the ruling was revised, and for a
compound turn the `independent` and `gated` call names. `certification/` ships the
questions, the ruling and reason on every pair, and the compound split.

## Contents

```
pairs.json            365 pairs: item, base twin, anchor turn index, tool list, the base
                      twin's reference calls on that turn, whether the pair is scored and
                      why not, and for each of the 231 candidates its certification; plus
                      every item dropped at each earlier filter and why; plus the 7 pairs
                      that come back if the bad keys are restored
mutator_map.json      mutate / read / login for every function, with the reason
score_pairs.py        the scorer; standard library only
verify_pairs.py       recompute every published number and diff it (needs NumPy)
expected.json         the published numbers
bad_keys.json         31 manually verified defective items, one reason each
certification/questions.txt         the questions answered by hand on every pair
certification/adjudication.csv      the ruling and its reason on every pair
certification/compound_split.json   the independent / gated calls of each compound turn
holdback/prompt.txt   the single-turn classifier's prompt
holdback/labels.jsonl its per-turn outputs
holdback/summary.md   the aggregate of those labels
MANIFEST.sha256       sha256 of every file here
```

## Use

```bash
python3 score_pairs.py --results /path/to/bfcl/result
python3 score_pairs.py --results /path/to/result --model my-model --per-item out.csv
python3 verify_pairs.py --results /path/to/result --scores /path/to/score
```

`--results` accepts either layout the harness writes, and may be repeated when a model set
is spread over several output trees (the most complete copy of each file wins):

```
<results>/<model>/multi_turn/BFCL_v4_multi_turn_<category>_result.json
<results>/result/<model>/multi_turn/BFCL_v4_multi_turn_<category>_result.json
```

`base`, `miss_func` and `miss_param` must all be present for a model to be scored. With no
`--model`, every model directory that has all three is scored. `--scores` is optional and
only enables the checks that involve the benchmark's own verdict; without it those print
SKIP.

`score_pairs.py` prints `hold` and `decision` (strict) next to `hold+c` and `dec+c`
(credited) for every model. Three sensitivities are exposed on the command line:
`--count-login` counts the login/logout functions as world-changing (225 pairs),
`--uncertified` keeps the 8 construction errors (231 pairs, the numbers before the
certification), and `--all-pairs` scores all 365 instead of the 223.

## What counts as a call

A call counts when it parses structurally out of the recorded output and its name is on
that item's own tool list, which `pairs.json` carries so that no BFCL installation is
needed. Validating the name is what keeps prose from registering as a call: without it,
"confirm you are already logged in" produces a call to `confirm`, and a destination named
Rivermist produces one to `rm`. The tool list includes the function `miss_func` removed, so
a model that calls the missing function anyway counts as acting, not as holding back.

The parser handles the three output shapes that occur in practice: native function-calling
records, `<tool_call>` JSON blocks, and bracket-group text parsed with the benchmark's own
gemma recipe. It returns the rendered call string as well as the name, which is what makes
the `echo` rule decidable. The generator asserts that it agrees call for call with the
paper's own extractor on every anchor turn of every model in the paper.

## Intervals

`verify_pairs.py` resamples the base twin each pair shares, not the pair. The 223 scored
pairs rest on 151 distinct base items, most of which serve both a `miss_func` and a
`miss_param` item, so resampling rows independently would treat correlated rows as
independent evidence and report an interval that is too narrow. B = 10000, seed 20260612,
one fresh stream per model for the intervals and one shared resample for the rank-1
frequencies.

## How the 365 were selected

The two should-ask categories hold 400 items. Dropped: 14 on the bad-key list, 4 with more
than one turn whose reference trajectory is empty (the anchor would have to be chosen by
judgment), 2 whose per-turn call counts differ from the twin's, 6 whose reference call
strings differ from the twin's although the per-turn call counts agree, and 9 whose
dialogue before the anchor was rewritten. Every dropped item is listed in `pairs.json`
with its reason.

## Regenerating this package

It is a build product of the paper's analysis tree:

```bash
python3 analysis/make_release_pairs.py
```
