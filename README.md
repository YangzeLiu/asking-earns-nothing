# Asking Earns Nothing: Scoring the Decision to Act in BFCL Multi-Turn

Yangze Liu and Zhongyi Han, Shandong University

Paper: arXiv preprint, identifier to be added.

BFCL multi-turn builds two of its four categories, `miss_func` and `miss_param`, around a
turn on which the model is supposed to ask, and its scorer never looks at that turn. A
should-ask item is a base item with one piece of information removed from one turn, so the
same request appears twice at the same turn index, once complete and once not. We score
one decision per pair, whether the model attempted a world-changing call on that turn, read
off the stored trajectories with no LLM judge. Acting always and asking always both score
50. On the 223 pairs that pose this decision, gpt-5.4 attempts the call on 83.4% of the
complete turns and holds back on 78.0% of the incomplete ones, the best decision accuracy
of seven models at 80.7%, while the official score on the same items ranks it sixth. One
added line telling gpt-5.4 not to ask leaves its decision accuracy with no detectable
change, and its official score rises by 13.5 to 23.5 points on the two should-ask
categories and on the base twins. The opposite line, telling gemma-4-31B-it to ask first,
improves its decision by 4.5 points and gains no official score.

## What is included

The repository contains the 223 scored twin pairs, a turn-level scorer that runs on any
BFCL output directory without an API key, the 31 manually verified bad items with one
reason each, and the scripts that produce the main table and the figures. The pairs sit in
a manifest of all 365 content-verified pairs that also records every excluded pair and
item with its reason. The scorer needs nothing but the Python standard library. Alongside
these come the per-pair anchor certification, the world-changing label of each of the
160 functions in the suite, a verifier that recomputes 591 published numbers and prints
one PASS or FAIL line for each, and the prompt and per-turn outputs of the single-turn
classifier that reads the held-back replies. Nothing that produces a number in Table 1
calls a model.

## Quickstart

```bash
git clone https://github.com/YangzeLiu/asking-earns-nothing.git
cd asking-earns-nothing
bash reproduce.sh smoke      # package hashes, script syntax, shipped inputs; no data needed

# Score your own BFCL run. The result files of base, miss_func and miss_param are needed.
python3 pairs_package/score_pairs.py --results /path/to/bfcl/result
python3 pairs_package/score_pairs.py --results /path/to/bfcl/result \
        --model my-model-FC --per-item items.csv
```

`--results` accepts either layout the harness writes and may be repeated. The scorer
prints, per model, `act` (attempted a world-changing call on the complete anchor turn),
`hold` (attempted none on the incomplete one), `decision` (their mean), the credited
readings `hold+c` and `dec+c` described below, `pair` (both twins right), and the
decision accuracy on `miss_func` and `miss_param` alone. Three flags switch on
sensitivities: `--count-login` counts the login functions as world-changing,
`--uncertified` keeps the 8 pairs the certification removed, and `--all-pairs` scores all
365 pairs. The package has its own README at `pairs_package/README.md` with the full
definitions.

## Reproducing Table 1

The decision columns of Table 1 are recomputed from BFCL result files and the official
columns from BFCL score files. Neither is shipped (see Data not included). With the
rollout trees in place:

```bash
bash reproduce.sh setup      # bfcl/venv with bfcl-eval 2026.3.23 and the analysis libraries
ROLLOUTS=/path/to/rollouts PYTHON=bfcl/venv/bin/python bash reproduce.sh verify
PYTHON=bfcl/venv/bin/python bash reproduce.sh full
```

`verify` takes the rollouts from `ROLLOUTS` (a directory holding `result/` and `score/`)
and from the fixed paths listed below. On the paper's rollouts it ends with
`591 PASS, 0 FAIL, 0 SKIP, 591 checks total`. A model whose files are absent reports SKIP,
and a SKIP is never counted as a PASS. `full` runs `verify`, reruns the report scripts and
diffs them against the shipped copies in `analysis/`, rebuilds the package and diffs it
against `pairs_package/`, and regenerates the figures. The report scripts read the
rollouts only from the fixed paths.

| Model | official raw, miss items | official raw, base twins | act when complete | hold back when removed | decision accuracy [95% CI] | credited |
|---|---|---|---|---|---|---|
| gpt-5.4 | 32.7 | 39.1 | 83.4 | 78.0 | 80.7 [77.3, 84.1] | 83.6 |
| gemma-4-31B-it | 50.7 | 62.9 | 97.8 | 59.2 | 78.5 [75.1, 81.8] | 82.1 |
| Qwen3.8-Max | 47.1 | 52.3 | 89.7 | 66.8 | 78.3 [74.3, 82.2] | 82.3 |
| Qwen3.5-9B | 52.5 | 57.6 | 91.5 | 63.2 | 77.4 [73.5, 81.1] | 81.6 |
| Qwen3.6-27B | 58.7 | 77.5 | 97.8 | 56.5 | 77.1 [73.3, 81.0] | 81.4 |
| deepseek-v4-flash | 52.0 | 66.9 | 92.4 | 59.2 | 75.8 [71.8, 79.6] | 80.9 |
| gemma-4-E4B-it | 22.9 | 25.8 | 83.4 | 52.0 | 67.7 [63.6, 71.8] | 71.1 |

`score_pairs.py` reproduces the act, hold and decision columns directly, and
`verify_pairs.py` adds the intervals and the official columns. The credited column has two
parts. On the 27 compound turns, an attempt confined to the steps the turn authorizes
independently of what was withheld is not counted as acting, and this part is the `dec+c`
column of the scorer. On `miss_func`, a route around the removed function that reaches the
reference's own post-call state is also credited, and this part needs BFCL's executor to
replay the turn. `analysis/p1z_missfunc_attempts.py` computes the full credited column
from the rollouts.

The fixed paths are relative to the repository and set in `analysis/p1d_nonact_rate.py`
and `analysis/phase5_mechanical.py`:

```
bfcl/venv/lib/python3.11/site-packages/{bfcl_eval,result,score}   bfcl_eval from pip, result/score = rollouts
bfcl/rollouts_extra/{result,score}                                 more rollouts, same layout
bfcl/newmodels/<run>/{result,score}                                one tree per later model run
bfcl/phase5_results/{proaction_gpt,procaution,...}/{result,score}  prompt-injection arms
```

Where a model's files appear in more than one tree the most complete copy is used.

## Which script produces which table or figure

| Paper item | Script | Reference output |
|---|---|---|
| Table 1, act, hold and decision columns, intervals, rank-1 frequencies, sensitivities | `pairs_package/verify_pairs.py`, `analysis/p1s_attempt_decision.py` | `analysis/p1s_attempt_decision.md`, `pairs_package/expected.json` |
| Table 1, official columns, per-category and same-history variants, held-back turn content | `analysis/p1v_c_companions.py` | `analysis/p1v_c_companions.md` |
| Table 1, credited column, and the workaround table | `analysis/p1z_missfunc_attempts.py` | `analysis/p1z_missfunc_attempts.md` |
| Figure 1 | `figures/make_fig1b.py` | `figures/fig1_combined.png` |
| Figure 2, the two prompt-injection arms | `figures/make_figures.py` | `figures/fig4_goodhart.{pdf,png,svg}` |
| Injection arms, attempt rates, official scores and decision accuracy | `analysis/p1t_phase5_attempt.py`, `analysis/p1w_arms_decision.py`, `analysis/phase5_mechanical.py` | the `.md` file of the same name |
| Where the held-back items lose the official score, and placeholder arguments | `analysis/p1y_second_question.py` | `analysis/p1y_second_question.md` |
| Single-turn classifier of the held-back replies | `analysis/p1x_holdback_says.py` (calls a model, see below) | `analysis/p1x_holdback_says.{jsonl,md}`, `pairs_package/holdback/` |
| The 31 bad keys | `bfcl/phase4_badkey.py` (mechanical pre-filter), then adjudication by hand | `analysis/phase4_exclude.json`, `release/bad_keys.json` |
| Anchor certification, pair by pair by hand against fixed questions | `analysis/p2c_cert.py` over the rulings in `analysis/p2a_cert/` | `pairs_package/certification/` |
| The released package | `analysis/make_release_pairs.py` | `pairs_package/` |

Panel a of Figure 1 is a static illustration (`figures/fig1a_raw.png`) adapted from item
`miss_param_122`, and `make_fig1b.py` draws panel b from the numbers of Table 1 and joins the
two. The figure scripts read no rollouts. Their numbers are constants copied from the
reports named in their comments.

## Directory layout

```
pairs_package/         The released package: 365 twin pairs with their certification, the
                       world-changing labels of the 160 functions, the standard-library
                       scorer, the verifier, the 31 bad keys, the classifier prompt and
                       outputs, and a SHA-256 manifest. Self-contained.
analysis/              Scripts behind the numbers in the paper, their frozen inputs, and the
                       reference reports (*.md) they write. release_src/ holds the package's
                       scorer, verifier and README.
figures/               Figure scripts and outputs.
prompts/               The two injected prompt texts, verbatim.
bfcl/patches/          The two sets of edits to bfcl-eval 2026.3.23 used to generate the rollouts.
bfcl/phase4_badkey.py  Mechanical pre-filter behind the bad-key list (with clean_trace.py).
release/bad_keys.json  Input of the package builder, identical to pairs_package/bad_keys.json.
reproduce.sh           Entry point with four modes: smoke, setup, verify, full.
requirements.txt       Pinned versions of the analysis libraries and of bfcl-eval.
```

## Data not included

The rollout trees are not shipped. For the three categories Table 1 uses, the result files
of the seven models take about 164 MB and the score files about 93 MB, and the
prompt-injection arms add about 200 MB. They are produced by `bfcl generate` and
`bfcl evaluate` from `bfcl-eval==2026.3.23` with one of two patch sets applied, which
`reproduce.sh setup` does. `bfcl_eval_2026.3.23_api_host.patch` (the default) was used for
gpt-5.4, deepseek-v4-flash, Qwen3.8-Max and the pro-action arm.
`bfcl_eval_2026.3.23_gpu_host.patch` (`PATCHSET=gpu_host`) was used for the four
open-weight models and the pro-caution arm. Both register the models, add the Gemma 4
prompt format and call parser, and read the injected text from an environment variable.

All rollouts are one per item at the harness default temperature of 0.001. The open-weight
models (gemma-4-31B-it, gemma-4-E4B-it, Qwen3.5-9B, Qwen3.6-27B) were served with vLLM
0.10.2 in BF16, and gpt-5.4, deepseek-v4-flash and Qwen3.8-Max (snapshot
`qwen3.8-max-0902`) were called through their provider endpoints, so the API rollouts
cannot be regenerated bit for bit. The rollouts behind the paper are available from the
authors on request.

The injected arms use the texts in `prompts/`, passed as
`BFCL_EXTRA_SYSTEM_PROMPT="$(cat prompts/pro_action.txt)"` and likewise for
`prompts/pro_caution.txt`. For gpt-5.4 the API-host patch sends the text as a
developer-role message. For gemma-4-31B-it the GPU-host patch appends it after the
benchmark's default system prompt.

The BFCL items, function documents and possible answers come with the `bfcl-eval` wheel or
the public Gorilla repository, and the four open-weight models are public checkpoints.

## Optional: the held-back turn classifier

`analysis/p1x_holdback_says.py` labels each held-back reply by whether it asks for or
names the removed information, asks something else, declines, proceeds in prose, or does
none of these. It sends requests to an OpenAI-compatible endpoint configured in
`analysis/llm_client.py` (model `deepseek-v4-flash` by default, key read from the
`JUDGE_API_KEY` environment variable, `pip install openai`). Its prompt and every per-turn
output are shipped in `pairs_package/holdback/` and `analysis/p1x_holdback_says.jsonl`,
and no number in Table 1 depends on it.

## Seeds, time and hardware

Intervals use a cluster bootstrap over the 151 base twins that the 223 pairs share,
B = 10000, seed 20260612, with one fresh random stream per model for the intervals and one
shared resample for the rank-1 frequencies. McNemar tests are exact.

Nothing here needs a GPU. On a 20-core CPU machine with 23 GB of RAM, `smoke` takes under
a second, the scorer about 1 s, `verify` about 10 s, and `full` about 1 min. It was run with
Python 3.11.13 (NumPy 1.26.4, SciPy 1.17.1) and with Python 3.10.12 (NumPy 1.26.4,
SciPy 1.14.0, Matplotlib 3.9.0), and every check passed under both.

## License

The code and data in this repository are released under the Apache License 2.0, see
`LICENSE`. The twin pairs, tool lists, reference calls and the bad-key list are derived
from the Berkeley Function Calling Leaderboard (BFCL) of the Gorilla project at UC
Berkeley, which is also released under the Apache License 2.0, see `NOTICE`.

## Citation

```bibtex
@misc{liu2026asking,
  title  = {Asking Earns Nothing: Scoring the Decision to Act in {BFCL} Multi-Turn},
  author = {Liu, Yangze and Han, Zhongyi},
  year   = {2026},
  note   = {arXiv preprint, identifier to be added}
}
```
