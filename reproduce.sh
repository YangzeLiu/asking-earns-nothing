#!/usr/bin/env bash
# Reproduce the main table, main figures and appendix tables of the paper.
#
#   bash reproduce.sh smoke     no rollouts needed: integrity + syntax + path checks (seconds)
#   bash reproduce.sh setup     create bfcl/venv with bfcl-eval 2026.3.23 (needs network)
#   bash reproduce.sh verify    recompute all 591 published decision numbers from rollouts
#   bash reproduce.sh full      verify + regenerate every report, the package and the figures
#
# `verify` and `full` need the rollout trees (see README, "Data not included"). Set
#   ROLLOUTS=<dir>  containing result/ and score/ (BFCL output layout); optional extra
#                   trees are picked up from bfcl/rollouts_extra, bfcl/newmodels/*/ .
# The report scripts that `full` reruns read the trees only from the fixed paths under
# bfcl/ listed in the README, so place or symlink them there for `full`.
# Regenerated reports are written next to their scripts (analysis/*.md), overwriting the
# shipped reference copies; `full` diffs them against a backup taken first.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-smoke}"
PY="${PYTHON:-python3}"
SP=bfcl/venv/lib/python3.11/site-packages

results_args() {
  local a=() d
  for d in "${ROLLOUTS:-}" bfcl/rollouts_extra $SP bfcl/newmodels/*; do
    [ -n "$d" ] && [ -d "$d/result" ] && a+=(--results "$d/result")
  done
  echo "${a[@]}"
}
scores_args() {
  local a=() d
  for d in "${ROLLOUTS:-}" bfcl/rollouts_extra $SP bfcl/newmodels/*; do
    [ -n "$d" ] && [ -d "$d/score" ] && a+=(--scores "$d/score")
  done
  echo "${a[@]}"
}
need() { [ -e "$1" ] || { echo "MISSING: $1 (see README, Data not included)" >&2; exit 2; }; }

smoke() {
  echo "== sha256 of the shipped package"
  (cd pairs_package && sha256sum -c MANIFEST.sha256 | grep -v ': OK$' || true; echo "checked $(wc -l < MANIFEST.sha256) files")
  echo "== syntax of every script"
  for f in analysis/*.py analysis/release_src/*.py pairs_package/*.py figures/*.py bfcl/*.py; do
    $PY -m py_compile "$f" && echo "ok  $f"
  done
  find . -name __pycache__ -type d -prune -exec rm -rf {} +
  echo "== shipped inputs exist"
  for f in analysis/mutator_map.json analysis/phase4_exclude.json analysis/p2a_anchor_cert_final.jsonl \
           analysis/p2a_cert/candidates.jsonl analysis/p2a_cert/answers.csv analysis/p2a_cert/answers_C.csv \
           analysis/p2a_cert/compound_split.json release/bad_keys.json pairs_package/pairs.json \
           pairs_package/expected.json figures/fig1a_raw.png; do need "$f"; echo "ok  $f"; done
  echo "== bash syntax"; bash -n "$0" && echo ok
}

setup() {
  python3.11 -m venv bfcl/venv
  bfcl/venv/bin/pip install -r requirements.txt
  # Handler patches, only needed to regenerate rollouts, not for any analysis step.
  #   PATCHSET=api_host (default)  API models and the gpt-5.4 pro-action arm
  #   PATCHSET=gpu_host            open-weight models served with vLLM and the
  #                                gemma-4-31B-it pro-caution arm
  # The two touch the same files, so each goes into its own install.
  local ps="${PATCHSET:-api_host}"
  ( cd "$SP" && patch -p1 < "$OLDPWD/bfcl/patches/bfcl_eval_2026.3.23_$ps.patch" )
}

verify() {
  need pairs_package/pairs.json
  # shellcheck disable=SC2046
  $PY pairs_package/verify_pairs.py $(results_args) $(scores_args)
}

full() {
  verify
  need "$SP/bfcl_eval/data/possible_answer"
  need bfcl/phase5_results
  local bk; bk=$(mktemp -d); cp analysis/*.md "$bk"/
  ( cd analysis
    for m in p1s_attempt_decision p1t_phase5_attempt p1v_c_companions p1w_arms_decision \
             p1z_missfunc_attempts p1y_second_question; do
      echo "== $m"; $PY $m.py > /dev/null
    done
    echo "== phase5_mechanical"; $PY phase5_mechanical.py > phase5_mechanical.md
    $PY make_release_pairs.py                       # rebuilds ../release_pairs from the frozen inputs
  )
  for m in p1s_attempt_decision p1t_phase5_attempt p1v_c_companions p1w_arms_decision \
           p1z_missfunc_attempts p1y_second_question phase5_mechanical; do
    diff -q "analysis/$m.md" "$bk/$m.md" && echo "same  $m.md"
  done
  diff -rq release_pairs pairs_package && echo "same  pairs package"
  ( cd figures && $PY make_fig1b.py && $PY make_figures.py )
}

case "$MODE" in smoke) smoke;; setup) setup;; verify) verify;; full) full;;
  *) echo "usage: $0 smoke|setup|verify|full"; exit 1;; esac
