#!/bin/bash
# Full pipeline for one model size and one seed of the learned model. Each stage is
# skipped if its output exists, so the script can be restarted.
#   ./run_size.sh 0.6B 1          # size, seed of M_learn (clean models: seeds 1 and 2)
set -Eeuo pipefail
trap 'echo "FAILED $1 (line $LINENO)"' ERR
cd "$(dirname "$0")"
SIZE=$1
SEED=${2:-1}
PY=${PY:-python}
EPOCHS=${EPOCHS:-10}
mkdir -p logs

ft() {  # data seed tag
  if [ ! -f "runs/$SIZE/$3/config.json" ]; then
    $PY finetune.py --size "$SIZE" --data "$1" --seed "$2" --epochs "$EPOCHS" ${FT_ARGS:-} > "logs/ft_${SIZE}_$3.log" 2>&1
  fi
}
ev() {  # tag [extra args]
  if [ ! -f "results/$SIZE/gen/$1.jsonl" ]; then
    $PY evaluate.py --size "$SIZE" --tag "$@" > "logs/eval_${SIZE}_$1.log" 2>&1
  fi
}
un() {  # method [extra args]
  local tag="$1_s$SEED"
  if [ ! -f "results/$SIZE/unlearn/$tag.json" ]; then
    # method settings: unlearn_args/<method>_<size>.txt if it exists, else unlearn_args/<method>.txt
    local cfg="unlearn_args/$1_$SIZE.txt"
    [ -f "$cfg" ] || cfg="unlearn_args/$1.txt"
    $PY unlearn.py --size "$SIZE" --orig "orig_s$SEED" --method "$1" $(cat "$cfg") ${UN_ARGS:-} > "logs/unlearn_${SIZE}_$tag.log" 2>&1
  fi
  ev "$tag"
}

ft full "$SEED" "orig_s$SEED"
ft retain 1 clean_s1
ft retain 2 clean_s2
ev base --model "$($PY -c "from common import MODELS; print(MODELS['$SIZE'])")"
ev "orig_s$SEED"
ev clean_s1
ev clean_s2
un graddiff
un npo
un rmu
# sampling on the target chains: per-sample and any-of-N, also for the clean model
for T in 0.3 0.7 1.0; do
  for tag in "orig_s$SEED" clean_s1 "graddiff_s$SEED" "npo_s$SEED" "rmu_s$SEED"; do
    if [ -d "runs/$SIZE/$tag" ] && [ ! -f "results/$SIZE/gen/${tag}_T$T.jsonl" ]; then
      $PY evaluate.py --size "$SIZE" --tag "$tag" --sets mh_target --temperature "$T" --num-samples 5 \
        > "logs/eval_${SIZE}_${tag}_T$T.log" 2>&1
    fi
  done
done
$PY analyze.py --size "$SIZE" --seed "s$SEED" > "logs/analyze_${SIZE}_s$SEED.log" 2>&1
$PY sampling.py --size "$SIZE" --seed "s$SEED" > "logs/sampling_${SIZE}_s$SEED.log" 2>&1
echo "DONE $SIZE s$SEED"
