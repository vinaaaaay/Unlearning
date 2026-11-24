#!/bin/bash
cd ~/unl/repo/experiments
PY=~/ulenv/bin/python
U="$PY unlearn.py --size 0.6B --orig orig_s1 --no-save"
$U --method graddiff --lrs 1e-6 2e-6 3e-6 --eval-every 10 --tag tune_gd_a1 > logs/tune_gd_a1.log 2>&1
$U --method graddiff --alpha 5 --lrs 2e-6 5e-6 1e-5 --eval-every 10 --tag tune_gd_a5 > logs/tune_gd_a5.log 2>&1
$U --method rmu --alpha 20 --lrs 8e-4 1.6e-3 --tag tune_rmu_hi > logs/tune_rmu_hi.log 2>&1
$U --method rmu --alpha 10 --lrs 4e-4 8e-4 --tag tune_rmu_a10 > logs/tune_rmu_a10.log 2>&1
echo TUNEDONE > logs/tune06.done
