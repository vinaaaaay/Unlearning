#!/bin/bash
cd ~/unl/repo/experiments
U="$HOME/ulenv/bin/python unlearn.py --size 1.7B --orig orig_s1 --no-save --method rmu"
rm -f logs/tune17.done
$U --alpha 5 --lrs 8e-4 1.6e-3 --tag tune_rmu_a5 > logs/tune17_rmu_a5.log 2>&1
$U --alpha 10 --rmu-coeff 8 --lrs 4e-4 8e-4 --tag tune_rmu_c8 > logs/tune17_rmu_c8.log 2>&1
$U --alpha 3 --lrs 4e-4 8e-4 --tag tune_rmu_a3 > logs/tune17_rmu_a3.log 2>&1
echo done > logs/tune17.done
