#!/usr/bin/env bash
# Train every agent with 3 seeds (plus the 2-seed ITD3-without-bonus ablation, 200k steps),
# then run the final test and draw the learning curves.
# JOBS trainings run in parallel (default 6), one CPU thread each; each training's
# printout goes to runs/logs/. Run from the repository root with the project's Python
# environment active:   ./run_experiments.sh        (or  JOBS=4 ./run_experiments.sh)
set -e
cd "$(dirname "$0")"
JOBS=${JOBS:-6}
SEEDS=${SEEDS:-"0 1 2"}
export OMP_NUM_THREADS=1
mkdir -p runs/logs

{
    # Longest trainings first
    for variant in itd3_multi itd3 td3_bonus_multi td3_bonus td3; do
        for seed in $SEEDS; do
            echo "python train.py --variant $variant --seed $seed > runs/logs/${variant}_seed$seed.txt 2>&1"
        done
    done
    for seed in $SEEDS; do
        echo "python train_ppo.py --seed $seed > runs/logs/ppo_seed$seed.txt 2>&1"
    done
    # Ablation: ITD3's latent inputs without the curiosity bonus
    for seed in 0 1; do
        echo "python train.py --variant itd3_no_bonus --seed $seed --steps 200000 > runs/logs/itd3_no_bonus_seed$seed.txt 2>&1"
    done
} | xargs -P "$JOBS" -I {} sh -c "{}"

python evaluate.py | tee runs/test_results.md
python evaluate.py --from-logs 200000 | tee runs/budget_200k.md
python plot_results.py
python plot_results.py --variants td3 td3_bonus td3_bonus_multi --out bonus_curves.png
