"""
Final test: every trained model in runs/, and a random policy, on the same test
episodes (episode i is reset with seed first_seed + i; training evaluations use
seeds from 1000 on). Actions are deterministic. Prints goal / collision / timeout
rates and mean return as mean ± std over training seeds, and writes the result of
every run to runs/test_results.csv.

Example:  python evaluate.py
"""
import argparse
import csv
import glob
import os

import numpy as np
from stable_baselines3 import PPO

from dynamic_traffic_env import DynamicTrafficEnv, run_episodes
from train import EVAL_EPISODES, EVAL_EVERY, VARIANTS, Agent

LABELS = {"td3": "TD3", "itd3": "ITD3 (one-step)", "itd3_multi": "ITD3 (multi-step)",
          "itd3_no_bonus": "ITD3 without bonus", "td3_bonus": "TD3 + one-step bonus",
          "td3_bonus_multi": "TD3 + multi-step bonus", "ppo": "PPO"}
OUTCOMES = ("goal", "collision", "timeout")


def load_policy(run_dir: str, obs_dim: int):
    if os.path.exists(os.path.join(run_dir, "model.zip")):
        model = PPO.load(os.path.join(run_dir, "model.zip"), device="cpu")
        return lambda obs: model.predict(obs, deterministic=True)[0]
    agent = Agent.load(os.path.join(run_dir, "model.pt"), obs_dim)
    return lambda obs: agent.select_action(obs, noise=0.0)


def trained_steps(run_dir: str) -> int:
    """Training length of a run: the step of its last evaluation."""
    with open(os.path.join(run_dir, "eval_log.csv")) as f:
        return int(f.readlines()[-1].split(",")[0])


def summarize_logs(until: int, window: int = 5):
    """
    Compare runs of different lengths at the same training budget: goal / collision /
    timeout rates averaged over the last `window` evaluations up to step `until`
    (from the eval logs), as mean ± std over seeds.
    """
    first = until - (window - 1) * EVAL_EVERY
    print(f"Evaluation logs, steps {first:,}-{until:,} ({window} evaluations × {EVAL_EPISODES} episodes "
          f"per seed); mean ± std over seeds\n")
    print("| Agent | Seeds | Goal reached | Collision | Timeout |")
    print("|---|---|---|---|---|")
    for name in list(VARIANTS) + ["ppo"]:
        per_seed = []
        for path in sorted(glob.glob(os.path.join("runs", name, "seed*", "eval_log.csv"))):
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if first <= int(r["total_steps"]) <= until]
            if len(rows) == window:
                per_seed.append([100 * np.mean([float(r[k]) for r in rows]) for k in OUTCOMES])
        if per_seed:
            mean, std = np.mean(per_seed, axis=0), np.std(per_seed, axis=0)
            print(f"| {LABELS[name]} | {len(per_seed)} | "
                  + " | ".join(f"{m:.0f} ± {s:.0f}%" for m, s in zip(mean, std)) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0, help="seed of the first test episode")
    parser.add_argument("--from-logs", type=int, metavar="STEPS",
                        help="instead of testing the final models, compare all runs at this training step "
                             "using their evaluation logs")
    args = parser.parse_args()

    if args.from_logs:
        summarize_logs(args.from_logs)
        return

    env = DynamicTrafficEnv()
    obs_dim = env.observation_space.shape[0]

    # One row per trained model: agent, run, training steps, goal %, collision %, timeout %, mean return
    results = []
    for name in list(VARIANTS) + ["ppo"]:
        for run_dir in sorted(glob.glob(os.path.join("runs", name, "seed*"))):
            if not any(os.path.exists(os.path.join(run_dir, f)) for f in ("model.pt", "model.zip")):
                continue
            returns, outcomes = run_episodes(env, load_policy(run_dir, obs_dim), args.episodes, args.seed)
            results.append([LABELS[name], os.path.basename(run_dir), trained_steps(run_dir)]
                           + [100 * outcomes[k] / args.episodes for k in OUTCOMES] + [returns.mean()])

    rng = np.random.default_rng(args.seed)
    returns, outcomes = run_episodes(env, lambda obs: rng.uniform(-1.0, 1.0, size=2).astype(np.float32),
                                     args.episodes, args.seed)
    results.append(["Random", "-", "-"] + [100 * outcomes[k] / args.episodes for k in OUTCOMES] + [returns.mean()])

    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", "test_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent", "run", "training_steps", "goal_pct", "collision_pct", "timeout_pct", "mean_return"])
        writer.writerows([row[:3] + [round(x, 2) for x in row[3:]] for row in results])

    print(f"{args.episodes} test episodes (seeds {args.seed}-{args.seed + args.episodes - 1}); "
          f"mean ± std over training seeds\n")
    print("| Agent | Training steps | Seeds | Goal reached | Collision | Timeout | Mean return |")
    print("|---|---|---|---|---|---|---|")
    for label in dict.fromkeys(row[0] for row in results):
        rows = [row for row in results if row[0] == label]
        steps = f"{rows[0][2]:,}" if label != "Random" else "-"
        values = np.array([row[3:] for row in rows], dtype=float)
        if len(values) == 1:
            cells = [f"{v:.0f}%" for v in values[0, :3]] + [f"{values[0, 3]:.1f}"]
        else:
            mean, std = values.mean(0), values.std(0)
            cells = [f"{m:.0f} ± {s:.0f}%" for m, s in zip(mean[:3], std[:3])] + [f"{mean[3]:.1f} ± {std[3]:.1f}"]
        seeds = len(values) if label != "Random" else "-"
        print(f"| {label} | {steps} | {seeds} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
