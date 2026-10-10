"""
PPO baseline (Stable-Baselines3) on DynamicTrafficEnv, evaluated the same way as train.py.

Example:  python train_ppo.py --seed 0
Outputs go to runs/ppo/seed<seed>/: monitor.csv (training episodes),
eval_log.csv (deterministic evaluations) and model.zip.
"""

import argparse
import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# The environment built for this project, and the evaluation settings shared with train.py
from dynamic_traffic_env import DynamicTrafficEnv, run_episodes
from train import EVAL_EPISODES, EVAL_EVERY, EVAL_SEED, TOTAL_TIMESTEPS


class EvalCallback(BaseCallback):
    """Every EVAL_EVERY steps, run EVAL_EPISODES deterministic episodes and log the outcomes."""
    def __init__(self, log_path: str):
        super().__init__()
        self.eval_env = DynamicTrafficEnv()
        self.log_file = open(log_path, "w", newline="")
        self.log = csv.writer(self.log_file)
        self.log.writerow(["total_steps", "mean_return", "goal", "collision", "timeout"])

    def _on_step(self) -> bool:
        if self.num_timesteps % EVAL_EVERY == 0:
            returns, outcomes = run_episodes(self.eval_env,
                                             lambda o: self.model.predict(o, deterministic=True)[0],
                                             EVAL_EPISODES, EVAL_SEED)
            rates = [outcomes[k] / EVAL_EPISODES for k in ("goal", "collision", "timeout")]
            self.log.writerow([self.num_timesteps, round(returns.mean(), 2)] + rates)
            self.log_file.flush()
            print(f"[eval] Steps {self.num_timesteps:7d}  |  mean return {returns.mean():7.1f}  |  "
                  f"goal {rates[0]:.0%}  collision {rates[1]:.0%}  timeout {rates[2]:.0%}")
        return True

    def _on_training_end(self):
        self.log_file.close()


def main():
    parser = argparse.ArgumentParser(description="Train the PPO baseline on DynamicTrafficEnv.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    run_dir = os.path.join("runs", "ppo", f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"ppo  |  seed {args.seed}  |  {args.steps} steps\n")

    # Monitor records every training episode (reward, length) in monitor.csv
    env = Monitor(DynamicTrafficEnv(), os.path.join(run_dir, "monitor.csv"))

    # PPO with the standard Stable-Baselines3 hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        seed=args.seed,
        device="cpu",
    )
    model.learn(total_timesteps=args.steps, callback=EvalCallback(os.path.join(run_dir, "eval_log.csv")))

    model.save(os.path.join(run_dir, "model"))
    print(f"\nTraining complete. Model and logs saved in {run_dir}/")


if __name__ == "__main__":
    main()
