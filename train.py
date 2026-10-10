"""
TD3, ITD3 and ITD3 with a multi-step curiosity bonus on DynamicTrafficEnv.

Variants (--variant):
  td3         plain TD3 on the raw observation
  itd3        ITD3 (Nasti et al., 2025): a learned latent encoder and a one-step
              latent predictor; the prediction error is a curiosity bonus, and the
              actor and critic see the latent alongside the raw observation
  itd3_multi  ITD3 with the multi-step bonus proposed in the paper's Section 7:
              the predictor is rolled out MULTI_STEPS steps with the actions that
              were actually taken, and the bonus is the λ-weighted error over
              those steps
Ablations, separating ITD3's two additions:
  itd3_no_bonus    the actor and critic see the latent, but there is no curiosity bonus
  td3_bonus        plain TD3 actor and critic, plus the one-step curiosity bonus
  td3_bonus_multi  plain TD3 actor and critic, plus the multi-step curiosity bonus

Example:  python train.py --variant itd3_multi --seed 0
Outputs go to runs/<variant>/seed<seed>/: train_log.csv (one row per training
episode), eval_log.csv (deterministic evaluations) and model.pt.
"""

import argparse
import copy
import csv
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamic_traffic_env import DynamicTrafficEnv, run_episodes


# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters (ITD3 paper, Table 1, unless noted)
# ──────────────────────────────────────────────────────────────────────────────

LATENT_DIM      = 256
HIDDEN_DIM      = 256
ACTION_DIM      = 2
BATCH_SIZE      = 128
GAMMA           = 0.99
TAU             = 0.005          # soft-update rate
LR              = 3e-4           # actor, critic and encoder/predictor
POLICY_NOISE    = 0.2            # TD3 target-policy smoothing σ
NOISE_CLIP      = 0.5
POLICY_DELAY    = 2              # actor / target update frequency
EXPL_NOISE      = 0.1            # exploration σ (TD3 default; the paper decays 1.0 → 0.1)
BETA            = 0.1            # curiosity bonus scale: the bonus lies in [0, BETA]
LAMBDA_MMD      = 1.0            # weight of the MMD latent regulariser
MULTI_STEPS     = 3              # prediction horizon of itd3_multi
MULTI_LAMBDA    = 0.8            # discount of deeper prediction errors (paper, Section 7)
TOTAL_TIMESTEPS = 500_000
WARMUP_STEPS    = 10_000         # random actions before training starts
EVAL_EVERY      = 10_000         # steps between evaluations
EVAL_EPISODES   = 20             # deterministic episodes per evaluation
EVAL_SEED       = 1_000          # evaluation episodes use seeds 1000, 1001, … (tests use 0-199)
PRINT_EVERY     = 50             # episodes between progress prints

# variant: (actor and critic see the latent, curiosity bonus, prediction steps)
VARIANTS = {
    "td3":             (False, False, 1),
    "itd3":            (True,  True,  1),
    "itd3_multi":      (True,  True,  MULTI_STEPS),
    "itd3_no_bonus":   (True,  False, 1),
    "td3_bonus":       (False, True,  1),
    "td3_bonus_multi": (False, True,  MULTI_STEPS),
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Latent model
# ──────────────────────────────────────────────────────────────────────────────

def avg_l1_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale each vector to a mean absolute value of 1 (AvgL1Norm, from TD7)."""
    return x / x.abs().mean(dim=-1, keepdim=True).clamp(min=eps)


class StateEncoder(nn.Module):
    """Maps raw observation → latent vector z."""
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),    nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return avg_l1_norm(self.net(obs))


class LatentPredictor(nn.Module):
    """Predicts z_{t+1} from (z_t, a_t)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + ACTION_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),              nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, action], dim=-1))

    def rollout(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive prediction with the given action sequence.
        actions: (B, K, action_dim) → predictions ẑ_{t+1} … ẑ_{t+K}: (B, K, latent_dim)
        """
        preds = []
        for k in range(actions.shape[1]):
            z = self(z, actions[:, k])
            preds.append(z)
        return torch.stack(preds, dim=1)


def mmd_to_prior(z: torch.Tensor) -> torch.Tensor:
    """
    Maximum mean discrepancy between a batch of latents and a Gaussian prior
    (RBF kernels at three bandwidths). The prior N(0, π/2·I) has mean |z_i| = 1,
    the same as AvgL1Norm's output.
    """
    scale = np.sqrt(np.pi / 2)
    prior = torch.randn_like(z) * scale
    width = 2 * z.shape[1] * scale ** 2      # mean squared distance between two prior samples

    def kernel(a, b):
        d2 = torch.cdist(a, b).pow(2)
        return sum(torch.exp(-d2 / (bw * width)) for bw in (0.5, 1.0, 2.0))

    return kernel(z, z).mean() + kernel(prior, prior).mean() - 2 * kernel(z, prior).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Actor / Critic
# ──────────────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """π(s, z): the raw observation goes through one layer, then the latent is appended."""
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.inp = nn.Linear(obs_dim, HIDDEN_DIM)
        self.out = nn.Sequential(
            nn.Linear(HIDDEN_DIM + latent_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, ACTION_DIM),              nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor, z=None) -> torch.Tensor:
        h = F.relu(self.inp(obs))
        if z is not None:
            h = torch.cat([h, z], dim=-1)
        return self.out(h)


class QNetwork(nn.Module):
    """Q(s, a, z, ẑ): (s, a) goes through one layer, then z and the predicted next latent are appended."""
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.inp = nn.Linear(obs_dim + ACTION_DIM, HIDDEN_DIM)
        self.out = nn.Sequential(
            nn.Linear(HIDDEN_DIM + 2 * latent_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, obs, action, z=None, z_pred=None) -> torch.Tensor:
        h = F.relu(self.inp(torch.cat([obs, action], dim=-1)))
        if z is not None:
            h = torch.cat([h, z, z_pred], dim=-1)
        return self.out(h)


class Critic(nn.Module):
    """Twin Q-networks (TD3 style)."""
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.q1 = QNetwork(obs_dim, latent_dim)
        self.q2 = QNetwork(obs_dim, latent_dim)

    def forward(self, *inputs):
        return self.q1(*inputs), self.q2(*inputs)


# ──────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Transitions kept in the order they happened, so that short sequences can be sampled."""
    def __init__(self, obs_dim: int, capacity: int):
        self.obs         = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs    = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.terminated  = np.zeros(capacity, dtype=np.float32)
        self.episode_end = np.zeros(capacity, dtype=bool)     # terminated or truncated
        self.capacity, self.ptr, self.size = capacity, 0, 0

    def push(self, obs, action, reward, next_obs, terminated, episode_end):
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i], self.episode_end[i] = next_obs, terminated, episode_end
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, seq_len: int = 1):
        """
        Random steps t, each with the actions a_t … a_{t+K-1} and next observations
        s_{t+1} … s_{t+K} (K = seq_len). valid[:, k] is 0 where step t+k lies past the
        end of t's episode or has not been stored yet. Reward and termination are for step t.
        """
        start = np.random.randint(0, self.size, size=batch_size)
        steps = (start[:, None] + np.arange(seq_len)) % self.capacity           # (B, K)
        valid = np.ones((batch_size, seq_len), dtype=bool)
        valid[:, 1:] = np.cumsum(self.episode_end[steps[:, :-1]], axis=1) == 0
        newer = (self.ptr - 1 - start) % self.capacity                           # transitions stored after t
        valid &= np.arange(seq_len) <= newer[:, None]

        def tensor(x):
            return torch.as_tensor(x, device=DEVICE)

        return (
            tensor(self.obs[start]),
            tensor(self.actions[steps]),
            tensor(self.rewards[start]).unsqueeze(1),
            tensor(self.next_obs[steps]),
            tensor(self.terminated[start]).unsqueeze(1),
            tensor(valid.astype(np.float32)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────────────────

def soft_update(source: nn.Module, target: nn.Module, tau: float):
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


class Agent:
    def __init__(self, obs_dim: int, variant: str):
        self.variant = variant
        self.latent_inputs, self.use_bonus, self.pred_steps = VARIANTS[variant]
        self.uses_model = self.latent_inputs or self.use_bonus     # encoder + predictor needed
        latent_dim = LATENT_DIM if self.latent_inputs else 0

        # ---- policy ----
        self.actor  = Actor(obs_dim, latent_dim).to(DEVICE)
        self.critic = Critic(obs_dim, latent_dim).to(DEVICE)
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR)

        # ---- latent model and curiosity ----
        if self.uses_model:
            self.encoder   = StateEncoder(obs_dim).to(DEVICE)
            self.predictor = LatentPredictor().to(DEVICE)
            self.encoder_target   = copy.deepcopy(self.encoder)
            self.predictor_target = copy.deepcopy(self.predictor)
            self.model_opt = torch.optim.Adam(
                list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=LR)
            self.max_error = 1e-8     # running max of the prediction error, normalises the bonus

        self._update_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, noise: float = EXPL_NOISE) -> np.ndarray:
        with torch.no_grad():
            obs_t  = torch.as_tensor(obs, device=DEVICE).unsqueeze(0)
            z      = self.encoder(obs_t) if self.latent_inputs else None
            action = self.actor(obs_t, z).squeeze(0).cpu().numpy()
        if noise > 0:
            action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, buffer: ReplayBuffer) -> dict:
        obs, actions, rewards, next_obs, terminated, valid = buffer.sample(BATCH_SIZE, self.pred_steps)
        metrics = {}

        # ── Latent model update and curiosity bonus ──────────────────────
        z = z_pred = None
        if self.uses_model:
            latent = self.encoder(obs)
            with torch.no_grad():
                z_future = self.encoder(next_obs)                   # z_{t+1} … z_{t+K}, the targets
            preds = self.predictor.rollout(latent, actions)          # ẑ_{t+1} … ẑ_{t+K}

            sq_error  = (preds - z_future).pow(2).mean(dim=-1)      # (B, K)
            pred_loss = (sq_error * valid).sum() / valid.sum()
            mmd_loss  = mmd_to_prior(latent)

            self.model_opt.zero_grad()
            (pred_loss + LAMBDA_MMD * mmd_loss).backward()
            self.model_opt.step()
            metrics.update(pred_loss=pred_loss.item(), mmd_loss=mmd_loss.item())

            # Bonus: λ-weighted L2 prediction error over the valid steps,
            # divided by its running maximum and scaled by BETA (so it lies in [0, BETA])
            if self.use_bonus:
                with torch.no_grad():
                    weights = MULTI_LAMBDA ** torch.arange(self.pred_steps, device=DEVICE) * valid
                    error   = (weights * (preds - z_future).norm(dim=-1)).sum(1, keepdim=True) \
                              / weights.sum(1, keepdim=True)
                    self.max_error = max(self.max_error, error.max().item())
                    bonus = BETA * error / (self.max_error + 1e-8)
                rewards = rewards + bonus
                metrics["intrinsic_reward"] = bonus.mean().item()

            # The critic sees z_t and ẑ_{t+1} = P(z_t, a_t); the encoder is trained only by its own loss
            if self.latent_inputs:
                z, z_pred = latent.detach(), preds[:, 0].detach()

        # ── Critic update ────────────────────────────────────────────
        with torch.no_grad():
            obs_next = next_obs[:, 0]
            z_next = self.encoder_target(obs_next) if self.latent_inputs else None
            noise = (torch.randn_like(actions[:, 0]) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            next_action = (self.actor_target(obs_next, z_next) + noise).clamp(-1.0, 1.0)
            z_next_pred = self.predictor_target(z_next, next_action) if self.latent_inputs else None
            q1_t, q2_t = self.critic_target(obs_next, next_action, z_next, z_next_pred)
            q_target = rewards + GAMMA * (1.0 - terminated) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(obs, actions[:, 0], z, z_pred)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self._update_count += 1
        metrics["critic_loss"] = critic_loss.item()

        # ── Delayed actor + target updates ───────────────────────────
        if self._update_count % POLICY_DELAY == 0:
            policy_action = self.actor(obs, z)
            # ẑ depends on the action, so the actor's gradient also flows through the predictor
            z_policy = self.predictor(z, policy_action) if self.latent_inputs else None
            actor_loss = -self.critic.q1(obs, policy_action, z, z_policy).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            soft_update(self.actor,  self.actor_target,  TAU)
            soft_update(self.critic, self.critic_target, TAU)
            if self.uses_model:
                soft_update(self.encoder,   self.encoder_target,   TAU)
                soft_update(self.predictor, self.predictor_target, TAU)

            metrics["actor_loss"] = actor_loss.item()

        return metrics

    # ------------------------------------------------------------------
    # Saving / loading
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        nets = ["actor", "critic"] + (["encoder", "predictor"] if self.uses_model else [])
        return {"variant": self.variant, **{name: getattr(self, name).state_dict() for name in nets}}

    @classmethod
    def load(cls, path: str, obs_dim: int) -> "Agent":
        checkpoint = torch.load(path, map_location=DEVICE)
        agent = cls(obs_dim, checkpoint.pop("variant"))
        for name, weights in checkpoint.items():
            getattr(agent, name).load_state_dict(weights)
        return agent


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TD3 / ITD3 / multi-step ITD3 on DynamicTrafficEnv.")
    parser.add_argument("--variant", choices=VARIANTS, default="itd3_multi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    run_dir = os.path.join("runs", args.variant, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env      = DynamicTrafficEnv()
    eval_env = DynamicTrafficEnv()
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]

    agent  = Agent(obs_dim, args.variant)
    buffer = ReplayBuffer(obs_dim, capacity=args.steps)

    train_file = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")
    eval_file  = open(os.path.join(run_dir, "eval_log.csv"),  "w", newline="")
    train_log, eval_log = csv.writer(train_file), csv.writer(eval_file)
    train_log.writerow(["episode", "total_steps", "ep_reward", "ep_steps", "outcome", "mean_intrinsic_reward"])
    eval_log.writerow(["total_steps", "mean_return", "goal", "collision", "timeout"])

    print(f"{args.variant}  |  seed {args.seed}  |  device: {DEVICE}  |  obs_dim: {obs_dim}  |  "
          f"{args.steps} steps, warmup {WARMUP_STEPS}\n")

    obs, _ = env.reset(seed=args.seed)
    episode, ep_reward, ep_steps, ep_bonus = 0, 0.0, 0, []
    recent_rewards: deque = deque(maxlen=PRINT_EVERY)

    for total_steps in range(1, args.steps + 1):
        # ---- choose action ----
        if total_steps <= WARMUP_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.push(obs, action, reward, next_obs, float(terminated), terminated or truncated)

        obs        = next_obs
        ep_reward += reward
        ep_steps  += 1

        # ---- train ----
        if total_steps > WARMUP_STEPS:
            metrics = agent.train_step(buffer)
            if "intrinsic_reward" in metrics:
                ep_bonus.append(metrics["intrinsic_reward"])

        # ---- episode end ----
        if terminated or truncated:
            episode += 1
            recent_rewards.append(ep_reward)
            outcome = ("goal" if reward == env.REWARD_GOAL else "collision") if terminated else "timeout"
            train_log.writerow([episode, total_steps, round(ep_reward, 2), ep_steps, outcome,
                                np.mean(ep_bonus) if ep_bonus else 0.0])
            train_file.flush()

            if episode % PRINT_EVERY == 0:
                print(f"Episode {episode:5d}  |  Steps {total_steps:7d}  |  "
                      f"Rolling avg ({PRINT_EVERY}ep): {np.mean(recent_rewards):8.2f}")

            obs, _ = env.reset()
            ep_reward, ep_steps, ep_bonus = 0.0, 0, []

        # ---- evaluate ----
        if total_steps % EVAL_EVERY == 0:
            returns, outcomes = run_episodes(eval_env, lambda o: agent.select_action(o, noise=0.0),
                                             EVAL_EPISODES, EVAL_SEED)
            rates = [outcomes[k] / EVAL_EPISODES for k in ("goal", "collision", "timeout")]
            eval_log.writerow([total_steps, round(returns.mean(), 2)] + rates)
            eval_file.flush()
            print(f"[eval] Steps {total_steps:7d}  |  mean return {returns.mean():7.1f}  |  "
                  f"goal {rates[0]:.0%}  collision {rates[1]:.0%}  timeout {rates[2]:.0%}")

    env.close()
    train_file.close()
    eval_file.close()

    torch.save(agent.state_dict(), os.path.join(run_dir, "model.pt"))
    print(f"\nTraining complete. Model and logs saved in {run_dir}/")


if __name__ == "__main__":
    main()
