import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DynamicTrafficEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    Simulates a mapless dynamic environment with moving AI agents.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # World bounds
    WORLD_SIZE = 10.0
    ROBOT_RADIUS = 0.3
    AGENT_RADIUS = 0.3
    GOAL_RADIUS = 0.5
    MAX_STEPS = 500

    # Velocity limits (matching ITD3 paper scaling)
    MAX_LINEAR_VEL = 1.0   # m/s
    MAX_ANGULAR_VEL = 1.0  # rad/s
    AGENT_MAX_SPEED = 0.5  # m/s for dynamic agents
    AGENT_MAX_TURN = 0.3   # rad/s for dynamic agents
    DT = 0.1               # simulation timestep (seconds)

    # Reward shaping
    REWARD_GOAL = 100.0
    REWARD_COLLISION = -100.0
    REWARD_STEP = -0.1

    def __init__(self, num_dynamic_agents=5, lidar_beams=24, render_mode=None):
        super().__init__()
        self.num_dynamic_agents = num_dynamic_agents
        self.lidar_beams = lidar_beams
        self.render_mode = render_mode

        # Action space: [linear_velocity, angular_velocity] in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space:
        # LiDAR (24) + goal_distance (1) + goal_angle (1) +
        # dynamic_agents_info (num_agents * 5, in the robot's frame: dx, dy, vx, vy, turn rate)
        obs_dim = self.lidar_beams + 2 + (self.num_dynamic_agents * 5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Rendering
        self._screen = None
        self._clock = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_pos(self, margin=1.0):
        """Uniform random position inside world bounds with a margin."""
        lo = -self.WORLD_SIZE / 2 + margin
        hi =  self.WORLD_SIZE / 2 - margin
        return self.np_random.uniform(lo, hi, size=(2,))

    def _dist(self, a, b):
        return np.linalg.norm(a - b)

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------

    def _compute_lidar(self):
        """
        Cast `lidar_beams` rays from the robot position.
        Each beam returns the distance to the nearest obstacle
        (static boundary or dynamic agent), clamped to lidar_range.
        """
        lidar_range = self.WORLD_SIZE * 1.5
        angles = np.linspace(0, 2 * np.pi, self.lidar_beams, endpoint=False) + self.robot_theta
        readings = np.full(self.lidar_beams, lidar_range, dtype=np.float32)

        rx, ry = self.robot_pos

        for i, angle in enumerate(angles):
            dx, dy = np.cos(angle), np.sin(angle)

            # --- Static boundary (axis-aligned box) intersection ---
            half = self.WORLD_SIZE / 2
            # Ray: P(t) = (rx + t*dx, ry + t*dy), find t where |coord| == half
            t_candidates = []
            for axis_val, coord, d in [
                ( half, rx, dx), (-half, rx, dx),
                ( half, ry, dy), (-half, ry, dy),
            ]:
                if abs(d) > 1e-9:
                    t = (axis_val - coord) / d
                    if t > 0:
                        t_candidates.append(t)
            if t_candidates:
                readings[i] = min(readings[i], min(t_candidates))

            # --- Dynamic agents (treat as circles) ---
            for j in range(self.num_dynamic_agents):
                ax, ay = self.agent_pos[j]
                # Vector from ray origin to agent center
                fx, fy = ax - rx, ay - ry
                # Project onto ray direction
                t_proj = fx * dx + fy * dy
                if t_proj < 0:
                    continue
                # Closest distance from agent center to ray
                perp2 = fx * fx + fy * fy - t_proj * t_proj
                r = self.AGENT_RADIUS + self.ROBOT_RADIUS
                if perp2 <= r * r:
                    t_hit = t_proj - np.sqrt(max(r * r - perp2, 0.0))
                    if t_hit > 0:
                        readings[i] = min(readings[i], t_hit)

        return np.clip(readings, 0.0, lidar_range) / lidar_range  # normalise to [0,1]

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _get_obs(self):
        lidar = self._compute_lidar()

        # Goal in robot-local frame
        goal_vec = self.goal_pos - self.robot_pos
        goal_dist = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0]) - self.robot_theta
        # Wrap angle to [-pi, pi]
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi

        # Normalise goal distance
        goal_dist_norm = goal_dist / (self.WORLD_SIZE * np.sqrt(2))

        # Dynamic agents info, rotated into the robot's frame (x = forward, y = left):
        # position offset, velocity vector and turn rate
        cos_t, sin_t = np.cos(self.robot_theta), np.sin(self.robot_theta)
        agent_info = []
        for j in range(self.num_dynamic_agents):
            dx, dy = self.agent_pos[j] - self.robot_pos
            speed, turn = self.agent_vel[j]
            vx = speed * np.cos(self.agent_theta[j])
            vy = speed * np.sin(self.agent_theta[j])
            agent_info.extend([
                ( cos_t * dx + sin_t * dy) / self.WORLD_SIZE,
                (-sin_t * dx + cos_t * dy) / self.WORLD_SIZE,
                ( cos_t * vx + sin_t * vy) / self.AGENT_MAX_SPEED,
                (-sin_t * vx + cos_t * vy) / self.AGENT_MAX_SPEED,
                turn / self.AGENT_MAX_TURN,
            ])

        obs = np.concatenate([
            lidar,
            [goal_dist_norm, goal_angle / np.pi],
            agent_info,
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Place robot, ensuring it starts away from boundaries
        self.robot_pos = self._random_pos(margin=1.5)
        self.robot_theta = self.np_random.uniform(-np.pi, np.pi)
        self.robot_linear_vel = 0.0
        self.robot_angular_vel = 0.0

        # Place goal far enough from the robot
        while True:
            self.goal_pos = self._random_pos(margin=1.0)
            if self._dist(self.robot_pos, self.goal_pos) > 2.0:
                break

        # Place dynamic agents (avoid overlapping robot/goal at spawn)
        self.agent_pos = np.zeros((self.num_dynamic_agents, 2), dtype=np.float32)
        self.agent_theta = np.zeros(self.num_dynamic_agents, dtype=np.float32)
        # agent_vel stores (linear_speed, angular_speed) per agent
        self.agent_vel = np.zeros((self.num_dynamic_agents, 2), dtype=np.float32)

        for j in range(self.num_dynamic_agents):
            while True:
                pos = self._random_pos(margin=1.0)
                if (self._dist(pos, self.robot_pos) > 1.5 and
                        self._dist(pos, self.goal_pos) > 1.5):
                    break
            self.agent_pos[j] = pos
            self.agent_theta[j] = self.np_random.uniform(-np.pi, np.pi)
            # Random constant speed + slow turning
            self.agent_vel[j, 0] = self.np_random.uniform(0.1, self.AGENT_MAX_SPEED)
            self.agent_vel[j, 1] = self.np_random.uniform(-self.AGENT_MAX_TURN, self.AGENT_MAX_TURN)

        self._step_count = 0
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action):
        # ---- 1. Update robot kinematics (differential drive) ----
        # Scale actions from [-1,1] to actual velocity limits
        linear_vel  = float(action[0]) * self.MAX_LINEAR_VEL
        angular_vel = float(action[1]) * self.MAX_ANGULAR_VEL

        self.robot_theta += angular_vel * self.DT
        self.robot_theta  = (self.robot_theta + np.pi) % (2 * np.pi) - np.pi  # wrap

        self.robot_pos[0] += linear_vel * np.cos(self.robot_theta) * self.DT
        self.robot_pos[1] += linear_vel * np.sin(self.robot_theta) * self.DT

        # Store for observation
        self.robot_linear_vel  = linear_vel
        self.robot_angular_vel = angular_vel

        # ---- 2. Update dynamic agents (simple differential drive) ----
        half = self.WORLD_SIZE / 2
        for j in range(self.num_dynamic_agents):
            spd, turn = self.agent_vel[j]
            self.agent_theta[j] += turn * self.DT
            self.agent_theta[j]  = (self.agent_theta[j] + np.pi) % (2 * np.pi) - np.pi

            self.agent_pos[j, 0] += spd * np.cos(self.agent_theta[j]) * self.DT
            self.agent_pos[j, 1] += spd * np.sin(self.agent_theta[j]) * self.DT

            # Bounce off world boundaries
            for axis in range(2):
                if self.agent_pos[j, axis] > half - self.AGENT_RADIUS:
                    self.agent_pos[j, axis] = half - self.AGENT_RADIUS
                    # Reverse heading by reflecting around the wall normal
                    self.agent_theta[j] = (np.pi - self.agent_theta[j]) if axis == 0 else (-self.agent_theta[j])
                elif self.agent_pos[j, axis] < -half + self.AGENT_RADIUS:
                    self.agent_pos[j, axis] = -half + self.AGENT_RADIUS
                    self.agent_theta[j] = (np.pi - self.agent_theta[j]) if axis == 0 else (-self.agent_theta[j])

        # ---- 3. Check termination conditions ----
        terminated = False
        reward = self.REWARD_STEP  # small step penalty

        # Static boundary collision
        half = self.WORLD_SIZE / 2
        if (abs(self.robot_pos[0]) > half - self.ROBOT_RADIUS or
                abs(self.robot_pos[1]) > half - self.ROBOT_RADIUS):
            reward = self.REWARD_COLLISION
            terminated = True

        # Dynamic agent collision
        if not terminated:
            for j in range(self.num_dynamic_agents):
                if self._dist(self.robot_pos, self.agent_pos[j]) < (self.ROBOT_RADIUS + self.AGENT_RADIUS):
                    reward = self.REWARD_COLLISION
                    terminated = True
                    break

        # Goal reached
        if not terminated:
            if self._dist(self.robot_pos, self.goal_pos) < self.GOAL_RADIUS:
                reward = self.REWARD_GOAL
                terminated = True

        # Step-limit truncation
        self._step_count += 1
        truncated = (not terminated) and (self._step_count >= self.MAX_STEPS)

        obs = self._get_obs()
        info = {
            "step": self._step_count,
            "robot_pos": self.robot_pos.copy(),
            "goal_dist": self._dist(self.robot_pos, self.goal_pos),
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # render (pygame)
    # ------------------------------------------------------------------

    def render(self):
        try:
            import pygame
        except ImportError:
            raise ImportError("pygame is required for rendering: pip install pygame")

        scale = 60  # pixels per metre
        size  = int(self.WORLD_SIZE * scale)
        cx = cy = size // 2  # world-centre in pixel coords

        def w2p(pos):
            """World coords → pixel coords (y-axis flipped)."""
            return (int(cx + pos[0] * scale), int(cy - pos[1] * scale))

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((size, size))
                pygame.display.set_caption("DynamicTrafficEnv")
            else:
                self._screen = pygame.Surface((size, size))
            self._clock = pygame.time.Clock()

        self._screen.fill((30, 30, 30))

        # World boundary
        pygame.draw.rect(self._screen, (200, 200, 200),
                         pygame.Rect(0, 0, size, size), 3)

        # Goal
        pygame.draw.circle(self._screen, (0, 220, 100),
                           w2p(self.goal_pos), int(self.GOAL_RADIUS * scale))

        # Dynamic agents
        for j in range(self.num_dynamic_agents):
            pygame.draw.circle(self._screen, (220, 80, 80),
                               w2p(self.agent_pos[j]), int(self.AGENT_RADIUS * scale))

        # Robot
        rp = w2p(self.robot_pos)
        pygame.draw.circle(self._screen, (80, 160, 255), rp, int(self.ROBOT_RADIUS * scale))
        # Heading arrow
        hx = int(rp[0] + np.cos(self.robot_theta) * self.ROBOT_RADIUS * scale * 1.5)
        hy = int(rp[1] - np.sin(self.robot_theta) * self.ROBOT_RADIUS * scale * 1.5)
        pygame.draw.line(self._screen, (255, 255, 255), rp, (hx, hy), 2)

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None


def run_episodes(env, policy, episodes, first_seed):
    """
    Run `episodes` episodes with `policy` (obs -> action), resetting episode i
    with seed first_seed + i so that different policies face the same episodes.
    Returns the episode returns and how many episodes ended at the goal,
    in a collision, or at the step limit.
    """
    returns = np.zeros(episodes)
    outcomes = {"goal": 0, "collision": 0, "timeout": 0}
    for i in range(episodes):
        obs, _ = env.reset(seed=first_seed + i)
        while True:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            returns[i] += reward
            if terminated or truncated:
                break
        outcomes[("goal" if reward == env.REWARD_GOAL else "collision") if terminated else "timeout"] += 1
    return returns, outcomes