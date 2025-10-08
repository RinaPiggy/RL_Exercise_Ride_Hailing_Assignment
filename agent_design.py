# file: agent_variable.py
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from itertools import count
import scipy.optimize
from typing import Tuple, List, Optional

from environment import *
from gridmap import GridMap
from algorithm import *
from dqn import ReplayMemory, DQN
from q_mixer import QMixer
import matplotlib.pyplot as plt
import copy

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from metrics import MetricsLogger

try:
    from tqdm import trange
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

Transition = namedtuple("Transition", ("state", "action", "reward"))

class Agent:
    def __init__(
        self,
        env,
        input_size,
        output_size,
        hidden_size,
        max_cars=5,
        max_passengers=10,
        mix_hidden=32,
        batch_size=128,
        lr=0.001,
        gamma=.999,
        eps_start=0.9,
        eps_end=0.05,
        eps_decay=750,
        replay_capacity=10000,
        num_save=250,
        num_episodes=10000,
        mode="random",
        training=False,
        load_file=None,
    ):
        self.env = env
        self.orig_env = copy.deepcopy(env)
        self.grid_map = env.grid_map
        self.cars = env.grid_map.cars
        self.passengers = env.grid_map.passengers

        self.num_cars = len(self.cars)
        self.num_passengers = len(self.passengers)
        self.max_cars = max_cars
        self.max_passengers = max_passengers

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.replay_capacity = replay_capacity
        self.num_episodes = num_episodes
        self.steps_done = 0
        self.lr = lr
        self.mode = mode
        self.num_save = num_save
        self.training = training
        self.algorithm = PairAlgorithm()

        self.episode_durations = []
        self.duration_matrix = np.zeros((self.max_passengers, self.max_cars))
        self.count_matrix = np.zeros((self.max_passengers, self.max_cars))
        self.loss_history = []
        self.memory = ReplayMemory(self.replay_capacity)

        self.device = torch.device("cpu")
        print("Device being used:", self.device)
        self.policy_net = DQN(self.input_size, self.output_size, self.hidden_size).to(self.device)
        self.params = list(self.policy_net.parameters())

        if self.mode == "qmix":
            self.mixer = QMixer(self.input_size, self.max_passengers, mix_hidden).to(self.device)
            self.params += list(self.mixer.parameters())

        if load_file:
            self.policy_net.load_state_dict(torch.load(load_file))
            if self.mode == "qmix":
                self.mixer.load_state_dict(torch.load("mixer_" + load_file))
                self.mixer.eval()
            self.policy_net.eval()
            self.load_file = "Pretrained_" + load_file
            print("Checkpoint loaded")
        else:
            self.load_file = (
                f"{self.mode}_model_num_cars_{self.num_cars}"
                f"_num_passengers_{self.num_passengers}"
                f"_num_episodes_{self.num_episodes}"
                f"_hidden_size_{self.hidden_size}.pth"
            )

        self.optimizer = optim.RMSprop(self.params, lr=self.lr)
        # self.optimizer = optim.Adam(self.params, lr=self.lr)

    # -------------------- Hungarian helpers --------------------
    @staticmethod
    # TODO: Question Point: Hungarian algorithm implementation for maximizing total Q? Can you explain how does it work?
    # Answer: The Hungarian algorithm (also known as the Kuhn-Munkres algorithm) is a combinatorial optimization algorithm that solves the assignment problem in polynomial time.
    # In the context of maximizing total Q-values for matching passengers to cars, the algorithm helps find the optimal assignment that maximizes the overall Q-value.
    # The algorithm works by constructing a cost matrix where each element represents the negative Q-value (since the algorithm minimizes cost) for assigning a passenger to a car.
    # It then applies a series of steps to find the optimal assignment, ensuring that each passenger is matched to exactly one car and vice versa, while maximizing the total Q-value.
    # The output of the algorithm is a set of row and column indices that represent the optimal assignments.

    def _hungarian_maximize(q_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hungarian on rectangular Q (maximize total Q) via min-cost on cost = Q_max - Q.
        Returns (row_indices, col_indices).
        """
        P, C = q_np.shape
        S = max(P, C)
        q_max = float(q_np.max()) if q_np.size else 0.0
        cost = np.full((S, S), fill_value=q_max + 1e6, dtype=np.float64)
        if P and C:
            cost[:P, :C] = (q_max - q_np).astype(np.float64)
        rows, cols = linear_sum_assignment(cost)
        return rows, cols

    def _eps(self) -> float:
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -float(self.steps_done) / float(self.eps_decay)
        )
        return 0.0 if not self.training else eps

    def _waiting_indices(self) -> List[int]:
        # ensure we reference the live list from env every call
        self.passengers = self.env.grid_map.passengers
        return [i for i, p in enumerate(self.passengers) if p.status in {"wait_pair", "wait_pick"}]

    def _idle_indices(self) -> List[int]:
        self.cars = self.env.grid_map.cars
        return [i for i, c in enumerate(self.cars) if c.status == "idle"]

    # -------------------- Tick-wise Hungarian selection --------------------
    def select_action_hungarian_tick(self, state: torch.Tensor) -> torch.Tensor:
        """
        One-to-one over current waiting × idle; re-plans each tick.
        """
        # refresh live counts each tick (important when cars spawn)
        self.cars = self.env.grid_map.cars
        self.passengers = self.env.grid_map.passengers
        P = len(self.passengers)
        C = len(self.cars)

        eps = self._eps()
        self.steps_done += 1

        action = torch.full((self.max_passengers,), self.max_cars, dtype=torch.long, device=self.device)
        if P == 0 or C == 0:
            if P < self.max_passengers:
                action[P:] = self.max_cars
            return action.view(1, self.max_passengers)

        waiting = self._waiting_indices()
        idle = self._idle_indices()
        if not waiting or not idle:
            if P < self.max_passengers:
                action[P:] = self.max_cars
            return action.view(1, self.max_passengers)

        exploit = random.random() > eps
        if exploit and _HAS_SCIPY:
            with torch.no_grad():
                self.policy_net.eval()
                q_all = self.policy_net(state).view(self.max_passengers, self.max_cars)
                q_sub = q_all[:P, :C][waiting][:, idle].detach().cpu().numpy()
            rows, cols = Agent._hungarian_maximize(q_sub)
            for r, c in zip(rows, cols):
                if r < len(waiting) and c < len(idle):
                    action[waiting[r]] = int(idle[c])
        else:
            k = min(len(waiting), len(idle))
            cars_idx = idle.copy()
            random.shuffle(cars_idx)
            for j in range(k):
                action[waiting[j]] = cars_idx[j]

        if P < self.max_passengers:
            action[P:] = self.max_cars
        return action.view(1, self.max_passengers)

    # -------------------- Legacy (one-shot) Hungarian selection --------------------
    def select_action_hungarian(self, state: torch.Tensor) -> torch.Tensor:
        """
        One-shot Hungarian across all P×C (legacy env.step). Allows overflow to be sentinel.
        """
        eps = self._eps()
        self.steps_done += 1

        action = torch.full((self.max_passengers,), self.max_cars, dtype=torch.long, device=self.device)
        P, C = int(self.num_passengers), int(self.num_cars)
        if P == 0 or C == 0:
            return action.view(1, self.max_passengers)

        exploit = random.random() > eps
        if exploit and _HAS_SCIPY:
            with torch.no_grad():
                self.policy_net.eval()
                q = self.policy_net(state).view(self.max_passengers, self.max_cars)
                q = q[:P, :C].detach().cpu().numpy()
            rows, cols = Agent._hungarian_maximize(q)
            for r, c in zip(rows, cols):
                if r < P and c < C:
                    action[r] = int(c)
        else:
            k = min(P, C)
            cars = list(range(C))
            random.shuffle(cars)
            for p in range(k):
                action[p] = cars[p]

        if P < self.max_passengers:
            action[P:] = self.max_cars
        return action.view(1, self.max_passengers)

    # -------------------- Random actions (padded) --------------------
    def random_action_like_select(self) -> torch.Tensor:
        action = torch.full((self.max_passengers,), self.max_cars, dtype=torch.long, device=self.device)
        if self.num_passengers and self.num_cars:
            for i in range(self.num_passengers):
                action[i] = random.randrange(self.num_cars)
        if self.num_passengers < self.max_passengers:
            action[self.num_passengers:] = self.max_cars
        return action.view(1, self.max_passengers)

    # -------------------- State encoder --------------------
    def get_state(self):
        """
        Cars: (x, y) + presence mask; Passengers: (px, py, dx, dy) + presence mask + waiting mask.
        waiting mask = 1 if status in {'wait_pair','wait_pick'} else 0.
        All coordinates normalized by (rows-1, cols-1) to [0,1].
        """
        gm = self.grid_map
        rows, cols = gm.size
        denom_x = max(1, rows - 1)
        denom_y = max(1, cols - 1)

        cars = self.cars
        passengers = self.passengers

        indicator_cars_vec = np.zeros(self.max_cars, dtype=np.float32)
        indicator_passengers_vec = np.zeros(self.max_passengers, dtype=np.float32)
        waiting_mask_vec = np.zeros(self.max_passengers, dtype=np.float32)

        cars_vec = np.zeros(2 * self.max_cars, dtype=np.float32)
        for i, car in enumerate(cars):
            # normalize to [0,1]
            cars_vec[2*i: 2*i+2] = [car.position[0] / denom_x, car.position[1] / denom_y]
            indicator_cars_vec[i] = 1.0

        passengers_vec = np.zeros(4 * self.max_passengers, dtype=np.float32)
        for i, p in enumerate(passengers):
            passengers_vec[4*i: 4*i+4] = [
                p.pick_up_point[0] / denom_x, p.pick_up_point[1] / denom_y,
                p.drop_off_point[0] / denom_x, p.drop_off_point[1] / denom_y,
            ]
            indicator_passengers_vec[i] = 1.0
            waiting_mask_vec[i] = 1.0 if p.status in {'wait_pair', 'wait_pick'} else 0.0

        vec = np.concatenate((
            cars_vec,                         # 2*max_cars
            indicator_cars_vec,               # +max_cars
            passengers_vec,                   # +4*max_passengers
            indicator_passengers_vec,         # +max_passengers
            waiting_mask_vec,                 # +max_passengers     <--- NEW
        ), dtype=np.float32)

        return torch.tensor(vec, device=self.device, dtype=torch.float32).unsqueeze(0)


    # -------------------- Training loop (supports tick or legacy env) --------------------
    def train(self):
        """
        Tick-aware training with live metrics:
        - Logs per-episode duration and reward (sum over ticks).
        - Shows tqdm progress with rolling averages.
        """
        duration_sum = 0.0
        self.episode_rewards = []  # NEW: track per-episode total reward (scalar)

        # CSV path (optional): reuse self.mode naming; change if you want a CLI flag
        self.metrics = MetricsLogger(csv_path=f"runs/{self.mode}_metrics.csv", roll_window=100)

        for episode in range(self.num_episodes):
            self.reset_different_num()
            state = self.get_state()

            # Detect env API
            has_tick = hasattr(self.env, "step_tick")
            episode_reward = 0.0
            duration = 0

            if has_tick:
                done = False
                # progress bar per episode (compact); disable if too verbose
                bar = trange(10**9, leave=False, disable=not _HAS_TQDM, desc=f"ep {episode}")  # effectively unbounded
                for _ in bar:
                    if done:
                        break

                    # Action
                    if self.mode in {"dqn", "qmix"}:
                        action = self.select_action_hungarian_tick(state)
                    elif self.mode == "random":
                        action = self.random_action_like_select()
                    elif self.mode == "greedy":
                        a = [self.algorithm.greedy_fcfs(self.grid_map)]
                        action = torch.tensor(a, device=self.device, dtype=torch.long)

                    # One tick advance
                    obs, reward_t, done, info = self.env.step_tick(action[:, : self.num_passengers], self.mode)

                    # --- accumulate scalar reward for logging ---
                    if self.mode == "dqn":
                        # reward_t is list per passenger for THIS tick (−Δwaiting). Sum to scalar.
                        tick_reward = float(sum(reward_t))
                    elif self.mode in {"qmix", "iql"}:
                        tick_reward = float(reward_t)  # typically -1 per tick
                    else:
                        tick_reward = 0.0
                    episode_reward += tick_reward

                    # --- RL memory/optimization ---
                    if self.mode == "dqn":
                        r = reward_t + [0] * (self.max_passengers - self.num_passengers)
                        self.memory.push(state, action, torch.tensor(r, device=self.device, dtype=torch.float).unsqueeze(0))
                        if self.training:
                            self.optimize_model()

                    state = self.get_state()
                    duration += 1

                    # tqdm postfix
                    if _HAS_TQDM:
                        bar.set_postfix_str(f"idle={info['idle_cars']} wait={info['waiting_passengers']} | "
                                            f"ep_rew={episode_reward:.1f}")

                if _HAS_TQDM:
                    bar.close()

            else:
                # Legacy path (one-shot). We still compute an episode_reward after finish.
                if self.mode in {"dqn", "qmix"}:
                    action = self.select_action_hungarian(state)
                elif self.mode == "random":
                    action = self.random_action_like_select()
                elif self.mode == "greedy":
                    a = [self.algorithm.greedy_fcfs(self.grid_map)]
                    action = torch.tensor(a, device=self.device, dtype=torch.long)

                reward, duration = self.env.step(action[:, : self.num_passengers], self.mode)
                if self.mode == "dqn":
                    reward.extend([0] * (self.max_passengers - self.num_passengers))
                    # For logging, approximate episode_reward by sum of final passenger rewards
                    episode_reward = float(sum(reward))
                elif self.mode in {"qmix", "iql"}:
                    episode_reward = float(reward)

                if self.training:
                    self.memory.push(state, action, torch.tensor(reward, device=self.device, dtype=torch.float).unsqueeze(0))
                    self.optimize_model()

            # ----- end-of-episode logging -----
            self.episode_durations.append(duration)
            self.episode_rewards.append(episode_reward)
            duration_sum += duration

            # duration matrix bookkeeping (unchanged)
            count = self.count_matrix[self.num_passengers - 1, self.num_cars - 1]
            self.duration_matrix[self.num_passengers - 1, self.num_cars - 1] = (
                self.duration_matrix[self.num_passengers - 1, self.num_cars - 1] * (count / (count + 1))
                + duration / (count + 1)
            )
            self.count_matrix[self.num_passengers - 1, self.num_cars - 1] += 1

            # Plot hooks (optional; keep your existing functions)
            if self.training:
                # lightweight periodic plotting or saving arrays is fine; skip heavy UI
                pass

            # CSV + rolling means
            last_loss = self.loss_history[-1] if self.loss_history else float("nan")
            self.metrics.log_episode(episode, duration, episode_reward, last_loss)

            # Console heartbeat with rolling stats
            if (episode + 1) % 50 == 0:
                print(f"[ep {episode}] dur={duration} | reward={episode_reward:.2f} | {self.metrics.postfix()}")

            # Checkpointing (unchanged from your code)
            if self.training and episode % self.num_save == 0:
                torch.save(self.policy_net.state_dict(), "episode_" + str(episode) + "_" + self.load_file)
                if self.mode == "qmix":
                    torch.save(self.mixer.state_dict(), "mixer_episode_" + str(episode) + "_" + self.load_file)
                print("Checkpoint saved")

            # episode end
            # print("Episode: ", episode)

        # final save (unchanged)
        if self.training:
            torch.save(self.policy_net.state_dict(), self.load_file)
            if self.mode == "qmix":
                torch.save(self.mixer.state_dict(), "mixer_" + self.load_file)
            print("Checkpoint saved")

        print("Average duration was ", duration_sum / self.num_episodes)
        print("Finished")
        np.save("Duration_matrix", self.duration_matrix)
        np.save("Count_matrix", self.count_matrix)
        print(self.duration_matrix)
        print(self.count_matrix)

    # -------------------- Resets --------------------
    def reset(self):
        self.env.reset()
        self.grid_map = self.env.grid_map
        self.cars = self.env.grid_map.cars
        self.passengers = self.env.grid_map.passengers

    def reset_different_num(self):
        self.env.grid_map.cars = []
        self.env.grid_map.passengers = []
        self.env.grid_map.num_passengers = random.randint(1, self.max_passengers)
        self.env.grid_map.num_cars = random.randint(1, self.max_cars)
        self.env.grid_map.add_passenger(self.env.grid_map.num_passengers)
        self.env.grid_map.add_cars(self.env.grid_map.num_cars)

        self.grid_map = self.env.grid_map
        self.num_passengers = self.env.grid_map.num_passengers
        self.num_cars = self.env.grid_map.num_cars
        self.cars = self.env.grid_map.cars
        self.passengers = self.env.grid_map.passengers

    def reset_orig_env(self):
        self.env = copy.deepcopy(self.orig_env)
        self.grid_map = self.env.grid_map
        self.cars = self.env.grid_map.cars
        self.passengers = self.env.grid_map.passengers
        self.grid_map.init_zero_map_cost()

    # -------------------- DQN optimize --------------------
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)      # [B, F]
        action_batch = torch.cat(batch.action)    # [B, 1, max_passengers]
        reward_batch = torch.cat(batch.reward)    # [B, 1, max_passengers]

        self.policy_net.train()

        # Infer sizes
        B = state_batch.size(0)
        P = self.max_passengers
        C = self.max_cars

        # Where is the waiting mask in the state vector?
        # Layout: [2C | C | 4P | P | P]  => offset = 2C + C + 4P + P
        offset = 2*C + C + 4*P + P
        waiting_mask = state_batch[:, offset: offset + P]            # [B, P]
        waiting_mask = waiting_mask.detach()                          # no grad
        # Avoid division by zero later
        valid_counts = waiting_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]

        # Q(s,·) and gather chosen Q(s,a)
        q_flat = self.policy_net(state_batch)                         # [B, P*C]
        q_values = q_flat.view(B, P, C)
        # Add sentinel column for unmatched/padded
        q_values = torch.cat((q_values, torch.zeros((B, P, 1), device=self.device)), dim=2)
        state_action_values = q_values.gather(2, action_batch.unsqueeze(2)).squeeze(2)  # [B, P]

        # Targets: immediate reward (contextual bandit)
        targets = reward_batch.squeeze(1)                              # [B, P]

        # --- Masked Huber loss over waiting passengers only ---
        diff = (state_action_values - targets) * waiting_mask
        loss = torch.nn.functional.smooth_l1_loss(
            state_action_values * waiting_mask,
            targets * waiting_mask,
            reduction='none'
        )
        # mean over valid positions per sample, then batch mean
        loss = (loss.sum(dim=1) / valid_counts.squeeze(1)).mean()

        self.loss_history.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.policy_net.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    # -------------------- Plots --------------------
    def plot_durations(self, filename):
        print("Saving durations plot ...")
        plt.figure(2)
        plt.clf()

        total_steps = np.array(self.episode_durations)
        N = len(total_steps)
        window_size = 200
        if N < window_size:
            total_steps_smoothed = total_steps
        else:
            total_steps_smoothed = np.zeros(N - window_size)
            for i in range(N - window_size):
                window_steps = total_steps[i : i + window_size]
                total_steps_smoothed[i] = np.average(window_steps)

        plt.title("Episode Duration history")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(total_steps_smoothed)
        np.save("Duration_" + filename, total_steps_smoothed)

    def plot_loss_history(self, filename):
        print("Saving loss history ...")
        plt.figure(2)
        plt.clf()

        total_loss = np.array(self.loss_history)
        N = len(total_loss)
        window_size = 50
        if N < window_size:
            total_loss_smoothed = total_loss
        else:
            total_loss_smoothed = np.zeros(N - window_size)
            for i in range(N - window_size):
                window_steps = total_loss[i : i + window_size]
                total_loss_smoothed[i] = np.average(window_steps)

        plt.title("Loss history")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.plot(self.loss_history)
        np.save("Loss_" + filename, total_loss_smoothed)
