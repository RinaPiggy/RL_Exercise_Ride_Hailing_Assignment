# file: main.py
import argparse
import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# --- Local imports (allow either agent_variable or agent_design) ---
from gridmap import GridMap
from environment import Environment

from agent_design import Agent  # preferred name in our discussion


# Optional tqdm

from tqdm import trange
# _HAS_TQDM = True

_HAS_TQDM = False


@dataclass
class Config:
    seed: int = 1
    rows: int = 5
    cols: int = 5
    init_cars: int = 2
    init_passengers: int = 3
    max_cars: int = 5
    max_passengers: int = 7
    hidden_size: int = 64
    mix_hidden: int = 64
    lr: float = 1e-3
    batch_size: int = 128
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 20000
    episodes: int = 1000
    mode: str = "dqn"        # dqn | qmix | random | greedy
    training: bool = True
    out_dir: str = "runs"
    ckpt_every: int = 250
    # demo
    run_demo: bool = True
    viz_every: int = 5
    demo_timeout_mult: float = 4.0  # cap multiplier for demo
    # logging
    log_csv: Optional[str] = None   # "runs/log.csv" if set


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_tick_cap(rows: int, cols: int, mult: float = 3.0) -> int:
    """Cap based on worst-case Manhattan traversal with unit edge costs."""
    edges = 2 * ((rows - 1) + (cols - 1))
    base = edges + 2  # +2 for pickup/dropoff service
    return int(max(50, base * mult))


def build(cfg: Config) -> tuple[Agent, Environment]:
    gm = GridMap(cfg.seed, (cfg.rows, cfg.cols), cfg.init_cars, cfg.init_passengers)
    env = Environment(gm)
    input_size = 3 * cfg.max_cars + 6 * cfg.max_passengers
    output_size = cfg.max_cars * cfg.max_passengers
    agent = Agent(
        env=env,
        input_size=input_size,
        output_size=output_size,
        hidden_size=cfg.hidden_size,
        max_cars=cfg.max_cars,
        max_passengers=cfg.max_passengers,
        mix_hidden=cfg.mix_hidden,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        gamma=.999,  # ignored per one-shot reward, kept for completeness
        eps_start=cfg.eps_start,
        eps_end=cfg.eps_end,
        eps_decay=cfg.eps_decay,
        replay_capacity=10000,
        num_save=cfg.ckpt_every,
        num_episodes=cfg.episodes,
        mode=cfg.mode,
        training=cfg.training,
        load_file=None,
    )
    return agent, env


def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def train(agent: Agent, cfg: Config):
    ensure_dir(cfg.out_dir)
    if cfg.log_csv:
        ensure_dir(os.path.dirname(cfg.log_csv))

    t0 = time.time()
    agent.train()  # uses tick env if available
    elapsed = time.time() - t0
    print(f"[train] finished {cfg.episodes} episodes in {elapsed/60:.2f} min")

    # Optional metrics dump to CSV (episode durations + loss) for quick analysis
    if cfg.log_csv:
        with open(cfg.log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "duration", "loss_or_nan"])
            losses = agent.loss_history
            for i, dur in enumerate(agent.episode_durations):
                loss_i = losses[i] if i < len(losses) else float("nan")
                w.writerow([i, dur, loss_i])

        print(f"[train] metrics saved to {cfg.log_csv}")


def demo_episode(agent: Agent, env: Environment, cfg: Config):
    """Visual demo with tick-wise Hungarian re-planning."""
    if not hasattr(env, "step_tick"):
        print("[demo] Environment has no step_tick; skipping demo.")
        return

    env.reset()
    # sync references once at demo start
    agent.grid_map = env.grid_map
    agent.cars = env.grid_map.cars
    agent.passengers = env.grid_map.passengers
    agent.num_cars = len(agent.cars)
    agent.num_passengers = len(agent.passengers)

    cap = compute_tick_cap(cfg.rows, cfg.cols, cfg.demo_timeout_mult)
    done = False
    ticks = 0
    print(f"[demo] running up to {cap} ticks...")

    bar = trange(cap, desc="demo", ncols=80) if _HAS_TQDM else range(cap)
    while not done and ticks < cap:
        state = agent.get_state()
        # Force exploitation for cleaner demo (no epsilon)
        was_training = agent.training
        agent.training = False
        action = agent.select_action_hungarian_tick(state)
        agent.training = was_training

        _, reward_t, done, info = env.step_tick(action[:, : agent.num_passengers], mode="dqn")

        ticks += 1
        if _HAS_TQDM:
            bar.update(1)
            bar.set_postfix_str(f"idle={info['idle_cars']} wait={info['waiting_passengers']}")
        if cfg.viz_every and ticks % cfg.viz_every == 0:
            print(f"\n[t={ticks}] idle={info['idle_cars']} waiting={info['waiting_passengers']}")
            env.grid_map.visualize()

    if done:
        print(f"[demo] completed in {ticks} ticks ✅")
    else:
        print(f"[demo] reached cap ({cap}) without finishing ⚠️")


def parse_args() -> Config:
    p = argparse.ArgumentParser("RL Ridesharing — train & visualize")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--init_cars", type=int, default=2)
    p.add_argument("--init_passengers", type=int, default=3)
    p.add_argument("--max_cars", type=int, default=5)
    p.add_argument("--max_passengers", type=int, default=7)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--mix_hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eps_start", type=float, default=0.9)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=int, default=20000)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--mode", type=str, default="dqn", choices=["dqn", "qmix", "random", "greedy"])
    p.add_argument("--no_train", action="store_true")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--ckpt_every", type=int, default=200)
    p.add_argument("--log_csv", type=str, default="")
    p.add_argument("--no_demo", action="store_true")
    p.add_argument("--viz_every", type=int, default=5)
    p.add_argument("--demo_timeout_mult", type=float, default=3.0)
    args = p.parse_args()

    return Config(
        seed=args.seed,
        rows=args.rows,
        cols=args.cols,
        init_cars=args.init_cars,
        init_passengers=args.init_passengers,
        max_cars=args.max_cars,
        max_passengers=args.max_passengers,
        hidden_size=args.hidden_size,
        mix_hidden=args.mix_hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        episodes=args.episodes,
        mode=args.mode,
        training=not args.no_train,
        out_dir=args.out_dir,
        ckpt_every=args.ckpt_every,
        run_demo=not args.no_demo,
        viz_every=args.viz_every,
        demo_timeout_mult=args.demo_timeout_mult,
        log_csv=(args.log_csv or None),
    )


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    agent, env = build(cfg)
    print(f"[config] grid={cfg.rows}x{cfg.cols} init_cars={cfg.init_cars} init_pax={cfg.init_passengers} "
          f"max_cars={cfg.max_cars} max_pax={cfg.max_passengers} mode={cfg.mode} training={cfg.training}")

    if cfg.training:
        train(agent, cfg)
    else:
        print("[train] skipped (no_train)")

    if cfg.run_demo:
        demo_episode(agent, env, cfg)
    else:
        print("[demo] skipped (no_demo)")


if __name__ == "__main__":
    main()


# import numpy as np
# import random
# import math
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import scipy.optimize
# import matplotlib.pyplot as plt
# import copy 
# from collections import namedtuple
# from itertools import count

# # import coded modules
# from environment import *
# from gridmap import GridMap
# from algorithm import *
# from dqn import ReplayMemory, DQN
# from q_mixer import QMixer
# from agent_design import Agent

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'reward'))


# if __name__ == '__main__':
#     init_cars = 2
#     init_passengers = 3
#     max_cars = 5
#     max_passengers = 10
    
    
    
#     grid_map = GridMap(1, (5,5), init_cars, init_passengers)
#     cars = grid_map.cars
#     passengers = grid_map.passengers
#     env = Environment(grid_map)


#     input_size = 3*max_cars + 5*max_passengers # cars (px, py), passengers(pickup_x, pickup_y, dest_x, dest_y)
#     output_size = max_cars * max_passengers  # num_cars * (num_passengers + 1)
#     hidden_size = 64
#     #load_file = "episode_50000_dqn_model_num_cars_2_num_passengers_7_num_episodes_100000_hidden_size_512.pth"
#     load_file = None
#     #greedy, random, dqn, qmix
#     agent = Agent(env, input_size, output_size, hidden_size, max_cars=max_cars, max_passengers = max_passengers, 
#                   load_file = load_file, lr=0.001, mix_hidden = 64, batch_size=128, eps_decay = 20000, num_episodes=1000, 
#                   mode = "dqn", training = True)
#     agent.train()
    
#     # qmix= np.load("Duration_matrix_qmix.npy")
#     # greedy = np.load("Duration_matrix_greedy.npy")
#     # random = np.load("Duration_matrix_random.npy")
    
    
#     # # qmix_count = np.load("Count_matrix_qmix.npy")
#     # greedy_count = np.load("Count_matrix_greedy.npy")
#     # qmix_count = np.load("Count_matrix_qmix.npy")
#     # random_count = np.load("Count_matrix_random.npy")
#     # print(np.sum(greedy*greedy_count)/10000)
#     # print(np.sum(qmix*qmix_count)/10000)
#     # print(np.sum(random*random_count)/10000)