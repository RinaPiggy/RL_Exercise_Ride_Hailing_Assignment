# RL Ridesharing — Tick-Based Hungarian Matching

A small research playground for dynamic ride-matching. Cars move on a grid; passengers request trips (origin→destination). Each **tick**, the agent assigns **idle cars** to **waiting passengers** using Q-values and **Hungarian** one-to-one matching.

Description about the NYC OD data:

To be added ...

---

## 🔍 Overview

- **Goal:** Minimize passenger waiting while serving trips efficiently.
- **State:** Normalized car & passenger positions + presence masks + a waiting mask.
- **Action:** For each passenger, choose a car id (or a sentinel meaning “no assignment this tick”).
- **Assignment:** We solve a **global** one-to-one assignment (Hungarian) on the **waiting×idle** submatrix of Q-values (not per-passenger argmax).
- **Dynamics:** Grid edges have unit cost (1 tick per move). Pickup and dropoff each cost 1 tick.
- **Rewards:** Base = negative waiting increment per tick; optional positive **drop bonus** proportional to trip length.

---
**Key design choices**
- **Tick-based** env (`step_tick`) with **replanning every tick** using Hungarian on *waiting × idle*.
- **Legacy** env (`step`) supports **re-queuing**, so multiple passengers can be served sequentially by the same car in one episode.
- **OD-aware sampling**: optional real-world OD probabilities for passenger origin–destination sampling.

---

## Problem Overview (Assignment We Solve)

At any tick, we have:
- A set of **waiting passengers** \(P_w\) and **idle cars** \(C_i\).
- A learned Q-value \(Q[p, c]\) estimating the value of assigning passenger \(p\) to car \(c\).

We solve:
$$\max_{X \in \{0,1\}^{|P_w|\times|C_i|}} 
\sum_{p,c} Q[p,c] \cdot X_{p,c}
\quad\text{s.t.}\quad
\sum_c X_{p,c} \le 1,\;
\sum_p X_{p,c} \le 1.$$
We implement this as a **min-cost** problem with cost \(= Q_{\max}-Q\) and run **Hungarian**. Unmatched passengers wait for later ticks when cars become idle (or new cars arrive, if enabled).

---

## RL Setup: Environment, State, Reward

### Environment (tick-based)
- **State evolution per tick**:
  1) Agent outputs an action tensor mapping **some passengers → car indices** (others can remain sentinel/unassigned).
  2) Env pairs valid `(idle car, waiting passenger)` and assigns a **pickup path** then **dropoff path**.
  3) `_advance_one_tick`: moves cars by one edge (unit-cost edges ⇒ **one grid per tick**), handles pickup/dropoff, and increments waiting counters.
  4) Episode ends when **all passengers are dropped** (no ongoing trips). With exogenous arrivals enabled, use **fixed-horizon** episodes.

- **Legacy mode**: single `step()` runs until all dropped. We maintain **per-car queues + global queue** so cars keep serving after each dropoff without new agent actions.

### Action
- Shape: `[1, max_passengers]` (LongTensor).  
- For each real passenger index `< num_passengers`:
  - `0..num_cars-1` → car id,
  - `>= num_cars` → sentinel (skip).  
- In tick-mode exploitation, actions come from a **Hungarian** assignment over the **waiting × idle** submatrix of Q.

### State (vectorized)
- **Cars**: `(x, y)` normalized to `[0,1]` for each slot + **presence mask**.
- **Passengers**: `(pickup_x, pickup_y, drop_x, drop_y)` normalized + **presence mask** + **waiting mask** (1 if `wait_pair|wait_pick`, else 0).  
- Total size = `3*max_cars + 6*max_passengers`.

### Reward
- **Base (DQN)**: per-tick, per-passenger **waiting penalty** `-Δwaiting_steps`.
- **Shaping (recommended)**: add a **drop-off bonus** `β * trip_len` to a passenger **on the tick they drop**.  
  - `β` in `[0.25, 1.0]` works well; `trip_len` is the assigned dropoff path length.
- **QMIX/IQL (optional)**: team reward per tick (e.g., `-1`) with mixer.

> If you need **lookahead** (value of freeing a car soon), switch to **TD(0) DQN** with a target net and bootstrap on `next_state`.

---

## About this Repository

This repository contains an exercise notebook designed for the course  
**CIEQ6002 / CIEM6000 – Transportation Modeling and Analysis**  
at **Delft University of Technology (TU Delft)**.  
It demonstrates a simplified **ride-hailing assignment problem** using reinforcement learning for dispatching and matching decisions.

If you find this repository useful in your work, please cite it as follows:

> **Reference**  
> Cheng, J., Lu, Y., Azadeh, S. S. (2025). *RL Exercise – Ride-Hailing Assignment Notebook*.  
> Delft University of Technology. GitHub repository: [https://github.com/RinaPiggy/RL_Exercise_Ride_Hailing_Assignment](https://github.com/RinaPiggy/RL_Exercise_Ride_Hailing_Assignment)

**BibTeX:**
```bibtex
@misc{cheng2025_rl_ridehailing,
  author       = {Cheng, Jingyi, Lu, Yahan, and Azadeh, Shadi Sharif},
  title        = {RL Exercise – Ride-Hailing Assignment Notebook},
  year         = {2025},
  institution  = {Delft University of Technology},
  howpublished = {\url{https://github.com/RinaPiggy/RL_Exercise_Ride_Hailing_Assignment}},
  note         = {RL Exercise notebook for CIEQ6002/CIEM6000 Transportation Modeling and Analysis course}
}
````
## Quick Start
```bash
### Setup (Python 3.9+)

pip install torch scipy numpy tqdm matplotlib
# optional
pip install pytest pandas

# (optional) create and activate venv
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1

pip install -U pip
pip install torch numpy scipy matplotlib tqdm

# train with defaults + run demo
python main.py

# faster smoke run
python main.py --episodes 50 --eps_decay 8000 --viz_every 5

# skip training, only run demo
python main.py --no_train

# log per-episode metrics to CSV
python main.py --log_csv runs/dqn_metrics.csv

# OD aware sampling for your own city data
import numpy as np
from gridmap import GridMap

rows, cols = 5, 5
P_od = np.load("my_city_od.npy")  # shape (rows*cols, rows*cols) or flat (N^2,)
gm = GridMap(seed=1, size=(rows, cols), num_cars=2, num_passengers=3,
             od_prob=P_od, allow_duplicate_pickups=True)

## Repository Structure

├── main.py # Entry point: train + demo visualization (tick-based)
├── agent_variable.py # Agent: ε-greedy, Hungarian matching, training loop, memory
├── dqn.py # DQN model (MLP)
├── q_mixer.py # (Optional) QMIX mixer for cooperative setting
├── environment.py # Environment: tick-wise step, legacy one-shot with re-queuing
├── gridmap.py # Grid + costs + pathing; OD-aware passenger sampling
├── car.py # Car dynamics: pair/assign/move/pick/drop
├── passenger.py # Passenger state: pickup/dropoff, waiting, trip_len
├── metrics.py # Rolling stats + CSV logger
├── plot_metrics.py # Offline plots for durations/rewards from CSV
├── tests/
│ └── smoke_test.py # Small tests for tick + legacy + sentinels
└── README.md # You are here
---


