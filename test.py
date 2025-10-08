# file: tests/smoke_test.py
import random
import time
import numpy as np
import torch

from gridmap import GridMap
from environment import Environment
from agent_design import Agent  # adjust import name if different

try:
    from tqdm import trange
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# ---------- helpers ----------
def _build_agent_env(seed=1, init_cars=2, init_passengers=3, max_cars=5, max_passengers=10):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gm = GridMap(seed, (5, 5), init_cars, init_passengers)
    env = Environment(gm)
    input_size = 3 * max_cars + 6 * max_passengers
    output_size = max_cars * max_passengers
    hidden_size = 64
    agent = Agent(
        env, input_size, output_size, hidden_size,
        max_cars=max_cars, max_passengers=max_passengers,
        load_file=None, lr=0.001, mix_hidden=64, batch_size=8,
        eps_decay=2000, num_episodes=1, mode="dqn", training=False
    )
    return agent, env

def _compute_max_ticks(env, multiplier: float = 3.0) -> int:
    """Cap based on worst-case Manhattan traversal with unit edges."""
    gm = env.grid_map
    rows, cols = gm.size
    # Worst-case edges = pickup path + dropoff path
    edges = 2 * ((rows - 1) + (cols - 1))
    # unit edges → move every tick; +2 for pickup/dropoff service
    base = edges + 2
    return int(max(50, base * multiplier))

# ---------- tests ----------
def test_tick_mode(progress: bool = True, viz_every: int = 0):
    agent, env = _build_agent_env()
    assert hasattr(env, "step_tick"), "Environment.step_tick not found"
    env.reset()

    # sync live counts
    agent.grid_map = env.grid_map
    agent.cars = env.grid_map.cars
    agent.passengers = env.grid_map.passengers
    agent.num_cars = len(agent.cars)
    agent.num_passengers = len(agent.passengers)

    max_ticks = _compute_max_ticks(env, multiplier=3.0)
    done = False
    ticks = 0
    start = time.time()
    bar = trange(max_ticks, desc="tick_mode", ncols=80) if (progress and _HAS_TQDM) else range(max_ticks)

    for _ in bar:
        if done: break
        state = agent.get_state()
        act = agent.select_action_hungarian_tick(state)

        assert act.shape == (1, agent.max_passengers)
        assert act.dtype == torch.long

        _, reward, done, info = env.step_tick(act[:, : agent.num_passengers], mode="dqn")
        assert isinstance(reward, list) and len(reward) == agent.num_passengers
        ticks += 1

        if progress and _HAS_TQDM:
            bar.set_postfix_str(f"idle={info['idle_cars']} wait={info['waiting_passengers']}")
        if viz_every and ticks % viz_every == 0:
            print(f"\n[t={ticks}] idle={info['idle_cars']} waiting={info['waiting_passengers']}")
            env.grid_map.visualize()

    elapsed = time.time() - start
    assert done, f"tick_mode: episode did not finish within {max_ticks} ticks"
    print(f"[tick_mode] finished in {ticks} ticks, {elapsed:.2f}s")

def test_legacy_one_shot(progress: bool = True):
    """
    Validates legacy env.step one-shot now terminates thanks to re-queuing:
    cars keep serving passengers sequentially after each dropoff.
    """
    agent, env = _build_agent_env()
    env.reset()

    # sync once (legacy runs to terminal internally)
    agent.grid_map = env.grid_map
    agent.cars = env.grid_map.cars
    agent.passengers = env.grid_map.passengers
    agent.num_cars = len(agent.cars)
    agent.num_passengers = len(agent.passengers)

    state = agent.get_state()
    act = agent.select_action_hungarian(state)  # may include sentinels; env now re-queues

    # Optional simple progress line
    t0 = time.time()
    if progress and _HAS_TQDM:
        print("legacy_mode: running (re-queuing enabled) ...")

    reward, duration = env.step(act[:, : agent.num_passengers], mode="dqn")

    # Sanity checks
    assert isinstance(reward, list) and len(reward) == agent.num_passengers
    assert duration > 0

    # Duration sanity cap (should be under a generous bound)
    cap = _compute_max_ticks(env, multiplier=6.0)  # legacy queues may take longer; give more slack
    assert duration <= cap, f"legacy_mode took too long: {duration} > {cap}"

    print(f"[legacy_mode] finished in {duration} ticks; reward_len={len(reward)}; took {(time.time()-t0):.2f}s")

def test_sentinel_guard():
    """Env should ignore invalid/sentinel indices in both modes."""
    agent, env = _build_agent_env()
    env.reset()

    P = len(env.grid_map.passengers)
    C = len(env.grid_map.cars)
    sentinel = C + 5
    act = torch.full((1, agent.max_passengers), sentinel, dtype=torch.long)
    if C > 0:
        act[0, 0] = 0  # only first pax uses a valid car

    # Tick mode: should not crash
    for _ in range(5):
        _, reward, _, _ = env.step_tick(act[:, :P], mode="dqn")
        assert isinstance(reward, list) and len(reward) == P

    # Legacy mode: should not crash and should still finish (thanks to re-queuing)
    reward, duration = env.step(act[:, :P], mode="dqn")
    print(f"[sentinel_guard] legacy_mode finished in {duration} ticks; reward_len={len(reward)} ({reward})")
    assert isinstance(reward, list) and len(reward) == P
    assert duration > 0
    print("[sentinel_guard] sentinel indices handled without error")

if __name__ == "__main__":
    test_tick_mode(progress=False, viz_every=0)     # set viz_every=5 to print grid snapshots
    print('Pass! ')
    test_legacy_one_shot(progress=True)
    print('Pass!! ')
    test_sentinel_guard()
    print("All smoke tests passed ✅")
