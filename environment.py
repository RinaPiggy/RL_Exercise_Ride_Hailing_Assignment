# =========================
# file: environment.py
# =========================
from typing import Any, Dict, Optional, Tuple, List
from collections import deque

class Environment:
    def __init__(self, grid_map, drop_bonus_scale: float = 0.5):
        self.grid_map = grid_map
        self.drop_bonus_scale = drop_bonus_scale  # β

    def _advance_one_tick(self) -> None:
        """
        Advance all cars exactly one simulation tick.
        Pickup/dropoff each consume one tick.
        Edge traversal uses gm.map_cost; with unit edges (1) cars move every tick.
        """
        gm = self.grid_map
        for car in gm.cars:
            if car.status == "idle":
                continue

            # Initialize traversal for the next edge when needed
            if car.required_steps is None and car.path:
                cost = gm.map_cost[(car.position, car.path[0])]
                car.required_steps = max(0, cost - 1)  # cost=1 ⇒ move this tick

            # Service events (consume this tick)
            if car.status == "picking_up" and car.passenger and car.position == car.passenger.pick_up_point:
                car.pick_passenger()
                # do not traverse further this tick
                continue
            if car.status == "dropping_off" and car.passenger and car.position == car.passenger.drop_off_point:
                car.drop_passenger()
                # do not traverse further this tick
                continue

            # Traverse current edge
            if car.required_steps is not None:
                if car.required_steps > 0:
                    car.required_steps -= 1
                else:
                    # Move to the next node
                    car.move()
                    # Preload next edge if a path remains
                    if car.path:
                        next_cost = gm.map_cost[(car.position, car.path[0])]
                        car.required_steps = max(0, next_cost - 1)
                    else:
                        car.required_steps = None  # at node; will handle service next tick

    def reset(self):
        gm = self.grid_map
        gm.cars = []
        gm.passengers = []
        gm.add_passenger(gm.num_passengers)
        gm.add_cars(gm.num_cars)
        return None

    def step(self, action, mode):
        """
        Legacy one-shot episode with re-queuing:
        - Build per-car queues from the provided 'action'.
        - When a car becomes idle, it pulls the next waiting passenger from its queue,
          otherwise from a global fallback queue (sentinels/unassigned).
        - Runs until all passengers dropped.
        """
        gm = self.grid_map
        cars = gm.cars
        passengers = gm.passengers

        # --- Build queues from the initial action ---
        P, C = len(passengers), len(cars)
        car_queues: Dict[int, deque] = {ci: deque() for ci in range(C)}
        global_queue: deque = deque()

        # Map each passenger i to the requested car if valid; else into global queue
        if action is not None and len(action) > 0:
            acts = action[0]
            for i in range(min(P, len(acts))):
                ci = int(acts[i])
                if 0 <= ci < C:
                    car_queues[ci].append(i)
                else:
                    global_queue.append(i)
        else:
            # No action provided → everyone queues globally
            for i in range(P):
                global_queue.append(i)

        def _assign_if_possible(car_idx: int):
            """Assign next waiting passenger to this idle car from its queue or global queue."""
            car = cars[car_idx]
            if car.status != "idle":
                return
            # Prefer this car's own queue
            while car_queues[car_idx]:
                p_idx = car_queues[car_idx].popleft()
                pax = passengers[p_idx]
                if pax.status == "wait_pair":
                    car.pair_passenger(pax)
                    pick_up_path = gm.plan_path(car.position, pax.pick_up_point)
                    drop_off_path = gm.plan_path(pax.pick_up_point, pax.drop_off_point)
                    car.assign_path(pick_up_path, drop_off_path)
                    return
            # Fallback to global queue
            while global_queue:
                p_idx = global_queue.popleft()
                pax = passengers[p_idx]
                if pax.status == "wait_pair":
                    car.pair_passenger(pax)
                    pick_up_path = gm.plan_path(car.position, pax.pick_up_point)
                    drop_off_path = gm.plan_path(pax.pick_up_point, pax.drop_off_point)
                    car.assign_path(pick_up_path, drop_off_path)
                    return

        # Initial assignment pass for any idle cars
        for ci in range(C):
            _assign_if_possible(ci)

        # --- Run the episode until all dropped ---
        done = False
        duration = 0

        while not done:
            # Progress one tick of movement/pickup/dropoff
            self._advance_one_tick()

            # Any car that became idle can grab the next passenger
            for ci in range(C):
                _assign_if_possible(ci)

            # Waiting penalty accumulation
            for p in passengers:
                if p.status in {"wait_pair", "wait_pick"}:
                    p.waiting_steps += 1

            done = all(p.status == "dropped" for p in passengers)
            duration += 1

        # Rewards (unchanged)
        if mode == "dqn":
            reward: List[float] = [-p.waiting_steps for p in passengers]
        elif mode in {"qmix", "iql"}:
            reward = -duration
        else:
            reward = 0

        return reward, duration

    def step_tick(self, action, mode: str = "dqn"):
        gm = self.grid_map
        cars = gm.cars
        passengers = gm.passengers

        P_now = len(passengers)

        # snapshot statuses to detect drops this tick
        status_before = [p.status for p in passengers]
        wait_before = [p.waiting_steps for p in passengers]

        # pairing (unchanged, with index guards)
        if action is not None and len(action) > 0:
            acts = action[0]
            for i in range(min(P_now, len(acts))):
                act = int(acts[i])
                if not (0 <= act < len(cars)):
                    continue
                car = cars[act]
                pax = passengers[i]
                if car.status == "idle" and pax.status == "wait_pair":
                    car.pair_passenger(pax)
                    pick_up_path = gm.plan_path(car.position, pax.pick_up_point)
                    drop_off_path = gm.plan_path(pax.pick_up_point, pax.drop_off_point)
                    car.assign_path(pick_up_path, drop_off_path)

        # advance one tick (cost=1 → move each tick if you applied that patch)
        self._advance_one_tick()

        # waiting penalty update
        for p in passengers:
            if p.status in {"wait_pair", "wait_pick"}:
                p.waiting_steps += 1

        # detect drops this tick
        dropped_idx: List[int] = []
        for i, p in enumerate(passengers):
            if status_before[i] != "dropped" and p.status == "dropped":
                dropped_idx.append(i)

        # rewards
        if mode == "dqn":
            # base: -Δwaiting per passenger
            reward = [-(p.waiting_steps - wait_before[i]) for i, p in enumerate(passengers)]
            # add drop bonus: β * trip_len to those who just dropped
            for i in dropped_idx:
                # guard if episode padded or passenger reused
                if 0 <= i < len(passengers):
                    reward[i] += self.drop_bonus_scale * float(passengers[i].trip_len)
        elif mode in {"qmix", "iql"}:
            reward = -1
        else:
            reward = 0

        done = all(p.status == "dropped" for p in passengers)
        info = {
            "idle_cars": sum(1 for c in cars if c.status == "idle"),
            "waiting_passengers": sum(1 for p in passengers if p.status in {"wait_pair", "wait_pick"}),
            "dropped_this_tick": len(dropped_idx),
        }
        obs = None
        return obs, reward, done, info