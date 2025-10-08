# file: gridmap.py (updated)
import math
import random
from typing import List, Tuple, Dict, Optional, Set

import numpy as np

from util import Util
from car import Car
from passenger import Passenger

class GridMap:
    def __init__(
        self,
        seed: int,
        size: Tuple[int, int],
        num_cars: int,
        num_passengers: int,
        rng: Optional[np.random.Generator] = None,
        allow_duplicate_pickups: bool = True,   # NEW: real-world often has many O overlaps
    ):
        random.seed(seed)
        self.seed = seed
        self.size = size  # (rows, cols)
        self.num_cars = num_cars
        self.num_passengers = num_passengers
        self.map_cost: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
        self.cars: List[Car] = []
        self.passengers: List[Passenger] = []

        # RNG & OD setup
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.allow_duplicate_pickups = allow_duplicate_pickups

        try:
            self.P_flat = np.load('P_flat.npy')
            print("Loaded P_flat.npy for OD probabilities.")
        except FileNotFoundError:
            print("P_flat.npy not found, using default OD probabilities.")
            self.P_flat = None
        self._init_od_prob(self.P_flat)

        self.add_passenger(num_passengers)
        self.add_cars(num_cars)
        self.init_map_cost()  # keep your unit-cost or randomized edges


    # ---------- Helpers ----------
    def is_valid(self, p: Tuple[int, int]) -> bool:
        return 0 <= p[0] < self.size[0] and 0 <= p[1] < self.size[1]

    def init_map_cost(self):
        """Unit cost example; change if you use random costs."""
        self.map_cost.clear()
        rows, cols = self.size
        for r in range(rows):
            for c in range(cols):
                u = (r, c)
                for v in [(r-1, c), (r, c+1), (r+1, c), (r, c-1)]:
                    if self.is_valid(v):
                        self.map_cost[(u, v)] = 1

    def _unique_positions(self, k: int, exclude: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        total = self.size[0] * self.size[1]
        if k > max(0, total - len(exclude)):
            raise AssertionError('number of items exceeds available grid cells')
        chosen = set()
        while len(chosen) < k:
            p = (random.randint(0, self.size[0] - 1), random.randint(0, self.size[1] - 1))
            if p not in chosen and p not in exclude:
                chosen.add(p)
        return list(chosen)

    def add_cars(self, num_cars: int):
        occupied = {c.position for c in self.cars}
        positions = self._unique_positions(num_cars, exclude=occupied)
        for pos in positions:
            self.cars.append(Car(pos))

    # ---------- OD probability setup ----------
    def _init_od_prob(self, od_prob: Optional[np.ndarray]):
        """
        Accepts:
          - None: use old uniform sampler.
          - 1D array of length (Ncells^2) with probabilities over OD pairs.
          - 2D array of shape [Ncells, Ncells] with P[orig, dest].
        Stores normalized flat vector self.P_flat and Ncells for decoding.
        """
        rows, cols = self.size
        ncells = rows * cols
        self.Ncells = ncells
        if od_prob is None:
            self.P_flat = None  # fallback to uniform sampler
            return
        arr = np.asarray(od_prob, dtype=np.float64)
        if arr.ndim == 2:
            assert arr.shape == (ncells, ncells), f"od_prob must be ({ncells},{ncells})"
            arr = arr.reshape(-1)
        else:
            assert arr.ndim == 1 and arr.size == ncells * ncells, \
                f"od_prob length must be (rows*cols)^2 = {ncells*ncells}"
        s = arr.sum()
        if s <= 0:
            raise ValueError("od_prob must have positive sum")
        self.P_flat = (arr / s).astype(np.float64)

    # ---------- Passenger sampling (OD-aware) ----------
    def add_passenger(self, num_passengers: int):
        """
        Sample passengers:
          - If self.P_flat is set: sample OD from that categorical distribution.
          - Else: fallback to uniform random O and D != O.
        allow_duplicate_pickups controls whether multiple passengers can share the same origin.
        """
        rows, cols = self.size
        ncells = self.Ncells
        # For legacy uniqueness: we only avoid duplicating *pickup* cells if the flag is False.
        occupied_pickups: Set[Tuple[int, int]] = set() if not self.allow_duplicate_pickups else set()

        # Include already-existing pickups in occupied set when uniqueness enforced
        if not self.allow_duplicate_pickups:
            occupied_pickups = {p.pick_up_point for p in self.passengers}

        for _ in range(num_passengers):
            if self.P_flat is not None:
                # Sample linear OD index over ncells*ncells possibilities
                od_lin = int(self.rng.choice(ncells * ncells, p=self.P_flat))
                o_idx = od_lin // ncells
                d_idx = od_lin % ncells
                O = (o_idx // cols, o_idx % cols)
                D = (d_idx // cols, d_idx % cols)

                # Enforce O != D; resample a few times, then adjust if needed
                tries = 0
                while (O == D) or (not self.allow_duplicate_pickups and O in occupied_pickups):
                    od_lin = int(self.rng.choice(ncells * ncells, p=self.P_flat))
                    o_idx = od_lin // ncells
                    d_idx = od_lin % ncells
                    O = (o_idx // cols, o_idx % cols)
                    D = (d_idx // cols, d_idx % cols)
                    tries += 1
                    if tries > 20:
                        # Fallback: shift D to a valid neighbor to break tie
                        for nbr in [(D[0]-1, D[1]), (D[0]+1, D[1]), (D[0], D[1]-1), (D[0], D[1]+1)]:
                            if self.is_valid(nbr) and nbr != O:
                                D = nbr
                                break
                        break
            else:
                # Uniform fallback (legacy behavior)
                # Optional uniqueness on O
                if not self.allow_duplicate_pickups:
                    choices = self._unique_positions(1, exclude=occupied_pickups)
                    O = choices[0]
                    occupied_pickups.add(O)
                else:
                    O = (random.randint(0, rows - 1), random.randint(0, cols - 1))
                # Draw D != O
                while True:
                    D = (random.randint(0, rows - 1), random.randint(0, cols - 1))
                    if D != O:
                        break

            self.passengers.append(Passenger(O, D))
            if not self.allow_duplicate_pickups:
                occupied_pickups.add(O)

    # ---------- Pathing & visualize unchanged ----------
    def plan_path(self, start_point: Tuple[int, int], end_point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Greedy Manhattan descent path."""
        def better(curr, best_d, best_n):
            if self.is_valid(curr):
                d = Util.cal_dist(curr, end_point)
                if d < best_d:
                    return d, curr
            return best_d, best_n

        path: List[Tuple[int, int]] = []
        curr = start_point
        while curr != end_point:
            best_d, best_n = math.inf, None
            for nxt in [(curr[0]-1, curr[1]), (curr[0]+1, curr[1]), (curr[0], curr[1]-1), (curr[0], curr[1]+1)]:
                best_d, best_n = better(nxt, best_d, best_n)
            assert best_n is not None, 'no valid position for next step'
            path.append(best_n); curr = best_n
        return path

    def visualize(self):
        m = [["     " for _ in range(self.size[1])] for _ in range(self.size[0])]
        for p in self.passengers:
            if p.status in {'wait_pair', 'wait_pick'}:
                m[p.pick_up_point[0]][p.pick_up_point[1]] = "p" + str(id(p))[-2:] + "  "
        for c in self.cars:
            if c.status == 'dropping_off':
                m[c.position[0]][c.position[1]] = "x" + str(id(c.passenger))[-2:] + ":" + str(c.required_steps or 0)
            elif c.status == 'picking_up':
                m[c.position[0]][c.position[1]] = "c" + str(id(c.passenger))[-2:] + ":" + str(c.required_steps or 0)
            else:
                m[c.position[0]][c.position[1]] = "c--  "
        for row in m:
            print(row)
