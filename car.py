# file: car.py
from typing import List, Optional, Tuple
from passenger import Passenger

class Car:
    """
    States:
      - 'idle': free; no passenger assigned.
      - 'picking_up': assigned; en route to passenger.pick_up_point or at pickup.
      - 'dropping_off': passenger onboard; en route to passenger.drop_off_point or at dropoff.
    """
    def __init__(self, start_pos: Tuple[int, int]):
        self.position: Tuple[int, int] = start_pos
        self.status: str = 'idle'
        self.passenger: Optional[Passenger] = None
        self.path: List[Tuple[int, int]] = []
        self.required_steps: Optional[int] = None  # edge traversal ticks
        self._drop_off_path: List[Tuple[int, int]] = []

    def __repr__(self) -> str:
        pid = id(self.passenger) if self.passenger else None
        return (f"Car(pos={self.position}, status={self.status}, "
                f"passenger_id={pid}, path_len={len(self.path)}, req={self.required_steps})")

    # -------- lifecycle --------
    def pair_passenger(self, passenger: Passenger) -> None:
        self.passenger = passenger
        if passenger.status == 'wait_pair':
            passenger.status = 'wait_pick'

    def assign_path(self, pick_up_path: List[Tuple[int, int]], drop_off_path: List[Tuple[int, int]]) -> None:
        self.required_steps = None
        self._drop_off_path = list(drop_off_path)
        if self.passenger is not None:
            # record trip length once, for reward shaping
            self.passenger.trip_len = len(drop_off_path)
        if self.position == (self.passenger.pick_up_point if self.passenger else self.position):
            self.status = 'dropping_off'
            self.path = list(drop_off_path)
        else:
            self.status = 'picking_up'
            self.path = list(pick_up_path)

    def pick_passenger(self) -> None:
        if self.passenger is None:
            self.status = 'idle'; self.path = []; self.required_steps = None
            return
        self.status = 'dropping_off'
        self.path = list(self._drop_off_path)
        self.required_steps = None
        self._drop_off_path = []

    def drop_passenger(self) -> None:
        if self.passenger is not None:
            self.passenger.status = 'dropped'
        self.passenger = None
        self.status = 'idle'
        self.path = []
        self.required_steps = None

    def move(self) -> None:
        if not self.path:
            if self.status == 'picking_up' and self.passenger and self.position == self.passenger.pick_up_point:
                self.pick_passenger()
            elif self.status == 'dropping_off' and self.passenger and self.position == self.passenger.drop_off_point:
                self.drop_passenger()
            return
        self.position = self.path.pop(0)