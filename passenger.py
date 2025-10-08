from typing import Tuple

class Passenger:
    """
    States:
      - 'wait_pair': not assigned yet.
      - 'wait_pick': assigned; waiting for pickup.
      - 'dropped': completed.
    """
    def __init__(self, pick_up_point: Tuple[int, int], drop_off_point: Tuple[int, int]):
        self.pick_up_point = pick_up_point
        self.drop_off_point = drop_off_point
        self.status = 'wait_pair'
        self.waiting_steps = 0
        self.trip_len = 0  # set at assignment

    def __repr__(self) -> str:
        return (f"Passenger(pickup={self.pick_up_point}, drop={self.drop_off_point}, "
                f"status={self.status}, wait={self.waiting_steps})")