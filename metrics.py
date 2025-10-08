from __future__ import annotations
import csv
import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque

@dataclass
class RollingStat:
    window: int
    vals: Deque[float]

    def __init__(self, window: int = 100):
        self.window = max(1, window)
        self.vals = deque(maxlen=self.window)

    def add(self, x: float) -> None:
        self.vals.append(float(x))

    def mean(self) -> float:
        return sum(self.vals) / len(self.vals) if self.vals else 0.0

class MetricsLogger:
    """
    Minimal CSV logger with rolling means; no external deps.
    Why: track episode_duration, episode_reward, loss.
    """
    def __init__(self, csv_path: Optional[str], roll_window: int = 100):
        self.csv_path = csv_path
        self.step = 0
        self.r_duration = RollingStat(roll_window)
        self.r_reward = RollingStat(roll_window)
        self.r_loss = RollingStat(roll_window)
        if csv_path:
            os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "duration", "reward", "loss"])

    def log_episode(self, episode: int, duration: float, reward: float, loss: float = float("nan")) -> None:
        self.r_duration.add(duration)
        self.r_reward.add(reward)
        if loss == loss:  # not NaN
            self.r_loss.add(loss)
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([episode, duration, reward, loss])

    def postfix(self) -> str:
        return (f"dur(avg{self.r_duration.window})={self.r_duration.mean():.1f} | "
                f"rew(avg{self.r_reward.window})={self.r_reward.mean():.2f} | "
                f"loss(avg{self.r_loss.window})={self.r_loss.mean():.4f}")