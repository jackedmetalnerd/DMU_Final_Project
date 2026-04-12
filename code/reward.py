"""
reward.py
=========
Reward class encapsulating all reward logic for the GameEnv MDP.

To switch the active reward function, change only the last line:
    Reward.ACTIVE = Reward.shaped_military_advantage
No other file needs to be edited.

Interface contract:
    Reward instance: reward.evaluate(s) -> float
    Reward.build_vector(states) -> np.ndarray  (populates env.R)
    s has fields: W1, M1, R1, W2, M2, R2, terminal (all int, range 0-10)
"""

import numpy as np


class Reward:

    # ── Reward functions ──────────────────────────────────────────────────────

    @staticmethod
    def terminal_only(s) -> float:
        """Baseline: sparse terminal reward only.
        +1.0 if P1 wins (M1 > 0, M2 == 0), -1.0 for any other terminal state.
        Non-terminal states return 0.0.
        """
        if s.terminal:
            return 1.0 if (s.M1 > 0 and s.M2 == 0) else -1.0
        return 0.0

    @staticmethod
    def shaped_military_advantage(s) -> float:
        """Terminal rewards + continuous shaping based on marine lead.
        Non-terminal signal: (M1 - M2) / 10, in range [-1, 1].
        Encourages building and maintaining a marine advantage.
        """
        if s.terminal:
            return 1.0 if (s.M1 > 0 and s.M2 == 0) else -1.0
        return (s.M1 - s.M2) / 10.0

    @staticmethod
    def shaped_combined(s) -> float:
        """Terminal rewards + weighted advantage across marines, resources, workers.
        Non-terminal signal: 0.5*(M1-M2)/10 + 0.3*(R1-R2)/10 + 0.2*(W1-W2)/10.
        All terms normalized to [-1, 1]; weights sum to 1.0.
        """
        if s.terminal:
            return 1.0 if (s.M1 > 0 and s.M2 == 0) else -1.0
        mil  = 0.5 * (s.M1 - s.M2) / 10.0
        res  = 0.3 * (s.R1 - s.R2) / 10.0
        work = 0.2 * (s.W1 - s.W2) / 10.0
        return mil + res + work

    @staticmethod
    def win_only(s) -> float:
        """Asymmetric: reward winning only, no penalty for losing.
        +1.0 for P1 win, 0.0 for everything else including loss.
        Tests whether removing the loss penalty changes agent aggression.
        """
        if s.terminal:
            return 1.0 if (s.M1 > 0 and s.M2 == 0) else 0.0
        return 0.0

    # ── Instance ──────────────────────────────────────────────────────────────

    def __init__(self, fn=None):
        self._fn = fn if fn is not None else Reward.ACTIVE

    def evaluate(self, s) -> float:
        return self._fn(s)

    def build_vector(self, states) -> np.ndarray:
        return np.array([self._fn(s) for s in states])

    @property
    def name(self) -> str:
        return self._fn.__name__


# ── Registry and active selector ──────────────────────────────────────────────

Reward.ALL = [
    Reward.terminal_only,
    Reward.shaped_military_advantage,
    Reward.shaped_combined,
    Reward.win_only,
]

# Change ONLY this line to switch the active reward function globally.
Reward.ACTIVE = Reward.terminal_only
