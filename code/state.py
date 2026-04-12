"""
state.py
========
State class for the two-player resource-and-combat MDP.

Replaces the namedtuple previously defined in game_env.py. Adds game-logic
predicates (is_win, is_loss, winner, terminal_value) that were previously
scattered across reward.py, game_env.py, and compare_rewards.py.

Fields: W1, M1, R1, W2, M2, R2, terminal — all int, range 0-10 (terminal 0/1).
"""


class State:
    __slots__ = ('W1', 'M1', 'R1', 'W2', 'M2', 'R2', 'terminal')

    def __init__(self, W1, M1, R1, W2, M2, R2, terminal):
        object.__setattr__(self, 'W1', W1)
        object.__setattr__(self, 'M1', M1)
        object.__setattr__(self, 'R1', R1)
        object.__setattr__(self, 'W2', W2)
        object.__setattr__(self, 'M2', M2)
        object.__setattr__(self, 'R2', R2)
        object.__setattr__(self, 'terminal', terminal)

    def __setattr__(self, name, value):
        raise AttributeError("State is immutable")

    # ── Game-logic predicates ─────────────────────────────────────────────────

    def is_terminal(self) -> bool:
        return bool(self.terminal)

    def is_win(self) -> bool:
        """P1 wins: has marines, P2 has none."""
        return bool(self.M1 > 0 and self.M2 == 0)

    def is_loss(self) -> bool:
        """Terminal state that is not a P1 win."""
        return bool(self.terminal and not self.is_win())

    def winner(self) -> str:
        """Returns 'P1', 'P2', or 'Draw'."""
        if self.M1 > 0 and self.M2 == 0:
            return 'P1'
        if self.M2 > 0 and self.M1 == 0:
            return 'P2'
        return 'Draw'

    def terminal_value(self) -> float:
        """Fixed ±1 terminal value used by VI for terminal pinning."""
        return 1.0 if self.is_win() else -1.0

    # ── Equality and hashing ──────────────────────────────────────────────────
    # Hash matches the old namedtuple so validate.py set comparisons pass.

    def __eq__(self, other) -> bool:
        try:
            return (self.W1 == other.W1 and self.M1 == other.M1 and
                    self.R1 == other.R1 and self.W2 == other.W2 and
                    self.M2 == other.M2 and self.R2 == other.R2 and
                    self.terminal == other.terminal)
        except AttributeError:
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.W1, self.M1, self.R1, self.W2, self.M2, self.R2, self.terminal))

    # ── Iteration (namedtuple compatibility) ──────────────────────────────────

    def __iter__(self):
        yield self.W1
        yield self.M1
        yield self.R1
        yield self.W2
        yield self.M2
        yield self.R2
        yield self.terminal

    def __repr__(self) -> str:
        return (f"State(W1={self.W1}, M1={self.M1}, R1={self.R1}, "
                f"W2={self.W2}, M2={self.M2}, R2={self.R2}, terminal={self.terminal})")

    # ── State space builder ───────────────────────────────────────────────────

    @classmethod
    def build_space(cls) -> list:
        """Return the full enumerated state space (all valid field combinations).
        Each variable ∈ [0, 10], terminal ∈ {0, 1} → 11^6 × 2 = 3,543,122 states.
        """
        return [
            cls(W1, M1, R1, W2, M2, R2, terminal)
            for W1 in range(11) for M1 in range(11) for R1 in range(11)
            for W2 in range(11) for M2 in range(11) for R2 in range(11)
            for terminal in range(2)
        ]
