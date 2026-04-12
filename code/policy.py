"""
policy.py
=========
Policy ABC and concrete subclasses for the two-player MDP.

Classes
-------
Policy          — abstract base; callable interface __call__(s) -> Action
DictPolicy      — wraps a pre-computed {State -> Action} dict (VI, QL output)
FunctionPolicy  — wraps a plain callable (hand-coded opponent policies)
SymmetricPolicy — mirrors a P1 Policy to P2 by swapping player fields;
                  replaces the P2_policy_converter() function in policies.py
MCTSPolicy      — lazy policy that runs MCTS planning on each call

All concrete policies are callable objects, so they work wherever a plain
callable is expected (GameEnv, simulate, update_P2_policy, etc.).
"""

from abc import ABC, abstractmethod
from action import Action
from state import State


class Policy(ABC):
    """Abstract base class for all MDP policies."""

    @abstractmethod
    def __call__(self, s: State) -> Action:
        """Return an action for state s."""


class DictPolicy(Policy):
    """Wraps a pre-computed {State -> Action} dict.

    Used by ValueIteration and QLearning to expose their solved policies.
    Supports dict-style lookup (policy[s]) in addition to callable (policy(s))
    so existing code that checks isinstance(policy, dict) still works.
    """

    def __init__(self, policy_dict: dict):
        self._dict = policy_dict

    def __call__(self, s: State) -> Action:
        return self._dict[s]

    def __getitem__(self, s: State) -> Action:
        return self._dict[s]

    def __contains__(self, s: State) -> bool:
        return s in self._dict

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)


class FunctionPolicy(Policy):
    """Wraps a plain callable.

    Used to adapt hand-coded opponent policies (alternating_training,
    alternating_training_attack) into the Policy ABC.
    """

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, s: State) -> Action:
        return self._fn(s)


class SymmetricPolicy(Policy):
    """Mirrors a P1 Policy to P2 by swapping player perspective.

    Replaces the P2_policy_converter() function in policies.py.
    Constructs an inverted state (P1 and P2 fields swapped), queries the
    underlying P1 policy, then maps the resulting P1 action to its P2 equivalent.
    """

    ACTION_MAP = {
        Action.P1_TRAIN_WORKERS: Action.P2_TRAIN_WORKERS,
        Action.P1_TRAIN_MARINES: Action.P2_TRAIN_MARINES,
        Action.P1_ATTACK:        Action.P2_ATTACK,
    }

    def __init__(self, p1_policy):
        """p1_policy may be a Policy subclass, a dict, or any callable."""
        self._p1_policy = p1_policy

    def __call__(self, s: State) -> Action:
        s_inv = State(W1=s.W2, M1=s.M2, R1=s.R2,
                      W2=s.W1, M2=s.M1, R2=s.R1, terminal=s.terminal)
        if isinstance(self._p1_policy, dict):
            a_p1 = self._p1_policy[s_inv]
        else:
            a_p1 = self._p1_policy(s_inv)
        return self.ACTION_MAP[a_p1]


class MCTSPolicy(Policy):
    """Lazy policy: runs MCTS planning on each call.

    Wraps an MCTSSolver instance and calls get_action(s) on demand.
    Returned by MCTSSolver so callers interact with a uniform Policy interface.
    """

    def __init__(self, solver):
        self._solver = solver

    def __call__(self, s: State) -> Action:
        return self._solver.get_action(s)
