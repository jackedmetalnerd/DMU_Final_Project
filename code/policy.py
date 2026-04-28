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
BeliefPolicy    - wrapper for belief-based policies
POMCPPolicy     - lazy policy that runs POMCP planning on each call

All concrete policies are callable objects, so they work wherever a plain
callable is expected (GameEnv, simulate, update_P2_policy, etc.).
"""

from abc import ABC, abstractmethod
from action import Action
from state import State
import numpy as np


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

class BeliefPolicy(Policy):
    # POMDP policy: takes a belief vector instead of a state
    def __init__(self, solver):
        self._solver = solver

    def __call__(self, b: np.ndarray) -> Action:
        return self._solver.get_action(b)


class POMCPPolicy(Policy):
    # Lazy belief policy for POMCP - tracks action/obs history
    def __init__(self, solver):
        self._solver = solver

    def __call__(self, b: np.ndarray) -> Action:
        return self._solver.get_action(b)

    def update(self, a, o):
        self._solver.update(a, o)

    def reset(self):
        self._solver.reset()


# ── Policy I/O ────────────────────────────────────────────────────────────────

def save_policy(policy, path: str, states: list, actions: list) -> None:
    """Save a deterministic policy to a .npy file as an array of action indices.

    Parameters
    ----------
    policy  : callable Policy (DictPolicy, FunctionPolicy, etc.)
    path    : file path, e.g. 'vi_policy_p1.npy'
    states  : ordered list of States (from State.build_space())
    actions : ordered list of Actions that index the array columns
    """
    indices = np.array([actions.index(policy(s)) for s in states], dtype=np.int16)
    np.save(path, indices)
    print(f"Policy saved to {path}  ({len(states)} states)")


def load_policy(path: str, states: list, actions: list) -> FunctionPolicy:
    """Load a deterministic policy saved by save_policy(). Returns a FunctionPolicy.

    Parameters
    ----------
    path    : file path, e.g. 'vi_policy_p1.npy'
    states  : ordered list of States (must match the list used when saving)
    actions : ordered list of Actions (must match the list used when saving)
    """
    indices     = np.load(path)
    state_index = {s: i for i, s in enumerate(states)}
    return FunctionPolicy(lambda s, idx=indices, si=state_index, a=actions: a[idx[si[s]]])


def save_mixed_policy(sigma: np.ndarray, path: str) -> None:
    """Save an FSP mixed policy sigma array (n_states, 3) to a .npy file."""
    np.save(path, sigma)
    print(f"Mixed policy saved to {path}  shape={sigma.shape}")


def load_mixed_policy(path: str, states: list, actions: list):
    """Load a mixed policy saved by save_mixed_policy(). Returns a MixedPolicy-like callable.

    The returned object samples an action according to the stored probabilities.
    """
    sigma       = np.load(path)
    state_index = {s: i for i, s in enumerate(states)}

    class _LoadedMixedPolicy(Policy):
        def __call__(self, s: State) -> Action:
            probs = sigma[state_index[s]]
            return actions[np.random.choice(len(actions), p=probs)]

    return _LoadedMixedPolicy()
