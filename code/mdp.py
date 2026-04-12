"""
mdp.py
======
MDP base class bundling the five formal MDP components: S, A, T, R, γ.

GameEnv inherits from this class and adds the RL/simulation interface
(act, observe, reset, simulate, update_P2_policy).

Attributes
----------
states          : list[State]           — ordered state space
actions         : list[Action]          — available actions for the agent (P1)
transition_model: TransitionModel       — transition logic; T[a] → sparse matrix
reward          : Reward                — reward function instance
gamma           : float                 — discount factor
state_index     : dict[State, int]      — maps each state to its position in `states`

Properties (backward-compat aliases used by solvers and validate.py)
---------------------------------------------------------------------
S   → states
A   → actions
T   → transition_model.T   (dict of sparse matrices, empty until build_matrices())
R   → reward.build_vector(states)
γ   → gamma
S_index → state_index
"""


class MDP:
    """Pure mathematical MDP: bundles (S, A, T, R, γ) as a formal structure."""

    def __init__(self, states: list, actions: list, transition_model, reward, gamma: float = 0.95):
        self.states           = states
        self.actions          = actions
        self.transition_model = transition_model
        self.reward           = reward
        self.gamma            = gamma
        self.state_index      = {s: i for i, s in enumerate(states)}

    # ── Standard MDP component aliases ───────────────────────────────────────

    @property
    def S(self):
        return self.states

    @property
    def A(self):
        return self.actions

    @property
    def T(self):
        """Sparse transition matrices (dict). Empty until build_matrices() is called."""
        return self.transition_model.T

    @property
    def R(self):
        """Reward vector over all states. Cached inside Reward after first call."""
        return self.reward.build_vector(self.states)

    @property
    def γ(self):
        return self.gamma

    @property
    def S_index(self):
        return self.state_index
