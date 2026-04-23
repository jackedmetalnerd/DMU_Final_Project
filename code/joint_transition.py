"""
joint_transition.py
===================
JointTransitionModel: extends TransitionModel for true 2-player Markov games.

The parent TransitionModel bakes P2's policy into transition(s, a1) by sampling
a2 from a fixed opponent_policy. JointTransitionModel adds joint_transition and
joint_sample that take both player actions explicitly, without any fixed policy.

Inherits from TransitionModel to reuse:
  _apply_action(s, a)    -- single-player action distribution (works for P1 or P2)
  _apply_resources(s)    -- deterministic resource update
  _precompute_combat()   -- binomial combat lookup table (symmetric, built once)
  _combat_lookup         -- dict[(m1, m2)] -> dict[(nm1, nm2), prob]
  valid_act(a, s)        -- action validity check for either player

Does NOT use the parent's transition(s, a1) or build_matrices() -- those embed
a fixed P2 policy. A dummy policy is passed to the parent constructor to satisfy
its signature; it raises RuntimeError if accidentally called.

Sequential action order (consistent with TransitionModel convention):
  P1 acts first → P2 acts on post-P1 state → resources update deterministically
"""

import numpy as np
from transition import TransitionModel
from state import State
from action import Action


class JointTransitionModel(TransitionModel):
    """Transition model for 2-player Markov games with explicit joint actions.

    Parameters
    ----------
    states      : list[State]    -- full state space (from State.build_space())
    state_index : dict[State, int]
    """

    def __init__(self, states: list, state_index: dict):
        def _dummy_policy(s: State) -> Action:
            raise RuntimeError(
                "JointTransitionModel: the inherited transition(s, a1) calls "
                "_opponent_policy internally and must not be used here. "
                "Use joint_transition(s, a1, a2) instead."
            )
        super().__init__(states, state_index, opponent_policy=_dummy_policy)

    # ── Public interface ──────────────────────────────────────────────────────

    def joint_transition(self, s: State, a1: Action, a2: Action) -> dict:
        """Return {next_State: probability} for explicit joint action (a1, a2).

        Steps (mirror parent's transition but with a2 explicit):
          1. Apply P1's action a1 to s         -> s1_dist
          2. Apply P2's action a2 to each s1   -> s2_dist  (explicit, not from policy)
          3. Deterministic resource update      -> result

        If s is already terminal, returns {s: 1.0} (absorbing).
        If s1 is terminal after P1's action, P2's action is skipped for that branch.

        Simultaneous-attack note: when both players attack, they are applied
        sequentially (P1 first). P2's attack is resolved on the post-P1-combat
        state. This matches the existing TransitionModel convention.
        """
        if s.terminal:
            return {s: 1.0}

        # Step 1: P1 action
        s1_dist = self._apply_action(s, a1)

        # Step 2: P2 action (explicit a2, not sampled from a policy)
        s2_dist = {}
        for s1, p1 in s1_dist.items():
            if s1.terminal:
                # P1 already won/lost; P2 cannot act
                s2_dist[s1] = s2_dist.get(s1, 0.0) + p1
            else:
                for s2, p2 in self._apply_action(s1, a2).items():
                    s2_dist[s2] = s2_dist.get(s2, 0.0) + p1 * p2

        # Step 3: Deterministic resource update
        result = {}
        for s2, p in s2_dist.items():
            sf = self._apply_resources(s2)
            result[sf] = result.get(sf, 0.0) + p

        return result

    def joint_sample(self, s: State, a1: Action, a2: Action) -> State:
        """Sample one next state from joint_transition(s, a1, a2)."""
        dist = self.joint_transition(s, a1, a2)
        states_list = list(dist.keys())
        probs = np.array(list(dist.values()), dtype=np.float64)
        probs /= probs.sum()  # guard against floating-point drift
        idx = np.random.choice(len(states_list), p=probs)
        return states_list[idx]
