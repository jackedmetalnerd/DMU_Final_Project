"""
joint_transition.py
===================
JointTransitionModel: extends TransitionModel for true 2-player Markov games.

The parent TransitionModel bakes P2's policy into transition(s, a1) by sampling
a2 from a fixed opponent_policy. JointTransitionModel adds joint_transition and
joint_sample that take both player actions explicitly, without any fixed policy.

Inherits from TransitionModel to reuse:
  _apply_resources(s)    -- deterministic resource update
  _precompute_combat()   -- binomial combat lookup table (symmetric, built once)
  _combat_lookup         -- dict[(m1, m2)] -> dict[(nm1, nm2), prob]

Actions are resolved SIMULTANEOUSLY (standard Markov game semantics):
  - Combat (if either player attacks) uses original (M1, M2)
  - Training uses original resources and troop counts
  - Effects are combined additively: M1_final = M1_post_combat + M1_trained
  - Resources update deterministically after all actions resolve
"""

import numpy as np
from transition import TransitionModel
from state import State
from action import Action


class JointTransitionModel(TransitionModel):
    """Transition model for 2-player Markov games with simultaneous joint actions.

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
        """Return {next_State: probability} for simultaneous joint action (a1, a2).

        All effects are evaluated from the original state s:
          1. Combat (if either player attacks): one round from original (M1, M2)
          2. Training deltas: computed from original resources/troops
          3. Effects combined: M1_final = M1_post_combat + dM1_train, etc.
          4. Resource update deterministically

        If s is already terminal, returns {s: 1.0} (absorbing).
        """
        if s.terminal:
            return {s: 1.0}

        p1_attacks = (a1 == Action.P1_ATTACK)
        p2_attacks = (a2 == Action.P2_ATTACK)

        # ── Step 1: Combat from original state (one round) ────────────────────
        if p1_attacks or p2_attacks:
            combat_dist = {}
            for (nm1, nm2), p in self._combat_lookup[(s.M1, s.M2)].items():
                combat_dist[(nm1, nm2)] = combat_dist.get((nm1, nm2), 0.0) + p
        else:
            combat_dist = {(s.M1, s.M2): 1.0}

        # ── Step 2: Training deltas from original state ───────────────────────
        p1_deltas = self._training_deltas_p1(s, a1) if not p1_attacks else {(0, 0, 0): 1.0}
        p2_deltas = self._training_deltas_p2(s, a2) if not p2_attacks else {(0, 0, 0): 1.0}

        # ── Step 3: Combine all effects ───────────────────────────────────────
        s2_dist = {}
        for (nm1, nm2), pc in combat_dist.items():
            for (dW1, dM1, dR1), pp1 in p1_deltas.items():
                for (dW2, dM2, dR2), pp2 in p2_deltas.items():
                    M1f = max(nm1 + dM1, 0)
                    M2f = max(nm2 + dM2, 0)
                    W1f = min(s.W1 + dW1, 10)
                    W2f = min(s.W2 + dW2, 10)
                    R1f = s.R1 + dR1
                    R2f = s.R2 + dR2
                    term = 1 if (M1f == 0 or M2f == 0) else 0
                    ns = State(W1f, M1f, R1f, W2f, M2f, R2f, term)
                    s2_dist[ns] = s2_dist.get(ns, 0.0) + pc * pp1 * pp2

        # ── Step 4: Deterministic resource update ─────────────────────────────
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
        probs /= probs.sum()
        idx = np.random.choice(len(states_list), p=probs)
        return states_list[idx]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _training_deltas_p1(self, s: State, a: Action) -> dict:
        """Return {(dW1, dM1, dR1): prob} for P1's training action from state s."""
        if a == Action.P1_TRAIN_WORKERS:
            if s.R1 < 1 or s.W1 > 9:
                return {(0, 0, 0): 1.0}
            dW1 = min(s.W1 + s.R1, 10) - s.W1
            dR1 = -s.R1
            return {(dW1, 0, dR1): 0.9, (0, 0, dR1): 0.1}
        elif a == Action.P1_TRAIN_MARINES:
            if s.R1 < 1 or s.M1 > 9:
                return {(0, 0, 0): 1.0}
            dM1 = min(s.M1 + s.R1, 10) - s.M1
            dR1 = -s.R1
            return {(0, dM1, dR1): 0.9, (0, 0, dR1): 0.1}
        return {(0, 0, 0): 1.0}

    def _training_deltas_p2(self, s: State, a: Action) -> dict:
        """Return {(dW2, dM2, dR2): prob} for P2's training action from state s."""
        if a == Action.P2_TRAIN_WORKERS:
            if s.R2 < 1 or s.W2 > 9:
                return {(0, 0, 0): 1.0}
            dW2 = min(s.W2 + s.R2, 10) - s.W2
            dR2 = -s.R2
            return {(dW2, 0, dR2): 0.9, (0, 0, dR2): 0.1}
        elif a == Action.P2_TRAIN_MARINES:
            if s.R2 < 1 or s.M2 > 9:
                return {(0, 0, 0): 1.0}
            dM2 = min(s.M2 + s.R2, 10) - s.M2
            dR2 = -s.R2
            return {(0, dM2, dR2): 0.9, (0, 0, dR2): 0.1}
        return {(0, 0, 0): 1.0}
