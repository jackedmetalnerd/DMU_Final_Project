"""
transition.py
=============
TransitionModel class for the two-player resource-and-combat MDP.

Encapsulates all transition logic extracted from GameEnv, including:
  - Stochastic combat resolution (precomputed binomial lookup)
  - Training action transitions (P1 and P2)
  - P2 policy application
  - Deterministic resource update

Public interface:
  transition(s, a) -> dict[State, float]   — single-step distribution, no matrices needed
  sample(s, a)     -> State                — sample one next state
  build_matrices() -> None                 — precompute sparse CSR matrices (VI only)
  update_P2_policy(π_P2) -> None           — recompose T after policy change
  T[action]        -> csr_matrix           — raises RuntimeError if not built
"""

import numpy as np
from scipy.stats import binom
from scipy.sparse import csr_matrix
from tqdm import tqdm
from state import State
from action import Action


class TransitionModel:
    ACTIONS_ALL = Action.ALL
    ACTIONS_P1  = Action.P1_ACTIONS

    def __init__(self, states: list, state_index: dict, opponent_policy):
        self._states        = states
        self._state_index   = state_index
        self._opponent_policy = opponent_policy
        # Combat lookup: 121-entry dict, fast to compute, always needed
        # by both transition() (for attack steps) and build_matrices().
        self._combat_lookup = self._precompute_combat()
        # Sparse matrices deferred — call build_matrices() when needed.
        self._T_base = None
        self._T_res  = None
        self._T_P2   = None
        self.T = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def transition(self, s: State, a) -> dict:
        """Return {next_State: probability} for a single (s, a) pair.
        Works without precomputed matrices. Suitable for QL and MCTS rollouts.
        Order of operations mirrors the matrix chain: P1 action → P2 action → resources.
        """
        if s.terminal:
            return {s: 1.0}

        # Step 1: P1's action
        s1_dist = self._apply_action(s, a)

        # Step 2: P2's action (based on intermediate state after P1 acts)
        s2_dist = {}
        for s1, p1 in s1_dist.items():
            if s1.terminal:
                s2_dist[s1] = s2_dist.get(s1, 0.0) + p1
            else:
                a2 = self._opponent_policy(s1)
                for s2, p2 in self._apply_action(s1, a2).items():
                    s2_dist[s2] = s2_dist.get(s2, 0.0) + p1 * p2

        # Step 3: Deterministic resource update
        result = {}
        for s2, p in s2_dist.items():
            sf = self._apply_resources(s2)
            result[sf] = result.get(sf, 0.0) + p

        return result

    def sample(self, s: State, a) -> State:
        """Sample one next state from transition(s, a)."""
        dist = self.transition(s, a)
        states = list(dist.keys())
        probs  = np.array(list(dist.values()), dtype=np.float64)
        probs /= probs.sum()  # normalize to guard against floating-point drift
        idx = np.random.choice(len(states), p=probs)
        return states[idx]

    def build_matrices(self) -> None:
        """Precompute full sparse transition matrices T[a] for all P1 actions.
        Expensive (several minutes). Only needed by ValueIteration for T[a] @ V
        matrix-vector products. QL and MCTS should use transition() / sample()."""
        self._T_base = self._build_base()
        self._T_res  = self._build_resource_matrix()
        self._T_P2   = self._build_P2_matrix(self._opponent_policy)
        self.T = {a: self._T_base[a] @ self._T_P2 @ self._T_res
                  for a in self.ACTIONS_P1}

    def update_P2_policy(self, opponent_policy) -> None:
        """Recompose T after a P2 policy change. Requires matrices already built."""
        self._opponent_policy = opponent_policy
        if self._T_base is not None:
            self._T_P2 = self._build_P2_matrix(opponent_policy)
            self.T = {a: self._T_base[a] @ self._T_P2 @ self._T_res
                      for a in self.ACTIONS_P1}

    def valid_act(self, a, s) -> bool:
        """Return True if action a is available from state s."""
        if s.terminal:
            return False
        if a == Action.P1_TRAIN_WORKERS:
            return s.R1 > 0 and s.W1 < 10
        if a == Action.P1_TRAIN_MARINES:
            return s.R1 > 0 and s.M1 < 10
        if a == Action.P2_TRAIN_WORKERS:
            return s.R2 > 0 and s.W2 < 10
        if a == Action.P2_TRAIN_MARINES:
            return s.R2 > 0 and s.M2 < 10
        return True  # attacks always valid

    def __getitem__(self, action):
        if not self.T:
            raise RuntimeError("Call build_matrices() before accessing T[action].")
        return self.T[action]

    # ── Single-step helpers (used by transition()) ────────────────────────────

    def _apply_action(self, s: State, a) -> dict:
        """Return {next_State: prob} for one player's action from state s."""
        if a == Action.P1_ATTACK or a == Action.P2_ATTACK:
            result = {}
            for (nm1, nm2), p in self._combat_lookup[(s.M1, s.M2)].items():
                term = 1 if (nm1 == 0 or nm2 == 0) else 0
                sp = State(s.W1, nm1, s.R1, s.W2, nm2, s.R2, term)
                result[sp] = result.get(sp, 0.0) + p
            return result

        invalid = (
            (a == Action.P1_TRAIN_WORKERS and (s.R1 < 1 or s.W1 > 9)) or
            (a == Action.P1_TRAIN_MARINES and (s.R1 < 1 or s.M1 > 9)) or
            (a == Action.P2_TRAIN_WORKERS and (s.R2 < 1 or s.W2 > 9)) or
            (a == Action.P2_TRAIN_MARINES and (s.R2 < 1 or s.M2 > 9))
        )
        if invalid:
            return {s: 1.0}

        dW1 = (min(s.W1 + s.R1, 10) - s.W1) if a == Action.P1_TRAIN_WORKERS else 0
        dM1 = (min(s.M1 + s.R1, 10) - s.M1) if a == Action.P1_TRAIN_MARINES else 0
        dR1 = -s.R1 if a == Action.P1_TRAIN_WORKERS or a == Action.P1_TRAIN_MARINES else 0
        dW2 = (min(s.W2 + s.R2, 10) - s.W2) if a == Action.P2_TRAIN_WORKERS else 0
        dM2 = (min(s.M2 + s.R2, 10) - s.M2) if a == Action.P2_TRAIN_MARINES else 0
        dR2 = -s.R2 if a == Action.P2_TRAIN_WORKERS or a == Action.P2_TRAIN_MARINES else 0

        sp_ok   = State(s.W1+dW1, s.M1+dM1, s.R1+dR1,
                        s.W2+dW2, s.M2+dM2, s.R2+dR2, 0)
        sp_fail = State(s.W1,     s.M1,     s.R1+dR1,
                        s.W2,     s.M2,     s.R2+dR2, 0)
        return {sp_ok: 0.9, sp_fail: 0.1}

    @staticmethod
    def _apply_resources(s: State) -> State:
        """Deterministic resource gain: R' = min(R + W, 10) for each player."""
        if s.terminal:
            return s
        return State(s.W1, s.M1, min(s.R1 + s.W1, 10),
                     s.W2, s.M2, min(s.R2 + s.W2, 10), s.terminal)

    # ── Sparse matrix builders ────────────────────────────────────────────────

    def _precompute_combat(self) -> dict:
        lookup = {}
        for m1 in range(11):
            for m2 in range(11):
                if m1 == 0 and m2 == 0:
                    lookup[(m1, m2)] = {(0, 0): 1.0}
                    continue
                outcomes = {}
                for l1 in range(m1 + 1):
                    for l2 in range(m2 + 1):
                        p = binom.pmf(l1, m2, 0.5) * binom.pmf(l2, m1, 0.5)
                        key = (max(m1 - l1, 0), max(m2 - l2, 0))
                        outcomes[key] = outcomes.get(key, 0) + p
                lookup[(m1, m2)] = outcomes
        return lookup

    def _build_base(self) -> dict:
        states, state_index = self._states, self._state_index
        n = len(states)
        T = {}
        for a in self.ACTIONS_ALL:
            rows, cols, vals = [], [], []
            for s_idx, s in enumerate(tqdm(states, desc=f"Building T[{a}]")):
                if s.terminal:
                    rows.append(s_idx); cols.append(s_idx); vals.append(1.0)
                    continue
                if a == Action.P1_ATTACK or a == Action.P2_ATTACK:
                    for (nm1, nm2), p in self._combat_lookup[(s.M1, s.M2)].items():
                        term = 1 if (nm1 == 0 or nm2 == 0) else 0
                        sp = State(s.W1, nm1, s.R1, s.W2, nm2, s.R2, term)
                        rows.append(s_idx); cols.append(state_index[sp]); vals.append(p)
                else:
                    invalid = (
                        (a == Action.P1_TRAIN_WORKERS and (s.R1 < 1 or s.W1 > 9)) or
                        (a == Action.P1_TRAIN_MARINES and (s.R1 < 1 or s.M1 > 9)) or
                        (a == Action.P2_TRAIN_WORKERS and (s.R2 < 1 or s.W2 > 9)) or
                        (a == Action.P2_TRAIN_MARINES and (s.R2 < 1 or s.M2 > 9))
                    )
                    if invalid:
                        rows.append(s_idx); cols.append(s_idx); vals.append(1.0)
                    else:
                        dW1 = min(s.W1+s.R1,10)-s.W1 if a == Action.P1_TRAIN_WORKERS else 0
                        dM1 = min(s.M1+s.R1,10)-s.M1 if a == Action.P1_TRAIN_MARINES else 0
                        dR1 = -s.R1 if a == Action.P1_TRAIN_WORKERS or a == Action.P1_TRAIN_MARINES else 0
                        dW2 = min(s.W2+s.R2,10)-s.W2 if a == Action.P2_TRAIN_WORKERS else 0
                        dM2 = min(s.M2+s.R2,10)-s.M2 if a == Action.P2_TRAIN_MARINES else 0
                        dR2 = -s.R2 if a == Action.P2_TRAIN_WORKERS or a == Action.P2_TRAIN_MARINES else 0
                        sp_ok   = State(s.W1+dW1, s.M1+dM1, s.R1+dR1,
                                        s.W2+dW2, s.M2+dM2, s.R2+dR2, 0)
                        sp_fail = State(s.W1,     s.M1,     s.R1+dR1,
                                        s.W2,     s.M2,     s.R2+dR2, 0)
                        rows.append(s_idx); cols.append(state_index[sp_ok]);   vals.append(0.9)
                        rows.append(s_idx); cols.append(state_index[sp_fail]); vals.append(0.1)
            T[a] = csr_matrix((vals, (rows, cols)), shape=(n, n))
        return T

    def _build_P2_matrix(self, opponent_policy) -> csr_matrix:
        states = self._states
        n = len(states)
        rows, cols, vals = [], [], []
        a2s = {a: [] for a in self.ACTIONS_ALL}
        for s_idx, s in enumerate(tqdm(states, desc="Applying P2 policy")):
            a2s[opponent_policy(s)].append(s_idx)
        for a, idxs in a2s.items():
            if not idxs:
                continue
            idxs_arr = np.array(idxs)
            sub = self._T_base[a][idxs_arr, :].tocoo()
            rows.extend(idxs_arr[sub.row].tolist())
            cols.extend(sub.col.tolist())
            vals.extend(sub.data.tolist())
        T_P2 = csr_matrix((vals, (rows, cols)), shape=(n, n))
        T_P2.eliminate_zeros()
        return T_P2

    def _build_resource_matrix(self) -> csr_matrix:
        states, state_index = self._states, self._state_index
        n = len(states)
        rows, cols, vals = [], [], []
        for s_idx, s in enumerate(tqdm(states, desc="Building resource matrix")):
            if s.terminal:
                rows.append(s_idx); cols.append(s_idx); vals.append(1.0)
            else:
                sp = State(s.W1, s.M1, min(s.R1 + s.W1, 10),
                           s.W2, s.M2, min(s.R2 + s.W2, 10), s.terminal)
                rows.append(s_idx); cols.append(state_index[sp]); vals.append(1.0)
        return csr_matrix((vals, (rows, cols)), shape=(n, n))
