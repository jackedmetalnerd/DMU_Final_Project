"""
transition.py
=============
TransitionModel class for the two-player resource-and-combat MDP.

Encapsulates all transition logic extracted from GameEnv, including:
  - Stochastic combat resolution (precomputed binomial lookup)
  - Training action transitions (P1 and P2)
  - P2 policy application
  - Deterministic resource update

Actions are resolved SIMULTANEOUSLY: both players observe the original state,
choose their actions, and all effects are combined before resources update.

Public interface:
  transition(s, a) -> dict[State, float]   — single-step distribution, no matrices needed
  sample(s, a)     -> State                — sample one next state
  build_matrices() -> None                 — precompute sparse CSR matrices (VI only)
  update_P2_policy(π_P2) -> None           — recompose T after policy change
  T[action]        -> csr_matrix           — raises RuntimeError if not built
"""

import numpy as np
from scipy.stats import binom
from scipy.sparse import csr_matrix, diags
from tqdm import tqdm
from state import State
from action import Action

# ── Index formula (lexicographic state ordering) ──────────────────────────────
# Ordering: W1, M1, R1, W2, M2, R2, terminal — each 0-10, terminal 0-1
_STRIDES = np.array([
    11**5 * 2,   # W1
    11**4 * 2,   # M1
    11**3 * 2,   # R1
    11**2 * 2,   # W2
    11**1 * 2,   # M2
    11**0 * 2,   # R2
    1,           # terminal
], dtype=np.int64)


def _to_idx(W1, M1, R1, W2, M2, R2, t):
    """State index from field values. All args may be numpy arrays."""
    return (W1 * _STRIDES[0] + M1 * _STRIDES[1] + R1 * _STRIDES[2]
          + W2 * _STRIDES[3] + M2 * _STRIDES[4] + R2 * _STRIDES[5]
          + t  * _STRIDES[6])


class TransitionModel:
    ACTIONS_ALL = Action.ALL
    ACTIONS_P1  = Action.P1_ACTIONS
    ACTIONS_P2  = Action.P2_ACTIONS

    def __init__(self, states: list, state_index: dict, opponent_policy):
        self._states          = states
        self._state_index     = state_index
        self._opponent_policy = opponent_policy
        self._combat_lookup   = self._precompute_combat()
        self._T_base = None
        self._T_res  = None
        self.T = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def transition(self, s: State, a) -> dict:
        """Return {next_State: probability} for a single (s, a) pair.
        Uses simultaneous action semantics: P2's action is chosen from the
        original state s, and all effects resolve together.
        """
        if s.terminal:
            return {s: 1.0}

        p1_attacks = (a == Action.P1_ATTACK)
        a2         = self._opponent_policy(s)      # P2 observes original s
        p2_attacks = (a2 == Action.P2_ATTACK)

        # Combat from original (M1, M2) if either player attacks
        if p1_attacks or p2_attacks:
            combat_dist = {}
            for (nm1, nm2), p in self._combat_lookup[(s.M1, s.M2)].items():
                combat_dist[(nm1, nm2)] = combat_dist.get((nm1, nm2), 0.0) + p
        else:
            combat_dist = {(s.M1, s.M2): 1.0}

        # Training deltas computed from original state
        p1_deltas = self._training_deltas_p1(s, a)  if not p1_attacks else {(0, 0, 0): 1.0}
        p2_deltas = self._training_deltas_p2(s, a2) if not p2_attacks else {(0, 0, 0): 1.0}

        # Combine all effects simultaneously
        s2_dist = {}
        for (nm1, nm2), pc in combat_dist.items():
            for (dW1, dM1, dR1), pp1 in p1_deltas.items():
                for (dW2, dM2, dR2), pp2 in p2_deltas.items():
                    M1f  = max(nm1 + dM1, 0)
                    M2f  = max(nm2 + dM2, 0)
                    W1f  = min(s.W1 + dW1, 10)
                    W2f  = min(s.W2 + dW2, 10)
                    R1f  = s.R1 + dR1
                    R2f  = s.R2 + dR2
                    term = 1 if (M1f == 0 or M2f == 0) else 0
                    ns   = State(W1f, M1f, R1f, W2f, M2f, R2f, term)
                    s2_dist[ns] = s2_dist.get(ns, 0.0) + pc * pp1 * pp2

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
        probs /= probs.sum()
        idx = np.random.choice(len(states), p=probs)
        return states[idx]

    def build_matrices(self) -> None:
        """Precompute full sparse transition matrices T[a] for all P1 actions.
        Expensive (several minutes). Only needed by ValueIteration."""
        self._T_base = self._build_base()
        self._T_res  = self._build_resource_matrix()
        self.T = self._build_simultaneous_T(self._opponent_policy)

    def update_P2_policy(self, opponent_policy) -> None:
        """Recompose T after a P2 policy change. Requires matrices already built.
        opponent_policy may be a callable or an (n_states, 3) numpy array (mixed policy)."""
        self._opponent_policy = opponent_policy
        if self._T_base is not None:
            self.T = self._build_simultaneous_T(opponent_policy)

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
    
    def build_uniform_P2(self) -> None:
        n = len(self._states)
        sigma = np.full((n, 3), 1.0 / 3.0)
        self.update_P2_policy(sigma)

    def __getitem__(self, action):
        if not self.T:
            raise RuntimeError("Call build_matrices() before accessing T[action].")
        return self.T[action]

    # ── Training delta helpers (shared with JointTransitionModel) ─────────────

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

    # ── Single-step helper (still used by _apply_action path) ─────────────────

    @staticmethod
    def _apply_resources(s: State) -> State:
        """Deterministic resource gain: R' = min(R + W, 10) for each player."""
        if s.terminal:
            return s
        return State(s.W1, s.M1, min(s.R1 + s.W1, 10),
                     s.W2, s.M2, min(s.R2 + s.W2, 10), s.terminal)

    # ── Sparse matrix builders ────────────────────────────────────────────────

    def _extract_fields(self) -> dict:
        """Extract all state fields as numpy int arrays. Called once per build."""
        return {f: np.array([getattr(s, f) for s in self._states], dtype=np.int64)
                for f in ('W1', 'M1', 'R1', 'W2', 'M2', 'R2', 'terminal')}

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
        f = self._extract_fields()
        n = len(self._states)
        src = np.arange(n, dtype=np.int64)
        term = f['terminal'].astype(bool)
        T = {}

        # ── Training actions ──────────────────────────────────────────────────
        for a in (Action.P1_TRAIN_WORKERS, Action.P1_TRAIN_MARINES,
                  Action.P2_TRAIN_WORKERS, Action.P2_TRAIN_MARINES):
            if a == Action.P1_TRAIN_WORKERS:
                invalid  = term | (f['R1'] < 1) | (f['W1'] > 9)
                R_spent  = np.zeros_like(f['R1'])
                stat_ok  = np.minimum(f['W1'] + f['R1'], 10)
                ok_tgt   = _to_idx(stat_ok, f['M1'], R_spent, f['W2'], f['M2'], f['R2'], 0)
                fail_tgt = _to_idx(f['W1'], f['M1'], R_spent, f['W2'], f['M2'], f['R2'], 0)
            elif a == Action.P1_TRAIN_MARINES:
                invalid  = term | (f['R1'] < 1) | (f['M1'] > 9)
                R_spent  = np.zeros_like(f['R1'])
                stat_ok  = np.minimum(f['M1'] + f['R1'], 10)
                ok_tgt   = _to_idx(f['W1'], stat_ok,  R_spent, f['W2'], f['M2'], f['R2'], 0)
                fail_tgt = _to_idx(f['W1'], f['M1'],  R_spent, f['W2'], f['M2'], f['R2'], 0)
            elif a == Action.P2_TRAIN_WORKERS:
                invalid  = term | (f['R2'] < 1) | (f['W2'] > 9)
                R_spent  = np.zeros_like(f['R2'])
                stat_ok  = np.minimum(f['W2'] + f['R2'], 10)
                ok_tgt   = _to_idx(f['W1'], f['M1'], f['R1'], stat_ok,  f['M2'], R_spent, 0)
                fail_tgt = _to_idx(f['W1'], f['M1'], f['R1'], f['W2'],  f['M2'], R_spent, 0)
            else:  # P2_TRAIN_MARINES
                invalid  = term | (f['R2'] < 1) | (f['M2'] > 9)
                R_spent  = np.zeros_like(f['R2'])
                stat_ok  = np.minimum(f['M2'] + f['R2'], 10)
                ok_tgt   = _to_idx(f['W1'], f['M1'], f['R1'], f['W2'], stat_ok,  R_spent, 0)
                fail_tgt = _to_idx(f['W1'], f['M1'], f['R1'], f['W2'], f['M2'],  R_spent, 0)

            ok_tgt   = np.where(invalid, src, ok_tgt)
            fail_tgt = np.where(invalid, src, fail_tgt)
            p_ok     = np.where(invalid, 1.0, 0.9)
            p_fail   = np.where(invalid, 0.0, 0.1)

            mat = csr_matrix(
                (np.concatenate([p_ok, p_fail]),
                 (np.tile(src, 2), np.concatenate([ok_tgt, fail_tgt]))),
                shape=(n, n))
            mat.sum_duplicates()
            mat.eliminate_zeros()
            T[a] = mat

        # ── Attack action (same matrix for P1 and P2) ─────────────────────────
        group_key = f['M1'] * 11 + f['M2']
        non_term  = ~term
        all_rows, all_cols, all_vals = [], [], []

        for key in range(121):
            m1, m2 = divmod(key, 11)
            idxs = np.where(non_term & (group_key == key))[0]
            if len(idxs) == 0:
                continue
            for (nm1, nm2), p in self._combat_lookup[(m1, m2)].items():
                t_new = np.int64(nm1 == 0 or nm2 == 0)
                tgt = _to_idx(f['W1'][idxs], nm1, f['R1'][idxs],
                              f['W2'][idxs], nm2, f['R2'][idxs], t_new)
                all_rows.append(idxs)
                all_cols.append(tgt)
                all_vals.append(np.full(len(idxs), p, dtype=np.float64))

        term_idxs = np.where(term)[0]
        if len(term_idxs):
            all_rows.append(term_idxs)
            all_cols.append(term_idxs)
            all_vals.append(np.ones(len(term_idxs), dtype=np.float64))

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        vals = np.concatenate(all_vals)
        T_atk = csr_matrix((vals, (rows, cols)), shape=(n, n))
        T_atk.sum_duplicates()
        T_atk.eliminate_zeros()
        T[Action.P1_ATTACK] = T_atk
        T[Action.P2_ATTACK] = T_atk

        return T

    def _build_resource_matrix(self) -> csr_matrix:
        f = self._extract_fields()
        n = len(self._states)
        src = np.arange(n, dtype=np.int64)
        R1_new = np.minimum(f['R1'] + f['W1'], 10)
        R2_new = np.minimum(f['R2'] + f['W2'], 10)
        tgt = _to_idx(f['W1'], f['M1'], R1_new, f['W2'], f['M2'], R2_new, f['terminal'])
        tgt = np.where(f['terminal'] == 1, src, tgt)
        return csr_matrix((np.ones(n, dtype=np.float64), (src, tgt)), shape=(n, n))

    def _build_simultaneous_T(self, policy) -> dict:
        """Build T_sim[a1] for all P1 actions using simultaneous joint semantics.

        T_sim[a1] = Σ_a2  diag(w_a2) @ JointBase(a1, a2) @ T_res

        For a deterministic policy, w_a2[s] = 1 if π(s)=a2 else 0.
        For a mixed policy (ndarray sigma), w_a2[s] = sigma[s, a2_idx].
        Terminal states self-loop and are handled separately.
        """
        n = len(self._states)
        term_arr = np.array([s.terminal for s in self._states], dtype=np.float64)
        T_term   = diags(term_arr, format='csr')

        if isinstance(policy, np.ndarray):
            sigma = policy
            # Zero terminal rows (terminals don't take actions)
            sigma_nt = sigma * (1.0 - term_arr[:, None])
            weights = {a2: sigma_nt[:, i] for i, a2 in enumerate(self.ACTIONS_P2)}
        else:
            w = {a2: np.zeros(n, dtype=np.float64) for a2 in self.ACTIONS_P2}
            for i, s in enumerate(tqdm(self._states, desc="Applying P2 policy")):
                if not s.terminal:
                    w[policy(s)][i] = 1.0
            weights = w

        T_sim = {}
        for a1 in self.ACTIONS_P1:
            mat = csr_matrix((n, n), dtype=np.float64)
            for a2 in self.ACTIONS_P2:
                J = self._build_joint_base(a1, a2)
                mat = mat + diags(weights[a2], format='csr') @ J
            mat = (mat + T_term) @ self._T_res
            T_sim[a1] = mat.tocsr()
        return T_sim

    def _build_joint_base(self, a1: Action, a2: Action) -> csr_matrix:
        """Simultaneous joint base matrix for action pair (a1, a2), without resource update.

        Most pairs affect independent state fields so the product T_base[a1] @ T_base[a2]
        gives the correct simultaneous result. Three pairs require custom builders:
          (ATK, ATK) — one round of combat, not two
          (TM,  ATK) — P1 marine delta from original M1 combined with combat outcome
          (ATK, TM)  — P2 marine delta from original M2 combined with combat outcome
        """
        p1_atk = (a1 == Action.P1_ATTACK)
        p2_atk = (a2 == Action.P2_ATTACK)
        p1_tm  = (a1 == Action.P1_TRAIN_MARINES)
        p2_tm  = (a2 == Action.P2_TRAIN_MARINES)

        if p1_atk and p2_atk:
            return self._T_base[Action.P1_ATTACK]   # single combat round
        if p1_tm and p2_atk:
            return self._build_joint_tm_atk()
        if p1_atk and p2_tm:
            return self._build_joint_atk_tm()
        # All remaining pairs touch independent fields — product is correct
        return (self._T_base[a1] @ self._T_base[a2]).tocsr()

    def _build_joint_tm_atk(self) -> csr_matrix:
        """Joint base: P1 trains marines + P2 attacks simultaneously.

        Combat resolves from original (M1, M2). P1's marine training delta is
        computed from original M1 and added to the post-combat nm1.
        """
        f        = self._extract_fields()
        n        = len(self._states)
        term     = f['terminal'].astype(bool)
        non_term = ~term

        valid    = non_term & (f['R1'] >= 1) & (f['M1'] <= 9)
        dM1_ok   = np.where(valid, np.minimum(f['M1'] + f['R1'], 10) - f['M1'], 0)
        R1_after = np.where(valid, np.int64(0), f['R1'])

        all_rows, all_cols, all_vals = [], [], []
        group_key = f['M1'] * 11 + f['M2']

        for key in range(121):
            m1, m2 = divmod(key, 11)
            idxs = np.where(non_term & (group_key == key))[0]
            if len(idxs) == 0:
                continue
            for (nm1, nm2), p in self._combat_lookup[(m1, m2)].items():
                v      = valid[idxs]
                v_idxs = idxs[v]

                if len(v_idxs):
                    # Training success
                    M1f_ok = np.maximum(nm1 + dM1_ok[v_idxs], 0)
                    M2f    = np.full(len(v_idxs), nm2, dtype=np.int64)
                    t_ok   = ((M1f_ok == 0) | (M2f == 0)).astype(np.int64)
                    tgt_ok = _to_idx(f['W1'][v_idxs], M1f_ok, R1_after[v_idxs],
                                     f['W2'][v_idxs], M2f, f['R2'][v_idxs], t_ok)
                    all_rows.append(v_idxs);  all_cols.append(tgt_ok)
                    all_vals.append(np.full(len(v_idxs), p * 0.9))

                    # Training fail
                    M1f_fail = np.full(len(v_idxs), nm1, dtype=np.int64)
                    t_fail   = ((M1f_fail == 0) | (M2f == 0)).astype(np.int64)
                    tgt_fail = _to_idx(f['W1'][v_idxs], M1f_fail, R1_after[v_idxs],
                                       f['W2'][v_idxs], M2f, f['R2'][v_idxs], t_fail)
                    all_rows.append(v_idxs);  all_cols.append(tgt_fail)
                    all_vals.append(np.full(len(v_idxs), p * 0.1))

                # Invalid training (no training attempted)
                inv_idxs = idxs[~v]
                if len(inv_idxs):
                    M1f = np.full(len(inv_idxs), nm1, dtype=np.int64)
                    M2f = np.full(len(inv_idxs), nm2, dtype=np.int64)
                    t   = ((M1f == 0) | (M2f == 0)).astype(np.int64)
                    tgt = _to_idx(f['W1'][inv_idxs], M1f, f['R1'][inv_idxs],
                                  f['W2'][inv_idxs], M2f, f['R2'][inv_idxs], t)
                    all_rows.append(inv_idxs);  all_cols.append(tgt)
                    all_vals.append(np.full(len(inv_idxs), p))

        term_idxs = np.where(term)[0]
        if len(term_idxs):
            all_rows.append(term_idxs);  all_cols.append(term_idxs)
            all_vals.append(np.ones(len(term_idxs)))

        mat = csr_matrix((np.concatenate(all_vals),
                          (np.concatenate(all_rows), np.concatenate(all_cols))),
                         shape=(n, n))
        mat.sum_duplicates()
        mat.eliminate_zeros()
        return mat

    def _build_joint_atk_tm(self) -> csr_matrix:
        """Joint base: P1 attacks + P2 trains marines simultaneously.

        Combat resolves from original (M1, M2). P2's marine training delta is
        computed from original M2 and added to the post-combat nm2.
        """
        f        = self._extract_fields()
        n        = len(self._states)
        term     = f['terminal'].astype(bool)
        non_term = ~term

        valid    = non_term & (f['R2'] >= 1) & (f['M2'] <= 9)
        dM2_ok   = np.where(valid, np.minimum(f['M2'] + f['R2'], 10) - f['M2'], 0)
        R2_after = np.where(valid, np.int64(0), f['R2'])

        all_rows, all_cols, all_vals = [], [], []
        group_key = f['M1'] * 11 + f['M2']

        for key in range(121):
            m1, m2 = divmod(key, 11)
            idxs = np.where(non_term & (group_key == key))[0]
            if len(idxs) == 0:
                continue
            for (nm1, nm2), p in self._combat_lookup[(m1, m2)].items():
                v      = valid[idxs]
                v_idxs = idxs[v]

                if len(v_idxs):
                    # Training success
                    M1f    = np.full(len(v_idxs), nm1, dtype=np.int64)
                    M2f_ok = np.maximum(nm2 + dM2_ok[v_idxs], 0)
                    t_ok   = ((M1f == 0) | (M2f_ok == 0)).astype(np.int64)
                    tgt_ok = _to_idx(f['W1'][v_idxs], M1f, f['R1'][v_idxs],
                                     f['W2'][v_idxs], M2f_ok, R2_after[v_idxs], t_ok)
                    all_rows.append(v_idxs);  all_cols.append(tgt_ok)
                    all_vals.append(np.full(len(v_idxs), p * 0.9))

                    # Training fail
                    M2f_fail = np.full(len(v_idxs), nm2, dtype=np.int64)
                    t_fail   = ((M1f == 0) | (M2f_fail == 0)).astype(np.int64)
                    tgt_fail = _to_idx(f['W1'][v_idxs], M1f, f['R1'][v_idxs],
                                       f['W2'][v_idxs], M2f_fail, R2_after[v_idxs], t_fail)
                    all_rows.append(v_idxs);  all_cols.append(tgt_fail)
                    all_vals.append(np.full(len(v_idxs), p * 0.1))

                # Invalid training
                inv_idxs = idxs[~v]
                if len(inv_idxs):
                    M1f = np.full(len(inv_idxs), nm1, dtype=np.int64)
                    M2f = np.full(len(inv_idxs), nm2, dtype=np.int64)
                    t   = ((M1f == 0) | (M2f == 0)).astype(np.int64)
                    tgt = _to_idx(f['W1'][inv_idxs], M1f, f['R1'][inv_idxs],
                                  f['W2'][inv_idxs], M2f, f['R2'][inv_idxs], t)
                    all_rows.append(inv_idxs);  all_cols.append(tgt)
                    all_vals.append(np.full(len(inv_idxs), p))

        term_idxs = np.where(term)[0]
        if len(term_idxs):
            all_rows.append(term_idxs);  all_cols.append(term_idxs)
            all_vals.append(np.ones(len(term_idxs)))

        mat = csr_matrix((np.concatenate(all_vals),
                          (np.concatenate(all_rows), np.concatenate(all_cols))),
                         shape=(n, n))
        mat.sum_duplicates()
        mat.eliminate_zeros()
        return mat
