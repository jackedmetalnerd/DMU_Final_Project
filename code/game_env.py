import numpy as np
from collections import namedtuple
from scipy.stats import binom
from scipy.sparse import csr_matrix
from tqdm import tqdm

State = namedtuple('State', ['W1', 'M1', 'R1', 'W2', 'M2', 'R2', 'terminal'])


class GameEnv:
    """MDP environment for the two-player resource-and-combat game."""

    ACTIONS_P1 = ['P1_train_workers', 'P1_train_marines', 'P1_attack']
    ACTIONS_ALL = ['P1_train_workers', 'P1_train_marines', 'P1_attack',
                   'P2_train_workers', 'P2_train_marines', 'P2_attack']

    def __init__(self, π_P2, s_init=None, γ=0.95):
        if s_init is None:
            s_init = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

        self.s_init = s_init
        self.s = s_init
        self.π_P2 = π_P2
        self.γ = γ
        
        self.S = self._build_state_space()
        self.S_index = {s: i for i, s in enumerate(self.S)}
        self.A = self.ACTIONS_P1

        self._combat_lookup = self._precompute_combat()
        T_base = self._build_transition_matrices()
        T_P2 = self._apply_P2_policy(T_base, π_P2)
        T_res = self._build_resource_matrix()
        self.T = {a: T_base[a] @ T_P2 @ T_res for a in self.A}
        self.T_base = T_base
        self.R = self._build_reward_vector()

    # ------------------------------------------------------------------
    # State / action helpers
    # ------------------------------------------------------------------

    def valid_act(self, a, s):
        if s.terminal:
            return False
        if a == 'P1_train_workers':
            return s.R1 > 0 and s.W1 < 10
        if a == 'P1_train_marines':
            return s.R1 > 0 and s.M1 < 10
        if a == 'P2_train_workers':
            return s.R2 > 0 and s.W2 < 10
        if a == 'P2_train_marines':
            return s.R2 > 0 and s.M2 < 10
        return True  # attacks always valid

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def act(self, a):
        s_idx = self.S_index[self.s]
        if self.s.terminal:
            return 0.0
        row = self.T[a].getrow(s_idx)
        probs = row.data / row.data.sum()
        sp_idx = np.random.choice(row.indices, p=probs)
        self.s = self.S[sp_idx]
        return self.R[self.S_index[self.s]]

    def observe(self):
        return self.s

    def reset(self):
        self.s = self.s_init

    def update_P2_policy(self, π_P2_new):
        T_P2 = self._apply_P2_policy(self.T_base, π_P2_new)
        T_res = self._build_resource_matrix()
        self.T = {a: self.T_base[a] @ T_P2 @ T_res for a in self.A}
        self.π_P2 = π_P2_new

    # ------------------------------------------------------------------
    # Game simulation
    # ------------------------------------------------------------------

    def simulate(self, π_P1, label='P1', max_turns=50):
        """Simulate a single game. π_P1 may be a dict or callable."""
        s = self.s_init
        print(f'P1 using {label} policy')
        print(f"{'Turn':<5} | {'P1 Action':<17} | {'P2 Action':<17} | "
              f"(W1,M1,R1 | W2,M2,R2 | terminal)")
        print("-" * 80)

        for turn in range(1, max_turns + 1):
            if s.terminal:
                winner = ("Winner: P1" if s.M1 > 0
                          else "Winner: P2" if s.M2 > 0 else "Draw")
                print(f"END   | {'TERMINAL':<17} | {'TERMINAL':<17} | "
                      f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | "
                      f"{s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")
                print(f"\nGame Over! {winner} in {turn - 1} turns\n")
                return

            a1 = π_P1[s] if isinstance(π_P1, dict) else π_P1(s)
            a2 = self.π_P2(s)
            print(f"{turn:<5} | {a1:<17} | {a2:<17} | "
                  f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | "
                  f"{s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")

            s_idx = self.S_index[s]
            row = self.T[a1].getrow(s_idx)
            probs = row.data / row.data.sum()
            s = self.S[np.random.choice(row.indices, p=probs)]

        print('\nGame Over! Draw - maximum turns reached\n')

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_state_space(self):
        return [
            State(W1, M1, R1, W2, M2, R2, terminal)
            for W1 in range(0, 11)
            for M1 in range(0, 11)
            for R1 in range(0, 11)
            for W2 in range(0, 11)
            for M2 in range(0, 11)
            for R2 in range(0, 11)
            for terminal in range(0, 2)
        ]

    def _precompute_combat(self):
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

    def _build_transition_matrices(self):
        S, S_index = self.S, self.S_index
        n = len(S)
        T = {}
        for a in self.ACTIONS_ALL:
            rows, cols, vals = [], [], []
            for s_idx, s in enumerate(tqdm(S, desc=f"Building T[{a}]")):
                if s.terminal:
                    rows.append(s_idx); cols.append(s_idx); vals.append(1.0)
                    continue
                if a in ('P1_attack', 'P2_attack'):
                    for (nm1, nm2), p in self._combat_lookup[(s.M1, s.M2)].items():
                        term = 1 if (nm1 == 0 or nm2 == 0) else 0
                        sp = State(s.W1, nm1, s.R1, s.W2, nm2, s.R2, term)
                        rows.append(s_idx); cols.append(S_index[sp]); vals.append(p)
                else:
                    invalid = (
                        (a == 'P1_train_workers' and (s.R1 < 1 or s.W1 > 9)) or
                        (a == 'P1_train_marines' and (s.R1 < 1 or s.M1 > 9)) or
                        (a == 'P2_train_workers' and (s.R2 < 1 or s.W2 > 9)) or
                        (a == 'P2_train_marines' and (s.R2 < 1 or s.M2 > 9))
                    )
                    if invalid:
                        rows.append(s_idx); cols.append(s_idx); vals.append(1.0)
                    else:
                        dW1 = min(s.W1 + s.R1, 10) - s.W1 if a == 'P1_train_workers' else 0
                        dM1 = min(s.M1 + s.R1, 10) - s.M1 if a == 'P1_train_marines' else 0
                        dR1 = -s.R1 if a in ('P1_train_workers', 'P1_train_marines') else 0
                        dW2 = min(s.W2 + s.R2, 10) - s.W2 if a == 'P2_train_workers' else 0
                        dM2 = min(s.M2 + s.R2, 10) - s.M2 if a == 'P2_train_marines' else 0
                        dR2 = -s.R2 if a in ('P2_train_workers', 'P2_train_marines') else 0
                        sp_ok = State(s.W1+dW1, s.M1+dM1, s.R1+dR1,
                                      s.W2+dW2, s.M2+dM2, s.R2+dR2, 0)
                        sp_fail = State(s.W1, s.M1, s.R1+dR1,
                                        s.W2, s.M2, s.R2+dR2, 0)
                        rows.append(s_idx); cols.append(S_index[sp_ok]);   vals.append(0.9)
                        rows.append(s_idx); cols.append(S_index[sp_fail]); vals.append(0.1)
            T[a] = csr_matrix((vals, (rows, cols)), shape=(n, n))
        return T

    def _apply_P2_policy(self, T_base, π_P2):
        S = self.S
        n = len(S)
        rows, cols, vals = [], [], []
        a2s = {a: [] for a in self.ACTIONS_ALL}
        for s_idx, s in enumerate(tqdm(S, desc="Applying P2 policy")):
            a2s[π_P2(s)].append(s_idx)
        for a, idxs in a2s.items():
            if not idxs:
                continue
            idxs_arr = np.array(idxs)
            sub = T_base[a][idxs_arr, :].tocoo()
            rows.extend(idxs_arr[sub.row].tolist())
            cols.extend(sub.col.tolist())
            vals.extend(sub.data.tolist())
        T_P2 = csr_matrix((vals, (rows, cols)), shape=(n, n))
        T_P2.eliminate_zeros()
        return T_P2

    def _build_resource_matrix(self):
        S, S_index = self.S, self.S_index
        n = len(S)
        rows, cols, vals = [], [], []
        for s_idx, s in enumerate(tqdm(S, desc="Building resource matrix")):
            if s.terminal:
                rows.append(s_idx); cols.append(s_idx); vals.append(1.0)
            else:
                sp = State(s.W1, s.M1, min(s.R1 + s.W1, 10),
                           s.W2, s.M2, min(s.R2 + s.W2, 10), s.terminal)
                rows.append(s_idx); cols.append(S_index[sp]); vals.append(1.0)
        return csr_matrix((vals, (rows, cols)), shape=(n, n))

    def _build_reward_vector(self):
        R = np.zeros(len(self.S))
        for idx, s in enumerate(self.S):
            if s.terminal:
                R[idx] = 1.0 if (s.M1 > 0 and s.M2 == 0) else -1.0
        return R
