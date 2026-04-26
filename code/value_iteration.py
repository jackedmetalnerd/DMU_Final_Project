import numpy as np
import time
from tqdm import tqdm
from game_env import GameEnv
from state import State
from policy import DictPolicy
from solver import Solver
from policies import alternating_training_attack, P2_policy_converter


class ValueIteration(Solver):
    """Solves a GameEnv with value iteration and extracts a greedy policy."""

    def __init__(self, env: GameEnv, tol=1e-9):
        super().__init__(env)
        self.tol = tol
        self.V = None
        self.v0_history = []
        self.policy = None

    def solve(self) -> DictPolicy:
        if not self.env.transition_model.T:
            self.env.transition_model.build_matrices()
        S, A, T, R, γ = self.env.S, self.env.A, self.env.T, self.env.R, self.env.γ

        term_mask = np.array([s.terminal == 1 for s in S])
        term_vals = np.array([s.terminal_value() if s.terminal else 0.0 for s in S])

        V = term_vals.copy()
        it, start = 0, time.time()

        R_nonterminal = np.where(term_mask, 0.0, R)
        while True:
            Vp = R_nonterminal + γ * (T[A[0]] @ V)
            for a in A[1:]:
                Vp = np.maximum(Vp, R_nonterminal + γ * (T[a] @ V))
            Vp[term_mask] = term_vals[term_mask]

            delta = np.max(np.abs(V - Vp))
            s0_idx = self.env.S_index[self.env.initial_state]
            self.v0_history.append(float(Vp[s0_idx]))

            V = Vp
            it += 1

            if it % 10 == 0:
                elapsed = time.time() - start
                m, sec = divmod(elapsed, 60)
                print(f"Iter {it:04d} | delta {delta:.8f} | {int(m):02d}:{sec:05.2f}")

            if delta < self.tol:
                break

        self.V = V
        self.policy = self._greedy(V)
        return self.policy

    def _greedy(self, V) -> DictPolicy:
        S, A, T, R, γ = self.env.S, self.env.A, self.env.T, self.env.R, self.env.γ
        Q = np.array([R + γ * (T[a] @ V) for a in A])
        best = np.argmax(Q, axis=0)
        policy_dict = {s: A[best[i]] for i, s in enumerate(tqdm(S, desc="Building greedy policy"))}
        return DictPolicy(policy_dict)


if __name__ == '__main__':
    s_init = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)
    env = GameEnv(opponent_policy=alternating_training_attack, initial_state=s_init)

    solver = ValueIteration(env)
    print("Running value iteration...")
    π_star = solver.solve()
    print("Done.")

    for _ in range(5):
        env.simulate(π_star, label='Value Iteration')

    # Also test against a P2 that uses the same VI-derived policy
    π_P2_vi = P2_policy_converter(π_star)
    env.update_P2_policy(π_P2_vi)
    print("\nVI policy vs VI-derived P2:")
    for _ in range(3):
        env.simulate(π_star, label='Value Iteration')
