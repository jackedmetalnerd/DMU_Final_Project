import numpy as np
import time
from tqdm import tqdm
from game_env import GameEnv
from state import State
from policy import DictPolicy, save_policy
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

    def solve(self, initial_V=None) -> DictPolicy:
        if not self.env.transition_model.T:
            self.env.transition_model.build_matrices()
        S, A, T, R, γ = self.env.S, self.env.A, self.env.T, self.env.R, self.env.γ

        term_mask = np.array([s.terminal == 1 for s in S])

        V = initial_V.copy() if initial_V is not None else np.zeros(len(S))
        V[term_mask] = 0.0
        it, start = 0, time.time()

        while True:
            RgV = R + γ * V          # expected next-state reward + discounted future value
            Vp = T[A[0]] @ RgV
            for a in A[1:]:
                Vp = np.maximum(Vp, T[a] @ RgV)
            Vp[term_mask] = 0.0

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
        Q = np.array([T[a] @ (R + γ * V) for a in A])
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
    save_policy(π_star, 'vi_policy_p1.npy', env.S, env.A)

    n_games = 20
    results = [env.simulate(π_star, label='Value Iteration') for _ in range(n_games)]

    p1_wins = results.count('P1')
    p2_wins = results.count('P2')
    draws   = results.count('Draw')
    print(f"\n{'='*40}")
    print(f"  Results over {n_games} games")
    print(f"{'='*40}")
    print(f"  P1 wins: {p1_wins}/{n_games} ({100*p1_wins/n_games:.0f}%)")
    print(f"  P2 wins: {p2_wins}/{n_games} ({100*p2_wins/n_games:.0f}%)")
    print(f"  Draws:   {draws}/{n_games} ({100*draws/n_games:.0f}%)")

    print("\nRunning matrix vs transition() diagnostic (1000 random states)...")
    import random
    mismatches = 0
    sample_states = [s for s in env.S if not s.terminal]
    for s in random.sample(sample_states, min(1000, len(sample_states))):
        for a in env.A:
            row = env.T[a][env.S_index[s], :].toarray().flatten()
            dist = env.transition_model.transition(s, a)
            for sp, p in dist.items():
                j = env.S_index[sp]
                if abs(row[j] - p) > 1e-9:
                    print(f"  MISMATCH: action={a}, state={s}")
                    print(f"    next={sp}: matrix={row[j]:.8f}, transition()={p:.8f}")
                    mismatches += 1
                    if mismatches >= 10:
                        break
            if mismatches >= 10:
                break
        if mismatches >= 10:
            break
    if mismatches == 0:
        print("  No mismatches found — matrices agree with transition().")
    else:
        print(f"  Found {mismatches}+ mismatches.")
