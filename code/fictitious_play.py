"""
fictitious_play.py
==================
Fictitious Self-Play (FSP) solver for the Markov game.

Each player iteratively best-responds to the **running average** of all past
opponent policies. The average is represented as a mixed (stochastic) policy
sigma(a|s) maintained via action count arrays. Convergence to a Nash equilibrium
is guaranteed for finite two-player zero-sum games.

Algorithm per iteration k:
  1. Normalize P2 counts → σ² (mixed policy over all states)
  2. Marginalize transition matrices over σ² → T̃¹[a¹]
  3. P1 runs Value Iteration → deterministic best response π¹_k
  4. Increment P1 count array from π¹_k
  5. Normalize P1 counts → σ¹
  6. Remap σ¹ to inverted state space (for P2's VI)
  7. P2 runs Value Iteration (inverted GameEnv) → deterministic best response π²_k
  8. Increment P2 count array from π²_k
  9. Record V¹(s₀) for convergence tracking

Usage:
    python fictitious_play.py
"""

import numpy as np
import time
from action import Action
from markov_game_env import MarkovGameEnv
from policies import alternating_training_attack
from state import State
from value_iteration import ValueIteration


# ── MixedPolicy callable ──────────────────────────────────────────────────────

class MixedPolicy:
    """Callable policy that samples actions from a mixed strategy σ.

    Parameters
    ----------
    sigma      : (n_states, 3) array  — sigma[s, i] = P(action_i | state s)
    state_index: dict State → int
    actions    : list of Actions (P1 or P2 labeled, matching sigma column order)
    """

    def __init__(self, sigma: np.ndarray, state_index: dict, actions: list):
        self._sigma       = sigma
        self._state_index = state_index
        self._actions     = actions

    def __call__(self, s: State) -> Action:
        i = self._state_index[s]
        probs = self._sigma[i]
        a_idx = np.random.choice(3, p=probs)
        return self._actions[a_idx]


# ── FictitiousPlay ────────────────────────────────────────────────────────────

class FictitiousPlay:
    """Fictitious Self-Play solver for MarkovGameEnv.

    Parameters
    ----------
    env      : MarkovGameEnv
    n_iters  : int    number of FSP iterations (VI solves per player = n_iters each)
    vi_tol   : float  VI convergence tolerance
    max_turns: int    max turns per simulated game
    """

    def __init__(
        self,
        env: MarkovGameEnv,
        n_iters: int = 5,
        vi_tol: float = 1e-9,
        max_turns: int = 50,
    ):
        self.env      = env
        self.n_iters  = n_iters
        self.vi_tol   = vi_tol
        self.max_turns = max_turns

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self):
        """Run FSP for n_iters iterations.

        Returns
        -------
        v1_history : list[float]  V¹(s₀) after each iteration
        sigma1     : (n_states, 3) array  final P1 average policy
        sigma2     : (n_states, 3) array  final P2 average policy
        """
        env = self.env
        s0  = env.initial_state

        # ── One-time setup ────────────────────────────────────────────────────
        print("Building P1 GameEnv and matrices (T_base + T_res, one-time)...")
        env_p1 = env.as_p1_gameenv(alternating_training_attack)
        t0 = time.time()
        env_p1.transition_model.build_matrices()
        print(f"  P1 env built in {time.time()-t0:.1f}s")

        print("Building P2 inverted GameEnv and matrices (one-time)...")
        env_p2 = env.as_p2_gameenv(_default_p1_policy)
        t0 = time.time()
        env_p2.transition_model.build_matrices()
        print(f"  P2 env built in {time.time()-t0:.1f}s")

        states      = env_p1.S
        state_index = {s: i for i, s in enumerate(states)}
        n           = len(states)
        s0_idx      = state_index[s0]

        perm = self._build_inversion_perm(states, state_index)

        # ── Count arrays and initial seed ─────────────────────────────────────
        n1 = np.zeros((n, 3), dtype=np.float64)   # P1 action counts
        n2 = np.zeros((n, 3), dtype=np.float64)   # P2 action counts

        # Seed n2 with the initial deterministic P2 policy
        for i, s in enumerate(states):
            a2_idx = Action.P2_ACTIONS.index(alternating_training_attack(s))
            n2[i, a2_idx] += 1.0

        V1_prev = None
        V2_prev = None
        v1_history = []

        # ── Main FSP loop ─────────────────────────────────────────────────────
        for k in range(1, self.n_iters + 1):
            print(f"\n{'='*65}")
            print(f"  FSP Iteration {k} / {self.n_iters}")
            print(f"{'='*65}")

            # ── P1 best response against σ² ───────────────────────────────────
            sigma2 = _normalize_counts(n2)
            print(f"  P1: updating T with mixed σ² ...")
            env_p1.update_P2_policy(sigma2)

            print(f"  P1: running Value Iteration...")
            vi1 = ValueIteration(env_p1, tol=self.vi_tol)
            pi1_k = vi1.solve(initial_V=V1_prev)
            V1_prev = vi1.V

            _increment_counts(n1, pi1_k, states, Action.P1_ACTIONS)

            # ── P2 best response against σ¹ ───────────────────────────────────
            sigma1     = _normalize_counts(n1)
            sigma1_inv = sigma1[perm, :]   # remap to inverted state indices

            print(f"  P2: updating inverted T with mixed σ¹ ...")
            env_p2.update_P2_policy(sigma1_inv)

            print(f"  P2: running Value Iteration...")
            vi2 = ValueIteration(env_p2, tol=self.vi_tol)
            pi2_k = vi2.solve(initial_V=V2_prev)
            V2_prev = vi2.V

            _increment_counts_inverted(n2, pi2_k, states, perm)

            v1_s0 = V1_prev[s0_idx]
            v1_history.append(v1_s0)
            print(f"\n  V¹(s₀) = {v1_s0:.6f}")

        sigma1 = _normalize_counts(n1)
        sigma2 = _normalize_counts(n2)
        return v1_history, sigma1, sigma2

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_inversion_perm(states: list, state_index: dict) -> np.ndarray:
        """Pre-compute inversion permutation: perm[i] = index of inverted form of state i."""
        perm = np.empty(len(states), dtype=np.int64)
        for i, s in enumerate(states):
            s_inv = State(W1=s.W2, M1=s.M2, R1=s.R2,
                          W2=s.W1, M2=s.M1, R2=s.R1,
                          terminal=s.terminal)
            perm[i] = state_index[s_inv]
        return perm


# ── Module-level helpers ──────────────────────────────────────────────────────

def _default_p1_policy(s: State) -> Action:
    """Default P1 bootstrap: train to max then attack."""
    if s.M1 == 10 and s.W1 == 10:
        return Action.P1_ATTACK
    if s.W1 < s.M1:
        return Action.P1_TRAIN_WORKERS
    return Action.P1_TRAIN_MARINES


def _normalize_counts(counts: np.ndarray) -> np.ndarray:
    """Normalize count rows to sum to 1 (probability distribution per state)."""
    totals = counts.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)   # avoid div-by-zero
    return counts / totals


def _increment_counts(counts: np.ndarray, policy, states: list, actions: list) -> None:
    """Increment count[s_idx, a_idx] for each state from a deterministic DictPolicy."""
    for i, s in enumerate(states):
        a_idx = actions.index(policy(s))
        counts[i, a_idx] += 1.0


def _increment_counts_inverted(counts: np.ndarray, policy, states: list,
                                perm: np.ndarray) -> None:
    """Increment P2 counts from a VI policy solved in the inverted state space.

    policy(s_inv) returns a P1-labeled action for inverted state s_inv at index j.
    The original state for j is perm[j]. Action index is the same for P1 and P2
    (0=train_workers, 1=train_marines, 2=attack).
    """
    for j, s_inv in enumerate(states):
        a = policy(s_inv)
        a_idx = Action.P1_ACTIONS.index(a)
        counts[perm[j], a_idx] += 1.0


# ── __main__ ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')   # non-interactive backend (saves to file)
    import matplotlib.pyplot as plt

    print("Initializing MarkovGameEnv...")
    mg_env = MarkovGameEnv()

    solver = FictitiousPlay(
        mg_env,
        n_iters=10,
        vi_tol=1e-9,
        max_turns=50,
    )

    v1_history, sigma1, sigma2 = solver.run()

    # ── Convergence plot ──────────────────────────────────────────────────────
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(v1_history) + 1), v1_history, marker='o')
    plt.xlabel('FSP Iteration')
    plt.ylabel('V¹(s₀)')
    plt.title('Fictitious Self-Play: P1 Value at Initial State')
    plt.grid(True)
    plt.tight_layout()
    plot_path = 'fsp_convergence.png'
    plt.savefig(plot_path)
    print(f"\nConvergence plot saved to {plot_path}")

    # ── Policy summary at initial state ──────────────────────────────────────
    s0 = mg_env.initial_state
    states     = MarkovGameEnv().joint_model._states   # reuse state list
    state_idx  = {s: i for i, s in enumerate(states)}
    s0_i = state_idx[s0]

    print(f"\nFinal mixed policies at s₀ = {s0}:")
    action_names = ['train_workers', 'train_marines', 'attack']
    print(f"  σ¹(a|s₀): " + ", ".join(
        f"{n}={sigma1[s0_i, j]:.3f}" for j, n in enumerate(action_names)))
    print(f"  σ²(a|s₀): " + ", ".join(
        f"{n}={sigma2[s0_i, j]:.3f}" for j, n in enumerate(action_names)))

    # ── Simulations with final mixed policies ─────────────────────────────────
    n_sims = 10
    print(f"\nSimulating {n_sims} games with final mixed policies...")
    p1_mixed = MixedPolicy(sigma1, state_idx, Action.P1_ACTIONS)
    p2_mixed = MixedPolicy(sigma2, state_idx, Action.P2_ACTIONS)

    results = [mg_env.simulate(p1_policy=p1_mixed, p2_policy=p2_mixed, max_turns=50)
               for _ in range(n_sims)]

    p1_wins = results.count('P1')
    p2_wins = results.count('P2')
    draws   = results.count('Draw')
    print(f"\n{'='*40}")
    print(f"  Results over {n_sims} games (mixed policies)")
    print(f"{'='*40}")
    print(f"  P1 wins: {p1_wins}/{n_sims} ({100*p1_wins/n_sims:.0f}%)")
    print(f"  P2 wins: {p2_wins}/{n_sims} ({100*p2_wins/n_sims:.0f}%)")
    print(f"  Draws:   {draws}/{n_sims} ({100*draws/n_sims:.0f}%)")
