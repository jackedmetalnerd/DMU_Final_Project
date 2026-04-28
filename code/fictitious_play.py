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
from policy import save_mixed_policy
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
        stats  : dict with keys:
                   v1_history, v2_history          — value at s₀ per iteration
                   entropy1_history, entropy2_history — avg policy entropy per iteration
                   sigma1_s0_history, sigma2_s0_history — action probs at s₀ (n_iters, 3)
        sigma1 : (n_states, 3) array  final P1 average policy
        sigma2 : (n_states, 3) array  final P2 average policy
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
        n1 = np.zeros((n, 3), dtype=np.float64)
        n2 = np.zeros((n, 3), dtype=np.float64)

        for i, s in enumerate(states):
            a2_idx = Action.P2_ACTIONS.index(alternating_training_attack(s))
            n2[i, a2_idx] += 1.0

        V1_prev = None
        V2_prev = None

        v1_history        = []
        v2_history        = []
        entropy1_history  = []
        entropy2_history  = []
        sigma1_s0_history = []
        sigma2_s0_history = []

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
            sigma1_inv = sigma1[perm, :]

            print(f"  P2: updating inverted T with mixed σ¹ ...")
            env_p2.update_P2_policy(sigma1_inv)

            print(f"  P2: running Value Iteration...")
            vi2 = ValueIteration(env_p2, tol=self.vi_tol)
            pi2_k = vi2.solve(initial_V=V2_prev)
            V2_prev = vi2.V

            _increment_counts_inverted(n2, pi2_k, states, perm)
            sigma2 = _normalize_counts(n2)

            # ── Record stats for this iteration ───────────────────────────────
            v1_s0 = V1_prev[s0_idx]
            v2_s0 = V2_prev[perm[s0_idx]]   # P2's value at inverted s₀

            v1_history.append(v1_s0)
            v2_history.append(v2_s0)
            entropy1_history.append(_policy_entropy(sigma1))
            entropy2_history.append(_policy_entropy(sigma2))
            sigma1_s0_history.append(sigma1[s0_idx].copy())
            sigma2_s0_history.append(sigma2[s0_idx].copy())

            print(f"\n  V¹(s₀) = {v1_s0:.6f}  |  V²(s₀) = {v2_s0:.6f}"
                  f"  |  H(σ¹) = {entropy1_history[-1]:.4f}"
                  f"  |  H(σ²) = {entropy2_history[-1]:.4f}")

        stats = {
            'v1_history':        np.array(v1_history),
            'v2_history':        np.array(v2_history),
            'entropy1_history':  np.array(entropy1_history),
            'entropy2_history':  np.array(entropy2_history),
            'sigma1_s0_history': np.array(sigma1_s0_history),
            'sigma2_s0_history': np.array(sigma2_s0_history),
        }
        sigma1 = _normalize_counts(n1)
        sigma2 = _normalize_counts(n2)
        return stats, sigma1, sigma2

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


def _policy_entropy(sigma: np.ndarray) -> float:
    """Mean entropy (nats) across all states for a mixed policy sigma (n, 3)."""
    eps = 1e-10
    return float(np.mean(-np.sum(sigma * np.log(sigma + eps), axis=1)))


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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Initializing MarkovGameEnv...")
    mg_env = MarkovGameEnv()

    solver = FictitiousPlay(mg_env, n_iters=100, vi_tol=1e-9, max_turns=1000)
    stats, sigma1, sigma2 = solver.run()

    # ── Save policies and stats ───────────────────────────────────────────────
    save_mixed_policy(sigma1, 'fsp_sigma_p1.npy')
    save_mixed_policy(sigma2, 'fsp_sigma_p2.npy')
    np.savez('fsp_stats.npz', **stats)
    print("Stats saved to fsp_stats.npz")

    # ── Build mixed policy callables ──────────────────────────────────────────
    states    = mg_env.joint_model._states
    state_idx = {s: i for i, s in enumerate(states)}
    s0        = mg_env.initial_state
    s0_i      = state_idx[s0]

    p1_mixed = MixedPolicy(sigma1, state_idx, Action.P1_ACTIONS)
    p2_mixed = MixedPolicy(sigma2, state_idx, Action.P2_ACTIONS)

    # ── Silent trace games (for plots) ───────────────────────────────────────
    n_traces = 5
    print(f"\nSimulating {n_traces} trace games...")
    traces = [mg_env.simulate_trace(p1_mixed, p2_mixed, max_turns=1000)
              for _ in range(n_traces)]
    np.savez('fsp_traces.npz',
             **{f'W1_{i}': t['W1'] for i, t in enumerate(traces)},
             **{f'M1_{i}': t['M1'] for i, t in enumerate(traces)},
             **{f'W2_{i}': t['W2'] for i, t in enumerate(traces)},
             **{f'M2_{i}': t['M2'] for i, t in enumerate(traces)},
             winners=np.array([t['winner'] for t in traces]))
    print("Traces saved to fsp_traces.npz")

    # ── Win/loss summary (bulk, silent) ──────────────────────────────────────
    n_sims = 1000
    print(f"\nSimulating {n_sims} games for win-rate estimate...")
    results = [mg_env.simulate_trace(p1_mixed, p2_mixed, max_turns=1000)['winner']
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

    # ── Policy summary at s₀ ─────────────────────────────────────────────────
    action_names = ['train_workers', 'train_marines', 'attack']
    print(f"\nFinal mixed policies at s₀ = {s0}:")
    print(f"  σ¹(a|s₀): " + ", ".join(
        f"{n}={sigma1[s0_i, j]:.3f}" for j, n in enumerate(action_names)))
    print(f"  σ²(a|s₀): " + ", ".join(
        f"{n}={sigma2[s0_i, j]:.3f}" for j, n in enumerate(action_names)))

    # ── 6-panel convergence plot ──────────────────────────────────────────────
    iters     = range(1, len(stats['v1_history']) + 1)
    c_p1      = ['#1f77b4', '#ff7f0e', '#2ca02c']
    c_p2      = ['#d62728', '#9467bd', '#8c564b']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Fictitious Self-Play Convergence', fontsize=14)

    # (0,0) Value at s₀
    ax = axes[0, 0]
    ax.plot(iters, stats['v1_history'], marker='.', label='V¹(s₀) — P1')
    ax.plot(iters, stats['v2_history'], marker='.', label='V²(s₀) — P2')
    ax.set_xlabel('FSP Iteration'); ax.set_ylabel('Value')
    ax.set_title('Value at Initial State'); ax.legend(); ax.grid(True)

    # (0,1) Policy entropy
    ax = axes[0, 1]
    ax.plot(iters, stats['entropy1_history'], marker='.', label='H(σ¹) — P1')
    ax.plot(iters, stats['entropy2_history'], marker='.', label='H(σ²) — P2')
    ax.set_xlabel('FSP Iteration'); ax.set_ylabel('Avg Entropy (nats)')
    ax.set_title('Policy Entropy'); ax.legend(); ax.grid(True)

    # (0,2) P1 action probs at s₀
    ax = axes[0, 2]
    for j, name in enumerate(action_names):
        ax.plot(iters, stats['sigma1_s0_history'][:, j],
                marker='.', color=c_p1[j], label=name)
    ax.set_xlabel('FSP Iteration'); ax.set_ylabel('Probability')
    ax.set_title('P1 Action Probs at s₀'); ax.legend(fontsize=8); ax.grid(True)

    # (1,0) P2 action probs at s₀
    ax = axes[1, 0]
    for j, name in enumerate(action_names):
        ax.plot(iters, stats['sigma2_s0_history'][:, j],
                marker='.', color=c_p2[j], label=name)
    ax.set_xlabel('FSP Iteration'); ax.set_ylabel('Probability')
    ax.set_title('P2 Action Probs at s₀'); ax.legend(fontsize=8); ax.grid(True)

    # (1,1) Marine counts over game turns
    ax = axes[1, 1]
    for i, t in enumerate(traces):
        turns = range(len(t['M1']))
        ax.plot(turns, t['M1'], color=c_p1[0], alpha=0.55,
                label=f'P1 ({t["winner"]} wins)' if i == 0 else None)
        ax.plot(turns, t['M2'], color=c_p2[0], alpha=0.55,
                label='P2' if i == 0 else None)
    ax.set_xlabel('Turn'); ax.set_ylabel('Marines')
    ax.set_title('Marine Counts — Sample Games'); ax.legend(fontsize=8); ax.grid(True)

    # (1,2) Worker counts over game turns
    ax = axes[1, 2]
    for i, t in enumerate(traces):
        turns = range(len(t['W1']))
        ax.plot(turns, t['W1'], color=c_p1[1], alpha=0.55,
                label='P1' if i == 0 else None)
        ax.plot(turns, t['W2'], color=c_p2[1], alpha=0.55,
                label='P2' if i == 0 else None)
    ax.set_xlabel('Turn'); ax.set_ylabel('Workers')
    ax.set_title('Worker Counts — Sample Games'); ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    plt.savefig('fsp_convergence.png', dpi=150)
    print(f"\nPlot saved to fsp_convergence.png")
