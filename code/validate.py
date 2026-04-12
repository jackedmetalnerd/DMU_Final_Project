"""
Validates that the OOP GameEnv produces the same results as project_mdp.py.
Run from the code/ directory:  python validate.py
"""

import numpy as np
import sys

# ── Originals ──────────────────────────────────────────────────────────────────
import project_mdp as orig
from project_mdp import (build_MDP, value_iteration as vi_orig, greedy as greedy_orig,
                          alternating_training_attack, State, Q_learning)

# ── OOP versions ───────────────────────────────────────────────────────────────
from game_env import GameEnv
from policies import alternating_training_attack as ata_oop
from value_iteration import ValueIteration
from q_learning import QLearning

PASS = "  PASS"
FAIL = "  FAIL"

def check(label, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"{tag} | {label}" + (f"  ({detail})" if detail else ""))
    return ok

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Building both MDPs ===")
s_init = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

print("\n[Original]")
orig_sim = build_MDP(π_P2=alternating_training_attack, s_init=s_init)

print("\n[OOP]")
env = GameEnv(opponent_policy=ata_oop, initial_state=s_init)

all_pass = True

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. State Space ===")
ok = len(orig_sim['S']) == len(env.S)
all_pass &= check("State count matches", ok,
                  f"orig={len(orig_sim['S'])} oop={len(env.S)}")

ok = set(orig_sim['S']) == set(env.S)
all_pass &= check("State sets identical", ok)

ok = orig_sim['S'] == env.S
all_pass &= check("State ordering identical", ok)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Reward Vector ===")
breakpoint()
max_diff = np.max(np.abs(orig_sim['R'] - env.R))
ok = max_diff < 1e-10
all_pass &= check("R vectors match", ok, f"max diff={max_diff:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Transition Matrices (combined T) ===")
env.transition_model.build_matrices()
for a in env.A:
    diff = (orig_sim['T'][a] - env.T[a])
    max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
    ok = max_diff < 1e-10
    all_pass &= check(f"T[{a}] matches", ok, f"max diff={max_diff:.2e}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. Value Iteration ===")
print("[Original VI]")
V_orig = vi_orig(orig_sim)
π_orig = greedy_orig(orig_sim, V_orig)

print("\n[OOP VI]")
solver = ValueIteration(env)
π_oop = solver.solve()
V_oop = solver.V

max_diff = np.max(np.abs(V_orig - V_oop))
ok = max_diff < 1e-6
all_pass &= check("Value functions match", ok, f"max diff={max_diff:.2e}")

# Policy agreement (non-terminal states only)
non_term = [s for s in env.S if not s.terminal]
agree = sum(π_orig[s] == π_oop[s] for s in non_term)
pct = 100.0 * agree / len(non_term)
ok = pct > 99.0
all_pass &= check("Greedy policies agree", ok,
                  f"{agree}/{len(non_term)} states = {pct:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. Win-rate comparison (50 simulated games each) ===")

def win_rate(sim_fn, n=50, seed=42):
    np.random.seed(seed)
    wins = 0
    for _ in range(n):
        result = sim_fn()
        if result == 'P1':
            wins += 1
    return wins / n

def run_orig_game(game_sim, π_P1, π_P2, s_init, max_turns=50):
    s = s_init
    S, T = game_sim['S'], game_sim['T']
    for _ in range(max_turns):
        if s.terminal:
            return 'P1' if s.M1 > 0 else 'P2'
        a1 = π_P1[s]
        s_idx = S.index(s)
        row = T[a1].getrow(s_idx)
        probs = row.data / row.data.sum()
        s = S[np.random.choice(row.indices, p=probs)]
    return 'draw'

def run_oop_game(env, π_P1, s_init, max_turns=50):
    s = s_init
    for _ in range(max_turns):
        if s.terminal:
            return 'P1' if s.M1 > 0 else 'P2'
        a1 = π_P1[s]
        s_idx = env.S_index[s]
        row = env.T[a1].getrow(s_idx)
        probs = row.data / row.data.sum()
        s = env.S[np.random.choice(row.indices, p=probs)]
    return 'draw'

np.random.seed(42)
wr_orig = sum(
    run_orig_game(orig_sim, π_orig, alternating_training_attack, s_init) == 'P1'
    for _ in range(50)) / 50

np.random.seed(42)
wr_oop = sum(
    run_oop_game(env, π_oop, s_init) == 'P1'
    for _ in range(50)) / 50

ok = abs(wr_orig - wr_oop) < 0.05
all_pass &= check("Win rates within 5%", ok,
                  f"orig={wr_orig:.2f} oop={wr_oop:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. Q-Learning (convergence check) ===")
print("[Original Q-learning]")
orig_eps = Q_learning(orig_sim, n_episodes=500, α=0.1)
Q_orig = orig_eps[-1][1]
π_Q_orig = lambda s: max(orig_sim['A'], key=lambda a: Q_orig[(s, a)])

print("\n[OOP Q-learning]")
ql = QLearning(env, γ=0.95, α=0.1, ϵ_start=0.2, ϵ_min=0.05)
ql.train(n_episodes=500)

np.random.seed(99)
wr_q_orig = sum(
    run_orig_game(orig_sim, {s: π_Q_orig(s) for s in orig_sim['S']},
                  alternating_training_attack, s_init) == 'P1'
    for _ in range(50)) / 50

np.random.seed(99)
wr_q_oop = sum(
    run_oop_game(env, {s: ql.policy(s) for s in env.S}, s_init) == 'P1'
    for _ in range(50)) / 50

ok = abs(wr_q_orig - wr_q_oop) < 0.10
all_pass &= check("Q-learning win rates within 10%", ok,
                  f"orig={wr_q_orig:.2f} oop={wr_q_oop:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 50)
if all_pass:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED — review output above")
    sys.exit(1)
