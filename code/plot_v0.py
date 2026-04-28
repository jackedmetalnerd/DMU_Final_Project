import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from game_env import GameEnv
from state import State
from policies import alternating_training_attack
from value_iteration import ValueIteration
from q_learning import QLearning
from dqn import DQNSolver
from mcts import MCTSSolver

S_INIT = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)
env = GameEnv(opponent_policy=alternating_training_attack, initial_state=S_INIT)

# Run all solvers
vi = ValueIteration(env)
π_vi = vi.solve()

ql = QLearning(env, alpha=0.1)
π_ql = ql.solve(n_episodes=10_000)

dqn = DQNSolver(env, batch_size=128, target_update_freq=200)
π_dqn = dqn.solve(n_episodes=10_000)

mcts = MCTSSolver(env, c=sqrt(2), depth=50, num_runs=10000)
mcts.get_action(S_INIT)

# Separate lighter MCTS solver for win-rate evaluation (1000 runs/move)
mcts_eval = MCTSSolver(env, c=sqrt(2), depth=50, num_runs=1000)
π_mcts_eval = mcts_eval.solve()

# ── Win-rate helpers ──────────────────────────────────────────────────────────

N_EVAL = 100

def _wilson_ci(p, n, z=1.96):
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half   = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - half, center + half

def _bar_yerr(proportions, n):
    lowers, uppers = [], []
    for p in proportions:
        lo, hi = _wilson_ci(p, n)
        lowers.append(p - lo)
        uppers.append(hi - p)
    return np.array([lowers, uppers])

def _measure_win_rate(policy, n_games=N_EVAL):
    wins = losses = draws = 0
    for _ in range(n_games):
        s = S_INIT
        for _ in range(500):
            if s.terminal:
                break
            if hasattr(policy, '_solver'):
                policy._solver.reset_tree()
            a = policy(s)
            s = env.transition_model.sample(s, a)
        if s.M1 > 0 and s.M2 == 0:
            wins += 1
        elif s.M2 > 0 and s.M1 == 0:
            losses += 1
        else:
            draws += 1
    return wins / n_games, losses / n_games, draws / n_games

# ── Figure 1: VI / QL / DQN training curves + MCTS reference line ─────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))

ax1.plot(np.linspace(0, 1, len(vi.v0_history)), vi.v0_history, label='Value Iteration')
ax1.plot(np.linspace(0, 1, len(ql.v0_history)), ql.v0_history, label='Q-Learning', alpha=0.7)
ax1.plot(np.linspace(0, 1, len(dqn.v0_history)), dqn.v0_history, label='DQN', alpha=0.7)
ax1.axhline(mcts.v0_history[-1], color='red', linestyle='--',
            label='MCTS (converged, 10_000 rollouts)', alpha=0.8)

ax1.set_xlabel('Normalized training budget')
ax1.set_ylabel('V(s₀) estimate')
ax1.set_title('Estimated value of initial state across solvers')
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.legend()
fig1.tight_layout()
fig1.savefig('v0_comparison.png', dpi=150)

# ── Figure 2: MCTS estimate sharpening with rollouts ──────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 4))

ax2.plot(range(1, len(mcts.v0_history) + 1), mcts.v0_history, color='red', alpha=0.7)
ax2.set_xlabel('Rollout count')
ax2.set_ylabel('V(s₀) estimate')
ax2.set_title('MCTS: estimate sharpening with rollouts')
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
fig2.tight_layout()
fig2.savefig('v0_mcts_rollouts.png', dpi=150)

# ── Figure 3: win rate bar chart with 95% Wilson CI ───────────────────────────
print("Measuring win rates...")
solvers   = ['VI', 'Q-Learning', 'DQN', 'MCTS\n(1000 runs/move)']
policies  = [π_vi, π_ql, π_dqn, π_mcts_eval]
wins3, losses3, draws3 = [], [], []
for label, π in zip(solvers, policies):
    w, l, d = _measure_win_rate(π)
    wins3.append(w); losses3.append(l); draws3.append(d)
    print(f"  {label.replace(chr(10), ' ')}: win={w:.1%}  loss={l:.1%}  draw={d:.1%}")

bw = 0.25
fig3, ax3 = plt.subplots(figsize=(9, 5))
x = np.arange(len(solvers))
c_win = ax3.bar(x - bw, wins3,   bw, label='Win',  color='steelblue',
               yerr=_bar_yerr(wins3,   N_EVAL), capsize=4, error_kw={'elinewidth': 1})
c_los = ax3.bar(x,      losses3, bw, label='Loss', color='tomato',
               yerr=_bar_yerr(losses3, N_EVAL), capsize=4, error_kw={'elinewidth': 1})
c_drw = ax3.bar(x + bw, draws3,  bw, label='Draw', color='gray',
               yerr=_bar_yerr(draws3,  N_EVAL), capsize=4, error_kw={'elinewidth': 1})
ax3.bar_label(c_win, labels=[str(round(p * N_EVAL)) for p in wins3],   padding=3, fontsize=8)
ax3.bar_label(c_los, labels=[str(round(p * N_EVAL)) for p in losses3], padding=3, fontsize=8)
ax3.bar_label(c_drw, labels=[str(round(p * N_EVAL)) for p in draws3],  padding=3, fontsize=8)
ax3.set_xticks(x)
ax3.set_xticklabels(solvers)
ax3.set_xlabel('Solver')
ax3.set_ylabel('Rate')
ax3.set_title(f'Win rates by solver (n={N_EVAL} games)')
ax3.set_ylim(0, 1)
ax3.legend()
fig3.tight_layout()
fig3.savefig('mdp_winrates.png', dpi=150)
print("Saved mdp_winrates.png")

plt.show()
