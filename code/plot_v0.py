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
vi.solve()

ql = QLearning(env, alpha=0.1)
ql.solve(n_episodes=10_000)

dqn = DQNSolver(env, batch_size=128, target_update_freq=200)
dqn.solve(n_episodes=10_000)

mcts = MCTSSolver(env, c=sqrt(2), depth=50, num_runs=10_000)
mcts.get_action(S_INIT)

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

plt.show()
