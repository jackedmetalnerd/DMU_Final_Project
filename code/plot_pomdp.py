"""
plot_pomdp.py
=============
Three diagnostic plots for the POMDP section. Run from the code/ directory.

Saves:
  pomdp_obs_levels.png  — QMDP win rate vs. observation granularity
  pomdp_entropy.png     — belief entropy over a game (mean + individual traces)
  pomdp_mismatch.png    — QMDP vs. POMCP under model mismatch
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from state import State
from pomdp_env import POMDPEnv
from policies import alternating_training_attack
from value_iteration import ValueIteration
from pomdp_solver import QMDPSolver, POMCPSolver
from policy import FunctionPolicy
from action import Action

S_INIT = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)
N_GAMES    = 100
STEP_LIMIT = 50

P2_ACTIONS = [Action.P2_TRAIN_WORKERS, Action.P2_TRAIN_MARINES, Action.P2_ATTACK]
random_P2  = FunctionPolicy(lambda s: np.random.choice(P2_ACTIONS))

# ── helpers ───────────────────────────────────────────────────────────────────

def _run_games(env, policy, n_games, label=''):
    wins = losses = draws = 0
    for _ in tqdm(range(n_games), desc=label, unit='game'):
        env.reset()
        if hasattr(policy, 'reset'):
            policy.reset()
        for _ in range(STEP_LIMIT):
            if env.observe_raw().terminal:
                break
            b = env.observe()
            a = policy(b)
            env.act(a)
            if hasattr(policy, 'update'):
                policy.update(a, env.observe_obs())
        s = env.observe_raw()
        if s.M1 > 0 and s.M2 == 0:
            wins += 1
        elif s.M2 > 0 and s.M1 == 0:
            losses += 1
        else:
            draws += 1
    return wins / n_games, losses / n_games, draws / n_games


def _run_mdp_games(env, policy, n_games):
    """Win rate for a full-info MDP DictPolicy (uses true state, not belief)."""
    wins = losses = draws = 0
    for _ in range(n_games):
        s = S_INIT
        for _ in range(STEP_LIMIT):
            if s.terminal:
                break
            a = policy(s)
            s = env.transition_model.sample(s, a)
        if s.M1 > 0 and s.M2 == 0:
            wins += 1
        elif s.M2 > 0 and s.M1 == 0:
            losses += 1
        else:
            draws += 1
    return wins / n_games


def _share_matrices(src, dst):
    """Copy pre-built transition matrices from src env to dst env."""
    sm, tm = src.transition_model, dst.transition_model
    tm._T_base = sm._T_base
    tm._T_res  = sm._T_res
    tm._T_P2   = sm._T_P2
    tm.T       = sm.T


def _wilson_ci(p, n, z=1.96):
    """95% Wilson score confidence interval for proportion p from n trials."""
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half   = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - half, center + half


def _bar_yerr(proportions, n):
    """Asymmetric Wilson CI error bars in the shape matplotlib expects (2, N)."""
    lowers, uppers = [], []
    for p in proportions:
        lo, hi = _wilson_ci(p, n)
        lowers.append(p - lo)
        uppers.append(hi - p)
    return np.array([lowers, uppers])


# ── shared setup: two VI solves ───────────────────────────────────────────────

print("VI solve 1/2: deterministic P2...")
env_det = POMDPEnv(opponent_policy=alternating_training_attack,
                   initial_state=S_INIT, n_obs_levels=1)
vi_det = ValueIteration(env_det)
vi_det.solve()

print("\nVI solve 2/2: random P2 (mismatch planning model)...")
env_plan = POMDPEnv(opponent_policy=random_P2, initial_state=S_INIT, n_obs_levels=1)
vi_plan  = ValueIteration(env_plan)
vi_plan.solve()
env_plan.transition_model.build_uniform_P2()


# ── Plot 1: QMDP win rate vs. observation granularity ────────────────────────

print("\n--- Plot 1: observation granularity sweep ---")
levels   = [1, 2, 4, 11]
xlabels  = ['n=1\n(blind)', 'n=2\n(presence/\nabsence)', 'n=4\n(bucketed)', 'n=11\n(exact)']
wins1, losses1, draws1 = [], [], []

for n in levels:
    env_n = POMDPEnv(opponent_policy=alternating_training_attack,
                     initial_state=S_INIT, n_obs_levels=n)
    _share_matrices(env_det, env_n)
    π = QMDPSolver(env_n, V=vi_det.V).solve()
    w, l, d = _run_games(env_n, π, N_GAMES, label=f'QMDP n_obs={n}')
    wins1.append(w); losses1.append(l); draws1.append(d)

vi_win_rate = _run_mdp_games(env_det, vi_det.policy, N_GAMES)

bw = 0.25
fig1, ax1 = plt.subplots(figsize=(8, 5))
x = np.arange(len(levels))
ax1.bar(x - bw, wins1,   bw, label='Win',  color='steelblue',
        yerr=_bar_yerr(wins1,   N_GAMES), capsize=4, error_kw={'elinewidth': 1})
ax1.bar(x,      losses1, bw, label='Loss', color='tomato',
        yerr=_bar_yerr(losses1, N_GAMES), capsize=4, error_kw={'elinewidth': 1})
ax1.bar(x + bw, draws1,  bw, label='Draw', color='gray',
        yerr=_bar_yerr(draws1,  N_GAMES), capsize=4, error_kw={'elinewidth': 1})
ax1.axhline(vi_win_rate, color='black', linestyle='--', linewidth=1.2,
            label='MDP VI win rate (full info)')
ax1.set_xticks(x)
ax1.set_xticklabels(xlabels)
ax1.set_xlabel('Observation granularity')
ax1.set_ylabel('Rate')
ax1.set_title('QMDP win rate vs. observation granularity')
ax1.set_ylim(0, 1)
ax1.legend()
fig1.tight_layout()
fig1.savefig('pomdp_obs_levels.png', dpi=150)
print("Saved pomdp_obs_levels.png")


# ── Plot 2: belief entropy over a game ───────────────────────────────────────

print("\n--- Plot 2: belief entropy traces ---")

def _entropy(b):
    nz = b[b > 0]
    return float(-np.sum(nz * np.log2(nz)))

def _entropy_trace(env, policy):
    env.reset()
    if hasattr(policy, 'reset'):
        policy.reset()
    trace = []
    for _ in range(STEP_LIMIT):
        if env.observe_raw().terminal:
            break
        b = env.observe()
        trace.append(_entropy(b))
        a = policy(b)
        env.act(a)
        if hasattr(policy, 'update'):
            policy.update(a, env.observe_obs())
    return trace

env_e = POMDPEnv(opponent_policy=alternating_training_attack,
                 initial_state=S_INIT, n_obs_levels=4)
_share_matrices(env_det, env_e)
π_e = QMDPSolver(env_e, V=vi_det.V).solve()

N_TRACE = 30
traces = [_entropy_trace(env_e, π_e)
          for _ in tqdm(range(N_TRACE), desc='Entropy traces')]

max_len = max(len(t) for t in traces)
padded  = np.full((N_TRACE, max_len), np.nan)
for i, t in enumerate(traces):
    padded[i, :len(t)] = t
mean_h = np.nanmean(padded, axis=0)
std_h  = np.nanstd(padded,  axis=0)
turns  = np.arange(1, max_len + 1)

fig2, ax2 = plt.subplots(figsize=(8, 4))
for t in traces:
    ax2.plot(range(1, len(t) + 1), t, color='steelblue', alpha=0.15, linewidth=0.8)
ax2.plot(turns, mean_h, color='steelblue', linewidth=2, label='Mean')
ax2.fill_between(turns, mean_h - std_h, mean_h + std_h,
                 alpha=0.2, color='steelblue', label='±1 std')
ax2.set_xlabel('Turn')
ax2.set_ylabel('Belief entropy H(b) (bits)')
ax2.set_title('Belief entropy over a game (QMDP, n_obs_levels=4)')
ax2.legend()
fig2.tight_layout()
fig2.savefig('pomdp_entropy.png', dpi=150)
print("Saved pomdp_entropy.png")


# ── Plot 3: QMDP vs. POMCP under model mismatch ──────────────────────────────

print("\n--- Plot 3: model mismatch ---")

π_qmdp_base  = QMDPSolver(env_det,  V=vi_det.V).solve()
π_qmdp_mis   = QMDPSolver(env_plan, V=vi_plan.V).solve()
π_pomcp_base = POMCPSolver(env_det,num_sims=5000).solve()
π_pomcp_mis  = POMCPSolver(env_plan,num_sims=5000).solve()

results3 = [
    _run_games(env_det, π_qmdp_base,  N_GAMES, label='QMDP baseline'),
    _run_games(env_det, π_qmdp_mis,   N_GAMES, label='QMDP mismatch'),
    _run_games(env_det, π_pomcp_base, N_GAMES, label='POMCP baseline'),
    _run_games(env_det, π_pomcp_mis,  N_GAMES, label='POMCP mismatch'),
]
conditions = ['QMDP\nbaseline', 'QMDP\nmismatch', 'POMCP\nbaseline', 'POMCP\nmismatch']
wins3   = [r[0] for r in results3]
losses3 = [r[1] for r in results3]
draws3  = [r[2] for r in results3]

fig3, ax3 = plt.subplots(figsize=(8, 5))
x = np.arange(len(conditions))
ax3.bar(x - bw, wins3,   bw, label='Win',  color='steelblue',
        yerr=_bar_yerr(wins3,   N_GAMES), capsize=4, error_kw={'elinewidth': 1})
ax3.bar(x,      losses3, bw, label='Loss', color='tomato',
        yerr=_bar_yerr(losses3, N_GAMES), capsize=4, error_kw={'elinewidth': 1})
ax3.bar(x + bw, draws3,  bw, label='Draw', color='gray',
        yerr=_bar_yerr(draws3,  N_GAMES), capsize=4, error_kw={'elinewidth': 1})
ax3.set_xticks(x)
ax3.set_xticklabels(conditions)
ax3.set_xlabel('Solver / planning model')
ax3.set_ylabel('Rate')
ax3.set_title('QMDP vs. POMCP under model mismatch')
ax3.set_ylim(0, 1)
ax3.legend()
fig3.tight_layout()
fig3.savefig('pomdp_mismatch.png', dpi=150)
print("Saved pomdp_mismatch.png")

plt.show()
