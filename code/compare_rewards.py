"""
compare_rewards.py
==================
Runs solvers under multiple reward functions and reports win rates.

Usage:
    python compare_rewards.py                              # all solvers, all reward fns
    python compare_rewards.py --solver vi                  # VI only
    python compare_rewards.py --solver vi ql               # VI and Q-learning
    python compare_rewards.py --rf terminal_only win_only  # specific reward fns
    python compare_rewards.py --games 100                  # more games per estimate
    python compare_rewards.py --ql-episodes 5000           # longer Q-learning training
"""

import argparse
import numpy as np
from math import sqrt

from game_env import GameEnv
from state import State
from policies import alternating_training_attack
from reward import Reward
from value_iteration import ValueIteration
from q_learning import QLearning
from mcts import MCTSSolver

# ── Configuration ─────────────────────────────────────────────────────────────

S_INIT = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

ALL_REWARD_FNS = Reward.ALL

# ── Win-rate measurement ───────────────────────────────────────────────────────

def measure_win_rate(env, policy_fn, n_games=50, seed=42):
    """Roll out n_games episodes using policy_fn(s) -> action. Returns (win, loss, draw) rates."""
    np.random.seed(seed)
    wins = losses = draws = 0
    for _ in range(n_games):
        s = S_INIT
        for _ in range(50):
            if s.terminal:
                break
            a = policy_fn(s)
            s_idx = env.S_index[s]
            row = env.T[a].getrow(s_idx)
            probs = row.data / row.data.sum()
            s = env.S[np.random.choice(row.indices, p=probs)]
        if s.terminal:
            if s.is_win():
                wins += 1
            elif s.is_loss():
                losses += 1
            else:
                draws += 1
        else:
            draws += 1
    total = n_games
    return wins / total, losses / total, draws / total

# ── Solver runners ─────────────────────────────────────────────────────────────

def run_vi(env):
    solver = ValueIteration(env)
    print(f"    [VI] solving...")
    π = solver.solve()
    return π

def run_ql(env, n_episodes=2000):
    agent = QLearning(env, γ=0.95, α=0.1, ϵ_start=0.2, ϵ_min=0.05)
    print(f"    [QL] training {n_episodes} episodes...")
    agent.train(n_episodes=n_episodes)
    return agent.policy

def run_mcts(env, num_runs=1000):
    agent = MCTSSolver(env, c=sqrt(2), depth=50, num_runs=num_runs)
    return agent.get_action

# ── Main comparison loop ───────────────────────────────────────────────────────

def compare(solvers_to_run, reward_fns, n_games, ql_episodes, mcts_runs):
    results = {}

    for rf_fn in reward_fns:
        rf_name = rf_fn.__name__
        print(f"\n{'='*60}")
        print(f"Reward: {rf_name}")
        first_line = rf_fn.__doc__.strip().splitlines()[0]
        print(f"  {first_line}")
        print(f"{'='*60}")

        env = GameEnv(opponent_policy=alternating_training_attack, initial_state=S_INIT, reward=Reward(rf_fn))

        if 'vi' in solvers_to_run:
            policy = run_vi(env)
            wr, lr, dr = measure_win_rate(env, policy, n_games=n_games)
            results[(rf_name, 'vi')] = (wr, lr, dr)
            print(f"    [VI]   win={wr:.1%}  loss={lr:.1%}  draw={dr:.1%}")

        if 'ql' in solvers_to_run:
            policy = run_ql(env, n_episodes=ql_episodes)
            wr, lr, dr = measure_win_rate(env, policy, n_games=n_games)
            results[(rf_name, 'ql')] = (wr, lr, dr)
            print(f"    [QL]   win={wr:.1%}  loss={lr:.1%}  draw={dr:.1%}")

        if 'mcts' in solvers_to_run:
            policy = run_mcts(env, num_runs=mcts_runs)
            wr, lr, dr = measure_win_rate(env, policy, n_games=n_games)
            results[(rf_name, 'mcts')] = (wr, lr, dr)
            print(f"    [MCTS] win={wr:.1%}  loss={lr:.1%}  draw={dr:.1%}")

    return results

def print_summary(results, solvers_to_run, reward_fns):
    rf_names = [f.__name__ for f in reward_fns]
    col_w = max(len(n) for n in rf_names) + 2
    solver_labels = {'vi': 'VI', 'ql': 'QL', 'mcts': 'MCTS'}

    print(f"\n{'='*60}")
    print("WIN RATE SUMMARY")
    print(f"{'='*60}")
    header = f"{'Reward Function':<{col_w}}" + "".join(f"  {solver_labels[s]:>6}" for s in solvers_to_run)
    print(header)
    print("-" * len(header))
    for rf_fn in reward_fns:
        rn = rf_fn.__name__
        row = f"{rn:<{col_w}}"
        for s in solvers_to_run:
            key = (rn, s)
            row += f"  {results[key][0]:>5.1%}" if key in results else f"  {'N/A':>5}"
        print(row)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare reward functions across solvers')
    parser.add_argument('--solver', nargs='+', default=['vi', 'ql', 'mcts'],
                        choices=['vi', 'ql', 'mcts'], metavar='SOLVER',
                        help='Solvers to run: vi, ql, mcts (default: all)')
    parser.add_argument('--rf', nargs='+', default=None, metavar='NAME',
                        help='Reward function names to test (default: all)')
    parser.add_argument('--games', type=int, default=50,
                        help='Games per win-rate estimate (default: 50)')
    parser.add_argument('--ql-episodes', type=int, default=2000,
                        help='Q-learning training episodes (default: 2000)')
    parser.add_argument('--mcts-runs', type=int, default=1000,
                        help='MCTS rollouts per move (default: 1000)')
    args = parser.parse_args()

    rf_map = {f.__name__: f for f in ALL_REWARD_FNS}
    if args.rf:
        unknown = [n for n in args.rf if n not in rf_map]
        if unknown:
            print(f"Unknown reward functions: {unknown}")
            print(f"Available: {list(rf_map.keys())}")
            raise SystemExit(1)
        selected_rfs = [rf_map[n] for n in args.rf]
    else:
        selected_rfs = ALL_REWARD_FNS

    results = compare(
        solvers_to_run=args.solver,
        reward_fns=selected_rfs,
        n_games=args.games,
        ql_episodes=args.ql_episodes,
        mcts_runs=args.mcts_runs,
    )
    print_summary(results, args.solver, selected_rfs)
