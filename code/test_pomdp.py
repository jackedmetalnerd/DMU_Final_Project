"""
test_pomdp.py
=============
Experiments comparing QMDP solver performance under different POMDP conditions.

Experiments
-----------
exp_obs_levels     — sweeps n_obs_levels (1, 2, 4, 11) to show how observation
                     granularity affects QMDP win rates against a fixed P2 policy
exp_P2_policy_type — compares QMDP against deterministic vs stochastic P2 policy;
                     stochastic case uses a uniform-mixture T_P2 via build_uniform_P2()
exp_model_mismatch — P1 plans using a uniform-random P2 model but simulates against
                     the true deterministic P2 policy; tests effect of model mismatch

Usage
-----
python test_pomdp.py                        # all experiments
python test_pomdp.py --exp obs              # obs granularity only
python test_pomdp.py --games 100 --seed 42  # larger sample, different seed
"""

from state import State
from pomdp_env import POMDPEnv
from policies import alternating_training_attack
from value_iteration import ValueIteration
from pomdp_solver import QMDPSolver
import numpy as np
import argparse
from action import Action
from policy import FunctionPolicy

S_INIT = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

P2_ACTIONS = [Action.P2_TRAIN_WORKERS, Action.P2_TRAIN_MARINES, Action.P2_ATTACK]

random_P2 = FunctionPolicy(lambda s: np.random.choice(P2_ACTIONS))

def _run_games(env, policy, n_games, step_limit=50): #game simulation helper
    wins = losses = draws = 0
    for _ in range(n_games):
        env.reset()
        for _ in range(step_limit):
            if env.observe_raw().terminal:
                break
            b = env.observe()
            a = policy(b)
            env.act(a)
        s = env.observe_raw()
        if s.M1 > 0 and s.M2 == 0:
            wins += 1
        elif s.M2 > 0 and s.M1 == 0:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws

def exp_obs_levels(n_games, seed):
    # QMDP policy performance against various observation qualities
    print(f"\n{'+'*60}")
    print("Experiment: QMDP policy performance with various observability levels")
    print(f"\n{'+'*60}")
    np.random.seed(seed)

    #Build matrices once (no T dependence on obs discretization)
    env_base = POMDPEnv(opponent_policy=alternating_training_attack,
                        initial_state=S_INIT, n_obs_levels=1)
    vi = ValueIteration(env_base)
    vi.solve()

    levels = [1,2,4,11] #no observation, presence/absence, bucketing, full observability
    labels = {1: "n=1  (blind)", 2: "n=2             ",
              4: "n=4          ", 11: "n=11 (perfect) "}
    for n in levels:
        env = POMDPEnv(opponent_policy=alternating_training_attack,
                       initial_state=S_INIT, n_obs_levels=n)
        #sub in pre-built matrices
        env.transition_model._T_base = env_base.transition_model._T_base
        env.transition_model._T_res = env_base.transition_model._T_res
        env.transition_model._T_P2 = env_base.transition_model._T_P2
        env.transition_model.T = env_base.transition_model.T
        π = QMDPSolver(env, V=vi.V).solve()
        wins, losses, draws = _run_games(env, π, n_games)
        w, l, d = wins/n_games, losses/n_games, draws/n_games
        print(f"  {labels[n]}win={w:.1%}  loss={l:.1%}  draw={d:.1%}")

def exp_P2_policy_type(n_games, seed):
    # Deterministic vs stochastic P2 policy using QMDP
    print(f"\n{'='*60}")
    print("Experiment: P2 policy type (deterministic vs stochastic)")
    print(f"\n{'='*60}")
    np.random.seed(seed)

    # Deterministic P2
    env_det = POMDPEnv(opponent_policy=alternating_training_attack, initial_state=S_INIT, n_obs_levels=1)
    vi_det = ValueIteration(env_det)
    vi_det.solve()
    π_det = QMDPSolver(env_det, V=vi_det.V).solve()
    wins, losses, draws = _run_games(env_det, π_det, n_games)
    w, l, d = wins/n_games, losses/n_games, draws/n_games
    print(f"  {"Deterministic P2"}  win={w:.1%}  loss={l:.1%}  draw={d:.1%}")

    # Stochastic P2
    env_sto = POMDPEnv(opponent_policy=random_P2, initial_state=S_INIT, n_obs_levels=1)
    vi_sto = ValueIteration(env_sto)
    vi_sto.solve()
    env_sto.transition_model.build_uniform_P2()
    π_sto = QMDPSolver(env_sto, V=vi_sto.V).solve()
    wins, losses, draws = _run_games(env_sto, π_sto, n_games)
    w, l, d = wins/n_games, losses/n_games, draws/n_games
    print(f"  {"Stochastic P2"}  win={w:.1%}  loss={l:.1%}  draw={d:.1%}")

def exp_model_mismatch(n_games, seed):
    # P1 plans based on wrong P2 policy
    print(f"\n{'='*60}")
    print(f"Experiment: Model mismatch (P1 plans on wrong P2 policy)")
    print(f"\n{'='*60}")
    np.random.seed(seed)

    # Baseline: plan=det, real=det
    env_det = POMDPEnv(opponent_policy=alternating_training_attack,
                       initial_state=S_INIT, n_obs_levels=1)
    vi_det = ValueIteration(env_det)
    vi_det.solve()
    π_baseline = QMDPSolver(env_det, V=vi_det.V).solve()
    wins, losses, draws = _run_games(env_det, π_baseline, n_games)
    w, l, d = wins/n_games, losses/n_games, draws/n_games
    print(f"  Baseline (plan=det,  sim=det)  win={w:.1%}  loss={l:.1%}  draw={d:.1%}")

    # Mismatch: plan=rand, sim=det
    env_plan = POMDPEnv(opponent_policy=random_P2,
                       initial_state=S_INIT, n_obs_levels=1)
    vi_plan = ValueIteration(env_plan)
    vi_plan.solve()
    env_plan.transition_model.build_uniform_P2()
    π_mismatch = QMDPSolver(env_plan, V=vi_plan.V).solve()

    wins, losses, draws = _run_games(env_det, π_mismatch, n_games)
    w, l, d = wins/n_games, losses/n_games, draws/n_games
    print(f"  Mismatch (plan=rand, sim=det)  win={w:.1%}  loss={l:.1%}  draw={d:.1%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POMDP experiments')
    parser.add_argument('--exp', nargs='+', default=['obs','P2','mismatch'],
                        choices=['obs','P2', 'mismatch'],
                        help='Experiments to run (default: all)')
    parser.add_argument('--games', type=int, default=50)
    parser.add_argument('--seed', type=int, default=16)
    args = parser.parse_args()

    if 'obs' in args.exp:
        exp_obs_levels(args.games, args.seed)
    if 'P2' in args.exp:
        exp_P2_policy_type(args.games, args.seed)
    if 'mismatch' in args.exp:
        exp_model_mismatch(args.games, args.seed)

    print()

# # Build POMDP
# print("Building POMDP environment...")
# env_pomdp = POMDPEnv(opponent_policy=alternating_training_attack,
#                      initial_state=s_init, n_obs_levels=1)
# # QMDP is very effective when n_obs_levels=4. Imperfect obs, but still enough info.
# # Slightly more difficult at n_obs_levels=2, but P1 still nearly always wins.
# # Same at n_obs_levels=1: P1 doesn't need obs of P2 to play well

# # Run Value Iteration on underlying MDP
# print("Running VI...")
# vi = ValueIteration(env_pomdp)
# vi.solve()
# V = vi.V

# # Create QMDP policy
# qmdp = QMDPSolver(env_pomdp, V=V)
# π_qmdp = qmdp.solve()

# # Game simulator - copied with minor updates from MDP version
# def simulate_pomdp(env, policy, label='no label provided', max_turns=50):
#     env.reset()
#     print(f'P1 using {label} policy (POMDP)')
#     print(f"{'Turn':<5} | {'P1 Action':<17} | {'P2 Action':<17} | "
#           f"(W1,M1,R1 | W2,M2,R2 | terminal) | Belief entropy")
#     print("-" * 95)

#     for turn in range(1, max_turns + 1):
#         s = env.observe_raw()
#         if s.terminal:
#             winner = "Winner: P1" if s.M1 > 0 and s.M2 == 0 else \
#                      "Winner: P2" if s.M2 > 0 and s.M1 == 0 else "Draw"
#             print(f"END   | {'TERMINAL':<17} | {'TERMINAL':<17} | "
#                   f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | {s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")
#             print(f"\nGame Over! {winner} in {turn - 1} turns\n")
#             return

#         b = env.observe()
#         a1 = policy(b)
#         a2 = env.opponent_policy(s)

#         # Belief entropy: how uncertain is P1? Higher = more uncertain
#         b_nonzero = b[b > 0]
#         entropy = -np.sum(b_nonzero * np.log2(b_nonzero))

#         print(f"{turn:<5} | {str(a1):<17} | {str(a2):<17} | "
#               f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | {s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})        | H={entropy:.2f}")

#         env.act(a1)

#     print('\nGame Over! Draw - maximum turns reached\n')

# # Simulate one game for visualization
# simulate_pomdp(env_pomdp, π_qmdp, 'QMDP')

# # Run multiple games with QMDP policy
# n_games = 25
# wins, losses, draws = 0, 0, 0
# for _ in range(n_games):
#     env_pomdp.reset()
#     for _ in range(200): #step limit
#         if env_pomdp.observe_raw().terminal:
#             break
#         b = env_pomdp.observe()
#         a = π_qmdp(b)
#         env_pomdp.act(a)
#     s = env_pomdp.observe_raw()
#     if s.M1 > 0 and s.M2 == 0:
#         wins += 1
#     elif s.M2 > 0 and s.M1 == 0:
#         losses += 1
#     else:
#         draws += 1

# print(f"QMDP results over {n_games} games: \nWins   - {wins} \nLosses - {losses} \nDraws  - {draws}")