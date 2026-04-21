from state import State
from pomdp_env import POMDPEnv
from policies import alternating_training_attack
from value_iteration import ValueIteration
from pomdp_solver import QMDPSolver
import numpy as np

s_init = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

# Build POMDP
print("Building POMDP environment...")
env_pomdp = POMDPEnv(opponent_policy=alternating_training_attack,
                     initial_state=s_init, n_obs_levels=1)
# QMDP is very effective when n_obs_levels=4. Imperfect obs, but still enough info.
# Slightly more difficult at n_obs_levels=2, but P1 still nearly always wins.

# Run Value Iteration on underlying MDP
print("Running VI...")
vi = ValueIteration(env_pomdp)
vi.solve()
V = vi.V

# Create QMDP policy
qmdp = QMDPSolver(env_pomdp, V=V)
π_qmdp = qmdp.solve()

# Game simulator - copied with minor updates from MDP version
def simulate_pomdp(env, policy, label='no label provided', max_turns=50):
    env.reset()
    print(f'P1 using {label} policy (POMDP)')
    print(f"{'Turn':<5} | {'P1 Action':<17} | {'P2 Action':<17} | "
          f"(W1,M1,R1 | W2,M2,R2 | terminal) | Belief entropy")
    print("-" * 95)

    for turn in range(1, max_turns + 1):
        s = env.observe_raw()
        if s.terminal:
            winner = "Winner: P1" if s.M1 > 0 and s.M2 == 0 else \
                     "Winner: P2" if s.M2 > 0 and s.M1 == 0 else "Draw"
            print(f"END   | {'TERMINAL':<17} | {'TERMINAL':<17} | "
                  f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | {s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")
            print(f"\nGame Over! {winner} in {turn - 1} turns\n")
            return

        b = env.observe()
        a1 = policy(b)
        a2 = env.opponent_policy(s)

        # Belief entropy: how uncertain is P1? Higher = more uncertain
        b_nonzero = b[b > 0]
        entropy = -np.sum(b_nonzero * np.log2(b_nonzero))

        print(f"{turn:<5} | {str(a1):<17} | {str(a2):<17} | "
              f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | {s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})        | H={entropy:.2f}")

        env.act(a1)

    print('\nGame Over! Draw - maximum turns reached\n')

# Simulate one game for visualization
simulate_pomdp(env_pomdp, π_qmdp, 'QMDP')

# Run multiple games with QMDP policy
n_games = 25
wins, losses, draws = 0, 0, 0
for _ in range(n_games):
    env_pomdp.reset()
    for _ in range(200): #step limit
        if env_pomdp.observe_raw().terminal:
            break
        b = env_pomdp.observe()
        a = π_qmdp(b)
        env_pomdp.act(a)
    s = env_pomdp.observe_raw()
    if s.M1 > 0 and s.M2 == 0:
        wins += 1
    elif s.M2 > 0 and s.M1 == 0:
        losses += 1
    else:
        draws += 1

print(f"QMDP results over {n_games} games: \nWins   - {wins} \nLosses - {losses} \nDraws  - {draws}")