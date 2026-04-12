import random
from math import sqrt, log
from game_env import GameEnv
from state import State
from policies import alternating_training_attack


class MCTSSolver:
    """Monte Carlo Tree Search agent for a GameEnv."""

    def __init__(self, env: GameEnv, c=sqrt(2), depth=50, num_runs=10000):
        self.env = env
        self.c = c
        self.depth = depth
        self.num_runs = num_runs
        self.n = {}  # visit counts  (s, a)
        self.q = {}  # action values (s, a)
        self.t = {}  # transition counts (s, a, s')

    def reset_tree(self):
        self.n, self.q, self.t = {}, {}, {}

    def get_action(self, s):
        for _ in range(self.num_runs):
            self._run(s, self.depth)
        A = self.env.A
        q_vals = np.array([self.q.get((s, a), 0.0) for a in A])
        best = np.flatnonzero(q_vals == q_vals.max())
        return A[random.choice(list(best))]

    # ------------------------------------------------------------------
    # Internal MCTS logic
    # ------------------------------------------------------------------

    def _ucb_action(self, s):
        A = self.env.A
        valid = [a for a in A if self.env.valid_act(a, s)]
        Ns = sum(self.n.get((s, a), 0) for a in valid)
        unvisited = [a for a in valid if self.n.get((s, a), 0) == 0]
        if unvisited:
            return random.choice(unvisited)
        return max(valid, key=lambda a: (
            self.q.get((s, a), 0.0) +
            self.c * (sqrt(log(Ns) / self.n[(s, a)]) if self.n[(s, a)] > 0 else float('inf'))
        ))

    def _rollout(self, s, max_steps=50):
        env = self.env
        r_total, t = 0.0, 0
        while t < max_steps:
            r_total += (env.γ ** t) * env.reward.evaluate(s)
            if s.terminal:
                break
            valid = [a for a in env.A if env.valid_act(a, s)]
            a = random.choice(valid)
            s = env._model.sample(s, a)
            t += 1
        return r_total

    def _run(self, s, depth):
        env = self.env

        if depth <= 0 or s.terminal:
            return env.reward.evaluate(s)

        if (s, env.A[0]) not in self.n:
            for a in [a for a in env.A if env.valid_act(a, s)]:
                self.n[(s, a)] = 0
                self.q[(s, a)] = 0.0
            return self._rollout(s, depth)

        a = self._ucb_action(s)
        sp = env._model.sample(s, a)

        q_it = env.reward.evaluate(s) + env.γ * self._run(sp, depth - 1)
        self.n[(s, a)] += 1
        self.q[(s, a)] += (q_it - self.q[(s, a)]) / self.n[(s, a)]
        key = (s, a, sp)
        self.t[key] = self.t.get(key, 0) + 1

        return q_it

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, s_init=None, max_turns=50):
        if s_init is None:
            s_init = self.env.s_init
        s = s_init
        env = self.env
        self.reset_tree()

        print('P1 using MCTS policy')
        print(f"{'Turn':<5} | {'P1 Action':<17} | {'P2 Action':<17} | "
              f"(W1,M1,R1 | W2,M2,R2 | terminal)")
        print("-" * 80)

        for turn in range(1, max_turns + 1):
            if s.terminal:
                print(f"END   | {'TERMINAL':<17} | {'TERMINAL':<17} | "
                      f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | "
                      f"{s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")
                print(f"\nGame Over! Winner: {s.winner()} in {turn - 1} turns\n")
                return

            a1 = self.get_action(s)
            a2 = env.π_P2(s)
            print(f"{turn:<5} | {a1:<17} | {a2:<17} | "
                  f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | "
                  f"{s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")

            s = env._model.sample(s, a1)

        print('\nGame Over! Draw - maximum turns reached\n')


if __name__ == '__main__':
    s_init = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)
    env = GameEnv(π_P2=alternating_training_attack, s_init=s_init)

    agent = MCTSSolver(env, c=sqrt(2), depth=50, num_runs=10000)
    print("Running MCTS simulations...")
    for _ in range(5):
        agent.simulate(s_init)
