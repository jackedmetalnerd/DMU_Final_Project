import numpy as np
from game_env import GameEnv
from state import State
from policy import DictPolicy
from solver import Solver
from policies import alternating_training_attack


class QLearning(Solver):
    """Tabular Q-learning agent for a GameEnv."""

    def __init__(self, env: GameEnv, gamma=0.95, alpha=0.05, epsilon_start=0.2, epsilon_min=0.05):
        super().__init__(env)
        self.gamma         = gamma
        self.alpha         = alpha
        self.epsilon_start = epsilon_start
        self.epsilon_min   = epsilon_min
        self.Q             = {(s, a): 0.0 for s in env.S for a in env.A}
        self.v0_history = []
        self._s0 = env.initial_state


    def solve(self, n_episodes=100) -> DictPolicy:
        """Train for n_episodes and return a greedy DictPolicy."""
        for i in range(1, n_episodes + 1):
            epsilon = max(self.epsilon_min, self.epsilon_start * (1 - i / n_episodes))
            self._episode(epsilon)
            s0 = self._s0
            self.v0_history.append(max(self.Q[(s0, a)] for a in self.env.A))
            
            if n_episodes >= 10_000: #cleaner output for large runs
                if i % 1000 == 0:
                    print(f"Episode {i}/{n_episodes}")
            else:
                if i % 100 == 0:
                    print(f"Episode {i}/{n_episodes}")
        return self._build_policy()

    def policy(self, s=None):
        """Return the greedy action for state s, or a DictPolicy over all states.

        Called with no arguments: returns a DictPolicy built from Q-table argmax.
        Called with a state: returns the greedy action for that single state.
        """
        if s is None:
            return self._build_policy()
        return max(self.env.A, key=lambda a: self.Q[(s, a)])

    # ------------------------------------------------------------------

    def _build_policy(self) -> DictPolicy:
        policy_dict = {s: max(self.env.A, key=lambda a: self.Q[(s, a)])
                       for s in self.env.S}
        return DictPolicy(policy_dict)

    def _episode(self, epsilon):
        env = self.env
        env.reset()
        s = env.observe()

        while not s.terminal:
            a = self._epsilon_greedy(s, epsilon)
            r = env.act(a)
            sp = env.observe()

            best_next = max(self.Q[(sp, ap)] for ap in env.A)
            self.Q[(s, a)] += self.alpha * (r + self.gamma * best_next - self.Q[(s, a)])
            s = sp

        # terminal update
        a = self._epsilon_greedy(s, epsilon)
        r = env.act(a)
        self.Q[(s, a)] += self.alpha * (r - self.Q[(s, a)])

    def _epsilon_greedy(self, s, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.env.A)
        return self.policy(s)


if __name__ == '__main__':
    s_init = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)
    env = GameEnv(opponent_policy=alternating_training_attack, initial_state=s_init)

    agent = QLearning(env, gamma=0.95, alpha=0.1, epsilon_start=0.2, epsilon_min=0.05)
    print("Training Q-learning agent...")
    π_star = agent.solve(n_episodes=2000)
    print("Training complete.")

    print("\nSimulating games with learned policy...")
    for _ in range(5):
        env.simulate(π_star, label='Q-Learning')
