import numpy as np
from game_env import GameEnv
from state import State
from policies import alternating_training_attack


class QLearning:
    """Tabular Q-learning agent for a GameEnv."""

    def __init__(self, env: GameEnv, γ=0.95, α=0.05, ϵ_start=0.2, ϵ_min=0.05):
        self.env = env
        self.γ = γ
        self.α = α
        self.ϵ_start = ϵ_start
        self.ϵ_min = ϵ_min
        self.Q = {(s, a): 0.0 for s in env.S for a in env.A}

    def train(self, n_episodes=100):
        for i in range(1, n_episodes + 1):
            ϵ = max(self.ϵ_min, self.ϵ_start * (1 - i / n_episodes))
            self._episode(ϵ)
            if i % 100 == 0:
                print(f"Episode {i}/{n_episodes}")

    def policy(self, s):
        """Return the greedy action from the learned Q table."""
        return max(self.env.A, key=lambda a: self.Q[(s, a)])

    # ------------------------------------------------------------------

    def _episode(self, ϵ):
        env = self.env
        env.reset()
        s = env.observe()

        while not s.terminal:
            a = self._ϵ_greedy(s, ϵ)
            r = env.act(a)
            sp = env.observe()

            best_next = max(self.Q[(sp, ap)] for ap in env.A)
            self.Q[(s, a)] += self.α * (r + self.γ * best_next - self.Q[(s, a)])
            s = sp

        # terminal update
        a = self._ϵ_greedy(s, ϵ)
        r = env.act(a)
        self.Q[(s, a)] += self.α * (r - self.Q[(s, a)])

    def _ϵ_greedy(self, s, ϵ):
        if np.random.rand() < ϵ:
            return np.random.choice(self.env.A)
        return self.policy(s)


if __name__ == '__main__':
    s_init = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)
    env = GameEnv(opponent_policy=alternating_training_attack, initial_state=s_init)

    agent = QLearning(env, γ=0.95, α=0.1, ϵ_start=0.2, ϵ_min=0.05)
    print("Training Q-learning agent...")
    agent.train(n_episodes=2000)
    print("Training complete.")

    print("\nSimulating games with learned policy...")
    for _ in range(5):
        env.simulate(agent.policy, label='Q-Learning')
