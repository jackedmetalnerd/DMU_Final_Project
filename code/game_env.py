from state import State
from transition import TransitionModel


class GameEnv:
    """MDP environment for the two-player resource-and-combat game."""

    S_INIT = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

    def __init__(self, π_P2, s_init=S_INIT, γ=0.95, reward=None):
        from reward import Reward

        self.s_init = s_init
        self.s = s_init
        self.π_P2 = π_P2
        self.γ = γ
        if reward is None:
            self.reward = Reward()
        elif isinstance(reward, Reward):
            self.reward = reward
        else:
            self.reward = Reward(fn=reward)

        self.S = State.build_space()
        self.S_index = {s: i for i, s in enumerate(self.S)}
        self._model = TransitionModel(self.S, self.S_index, π_P2)
        self.A = self._model.ACTIONS_P1
        self._R = None  # built lazily on first access via env.R

    @property
    def T(self):
        return self._model.T

    @property
    def R(self):
        if self._R is None:
            self._R = self.reward.build_vector(self.S)
        return self._R

    def valid_act(self, a, s) -> bool:
        return self._model.valid_act(a, s)

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------

    def act(self, a):
        if self.s.terminal:
            return 0.0
        self.s = self._model.sample(self.s, a)
        return self.reward.evaluate(self.s)

    def observe(self):
        return self.s

    def reset(self):
        self.s = self.s_init

    def update_P2_policy(self, π_P2_new):
        self._model.update_P2_policy(π_P2_new)
        self.π_P2 = π_P2_new

    # ------------------------------------------------------------------
    # Game simulation
    # ------------------------------------------------------------------

    def simulate(self, π_P1, label='P1', max_turns=50):
        """Simulate a single game. π_P1 may be a dict or callable."""
        s = self.s_init
        print(f'P1 using {label} policy')
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

            a1 = π_P1[s] if isinstance(π_P1, dict) else π_P1(s)
            a2 = self.π_P2(s)
            print(f"{turn:<5} | {a1:<17} | {a2:<17} | "
                  f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | "
                  f"{s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")

            s = self._model.sample(s, a1)

        print('\nGame Over! Draw - maximum turns reached\n')


