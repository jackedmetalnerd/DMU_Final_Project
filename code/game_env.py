from state import State
from transition import TransitionModel


class GameEnv:
    """MDP environment for the two-player resource-and-combat game."""

    S_INIT = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

    def __init__(self, opponent_policy, initial_state=S_INIT, gamma=0.95, reward=None):
        from reward import Reward

        self.initial_state = initial_state
        self.state         = initial_state
        self.opponent_policy = opponent_policy
        self.gamma         = gamma

        if reward is None:
            self.reward = Reward()
        elif isinstance(reward, Reward):
            self.reward = reward
        else:
            self.reward = Reward(fn=reward)

        self.states      = State.build_space()
        self.state_index = {s: i for i, s in enumerate(self.states)}
        self.transition_model = TransitionModel(self.states, self.state_index, opponent_policy)
        self.actions     = self.transition_model.ACTIONS_P1

    # ── MDP components (standard letter aliases used by solvers / validate) ──

    @property
    def S(self):
        return self.states

    @property
    def A(self):
        return self.actions

    @property
    def T(self):
        return self.transition_model.T

    @property
    def R(self):
        return self.reward.build_vector(self.states)

    @property
    def γ(self):
        return self.gamma

    # ── S_index alias for validate.py / solver compatibility ─────────────────

    @property
    def S_index(self):
        return self.state_index

    # ── Action validity ───────────────────────────────────────────────────────

    def valid_act(self, a, s) -> bool:
        return self.transition_model.valid_act(a, s)

    # ── RL interface ──────────────────────────────────────────────────────────

    def act(self, a):
        if self.state.terminal:
            return 0.0
        self.state = self.transition_model.sample(self.state, a)
        return self.reward.evaluate(self.state)

    def observe(self):
        return self.state

    def reset(self):
        self.state = self.initial_state

    def update_P2_policy(self, new_opponent_policy):
        self.transition_model.update_P2_policy(new_opponent_policy)
        self.opponent_policy = new_opponent_policy

    # ── Game simulation ───────────────────────────────────────────────────────

    def simulate(self, p1_policy, label='P1', max_turns=50):
        """Simulate a single game. p1_policy may be a dict or callable."""
        s = self.initial_state
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

            a1 = p1_policy[s] if isinstance(p1_policy, dict) else p1_policy(s)
            a2 = self.opponent_policy(s)
            print(f"{turn:<5} | {a1:<17} | {a2:<17} | "
                  f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | "
                  f"{s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")

            s = self.transition_model.sample(s, a1)

        print('\nGame Over! Draw - maximum turns reached\n')
