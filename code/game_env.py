from state import State
from transition import TransitionModel
from mdp import MDP


class GameEnv(MDP):
    """MDP environment for the two-player resource-and-combat game.

    Inherits the formal MDP structure (S, A, T, R, γ) from MDP and adds
    the RL/simulation interface: act(), observe(), reset(), simulate(),
    and update_P2_policy().
    """

    S_INIT = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

    def __init__(self, opponent_policy, initial_state=S_INIT, gamma=0.95, reward=None):
        from reward import Reward

        if reward is None:
            reward_obj = Reward()
        elif isinstance(reward, Reward):
            reward_obj = reward
        else:
            reward_obj = Reward(fn=reward)

        states           = State.build_space()
        transition_model = TransitionModel(states, {s: i for i, s in enumerate(states)}, opponent_policy)

        super().__init__(
            states           = states,
            actions          = transition_model.ACTIONS_P1,
            transition_model = transition_model,
            reward           = reward_obj,
            gamma            = gamma,
        )

        self.initial_state   = initial_state
        self.state           = initial_state
        self.opponent_policy = opponent_policy

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
