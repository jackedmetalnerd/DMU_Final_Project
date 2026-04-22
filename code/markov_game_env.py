"""
markov_game_env.py
==================
MarkovGameEnv: two-player Markov game (stochastic game) environment.

Unlike GameEnv (which fixes P2's policy at construction time), MarkovGameEnv
has no fixed opponent. Both players supply actions each step. This is the
natural formulation for solving true Markov games via Nash equilibrium methods.

Does NOT inherit from MDP -- the MDP base class assumes a single agent with a
single action set and scalar reward vector. MarkovGameEnv holds a
JointTransitionModel directly and defines separate reward functions for each
player by exploiting the game's symmetry (R2 = invert-state then apply R1).

Key methods:
  step(a1, a2)             -- advance game state, return (r1, r2)
  reward_p1(s) / reward_p2(s) -- per-player rewards
  as_p1_gameenv(p2_policy) -- 1-player GameEnv view for single-agent solvers
  simulate(p1_policy, p2_policy) -- full game trace with joint transitions
"""

from state import State
from action import Action
from reward import Reward
from joint_transition import JointTransitionModel
from game_env import GameEnv


class MarkovGameEnv:
    """Two-player Markov game environment with no fixed opponent policy.

    Parameters
    ----------
    initial_state : State
        Starting state (default: symmetric 1W/1M/1R for each player).
    gamma : float
        Discount factor. Used by solvers operating on this environment.
    reward : Reward, optional
        Reward function from P1's perspective. P2's reward is derived by
        state inversion. Defaults to Reward() which uses ACTIVE reward fn.
    """

    S_INIT = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

    def __init__(self, initial_state: State = S_INIT, gamma: float = 0.95,
                 reward: Reward = None):
        states = State.build_space()
        state_index = {s: i for i, s in enumerate(states)}
        self.joint_model = JointTransitionModel(states, state_index)
        self.initial_state = initial_state
        self.state = initial_state
        self.gamma = gamma
        self._reward = reward if reward is not None else Reward()

    # ── Reward ────────────────────────────────────────────────────────────────

    def reward_p1(self, s: State) -> float:
        """P1's reward at state s (standard perspective)."""
        return self._reward.evaluate(s)

    def reward_p2(self, s: State) -> float:
        """P2's reward at state s, derived by state inversion.

        Swaps W1↔W2, M1↔M2, R1↔R2 then evaluates the same reward function.
        Gives +1 when P2 wins, -1 when P1 wins (zero-sum symmetric game).
        """
        s_inv = State(W1=s.W2, M1=s.M2, R1=s.R2,
                      W2=s.W1, M2=s.M1, R2=s.R1,
                      terminal=s.terminal)
        return self._reward.evaluate(s_inv)

    # ── RL interface ──────────────────────────────────────────────────────────

    def step(self, a1: Action, a2: Action) -> tuple:
        """Apply joint action (a1, a2), advance internal state.

        Returns
        -------
        (r1, r2) : tuple[float, float]
            Rewards for P1 and P2 after the transition.
            Returns (0.0, 0.0) without advancing if the state is already terminal.
        """
        if self.state.terminal:
            return 0.0, 0.0
        self.state = self.joint_model.joint_sample(self.state, a1, a2)
        return self.reward_p1(self.state), self.reward_p2(self.state)

    def observe(self) -> State:
        """Return the current game state."""
        return self.state

    def reset(self) -> None:
        """Reset to the initial state."""
        self.state = self.initial_state

    # ── Single-player GameEnv factory ─────────────────────────────────────────

    def as_p1_gameenv(self, p2_policy) -> GameEnv:
        """Create a 1-player GameEnv for P1 with P2's policy fixed.

        Useful for running single-agent solvers (VI, QL, MCTSSolver) from P1's
        perspective against a known opponent. p2_policy must accept a real State
        and return a P2-labeled Action.
        """
        return GameEnv(
            opponent_policy=p2_policy,
            initial_state=self.initial_state,
            reward=self._reward,
        )

    # ── Game simulation ───────────────────────────────────────────────────────

    def simulate(self, p1_policy, p2_policy, max_turns: int = 50) -> str:
        """Run one complete game and print a turn-by-turn trace.

        Uses joint_model.joint_sample for transitions (both actions applied jointly).

        Parameters
        ----------
        p1_policy : callable(State) -> Action  (P1-labeled action)
        p2_policy : callable(State) -> Action  (P2-labeled action)
        max_turns : int

        Returns
        -------
        str : 'P1', 'P2', or 'Draw'
        """
        s = self.initial_state
        print(f"{'Turn':<5} | {'P1 Action':<22} | {'P2 Action':<22} | "
              f"(W1,M1,R1 | W2,M2,R2 | term)")
        print("-" * 90)

        for turn in range(1, max_turns + 1):
            if s.terminal:
                print(f"END   | {'TERMINAL':<22} | {'TERMINAL':<22} | "
                      f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | "
                      f"{s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")
                winner = s.winner()
                print(f"\nGame Over! Winner: {winner} in {turn - 1} turns\n")
                return winner

            a1 = p1_policy(s)
            a2 = p2_policy(s)
            print(f"{turn:<5} | {str(a1):<22} | {str(a2):<22} | "
                  f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | "
                  f"{s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")

            s = self.joint_model.joint_sample(s, a1, a2)

        print(f"\nGame Over! Draw - maximum turns reached\n")
        return 'Draw'
