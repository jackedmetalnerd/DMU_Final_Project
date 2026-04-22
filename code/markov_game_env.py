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
  step(a1, a2)            -- advance game state, return (r1, r2)
  as_p1_gameenv(p2_policy) -- 1-player GameEnv view for P1's MCTS
  as_p2_gameenv(p1_policy) -- 1-player GameEnv view for P2's MCTS (inverted state)
  simulate(p1_policy, p2_policy) -- full game trace with joint transitions

The as_p2_gameenv inversion:
  MCTSSolver always plans for "P1" (uses Action.P1_ACTIONS, P1-perspective reward).
  To plan for P2, we invert the state (swap W1/M1/R1 ↔ W2/M2/R2) so P2's data
  occupies the P1 slots. The opponent policy in that inverted env is P1's real
  policy, wrapped in SymmetricPolicy to handle the state/action relabeling.
"""

from state import State
from action import Action
from reward import Reward
from joint_transition import JointTransitionModel
from game_env import GameEnv
from policy import SymmetricPolicy


class MarkovGameEnv:
    """Two-player Markov game environment with no fixed opponent policy.

    Parameters
    ----------
    initial_state : State
        Starting state (default: symmetric 1W/1M/1R for each player).
    reward : Reward, optional
        Reward function from P1's perspective. P2's reward is derived by
        state inversion. Defaults to Reward() which uses ACTIVE reward fn.
    """

    S_INIT = State(W1=1, M1=1, R1=1, W2=1, M2=1, R2=1, terminal=0)

    def __init__(self, initial_state: State = S_INIT, reward: Reward = None):
        states = State.build_space()
        state_index = {s: i for i, s in enumerate(states)}
        self.joint_model = JointTransitionModel(states, state_index)
        self.initial_state = initial_state
        self.state = initial_state
        self._reward = reward if reward is not None else Reward()

    # ── Reward ────────────────────────────────────────────────────────────────

    def reward_p1(self, s: State) -> float:
        """P1's reward at state s (standard perspective)."""
        return self._reward.evaluate(s)

    def reward_p2(self, s: State) -> float:
        """P2's reward at state s, derived by state inversion.

        Swaps W1↔W2, M1↔M2, R1↔R2, then evaluates the same reward function.
        In the terminal-only reward case this gives +1 when P2 wins, -1 when P1 wins.
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

    # ── Single-player GameEnv factories ──────────────────────────────────────

    def as_p1_gameenv(self, p2_policy) -> GameEnv:
        """Create a 1-player GameEnv for P1's MCTS with P2's policy fixed.

        P1 is the protagonist; MCTS uses Action.P1_ACTIONS and standard state.
        p2_policy must accept a real (non-inverted) State and return a P2 Action.
        """
        return GameEnv(
            opponent_policy=p2_policy,
            initial_state=self.initial_state,
            reward=self._reward,
        )

    def as_p2_gameenv(self, p1_policy) -> GameEnv:
        """Create a 1-player GameEnv for P2's MCTS with P1's policy fixed.

        Since MCTSSolver always plans for "P1", we make P2 appear as P1 by
        inverting the state: W1↔W2, M1↔M2, R1↔R2. In the inverted env:
          - The "P1 slot" contains P2's resources (MCTS plans P2's moves)
          - The "P2 slot" (opponent) contains P1's resources
          - SymmetricPolicy(p1_policy) adapts p1_policy for the inverted state:
              1. Receives s_inv (P2's data in P1 slots)
              2. Reinverts to get the real state
              3. Calls p1_policy(s_real) -> P1 action
              4. Maps P1 action to P2 equivalent (for the inverted env's opponent)

        p1_policy must accept a real (non-inverted) State and return a P1 Action.

        The reward function evaluates the inverted state, correctly computing
        advantage from P2's perspective (P2's data is in the P1 slot).
        """
        inverted_initial = State(
            W1=self.initial_state.W2, M1=self.initial_state.M2, R1=self.initial_state.R2,
            W2=self.initial_state.W1, M2=self.initial_state.M1, R2=self.initial_state.R1,
            terminal=self.initial_state.terminal,
        )
        return GameEnv(
            opponent_policy=SymmetricPolicy(p1_policy),
            initial_state=inverted_initial,
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

        winner = s.winner() if s.terminal else 'Draw'
        print(f"\nGame Over! Draw - maximum turns reached\n")
        return 'Draw'
