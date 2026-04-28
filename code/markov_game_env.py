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
from policy import SymmetricPolicy
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

    def as_p2_gameenv(self, p1_policy) -> GameEnv:
        """Create a 1-player GameEnv for P2 with P1's policy fixed.

        Uses state inversion so single-agent solvers (VI, QL) can compute P2's
        best response. The returned GameEnv has P1/P2 fields swapped in the
        initial state; VI results (DictPolicy over inverted states → P1-labeled
        actions) should be wrapped in SymmetricPolicy to get a callable P2 policy
        for the original game.
        """
        s0 = self.initial_state
        inv_initial = State(W1=s0.W2, M1=s0.M2, R1=s0.R2,
                            W2=s0.W1, M2=s0.M1, R2=s0.R1,
                            terminal=s0.terminal)
        return GameEnv(
            opponent_policy=SymmetricPolicy(p1_policy),
            initial_state=inv_initial,
            reward=self._reward,
        )

    # ── Game simulation ───────────────────────────────────────────────────────

    def simulate(self, p1_policy, p2_policy, max_turns: int = 50,
                 save_path: str = 'auto') -> str:
        """Run one complete game and print a turn-by-turn trace.

        Parameters
        ----------
        p1_policy : callable(State) -> Action
        p2_policy : callable(State) -> Action
        max_turns : int
        save_path : str
            Path for the .npz trace file. Defaults to 'auto', which generates
            a unique timestamped filename. Pass None to disable saving.

        Returns
        -------
        str : 'P1', 'P2', or 'Draw'
        """
        import numpy as np, time as _time
        if save_path == 'auto':
            save_path = f'mg_trace_{int(_time.time()*1000)}.npz'
        s = self.initial_state
        W1_h, M1_h, W2_h, M2_h = [s.W1], [s.M1], [s.W2], [s.M2]

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
                if save_path:
                    np.savez(save_path,
                             W1=np.array(W1_h), M1=np.array(M1_h),
                             W2=np.array(W2_h), M2=np.array(M2_h),
                             winner=np.array([winner]))
                return winner

            a1 = p1_policy(s)
            a2 = p2_policy(s)
            print(f"{turn:<5} | {str(a1):<22} | {str(a2):<22} | "
                  f"({s.W1:02d},{s.M1:02d},{s.R1:02d} | "
                  f"{s.W2:02d},{s.M2:02d},{s.R2:02d} | {s.terminal})")

            s = self.joint_model.joint_sample(s, a1, a2)
            W1_h.append(s.W1); M1_h.append(s.M1)
            W2_h.append(s.W2); M2_h.append(s.M2)

        winner = 'Draw'
        print(f"\nGame Over! Draw - maximum turns reached\n")
        if save_path:
            np.savez(save_path,
                     W1=np.array(W1_h), M1=np.array(M1_h),
                     W2=np.array(W2_h), M2=np.array(M2_h),
                     winner=np.array([winner]))
        return winner

    def simulate_trace(self, p1_policy, p2_policy, max_turns: int = 50) -> dict:
        """Run one game silently and return turn-by-turn unit counts.

        Returns
        -------
        dict with keys 'W1', 'M1', 'W2', 'M2' (np.ndarray, length = turns+1)
        and 'winner' ('P1', 'P2', or 'Draw').
        """
        import numpy as np
        s = self.initial_state
        W1, M1 = [s.W1], [s.M1]
        W2, M2 = [s.W2], [s.M2]

        for _ in range(max_turns):
            if s.terminal:
                break
            s = self.joint_model.joint_sample(s, p1_policy(s), p2_policy(s))
            W1.append(s.W1); M1.append(s.M1)
            W2.append(s.W2); M2.append(s.M2)

        winner = s.winner() if s.terminal else 'Draw'
        return {
            'W1': np.array(W1, dtype=np.int32), 'M1': np.array(M1, dtype=np.int32),
            'W2': np.array(W2, dtype=np.int32), 'M2': np.array(M2, dtype=np.int32),
            'winner': winner,
        }
