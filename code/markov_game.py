"""
markov_game.py
==============
Dual-MCTS Alternating Best Response solver for MarkovGameEnv.

Implements Alternating Best Response (ABR) using MarkovGameMCTSSolver as the
best-response oracle. Each game, P1 and P2 independently run MCTS against
the other player's policy estimate from the previous game. After each game the
estimates update to greedy policies derived from the MCTS Q-tables (O(1) lookup,
no nested MCTS calls during opponent rollouts).

This is one of several methods for solving Markov (stochastic) games:

  Method                    | Description
  ──────────────────────────┼────────────────────────────────────────────────
  Minimax Value Iteration   | Solve LP at each state: V(s) = maximin_a1
                            | minimax_a2 [R + γ·ΣT·V]. Exact Nash for zero-sum.
  Nash Q-Learning           | Both agents maintain Q[s,a1,a2]; solve matrix
                            | game each step for Nash equilibrium values.
  Alternating Best Response | Each player best-responds to opponent's current
  (this file)               | policy. Converges to Nash in zero-sum games.
  Correlated Equilibrium    | Single LP over joint action profiles; easier
                            | than Nash but weaker for zero-sum games.
  Multi-Agent DQN           | Independent DQN agents; non-stationary but
                            | practical for large state spaces.
"""

from math import sqrt
from action import Action
from markov_game_env import MarkovGameEnv
from markov_game_mcts import MarkovGameMCTSSolver
from policies import alternating_training_attack


# ── Default bootstrap policies ────────────────────────────────────────────────

def _default_p1_policy(s):
    """P1 mirror of alternating_training_attack: train to max then attack."""
    if s.M1 == 10 and s.W1 == 10:
        return Action.P1_ATTACK
    if s.W1 < s.M1:
        return Action.P1_TRAIN_WORKERS
    return Action.P1_TRAIN_MARINES


# ── MarkovGameSolver ──────────────────────────────────────────────────────────

class MarkovGameSolver:
    """Dual-MCTS Alternating Best Response solver for MarkovGameEnv.

    Each game:
      1. p1_mcts = MarkovGameMCTSSolver(env, 'P1', opponent_policy=p2_estimate)
      2. p2_mcts = MarkovGameMCTSSolver(env, 'P2', opponent_policy=p1_estimate)
      3. Simulate using env.simulate(p1_mcts.get_action, p2_mcts.get_action)
      4. Update estimates to greedy policies from each solver's Q-table

    Policy estimates are greedy lookups into the previous game's Q-table (O(1)
    per call), not full MCTS re-runs, so they're safe to use as opponent policies
    inside subsequent MCTS rollouts.

    Parameters
    ----------
    env        : MarkovGameEnv
    num_games  : int   -- number of ABR games to simulate
    mcts_c     : float -- UCB1 exploration constant
    mcts_depth : int   -- MCTS rollout depth
    mcts_runs  : int   -- MCTS simulations per action
    max_turns  : int   -- max turns per game before draw
    """

    def __init__(
        self,
        env: MarkovGameEnv,
        num_games: int = 3,
        mcts_c: float = sqrt(2),
        mcts_depth: int = 50,
        mcts_runs: int = 10_000,
        max_turns: int = 50,
    ):
        self.env        = env
        self.num_games  = num_games
        self.mcts_c     = mcts_c
        self.mcts_depth = mcts_depth
        self.mcts_runs  = mcts_runs
        self.max_turns  = max_turns

    def run(self, p1_estimate=None, p2_estimate=None) -> list:
        """Run num_games of dual-MCTS best response and return list of winners.

        Parameters
        ----------
        p1_estimate : callable(State) -> P1 Action, optional
            Initial estimate of P1's policy (used as P2's opponent in MCTS).
            Defaults to _default_p1_policy.
        p2_estimate : callable(State) -> P2 Action, optional
            Initial estimate of P2's policy (used as P1's opponent in MCTS).
            Defaults to alternating_training_attack.

        Returns
        -------
        list[str] : winner of each game ('P1', 'P2', or 'Draw')
        """
        if p1_estimate is None:
            p1_estimate = _default_p1_policy
        if p2_estimate is None:
            p2_estimate = alternating_training_attack

        results = []

        for game_num in range(1, self.num_games + 1):
            print(f"\n{'=' * 70}")
            print(f"  GAME {game_num} / {self.num_games}")
            print(f"  P1 MCTS vs P2 MCTS (Alternating Best Response)")
            print(f"{'=' * 70}")

            # Each player's MCTS plans against the opponent's current estimate
            p1_mcts = MarkovGameMCTSSolver(
                self.env, player='P1', opponent_policy=p2_estimate,
                c=self.mcts_c, depth=self.mcts_depth, num_runs=self.mcts_runs,
            )
            p2_mcts = MarkovGameMCTSSolver(
                self.env, player='P2', opponent_policy=p1_estimate,
                c=self.mcts_c, depth=self.mcts_depth, num_runs=self.mcts_runs,
            )

            winner = self.env.simulate(
                p1_policy=p1_mcts.get_action,
                p2_policy=p2_mcts.get_action,
                max_turns=self.max_turns,
            )
            results.append(winner)

            # Update estimates to greedy Q-table lookups (O(1), no nested MCTS)
            p1_estimate = p1_mcts.get_greedy_policy()
            p2_estimate = p2_mcts.get_greedy_policy()

        p1_wins = results.count('P1')
        p2_wins = results.count('P2')
        draws   = results.count('Draw')
        print(f"\n{'=' * 70}")
        print(f"  RESULTS SUMMARY ({self.num_games} games)")
        print(f"{'=' * 70}")
        for i, r in enumerate(results, 1):
            print(f"  Game {i}: {r}")
        print(f"\n  P1 wins: {p1_wins}/{self.num_games}")
        print(f"  P2 wins: {p2_wins}/{self.num_games}")
        print(f"  Draws:   {draws}/{self.num_games}")

        return results


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Initializing MarkovGameEnv...")
    env = MarkovGameEnv()

    print("Running dual-MCTS Markov game simulation (Alternating Best Response)...")
    solver = MarkovGameSolver(
        env,
        num_games= 10,
        mcts_c=sqrt(3),
        mcts_depth=50,
        mcts_runs=5_000,
        max_turns=50,
    )
    solver.run()
