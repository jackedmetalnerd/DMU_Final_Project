"""
markov_game.py
==============
Dual-MCTS Alternating Best Response solver for MarkovGameEnv.

Implements Alternating Best Response (ABR) using MCTS as the best-response
oracle. Each game, P1 and P2 independently run MCTS against the other player's
policy estimate from the previous game. After each game the estimates update to
the solved MCTS policies.

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

The ABR loop here uses the existing MCTSSolver unchanged. Each player's MCTS
runs on a single-player GameEnv view (as_p1_gameenv / as_p2_gameenv) with the
opponent's current policy frozen.

Helper functions:
  _p2_policy_to_p1(p2_policy)          -- convert P2-labeled policy to P1-labeled
  _inverted_mcts_to_p2_policy(solver)  -- wrap P2's MCTS (inverted env) as P2 policy
"""

from math import sqrt
from state import State
from action import Action
from policy import MCTSPolicy
from markov_game_env import MarkovGameEnv
from mcts import MCTSSolver
from policies import alternating_training_attack


# ── Policy conversion helpers ─────────────────────────────────────────────────

def _p2_policy_to_p1(p2_policy):
    """Convert a P2-labeled policy to a P1-labeled policy via state inversion.

    Used to bootstrap p1_estimate from a hand-coded P2 policy such as
    alternating_training_attack.

    Takes a real State, returns a P1-labeled Action by:
      1. Inverting state (W1↔W2, M1↔M2, R1↔R2)
      2. Calling p2_policy on the inverted state -> P2 action
      3. Remapping P2 action label to P1 equivalent
    """
    _P2_TO_P1 = {
        Action.P2_TRAIN_WORKERS: Action.P1_TRAIN_WORKERS,
        Action.P2_TRAIN_MARINES: Action.P1_TRAIN_MARINES,
        Action.P2_ATTACK:        Action.P1_ATTACK,
    }

    def p1_policy(s: State) -> Action:
        s_inv = State(W1=s.W2, M1=s.M2, R1=s.R2,
                      W2=s.W1, M2=s.M1, R2=s.R1, terminal=s.terminal)
        a2 = p2_policy(s_inv)
        return _P2_TO_P1[a2]

    return p1_policy


def _inverted_mcts_to_p2_policy(p2_mcts_solver):
    """Wrap P2's MCTSSolver (trained on an inverted GameEnv) as a P2-labeled policy.

    P2's MCTSSolver was built on an inverted GameEnv (P2 in P1 slots), so
    get_action() expects an inverted State and returns a P1-labeled Action.
    This wrapper handles the inversion/relabeling so callers can query it with
    a real State and receive a P2-labeled Action.

    Steps:
      1. Invert the real state (W1↔W2, M1↔M2, R1↔R2)
      2. Call p2_mcts_solver.get_action(s_inv) -> P1-labeled Action
      3. Remap P1 label to P2 equivalent
    """
    _P1_TO_P2 = {
        Action.P1_TRAIN_WORKERS: Action.P2_TRAIN_WORKERS,
        Action.P1_TRAIN_MARINES: Action.P2_TRAIN_MARINES,
        Action.P1_ATTACK:        Action.P2_ATTACK,
    }

    def p2_policy(s: State) -> Action:
        s_inv = State(W1=s.W2, M1=s.M2, R1=s.R2,
                      W2=s.W1, M2=s.M1, R2=s.R1, terminal=s.terminal)
        a1 = p2_mcts_solver.get_action(s_inv)
        return _P1_TO_P2[a1]

    return p2_policy


# ── MarkovGameSolver ──────────────────────────────────────────────────────────

class MarkovGameSolver:
    """Dual-MCTS Alternating Best Response solver for MarkovGameEnv.

    Each game:
      1. P1 runs MCTS on as_p1_gameenv(p2_estimate) -- best response to P2's policy
      2. P2 runs MCTS on as_p2_gameenv(p1_estimate) -- best response to P1's policy
      3. Simulate the game using both MCTS policies and joint transitions
      4. Update p1_estimate = MCTSPolicy(p1_mcts)
               p2_estimate = _inverted_mcts_to_p2_policy(p2_mcts)

    Fresh MCTSSolver objects are created each game. The frozen policy closures
    from the previous game carry forward as opponent estimates for the next game.

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
            Initial estimate of P1's policy (used by P2's MCTS as opponent).
            Defaults to the symmetric mirror of alternating_training_attack.
        p2_estimate : callable(State) -> P2 Action, optional
            Initial estimate of P2's policy (used by P1's MCTS as opponent).
            Defaults to alternating_training_attack.

        Returns
        -------
        list[str] : winner of each game ('P1', 'P2', or 'Draw')
        """
        if p2_estimate is None:
            p2_estimate = alternating_training_attack
        if p1_estimate is None:
            p1_estimate = _p2_policy_to_p1(alternating_training_attack)

        results = []

        for game_num in range(1, self.num_games + 1):
            print(f"\n{'=' * 70}")
            print(f"  GAME {game_num} / {self.num_games}")
            print(f"  P1 MCTS vs P2 MCTS (Alternating Best Response)")
            print(f"{'=' * 70}")

            # Build per-game single-player envs for each MCTS solver
            p1_env = self.env.as_p1_gameenv(p2_estimate)
            p2_env = self.env.as_p2_gameenv(p1_estimate)

            # Create fresh MCTS solvers for this game
            p1_mcts = MCTSSolver(
                p1_env,
                c=self.mcts_c,
                depth=self.mcts_depth,
                num_runs=self.mcts_runs,
            )
            p2_mcts = MCTSSolver(
                p2_env,
                c=self.mcts_c,
                depth=self.mcts_depth,
                num_runs=self.mcts_runs,
            )

            # Build callable policies for the joint simulation
            # P1: direct MCTS queries on real states (P1-labeled actions)
            def p1_policy(s: State, solver=p1_mcts) -> Action:
                return solver.get_action(s)

            # P2: MCTS on inverted env, wrapped to return P2-labeled actions
            p2_policy = _inverted_mcts_to_p2_policy(p2_mcts)

            # Simulate using joint transitions (env.simulate handles the trace)
            winner = self.env.simulate(
                p1_policy=p1_policy,
                p2_policy=p2_policy,
                max_turns=self.max_turns,
            )
            results.append(winner)

            # Update policy estimates for the next game:
            # P1's estimate: MCTSPolicy wraps the solver (direct callable on real states)
            p1_estimate = MCTSPolicy(p1_mcts)
            # P2's estimate: inverted MCTS wrapper (handles state inversion + relabeling)
            p2_estimate = _inverted_mcts_to_p2_policy(p2_mcts)

        # Summary
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
        num_games=3,
        mcts_c=sqrt(2),
        mcts_depth=50,
        mcts_runs=10_000,
        max_turns=50,
    )
    solver.run()
