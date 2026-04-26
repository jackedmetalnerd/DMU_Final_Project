"""
markov_game_mcts.py
===================
MarkovGameMCTSSolver: MCTS for one player in a two-player MarkovGameEnv.

Subclasses MCTSSolver to operate directly on MarkovGameEnv rather than the
single-agent GameEnv. Takes a `player` parameter ('P1' or 'P2') and an
`opponent_policy` callable. No state inversion or action label remapping needed.

Design compared to using MCTSSolver on a GameEnv:
  Old approach: invert state so P2 looks like P1, wrap MCTSSolver on GameEnv
    → required _inverted_mcts_to_p2_policy and _p2_policy_to_p1 helpers
    → actions and rewards needed remapping at every step

  New approach: MarkovGameMCTSSolver(env, player='P2', opponent_policy=...)
    → P2's MCTS uses Action.P2_ACTIONS and env.reward_p2 directly
    → _sample_step routes (self_action, opponent_action) correctly to joint_sample
    → no conversion needed anywhere

Opponent policy note:
  The opponent_policy is called once per tree step (not per rollout step for
  random rollouts). For ABR between games, update opponent_policy to the
  previous game's get_greedy_policy() result — this is an O(1) dict lookup
  rather than re-running MCTS, avoiding nested MCTS calls during rollouts.
"""

import random
import numpy as np
from math import sqrt, log

from mcts import MCTSSolver
from markov_game_env import MarkovGameEnv
from state import State
from action import Action


class MarkovGameMCTSSolver(MCTSSolver):
    """MCTS solver for one player in a MarkovGameEnv.

    Parameters
    ----------
    env             : MarkovGameEnv
    player          : str   -- 'P1' or 'P2'
    opponent_policy : callable(State) -> Action
        Policy for the opposing player. Used in _sample_step to complete the
        joint transition when the solver's player chooses an action.
        Should return P1-labeled actions if player='P2', and vice versa.
    c               : float -- UCB1 exploration constant
    depth           : int   -- MCTS rollout depth
    num_runs        : int   -- MCTS simulations per get_action call
    """

    def __init__(self, env: MarkovGameEnv, player: str, opponent_policy,
                 c: float = sqrt(2), depth: int = 50, num_runs: int = 10_000):
        # Call grandparent Solver.__init__ directly to set self.env,
        # then set MCTSSolver attributes manually — avoids GameEnv type assumption.
        super(MCTSSolver, self).__init__(env)
        self.c        = c
        self.depth    = depth
        self.num_runs = num_runs
        self.n = {}
        self.q = {}
        self.t = {}

        assert player in ('P1', 'P2'), f"player must be 'P1' or 'P2', got '{player}'"
        self.player          = player
        self.opponent_policy = opponent_policy

        if player == 'P1':
            self.actions   = Action.P1_ACTIONS
            self.reward_fn = env.reward_p1
        else:
            self.actions   = Action.P2_ACTIONS
            self.reward_fn = env.reward_p2

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sample_step(self, s: State, a: Action) -> State:
        """Sample next state: self plays `a`, opponent plays their policy."""
        a_opp = self.opponent_policy(s)
        if self.player == 'P1':
            return self.env.joint_model.joint_sample(s, a, a_opp)
        else:
            return self.env.joint_model.joint_sample(s, a_opp, a)

    def _valid_actions(self, s: State) -> list:
        """Return valid actions for this player from state s."""
        return [a for a in self.actions
                if self.env.joint_model.valid_act(a, s)]

    # ── Public interface ──────────────────────────────────────────────────────

    def get_action(self, s: State) -> Action:
        """Run num_runs MCTS simulations from s, return best action."""
        for _ in range(self.num_runs):
            self._run(s, self.depth)
        q_vals = np.array([self.q.get((s, a), 0.0) for a in self.actions])
        best = np.flatnonzero(q_vals == q_vals.max())
        return self.actions[random.choice(list(best))]

    def get_greedy_policy(self):
        """Return a fast greedy policy from the current Q-table (no new MCTS runs).

        Useful as an opponent_policy for the next game's solver — O(1) dict lookup
        instead of re-running num_runs simulations per step.
        Falls back to a random valid action for states not in the tree.
        """
        q_snapshot = dict(self.q)
        actions    = self.actions

        def policy(s: State) -> Action:
            valid = [a for a in actions
                     if self.env.joint_model.valid_act(a, s)]
            if not valid:
                return actions[0]
            known = [a for a in valid if (s, a) in q_snapshot]
            if known:
                return max(known, key=lambda a: q_snapshot[(s, a)])
            return random.choice(valid)

        return policy

    # ── MCTS internals (override parent to use player-specific attributes) ────

    def _ucb_action(self, s: State) -> Action:
        valid = self._valid_actions(s)
        Ns = sum(self.n.get((s, a), 0) for a in valid)
        unvisited = [a for a in valid if self.n.get((s, a), 0) == 0]
        if unvisited:
            return random.choice(unvisited)
        return max(valid, key=lambda a: (
            self.q.get((s, a), 0.0) +
            self.c * (sqrt(log(Ns) / self.n[(s, a)]) if self.n[(s, a)] > 0 else float('inf'))
        ))

    def _rollout(self, s: State, max_steps: int = 50) -> float:
        r_total, t = 0.0, 0
        while t < max_steps:
            r_total += (self.env.gamma ** t) * self.reward_fn(s)
            if s.terminal:
                break
            valid = self._valid_actions(s)
            a = random.choice(valid)
            s = self._sample_step(s, a)
            t += 1
        return r_total

    def _run(self, s: State, depth: int) -> float:
        if depth <= 0 or s.terminal:
            return self.reward_fn(s)

        if (s, self.actions[0]) not in self.n:
            for a in self._valid_actions(s):
                self.n[(s, a)] = 0
                self.q[(s, a)] = 0.0
            return self._rollout(s, depth)

        a  = self._ucb_action(s)
        sp = self._sample_step(s, a)

        q_it = self.reward_fn(s) + self.env.gamma * self._run(sp, depth - 1)
        self.n[(s, a)] += 1
        self.q[(s, a)] += (q_it - self.q[(s, a)]) / self.n[(s, a)]
        key = (s, a, sp)
        self.t[key] = self.t.get(key, 0) + 1

        return q_it
