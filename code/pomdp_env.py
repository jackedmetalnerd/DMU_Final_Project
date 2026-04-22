"""
pomdp_env.py
============
POMDP environment extending GameEnv with belief-state tracking.

Wraps the underlying MDP with an ObservationModel and maintains a
probability distribution (belief) over the state space. Belief is
updated automatically on each act() call using Bayes' rule.

Classes
-------
BeliefCollapseError — raised when belief update yields zero probability
                      for all states consistent with the observation
POMDPEnv(GameEnv)   — adds observe(), belief_update(), and step() to GameEnv;
                      observe() returns the current belief vector instead of
                      the true state
"""

import numpy as np
from game_env import GameEnv
from observation_model import ObservationModel
from state import State

class BeliefCollapseError(RuntimeError): #check for 0 belief
    pass

class POMDPEnv(GameEnv):
    def __init__(self, opponent_policy, initial_state=GameEnv.S_INIT,
                 γ=0.95, reward=None, n_obs_levels=4, initial_belief=None):
        super().__init__(opponent_policy, initial_state, γ, reward) #use GameEnv builder for main setup
        self.observation_model = ObservationModel(
            self.states, self.state_index, n_levels=n_obs_levels
        )
        self._belief = self._make_initial_belief(initial_belief)
        self._last_obs = None

    def _make_initial_belief(self, initial_belief): 
        if initial_belief is not None:
            b = np.array(initial_belief, dtype=np.float64)
            return b / b.sum() #ensure normalized
        b = np.zeros(len(self.states), dtype=np.float64)
        b[self.state_index[self.initial_state]] = 1.0 #default belief is initial state
        return b
    
    @property
    def belief(self): #read-only property
        return self._belief
    
    def belief_update(self, a, o):
        pred = np.asarray(self._belief @ self.transition_model.T[a]).flatten() #prediction
        pred *= self.observation_model.obs_masks[o] #apply possibility mask
        total = pred.sum()
        if total < 1e-15:
            raise BeliefCollapseError(
                f"Belief collapes: no states consistent with observation {o} after action {a}"
            )
        self._belief = pred / total #normalize

    def act(self, a): #true state update
        if self.state.terminal: #no transition out of terminal state
            return 0.0 
        self.state = self.transition_model.sample(self.state, a) #state update
        o = self.observation_model.sample_obs(self.state, a) #received observation
        self._last_obs = o
        self.belief_update(a, o) #automatic belief update
        return self.reward.evaluate(self.state) #return reward
    
    def observe(self): #overrides MDP observation
        return self._belief
    
    def observe_obs(self):
        return self._last_obs
    
    def observe_raw(self): #backdoor for error checking
        return self.state
    
    def reset(self):
        super().reset() #mdp reset
        self._belief = self._make_initial_belief(None) #belief reset
        self._last_obs = None

    def step(self, s, a): #for use with POMCP or similar methods for rollouts - no true state update
        sp = self.transition_model.sample(s, a)
        o = self.observation_model.sample_obs(sp, a)
        r = self.reward.evaluate(sp)
        return sp, o, r
    
