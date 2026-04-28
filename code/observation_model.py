"""
observation_model.py
====================
Observation model for the POMDP extension of the two-player MDP.

P1 observes its own state exactly (W1, M1, R1) and receives bucketed
observations of P2's forces (M2, W2) at configurable granularity.

Classes
-------
ObservationModel — maps states to observations; builds obs_masks for
                   efficient belief updates in POMDPEnv

Functions
---------
_bucket(x, n_levels) — discretizes a continuous count into n_levels bins;
                        n_levels=1 gives blind obs, n_levels>=11 gives exact obs
"""

from collections import namedtuple
import numpy as np
from state import State

Obs = namedtuple('Obs', ['W1','M1','R1','M2_level','W2_level','terminal']) #observation tuple

def _bucket(x, n_levels): #bucket adversary marine and worker counts
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    if n_levels == 1:
        return 0 #no observation of enemy
    if n_levels >= 11: #exact observation
        return x 
    if x == 0: #bucket zero when no workers/marines
        return 0
    return min(1+(x-1)*(n_levels-1) // 10, n_levels-1) #split into buckets

class ObservationModel:
    def __init__(self, states, state_index, n_levels=4):
        self._states = states
        self._state_index = state_index
        self._n_levels = n_levels
        self._obs_masks = None #wait to initialize until needed

    def obs_fn(self, s):
        # Deterministic observation: P1 sees own marines and workers; bucketed P2 marines and workers
        return Obs(
            W1=s.W1, M1=s.M1, R1=s.R1,
            M2_level = _bucket(s.M2, self._n_levels),
            W2_level = _bucket(s.W2, self._n_levels),
            terminal = s.terminal
        )
    
    def sample_obs(self, sp, a):
        return self.obs_fn(sp)
    
    @property
    def obs_masks(self): #maps obs to consistent states - faster updates than checking all
        if self._obs_masks is None:
            self._obs_masks = self._build_obs_masks()
        return self._obs_masks
    
    def _build_obs_masks(self): #builds when first needed
        masks = {}
        for i, s in enumerate(self._states):
            o = self.obs_fn(s)
            if o not in masks:
                masks[o] = np.zeros(len(self._states), dtype=bool)
            masks[o][i] = True
        return masks