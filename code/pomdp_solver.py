"""
pomdp_solver.py
===============
QMDP solver for the POMDP extension of the two-player MDP.

QMDP approximates the POMDP value function using the underlying MDP
solution: Q(b, a) = b @ (R + γ · T[a] · V), where V is the MDP value
vector from ValueIteration. Effective when uncertainty is low or resolves
quickly; may underperform when the agent must act to gain information.

Classes
-------
QMDPSolver(Solver) — solve() returns a BeliefPolicy; planning is done
                     via get_action(belief) on each step
"""

import numpy as np
from math import sqrt
from solver import Solver
from policy import BeliefPolicy

class QMDPSolver(Solver):
    # Implement QMDP solver
    def __init__(self, env, V, γ=0.95):
        super().__init__(env)
        self.V = V
        self.γ = γ

    def solve(self):
        return BeliefPolicy(self)
    
    def get_action(self, belief):
        best_a = None
        best_Q = -np.inf
        for a in self.env.A:
            T = self.env.transition_model.T[a] #transition for this action
            Q = belief @ (self.env.reward.build_vector(self.env.states)
                          + self.γ * T.dot(self.V)) #matrix Q update
            if Q > best_Q:
                best_Q = Q
                best_a = a
        return best_a
    
    @property
    def name(self):
        return 'QMDP'