"""
pomdp_solver.py
===============
QMDP and POMCP solvers for the POMDP extension of the two-player MDP.

QMDP approximates the POMDP value function using the underlying MDP
solution: Q(b, a) = b @ (R + γ · T[a] · V), where V is the MDP value
vector from ValueIteration. Effective when uncertainty is low or resolves
quickly; may underperform when the agent must act to gain information.

POMCP is an online planner that represents belief via a particle filter. 
Builds a search tree over action/observation histories using UCB to choose
actions and random rollouts for value estimation. More accurate than QMDP 
under persistent uncertainty; higher computational cost per step.

Classes
-------
QMDPSolver(Solver) - solve() returns a BeliefPolicy; planning is done
                     via get_action(belief) on each step

POMCPSolver(Solver) - solve() returns a POMCPPolicy that replans lazily on
                     each step, tracking history with update(a, o)
"""

import numpy as np
import random
from math import sqrt, log
from solver import Solver
from policy import BeliefPolicy, POMCPPolicy

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
    
class POMCPSolver(Solver):
    # Implement Partially Observable Monte Carlo Planning (POMCP)
    def __init__(self, env, c=sqrt(2), depth=20, num_sims=1000, n_particles=200):
        super().__init__(env)
        self.c = c
        self.depth = depth
        self.num_sims = num_sims
        self.n_particles = n_particles
        self._N = {} #visit count (history, action)
        self._V = {} #value estimate (history, action)
        self._Nh = {} #node visit count
        self._B = {} #particle belief state
        self._h = () #history tuple (str(a), obs)

    def solve(self):
        return POMCPPolicy(self)
    
    def reset(self):
        self._N, self._V, self._Nh, self._B, self._h = {}, {}, {}, {}, ()

    @property
    def name(self):
        return 'POMCP'
    
    # Out-facing fcts
    def get_action(self, belief):
        #choose action based on current particle belief history
        h = self._h
        if h not in self._B or len(self._B[h]) == 0:
            self._B[h] = self._sample_particles(belief, self.n_particles)
        for _ in range(self.num_sims):
            s = random.choice(self._B[h])
            self._simulate(s, h, self.depth)
        return max(self.env.A, key=lambda a: self._V.get((h, a), -np.inf))
    
    def update(self, a, o):
        #history stepper
        self._h = self._h + ((str(a), o),)

    # In-facing fcts
    def _sample_particles(self, belief, n):
        #samples n belief particles
        indxs = np.random.choice(len(self.env.states), size=n, p=belief)
        return [self.env.states[i] for i in indxs]
    
    def _ucb_action(self, h):
        #Use UCB to choose MCTS action
        Nh = self._Nh.get(h, 1)
        unvis = [a for a in self.env.A if self._N.get((h, a), 0) == 0]
        if unvis:
            return random.choice(unvis)
        return max(self.env.A, key=lambda a:
                    self._V[(h, a)] + self.c * sqrt(log(Nh) /self._N[(h, a)])
                    )
    
    def _rollout(self, s, depth):
        #recursive rollout fct for MCTS
        if depth <= 0 or s.terminal:
            return self.env.reward.evaluate(s)
        valid = [a for a in self.env.A if self.env.valid_act(a, s)]
        a = random.choice(valid)
        sp, o, r = self.env.step(s, a)
        return  r + self.env.γ * self._rollout(sp, depth-1)
    
    def _simulate(self, s, h, depth):
        #run one MCTS simulation
        if depth <= 0 or s.terminal: #end of sim
            return self.env.reward.evaluate(s)
        if (h, self.env.A[0]) not in self._N: #leaf node
            for a in self.env.A:
                self._N[(h, a)] = 0
                self._V[(h, a)] = 0.0
            self._Nh[h] = 0
            return self._rollout(s, depth)
        a = self._ucb_action(h)
        sp, o, r = self.env.step(s, a)
        h_child = h + ((str(a), o),)
        if h_child not in self._B:
            self._B[h_child] = []
        self._B[h_child].append(sp)
        R = r + self.env.γ * self._simulate(sp, h_child, depth -1)
        self._Nh[h] = self._Nh.get(h, 0) + 1
        self._N[(h, a)] += 1
        self._V[(h, a)] += (R - self._V[(h, a)]) / self._N[(h, a)]
        return R