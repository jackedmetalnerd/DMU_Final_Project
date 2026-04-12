"""
solver.py
=========
Solver abstract base class for all MDP solving algorithms.

Subclasses implement solve() and return a Policy object.

Classes
-------
Solver — abstract base; requires solve() to be implemented

Concrete subclasses (each in their own file):
  ValueIteration  (value_iteration.py)
  QLearning       (q_learning.py)
  MCTSSolver      (mcts.py)
"""

from abc import ABC, abstractmethod
from policy import Policy


class Solver(ABC):
    """Abstract base for all MDP solvers."""

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def solve(self, **kwargs) -> Policy:
        """Run the algorithm; return an executable Policy."""

    @property
    def name(self) -> str:
        return self.__class__.__name__
