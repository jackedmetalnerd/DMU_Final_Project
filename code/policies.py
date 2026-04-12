from state import State
from action import Action
from policy import FunctionPolicy, SymmetricPolicy


def alternating_training(s):
    """Train workers and marines in alternation."""
    if s.W2 < s.M2:
        return Action.P2_TRAIN_WORKERS
    return Action.P2_TRAIN_MARINES


def alternating_training_attack(s):
    """Train to max, then attack."""
    if s.M2 == 10 and s.W2 == 10:
        return Action.P2_ATTACK
    if s.W2 < s.M2:
        return Action.P2_TRAIN_WORKERS
    return Action.P2_TRAIN_MARINES


def P2_policy_converter(p1_policy):
    """Wrap a P1 policy (dict, Policy, or callable) to act as P2 by inverting the state.

    Delegates to SymmetricPolicy for the actual implementation.
    Kept for backward compatibility with existing call sites.
    """
    return SymmetricPolicy(p1_policy)
