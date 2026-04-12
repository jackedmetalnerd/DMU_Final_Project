from state import State
from action import Action


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
    """Wrap a P1 policy (dict or callable) to act as P2 by inverting the state."""
    ACTION_MAP = {
        Action.P1_TRAIN_WORKERS: Action.P2_TRAIN_WORKERS,
        Action.P1_TRAIN_MARINES: Action.P2_TRAIN_MARINES,
        Action.P1_ATTACK:        Action.P2_ATTACK,
    }

    def p2_policy(s):
        s_inv = State(W1=s.W2, M1=s.M2, R1=s.R2,
                      W2=s.W1, M2=s.M1, R2=s.R1, terminal=s.terminal)
        a_p1 = p1_policy[s_inv] if isinstance(p1_policy, dict) else p1_policy(s_inv)
        return ACTION_MAP[a_p1]

    return p2_policy
