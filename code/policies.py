from state import State


def alternating_training(s):
    """Train workers and marines in alternation."""
    if s.W2 < s.M2:
        return 'P2_train_workers'
    return 'P2_train_marines'


def alternating_training_attack(s):
    """Train to max, then attack."""
    if s.M2 == 10 and s.W2 == 10:
        return 'P2_attack'
    if s.W2 < s.M2:
        return 'P2_train_workers'
    return 'P2_train_marines'


def P2_policy_converter(π_P1):
    """Wrap a P1 policy (dict or callable) to act as P2 by inverting the state."""
    ACTION_MAP = {
        'P1_train_workers': 'P2_train_workers',
        'P1_train_marines': 'P2_train_marines',
        'P1_attack':        'P2_attack',
    }

    def π_P2(s):
        s_inv = State(W1=s.W2, M1=s.M2, R1=s.R2,
                      W2=s.W1, M2=s.M1, R2=s.R1, terminal=s.terminal)
        a_p1 = π_P1[s_inv] if isinstance(π_P1, dict) else π_P1(s_inv)
        return ACTION_MAP[a_p1]

    return π_P2
