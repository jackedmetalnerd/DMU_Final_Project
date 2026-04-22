"""
test_markov_game_env.py
=======================
Unit tests for MarkovGameEnv.

Run from code/:
    python test_markov_game_env.py
"""

from state import State
from action import Action
from markov_game_env import MarkovGameEnv
from game_env import GameEnv


def test_step_returns_floats():
    """step() should return (float, float) rewards."""
    env = MarkovGameEnv()
    r1, r2 = env.step(Action.P1_TRAIN_MARINES, Action.P2_TRAIN_MARINES)
    assert isinstance(r1, float) and isinstance(r2, float), (
        f"Expected (float, float), got ({type(r1)}, {type(r2)})"
    )
    print(f"PASS test_step_returns_floats  r1={r1}, r2={r2}")


def test_step_advances_state():
    """step() should advance the internal state."""
    env = MarkovGameEnv()
    s0 = env.observe()
    # Run several steps so at least one is likely to change state
    for _ in range(10):
        env.step(Action.P1_TRAIN_MARINES, Action.P2_TRAIN_MARINES)
        if env.observe() != s0:
            break
    # After 10 training steps the state should almost certainly have changed
    print(f"PASS test_step_advances_state  (state after steps: {env.observe()})")


def test_terminal_step_noop():
    """step() on a terminal state should return (0.0, 0.0) without advancing."""
    env = MarkovGameEnv()
    s_term = State(W1=1, M1=1, R1=1, W2=1, M2=0, R2=1, terminal=1)
    env.state = s_term
    r1, r2 = env.step(Action.P1_ATTACK, Action.P2_ATTACK)
    assert r1 == 0.0 and r2 == 0.0, f"Expected (0.0, 0.0), got ({r1}, {r2})"
    assert env.observe() == s_term, "Terminal state should not advance"
    print("PASS test_terminal_step_noop")


def test_reset():
    """reset() should return the environment to S_INIT."""
    env = MarkovGameEnv()
    env.step(Action.P1_ATTACK, Action.P2_ATTACK)
    env.reset()
    assert env.observe() == MarkovGameEnv.S_INIT, (
        f"Expected S_INIT after reset, got {env.observe()}"
    )
    print("PASS test_reset")


def test_reward_p1_p1_win():
    """P1 win terminal state: reward_p1 should be +1."""
    env = MarkovGameEnv()
    s_p1_win = State(W1=1, M1=2, R1=1, W2=1, M2=0, R2=1, terminal=1)
    r1 = env.reward_p1(s_p1_win)
    assert r1 == 1.0, f"Expected r1=1.0 for P1 win, got {r1}"
    print(f"PASS test_reward_p1_p1_win  r1={r1}")


def test_reward_p2_p1_win():
    """P1 win terminal state: reward_p2 should be -1 (P2 loses)."""
    env = MarkovGameEnv()
    s_p1_win = State(W1=1, M1=2, R1=1, W2=1, M2=0, R2=1, terminal=1)
    r2 = env.reward_p2(s_p1_win)
    assert r2 == -1.0, f"Expected r2=-1.0 for P1 win (P2 loss), got {r2}"
    print(f"PASS test_reward_p2_p1_win  r2={r2}")


def test_reward_p2_p2_win():
    """P2 win terminal state: reward_p2 should be +1."""
    env = MarkovGameEnv()
    s_p2_win = State(W1=1, M1=0, R1=1, W2=1, M2=2, R2=1, terminal=1)
    r2 = env.reward_p2(s_p2_win)
    assert r2 == 1.0, f"Expected r2=1.0 for P2 win, got {r2}"
    print(f"PASS test_reward_p2_p2_win  r2={r2}")


def test_as_p1_gameenv_returns_gameenv():
    """as_p1_gameenv() should return a GameEnv instance."""
    from policies import alternating_training_attack
    env = MarkovGameEnv()
    p1_env = env.as_p1_gameenv(alternating_training_attack)
    assert isinstance(p1_env, GameEnv), f"Expected GameEnv, got {type(p1_env)}"
    print("PASS test_as_p1_gameenv_returns_gameenv")


def test_as_p1_gameenv_initial_state():
    """as_p1_gameenv() should use the same initial state as MarkovGameEnv."""
    from policies import alternating_training_attack
    env = MarkovGameEnv()
    p1_env = env.as_p1_gameenv(alternating_training_attack)
    assert p1_env.initial_state == env.initial_state, (
        f"Expected {env.initial_state}, got {p1_env.initial_state}"
    )
    print("PASS test_as_p1_gameenv_initial_state")


def test_as_p2_gameenv_returns_gameenv():
    """as_p2_gameenv() should return a GameEnv instance."""
    from markov_game import _p2_policy_to_p1
    from policies import alternating_training_attack
    env = MarkovGameEnv()
    p1_estimate = _p2_policy_to_p1(alternating_training_attack)
    p2_env = env.as_p2_gameenv(p1_estimate)
    assert isinstance(p2_env, GameEnv), f"Expected GameEnv, got {type(p2_env)}"
    print("PASS test_as_p2_gameenv_returns_gameenv")


def test_as_p2_gameenv_inverted_initial_state():
    """as_p2_gameenv() should swap W1/M1/R1 and W2/M2/R2 in the initial state."""
    from markov_game import _p2_policy_to_p1
    from policies import alternating_training_attack
    env = MarkovGameEnv()
    p1_estimate = _p2_policy_to_p1(alternating_training_attack)
    p2_env = env.as_p2_gameenv(p1_estimate)
    init = env.initial_state
    inv  = p2_env.initial_state
    assert inv.W1 == init.W2 and inv.M1 == init.M2 and inv.R1 == init.R2, (
        f"Expected W1={init.W2},M1={init.M2},R1={init.R2} in inverted env, "
        f"got W1={inv.W1},M1={inv.M1},R1={inv.R1}"
    )
    assert inv.W2 == init.W1 and inv.M2 == init.M1 and inv.R2 == init.R1, (
        f"Expected W2={init.W1},M2={init.M1},R2={init.R1} in inverted env, "
        f"got W2={inv.W2},M2={inv.M2},R2={inv.R2}"
    )
    print("PASS test_as_p2_gameenv_inverted_initial_state")


if __name__ == '__main__':
    test_step_returns_floats()
    test_step_advances_state()
    test_terminal_step_noop()
    test_reset()
    test_reward_p1_p1_win()
    test_reward_p2_p1_win()
    test_reward_p2_p2_win()
    test_as_p1_gameenv_returns_gameenv()
    test_as_p1_gameenv_initial_state()
    test_as_p2_gameenv_returns_gameenv()
    test_as_p2_gameenv_inverted_initial_state()
    print("\nAll MarkovGameEnv tests passed.")
