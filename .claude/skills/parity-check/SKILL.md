---
name: parity-check
description: Verify that the C++ pybind11 implementation produces identical outputs to the Python reference implementation. Use after making changes to C++ transition matrix builders, VI, or MCTS code.
disable-model-invocation: true
---

Compare the C++ (`game_env_cpp`) and Python (`game_env.py`) implementations:

1. **State space** — sizes match, sets are equal, ordering is identical
2. **Reward vector** — max absolute difference < 1e-10
3. **Transition matrices** — per action, max absolute difference < 1e-10
4. **Value iteration** — V-function max diff < 1e-6, greedy policy agreement > 99%
5. **Win rates** — play 50 games with each implementation using the same VI policy, within 5%

The Python implementation in `code/project_mdp.py` is the ground truth reference. The C++ module must match it within the tolerances above before any solver results are trusted.

Note: requires a successful `cmake --build` before running. If the C++ module is not yet built, run the build first.
