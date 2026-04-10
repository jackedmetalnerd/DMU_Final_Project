---
name: validate
description: Run the MDP validation suite comparing Python implementations. Use when verifying correctness after changes to game_env.py, transition logic, or solver code.
disable-model-invocation: true
---

Run from the `code/` directory:

```bash
cd code && python validate.py
```

There is a `breakpoint()` at line 54 — type `c` to continue when the debugger pauses.

Checks run in this order:
1. State space size and set equality
2. Reward vector (max diff < 1e-10)
3. Transition matrices per action (max diff < 1e-10)
4. Value iteration — V-function (max diff < 1e-6) and policy agreement (> 99%)
5. Win-rate comparison over 50 games (within 5%)
6. Q-learning win-rates over 500 training episodes (within 10%)

Exits with code 1 and prints "SOME CHECKS FAILED" if any check fails.
