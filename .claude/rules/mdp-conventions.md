---
paths:
  - "code/**/*.py"
  - "src/**/*.cpp"
  - "src/**/*.h"
---

## Ground Truth Reference

`project_mdp.py` is the original monolithic implementation and must never be modified.
It is the reference used by `validate.py` to verify all other implementations.

## MDP Numerical Conventions

- Transition matrix comparisons: tolerance 1e-10
- Value iteration convergence: tolerance 1e-9
- Policy agreement threshold: 99% of states must agree
- Win-rate parity between implementations: within 5% over 50 games

## State Representation

State is always a 7-tuple `(W1, M1, R1, W2, M2, R2, terminal)`:
- W: workers, M: marines, R: resources — each ∈ [0, 10]
- terminal ∈ {0, 1}
- Total states: 11^6 × 2 = 3,543,122 (11 values each for 6 vars, × 2 for terminal flag)

## Transition Matrix Format

- Python: scipy `csr_matrix`, shape `(n_states, n_states)`
- Never use dense matrices — state space is too large

## Opponent Policy

Any change to the P2 policy requires rebuilding `T_combined` via `update_P2_policy()`. The combined transition is:

```
T_combined[a] = T_base[a] @ T_P2 @ T_resource
```

Do not modify `T_base` or `T_resource` individually without rebuilding the full chain.
