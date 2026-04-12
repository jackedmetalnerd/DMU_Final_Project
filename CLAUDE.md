# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two-player strategy game formulated as a Markov Decision Process (MDP) for ASEN 5264 (Decision Making Under Uncertainty) at CU Boulder. Player 1 (the agent) tries to eliminate all of Player 2's marines; Player 2 uses a fixed policy, allowing the game to be solved as a 1-player MDP from P1's perspective.

**State:** `(W1, M1, R1, W2, M2, R2, terminal)` — workers, marines, resources for each player + terminal flag. Each variable ∈ [0,10], terminal ∈ {0,1}, giving 11^6 × 2 = 3,543,122 states.

**Actions per player:** `train_workers`, `train_marines`, `attack`

**Transition dynamics:**
- Training: 90% success (costs 1 or all resources), 10% failure
- Combat (attack): Marine losses ~ Binomial(opponent_marines, 0.5) for both sides
- Resource gain (deterministic, every turn): `R' = min(R + W, 10)`

**Rewards:** +1 if P1 wins (M1 > 0, M2 = 0), −1 if P1 loses (M1 = 0), 0 otherwise. Discount γ = 0.95.

## Running Code

All scripts are run from the `code/` directory:

```bash
cd code

# Validate OOP refactor correctness against original monolithic implementation
python validate.py        # Note: has a breakpoint() at line 54 — press 'c' to continue

# Run individual solvers (each has a __main__ block with demo simulations)
python value_iteration.py   # Solves VI, runs 5 simulations
python q_learning.py        # Trains Q-agent 2000 episodes, runs 5 simulations
python mcts.py              # Runs 5 MCTS game simulations
```

There is no build system, package manager, or test framework. Dependencies: `numpy`, `scipy`, `tqdm`, `matplotlib`.

## Architecture

### Key Files

| File | Role |
|------|------|
| `project_mdp.py` | Original monolithic implementation — ground truth reference |
| `state.py` | `State` class — immutable game state with game-logic predicates (`is_win`, `is_loss`, `winner`, `terminal_value`) |
| `transition.py` | `TransitionModel` class — all transition logic; exposes `transition(s,a)`, `sample(s,a)`, `build_matrices()` |
| `game_env.py` | OOP refactor: `GameEnv` class, the canonical environment interface |
| `reward.py` | `Reward` class — reward functions and active selector |
| `policies.py` | Fixed P2 opponent policies (`alternating_training`, `alternating_training_attack`) |
| `value_iteration.py` | `ValueIteration` solver class |
| `q_learning.py` | `QLearning` agent class (tabular, epsilon-greedy) |
| `mcts.py` | `MCTSSolver` class (UCB1, recursive rollouts) |
| `validate.py` | Integration test suite comparing old vs. new implementations |

### Solver Pipeline

```
GameEnv(π_P2, s_init)        # Build environment with opponent policy baked in
    ↓
Solver(env).solve()           # ValueIteration / QLearning.train() / MCTSSolver
    ↓
π_star: dict or callable      # Policy mapping state → action
    ↓
env.simulate(π_star, label)   # Evaluate policy
```

### TransitionModel (`transition.py`)

All transition logic lives in `TransitionModel`, which `GameEnv` instantiates as `env._model`. Two public interfaces:

- **`transition(s, a) -> dict[State, float]`** — returns next-state distribution for one (state, action) pair without using precomputed matrices. Implements the full P1-action → P2-action → resource-update chain directly. Used by QL and MCTS for episode rollouts.
- **`sample(s, a) -> State`** — samples one next state from `transition(s, a)`. Used by `GameEnv.act()`, `GameEnv.simulate()`, and MCTS internally.
- **`build_matrices() -> None`** — precomputes scipy sparse CSR matrices for all 3 P1 actions. Expensive (several minutes). Called once in `GameEnv.__init__()`. Only needed by VI for `T[a] @ V` matrix-vector products.

Combat outcomes (binomial losses for both sides) are precomputed in a 121-entry lookup dict at construction time — fast enough to always build, and needed by both `transition()` and `build_matrices()`.

The combined transition matrix chain is:
```
T_combined[a] = T_base[a] @ T_P2 @ T_resource
```

Any P2 policy change requires calling `env.update_P2_policy(π_P2_new)`, which rebuilds `T_P2` and recomposes `T`.

### State Class (`state.py`)

`State` is an immutable class (not a namedtuple) with `__slots__` for memory efficiency. It is hashable and dict-key compatible, with hash matching the old namedtuple so validate.py comparisons work. Key methods:
- `is_win()` / `is_loss()` / `is_terminal()` — game outcome predicates
- `winner()` → `'P1'`, `'P2'`, or `'Draw'`
- `terminal_value()` → fixed `±1.0` used by VI for terminal pinning

All win/loss checks throughout the codebase use these methods rather than inline field comparisons.

### Policy Conventions

- Policies are callables: `π(s: State) -> str (action name)`
- `policies.py::P2_policy_converter(π_P1)` mirrors a P1 policy to P2 by inverting state (swapping P1/P2 fields) and remapping action names

## Reward Function Testing (reward_testing branch)

**Goal:** Test different reward functions across all solvers without changing `game_env.py` or solver files.

### Switching the active reward function

Edit `code/reward_functions.py` — change only the last line:
```python
ACTIVE = shaped_military_advantage   # was: terminal_only
```
No other file needs to change. All solvers pick up the new reward automatically.

### Adding a new reward function

1. Add a function to `code/reward_functions.py` following the interface: `reward_fn(s) -> float` where `s` has fields `W1, M1, R1, W2, M2, R2, terminal`
2. Add it to `ALL_REWARD_FNS` in `code/compare_rewards.py`
3. Set `ACTIVE = your_new_fn` to use it

### Running comparisons

```bash
cd code
python compare_rewards.py                              # all solvers × all reward fns
python compare_rewards.py --solver vi                  # VI only
python compare_rewards.py --rf terminal_only win_only  # specific reward fns
python compare_rewards.py --games 100                  # larger sample
```

### VI Bellman fix

`value_iteration.py`'s solve loop was updated to include `R` in the Bellman backup (`R_nonterminal + γ * T[a] @ V`) so shaped rewards properly influence VI convergence — not just policy extraction. Terminal state values are still pinned to ±1 each iteration.

### validate.py Checks

In order: state space equality → reward vector → transition matrices (per action, tol 1e-10) → value iteration (V diff tol 1e-6, policy agreement >99%) → win-rate comparison (50 games, within 5%) → Q-learning win-rates (500 ep training, within 10%).