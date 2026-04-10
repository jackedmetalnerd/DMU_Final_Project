# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two-player strategy game formulated as a Markov Decision Process (MDP) for ASEN 5264 (Decision Making Under Uncertainty) at CU Boulder. Player 1 (the agent) tries to eliminate all of Player 2's marines; Player 2 uses a fixed policy, allowing the game to be solved as a 1-player MDP from P1's perspective.

**State:** `(W1, M1, R1, W2, M2, R2, terminal)` — workers, marines, resources for each player + terminal flag. Each variable ∈ [0,10], giving ~332,640 states.

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
| `game_env.py` | OOP refactor: `GameEnv` class, the canonical environment interface |
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

### Transition Matrix Design

`GameEnv` precomputes scipy sparse CSR matrices for all 6 actions (P1+P2 combined). P2's policy is baked in by selecting only P2's chosen actions per state (`_apply_P2_policy`), reducing the 2-player game to a 1-player MDP. The full transition is a chain:

```
T_combined[a] = T_base[a] @ T_P2 @ T_resource
```

This means that any change to the opponent policy requires calling `env.update_P2_policy(π_P2_new)` and rebuilding these matrices.

### Policy Conventions

- Policies are callables: `π(s: State) -> str (action name)`
- `policies.py::P2_policy_converter(π_P1)` mirrors a P1 policy to P2 by inverting state (swapping P1/P2 fields) and remapping action names

### validate.py Checks

In order: state space equality → reward vector → transition matrices (per action, tol 1e-10) → value iteration (V diff tol 1e-6, policy agreement >99%) → win-rate comparison (50 games, within 5%) → Q-learning win-rates (500 ep training, within 10%).

## Collaboration Style (c++_pybind branch)

Work alongside the user — do not simply implement things. The goal is to build the C++ port together so the user learns through the process.

**User's background:**
- Strong OOP intuition — no need to explain class design concepts
- Has written C++ before but needs reminders on C++ nuances (memory model, const correctness, references vs pointers, .cpp vs .h, etc.)
- Has never used pybind11 — explain pybind11 patterns in detail before writing them, not after

**How to work:**
- Before writing a new pybind11 construct, explain what it does and why
- When C++ syntax differs meaningfully from Python (e.g. move semantics, const, templates), call it out explicitly
- Prefer showing a small piece then pausing over writing large blocks silently
- When there are multiple valid C++ approaches, present the tradeoffs rather than picking one unilaterally

## C++/pybind11 Architecture (c++_pybind branch)

**Goal:** All heavy computation runs in C++, exposed to Python via pybind11. Python handles orchestration, policy definitions, and simulation display.

**C++ owns:**
- State space construction (`_build_state_space`)
- Combat probability precomputation (`_precompute_combat`)
- Transition matrix construction (`_build_transition_matrices`, `_apply_P2_policy`, `_build_resource_matrix`)
- Value iteration inner loop (Bellman backup)
- MCTS rollouts

**Python owns:**
- `GameEnv` interface (wraps the C++ module)
- Policy callables (`π(s) -> action`)
- Solver orchestration and result handling
- Simulation display and evaluation

The C++ module (`game_env_cpp`) must expose an interface compatible with existing Python callers — `act()`, `observe()`, `reset()`, `simulate()` signatures must not change.
