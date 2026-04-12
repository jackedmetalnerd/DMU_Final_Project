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
| `state.py` | `State` class — immutable game state with game-logic predicates |
| `action.py` | `Action` class — immutable action with string interoperability |
| `transition.py` | `TransitionModel` class — all transition logic |
| `reward.py` | `Reward` class — reward functions and active selector |
| `mdp.py` | `MDP` base class — bundles (states, actions, transition_model, reward, gamma) |
| `game_env.py` | `GameEnv(MDP)` — adds RL/simulation interface to MDP |
| `policy.py` | `Policy` ABC + `DictPolicy`, `FunctionPolicy`, `SymmetricPolicy`, `MCTSPolicy` |
| `policies.py` | Hand-coded P2 opponent policies; `P2_policy_converter` wraps `SymmetricPolicy` |
| `solver.py` | `Solver` ABC — requires `solve() -> Policy` |
| `value_iteration.py` | `ValueIteration(Solver)` — sparse matrix VI |
| `q_learning.py` | `QLearning(Solver)` — tabular epsilon-greedy Q-learning |
| `mcts.py` | `MCTSSolver(Solver)` — UCB1 Monte Carlo Tree Search |
| `validate.py` | Integration test suite comparing old vs. new implementations |
| `compare_rewards.py` | Runs solvers across reward functions; reports win rates |

### Class Hierarchy

```
MDP
└── GameEnv          # adds act(), observe(), reset(), simulate(), update_P2_policy()

Solver (ABC)
├── ValueIteration   # solve() → DictPolicy; uses T[a] @ V sparse matrix products
├── QLearning        # solve(n_episodes) → DictPolicy; uses sample() rollouts
└── MCTSSolver       # solve() → MCTSPolicy; uses sample() rollouts

Policy (ABC)
├── DictPolicy       # wraps {State → Action} dict; __getitem__ for dict-style access
├── FunctionPolicy   # wraps a plain callable
├── SymmetricPolicy  # mirrors P1 Policy to P2 by swapping player perspective
└── MCTSPolicy       # lazy; calls MCTSSolver.get_action(s) on each step
```

### Solver Pipeline

```
GameEnv(opponent_policy, initial_state)   # Build environment
    ↓
Solver(env).solve()                       # Returns a Policy object
    ↓
policy: Policy                            # Callable: policy(s) -> Action
    ↓
env.simulate(policy, label)               # Evaluate policy
```

### TransitionModel (`transition.py`)

All transition logic lives in `TransitionModel`, which `GameEnv` instantiates as `env.transition_model`. Two public interfaces:

- **`transition(s, a) -> dict[State, float]`** — returns next-state distribution without precomputed matrices. Used by QL and MCTS rollouts.
- **`sample(s, a) -> State`** — samples one next state. Used by `GameEnv.act()`, `GameEnv.simulate()`, and MCTS.
- **`build_matrices() -> None`** — precomputes scipy sparse CSR matrices for all 3 P1 actions. Expensive. Called explicitly by VI before solving; not called during `GameEnv.__init__()`.

The combined transition matrix chain is:
```
T_combined[a] = T_base[a] @ T_P2 @ T_resource
```

Any P2 policy change requires calling `env.update_P2_policy(new_policy)`, which rebuilds `T_P2` and recomposes `T`.

### State Class (`state.py`)

`State` is an immutable class (not a namedtuple) with `__slots__` for memory efficiency. It is hashable and dict-key compatible. Key methods:
- `is_win()` / `is_loss()` / `is_terminal()` — game outcome predicates
- `winner()` → `'P1'`, `'P2'`, or `'Draw'`
- `terminal_value()` → fixed `±1.0` used by VI for terminal pinning
- `build_space()` — classmethod returning the full 3,543,122-state list

### Action Class (`action.py`)

`Action` is immutable with `player` and `type` fields. Key design: `__hash__ = hash(str(self))` and `__eq__` handles string comparison, so Action objects and string literals are interchangeable as dict keys — Q-tables and T-matrices built with either type work with both.

### Policy Classes (`policy.py`)

- `DictPolicy` wraps a `{State → Action}` dict with both callable and `__getitem__` access
- `SymmetricPolicy` mirrors a P1 policy to P2 by swapping state fields; replaces `P2_policy_converter`
- `MCTSPolicy` calls `MCTSSolver.get_action(s)` on each invocation

### MDP and GameEnv (`mdp.py`, `game_env.py`)

`MDP` is the formal structure holding (states, actions, transition_model, reward, gamma). `GameEnv` inherits from `MDP` and adds the RL interface. Backward-compat property aliases on `MDP`:
- `S`, `A`, `T`, `R`, `γ`, `S_index`

`T` returns `transition_model.T` — an empty dict until `build_matrices()` is called.

### Solver and Algorithm Classes (`solver.py`, solver files)

`Solver` is the ABC; `solve()` is the only required method. All three solvers inherit from it:
- `ValueIteration`: `self.V` stores value function after solve; returns `DictPolicy`
- `QLearning`: `self.Q` stores Q-table; `solve(n_episodes)` returns `DictPolicy`; `policy(s)` still works for single-state lookup
- `MCTSSolver`: `solve()` returns `MCTSPolicy(self)`; planning runs lazily on each `get_action(s)` call

## Reward Function Testing

**Goal:** Test different reward functions across all solvers without changing `game_env.py` or solver files.

### Switching the active reward function

Edit `code/reward.py` — change only the last line:
```python
Reward.ACTIVE = Reward.shaped_military_advantage   # was: Reward.terminal_only
```
No other file needs to change. All solvers pick up the new reward automatically.

### Adding a new reward function

1. Add a `@staticmethod` to `Reward` in `code/reward.py` with signature `fn(s) -> float`
2. Add it to `Reward.ALL` at the bottom of `reward.py`
3. Set `Reward.ACTIVE = Reward.your_new_fn` to use it

### Running comparisons

```bash
cd code
python compare_rewards.py                              # all solvers × all reward fns
python compare_rewards.py --solver vi                  # VI only
python compare_rewards.py --rf terminal_only win_only  # specific reward fns
python compare_rewards.py --games 100                  # larger sample
```

### VI Bellman fix

`value_iteration.py`'s solve loop includes `R` in the Bellman backup (`R_nonterminal + γ * T[a] @ V`) so shaped rewards properly influence VI convergence. Terminal state values are pinned to ±1 each iteration.

### validate.py Checks

In order: state space equality → reward vector → transition matrices (per action, tol 1e-10) → value iteration (V diff tol 1e-6, policy agreement >99%) → win-rate comparison (50 games, within 5%) → Q-learning win-rates (500 ep training, within 10%).

**Known pre-existing FAIL:** R vector diff = 1.00 on unreachable terminal states (terminal=1, M1>0, M2>0). All substantive checks (T, VI, win rates) pass.
