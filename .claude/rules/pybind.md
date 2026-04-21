---
paths:
  - "src/**/*.cpp"
  - "src/**/*.h"
  - "CMakeLists.txt"
---

## pybind11 Interface Rules

- The C++ module must be importable as `import game_env_cpp`
- Exposed methods must match the Python `GameEnv` interface exactly: same names, same argument order, same return types
  - `act(a: str) -> float`
  - `observe() -> State`
  - `reset() -> None`
  - `valid_act(a: str, s: State) -> bool`
  - `update_P2_policy(π: callable) -> None`

## State Struct

Define a C++ struct that maps directly to the Python `State` namedtuple:

```cpp
struct State {
    int W1, M1, R1, W2, M2, R2, terminal;
};
```

Expose it to Python via pybind11 so it is interchangeable with the namedtuple on the Python side.

## Sparse Matrices

Use `Eigen::SparseMatrix<double, RowMajor>` — row-major matches scipy CSR layout and avoids conversion overhead when returning matrices to Python.

## Policy Callables

Accept P2 policy as `pybind11::function` (or `std::function<std::string(State)>`). Keep policy logic in Python; C++ calls back into Python for policy evaluation. Do not reimplement policies in C++.

## Naming

- C++ source files: `src/game_env.cpp`, `src/value_iteration.cpp`, `src/mcts.cpp`
- Pybind11 module definition: `src/bindings.cpp`
- Python wrapper: `code/game_env.py` (unchanged interface, imports `game_env_cpp` internally)
