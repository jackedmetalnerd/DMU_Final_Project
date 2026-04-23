"""
test_vectorized_transition.py
==============================
Phase 1: implement vectorized matrix builders as standalone functions,
         compare against current TransitionModel, save checksums to JSON.
Phase 3: load saved checksums, verify class output matches them.

Run from code/ directory:
    python test_vectorized_transition.py
"""

import numpy as np
import json
import os
import time
from scipy.sparse import csr_matrix
from game_env import GameEnv
from policies import alternating_training_attack

BASELINE_FILE = os.path.join(os.path.dirname(__file__),
                             'test_vectorized_transition_baseline.json')

# ── Index formula (lexicographic state ordering) ──────────────────────────────
# State ordering: W1, M1, R1, W2, M2, R2, terminal — each 0-10, terminal 0-1
# Strides derived from build_space() loop nesting order.
_STRIDES = np.array([
    11**5 * 2,   # W1
    11**4 * 2,   # M1
    11**3 * 2,   # R1
    11**2 * 2,   # W2
    11**1 * 2,   # M2
    11**0 * 2,   # R2
    1,           # terminal
], dtype=np.int64)


def _to_idx(W1, M1, R1, W2, M2, R2, t):
    """Compute state index from field values. All args may be numpy arrays."""
    return (W1 * _STRIDES[0] + M1 * _STRIDES[1] + R1 * _STRIDES[2]
          + W2 * _STRIDES[3] + M2 * _STRIDES[4] + R2 * _STRIDES[5]
          + t  * _STRIDES[6])


def _extract_fields(states):
    """Extract all state fields as numpy int arrays. O(n) once per build."""
    return {f: np.array([getattr(s, f) for s in states], dtype=np.int64)
            for f in ('W1', 'M1', 'R1', 'W2', 'M2', 'R2', 'terminal')}


# ── Vectorized resource matrix ─────────────────────────────────────────────────

def build_resource_matrix_vec(states, fields=None) -> csr_matrix:
    """R' = min(R + W, 10) for each player. Terminal states self-loop."""
    if fields is None:
        fields = _extract_fields(states)
    f = fields
    n = len(states)
    src = np.arange(n, dtype=np.int64)

    R1_new = np.minimum(f['R1'] + f['W1'], 10)
    R2_new = np.minimum(f['R2'] + f['W2'], 10)

    tgt = _to_idx(f['W1'], f['M1'], R1_new, f['W2'], f['M2'], R2_new, f['terminal'])
    # Terminal states self-loop
    tgt = np.where(f['terminal'] == 1, src, tgt)

    return csr_matrix((np.ones(n, dtype=np.float64), (src, tgt)), shape=(n, n))


# ── Vectorized training actions ────────────────────────────────────────────────

def build_base_training_vec(states, action: str, fields=None) -> csr_matrix:
    """Build T_base for one training action via numpy vectorization.

    For all non-terminal valid states:
      success (0.9): relevant stat += resources, resources → 0
      failure (0.1): stat unchanged, resources → 0
    Terminal or invalid states: self-loop.
    """
    if fields is None:
        fields = _extract_fields(states)
    f = fields
    n = len(states)
    src = np.arange(n, dtype=np.int64)

    term = f['terminal'].astype(bool)

    # Determine which field changes and what the invalidity condition is
    if action == 'P1_train_workers':
        invalid = term | (f['R1'] < 1) | (f['W1'] > 9)
        W1_ok = np.minimum(f['W1'] + f['R1'], 10)
        R1_spent = np.zeros_like(f['R1'])
        ok_tgt   = _to_idx(W1_ok,   f['M1'], R1_spent, f['W2'], f['M2'], f['R2'], 0)
        fail_tgt = _to_idx(f['W1'], f['M1'], R1_spent, f['W2'], f['M2'], f['R2'], 0)

    elif action == 'P1_train_marines':
        invalid = term | (f['R1'] < 1) | (f['M1'] > 9)
        M1_ok = np.minimum(f['M1'] + f['R1'], 10)
        R1_spent = np.zeros_like(f['R1'])
        ok_tgt   = _to_idx(f['W1'], M1_ok,   R1_spent, f['W2'], f['M2'], f['R2'], 0)
        fail_tgt = _to_idx(f['W1'], f['M1'], R1_spent, f['W2'], f['M2'], f['R2'], 0)

    elif action == 'P2_train_workers':
        invalid = term | (f['R2'] < 1) | (f['W2'] > 9)
        W2_ok = np.minimum(f['W2'] + f['R2'], 10)
        R2_spent = np.zeros_like(f['R2'])
        ok_tgt   = _to_idx(f['W1'], f['M1'], f['R1'], W2_ok,   f['M2'], R2_spent, 0)
        fail_tgt = _to_idx(f['W1'], f['M1'], f['R1'], f['W2'], f['M2'], R2_spent, 0)

    elif action == 'P2_train_marines':
        invalid = term | (f['R2'] < 1) | (f['M2'] > 9)
        M2_ok = np.minimum(f['M2'] + f['R2'], 10)
        R2_spent = np.zeros_like(f['R2'])
        ok_tgt   = _to_idx(f['W1'], f['M1'], f['R1'], f['W2'], M2_ok,   R2_spent, 0)
        fail_tgt = _to_idx(f['W1'], f['M1'], f['R1'], f['W2'], f['M2'], R2_spent, 0)

    else:
        raise ValueError(f"Not a training action: {action}")

    # Self-loop for terminal / invalid states
    ok_tgt   = np.where(invalid, src, ok_tgt)
    fail_tgt = np.where(invalid, src, fail_tgt)
    p_ok     = np.where(invalid, 1.0, 0.9)
    p_fail   = np.where(invalid, 0.0, 0.1)

    rows = np.tile(src, 2)
    cols = np.concatenate([ok_tgt, fail_tgt])
    vals = np.concatenate([p_ok, p_fail])

    T = csr_matrix((vals, (rows, cols)), shape=(n, n))
    T.sum_duplicates()   # merge self-loop rows (terminal/invalid: 1.0+0.0=1.0)
    T.eliminate_zeros()
    return T


# ── Vectorized attack action ───────────────────────────────────────────────────

def build_base_attack_vec(states, combat_lookup, fields=None) -> dict:
    """Build T_base for P1_attack and P2_attack (same matrix — symmetric combat).

    Groups states by (M1, M2) pair (121 unique groups) and processes each
    in bulk with numpy, replacing the O(n * avg_outcomes) Python loop.

    Returns dict with keys 'P1_attack' and 'P2_attack' (identical matrices).
    """
    if fields is None:
        fields = _extract_fields(states)
    f = fields
    n = len(states)
    src = np.arange(n, dtype=np.int64)

    # Group non-terminal states by (M1, M2)
    group_key = f['M1'] * 11 + f['M2']   # unique int in [0, 120]
    non_term  = f['terminal'] == 0

    all_rows = []
    all_cols = []
    all_vals = []

    for key in range(121):
        m1, m2 = divmod(key, 11)
        mask = non_term & (group_key == key)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        for (nm1, nm2), p in combat_lookup[(m1, m2)].items():
            term_new = np.int64(nm1 == 0 or nm2 == 0)
            tgt = _to_idx(f['W1'][idxs], nm1, f['R1'][idxs],
                          f['W2'][idxs], nm2, f['R2'][idxs], term_new)
            all_rows.append(idxs)
            all_cols.append(tgt)
            all_vals.append(np.full(len(idxs), p, dtype=np.float64))

    # Terminal states: self-loop
    term_idxs = np.where(~non_term)[0]
    if len(term_idxs):
        all_rows.append(term_idxs)
        all_cols.append(term_idxs)
        all_vals.append(np.ones(len(term_idxs), dtype=np.float64))

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)

    T = csr_matrix((vals, (rows, cols)), shape=(n, n))
    T.sum_duplicates()
    T.eliminate_zeros()
    return {'P1_attack': T, 'P2_attack': T}


# ── Checksum helpers ───────────────────────────────────────────────────────────

def _checksums(T: csr_matrix) -> dict:
    T_csr = T.tocsr()
    T_csr.sort_indices()
    return {
        'shape':  list(T_csr.shape),
        'nnz':    int(T_csr.nnz),
        'sum':    float(T_csr.sum()),
        'frob':   float(np.sqrt((T_csr.data ** 2).sum())),
    }


def _check_match(cs_new: dict, cs_ref: dict, label: str, tol: float = 1e-6):
    assert cs_new['shape'] == cs_ref['shape'],  f"{label} shape mismatch"
    assert cs_new['nnz']   == cs_ref['nnz'],    f"{label} nnz mismatch"
    assert abs(cs_new['sum']  - cs_ref['sum'])  < tol, \
        f"{label} sum mismatch: {cs_new['sum']:.10f} vs {cs_ref['sum']:.10f}"
    assert abs(cs_new['frob'] - cs_ref['frob']) < tol, \
        f"{label} frob mismatch: {cs_new['frob']:.10f} vs {cs_ref['frob']:.10f}"


# ── Entry point ────────────────────────────────────────────────────────────────

def _diff(A: csr_matrix, B: csr_matrix) -> float:
    return float(abs(A - B).max())


if __name__ == '__main__':
    print("Building GameEnv (state space + opponent policy)...")
    env = GameEnv(π_P2=alternating_training_attack)
    model = env._model
    S = env.S

    if not os.path.exists(BASELINE_FILE):
        # ── Phase 1: compare standalone vectorized functions vs current class ──
        print("\nPhase 1: building current (reference) matrices via TransitionModel...")
        t0 = time.time()
        model.build_matrices()
        print(f"  Reference build time: {time.time()-t0:.1f}s")

        print("\nPhase 1: building and comparing vectorized matrices...")
        t1 = time.time()
        fields = _extract_fields(S)

        checksums = {}
        all_pass = True

        # Resource matrix
        T_res_new = build_resource_matrix_vec(S, fields)
        d = _diff(T_res_new, model._T_res)
        ok = d < 1e-10
        print(f"  {'PASS' if ok else 'FAIL'} T_res              max_diff={d:.2e}")
        if not ok: all_pass = False
        checksums['T_res'] = _checksums(T_res_new)

        # Training matrices
        for a in ['P1_train_workers', 'P1_train_marines',
                  'P2_train_workers', 'P2_train_marines']:
            T_new = build_base_training_vec(S, a, fields)
            d = _diff(T_new, model._T_base[a])
            ok = d < 1e-10
            print(f"  {'PASS' if ok else 'FAIL'} T_base[{a:<20}] max_diff={d:.2e}")
            if not ok: all_pass = False
            checksums[f'T_base_{a}'] = _checksums(T_new)

        # Attack matrices
        T_atk = build_base_attack_vec(S, model._combat_lookup, fields)
        for a in ['P1_attack', 'P2_attack']:
            d = _diff(T_atk[a], model._T_base[a])
            ok = d < 1e-10
            print(f"  {'PASS' if ok else 'FAIL'} T_base[{a:<20}] max_diff={d:.2e}")
            if not ok: all_pass = False
            checksums[f'T_base_{a}'] = _checksums(T_atk[a])

        print(f"\n  Vectorized build time: {time.time()-t1:.1f}s")

        if all_pass:
            with open(BASELINE_FILE, 'w') as fp:
                json.dump(checksums, fp, indent=2)
            print(f"\nAll Phase 1 checks PASSED. Baseline saved to {BASELINE_FILE}")
        else:
            print("\nSome Phase 1 checks FAILED — baseline NOT saved.")

    else:
        # ── Phase 3: verify class output against saved baseline ────────────────
        print(f"\nPhase 3: loading baseline from {BASELINE_FILE}...")
        with open(BASELINE_FILE) as fp:
            baseline = json.load(fp)

        print("Phase 3: building matrices via updated TransitionModel...")
        t0 = time.time()
        model.build_matrices()
        print(f"  Build time: {time.time()-t0:.1f}s")

        all_pass = True

        # Resource matrix
        cs = _checksums(model._T_res)
        try:
            _check_match(cs, baseline['T_res'], 'T_res')
            print(f"  PASS T_res")
        except AssertionError as e:
            print(f"  FAIL T_res: {e}")
            all_pass = False

        # All T_base matrices
        for a in model.ACTIONS_ALL:
            key = f'T_base_{a}'
            cs = _checksums(model._T_base[a])
            try:
                _check_match(cs, baseline[key], key)
                print(f"  PASS T_base[{a}]")
            except AssertionError as e:
                print(f"  FAIL T_base[{a}]: {e}")
                all_pass = False

        if all_pass:
            print("\nAll Phase 3 checks PASSED — class output matches baseline.")
        else:
            print("\nSome Phase 3 checks FAILED.")
