# tests/test_bnb.py
import numpy as np
from bnb import branch_and_bound

def small_instance():
    # 0=depot, 1..3 pontos
    values = np.array([0, 10, 8, 7], dtype=float)
    visit = np.array([0, 20, 20, 20], dtype=float)
    T = np.array([
        [0, 10, 10, 10],
        [10, 0, 10, 10],
        [10, 10, 0, 10],
        [10, 10, 10, 0]
    ], dtype=float)
    return values, visit, T

def test_bnb_finds_better_or_equal_than_greedy():
    v, vis, T = small_instance()
    res = branch_and_bound(v, vis, T, time_limit=70, max_nodes=10000)
    assert res["best_value"] >= 17.0  # deve permitir 1 e 2 por exemplo
    assert res["best_time"] <= 70 + 1e-6
