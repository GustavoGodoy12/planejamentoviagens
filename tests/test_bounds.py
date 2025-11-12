# tests/test_bounds.py
import numpy as np
from bnb import fractional_bound, Node

def test_fractional_bound_monotonic():
    values = np.array([0, 10, 9], float)
    visit = np.array([0, 10, 10], float)
    T = np.array([
        [0, 10, 10],
        [10, 0, 10],
        [10, 10, 0]
    ], float)
    # nó raiz
    node = Node(priority=0, bound=0, value=0, time_used=0, current=0, visited=(0,), depth=0)
    b1 = fractional_bound(values, visit, T, node, time_limit=40)
    b2 = fractional_bound(values, visit, T, node, time_limit=60)
    assert b2 >= b1
