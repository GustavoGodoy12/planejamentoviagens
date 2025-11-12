# bnb.py
from __future__ import annotations
import time
import heapq
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict
import numpy as np

@dataclass(order=True)
class Node:
    priority: float
    bound: float = field(compare=False)
    value: float = field(compare=False)
    time_used: float = field(compare=False)
    current: int = field(compare=False)
    visited: Tuple[int, ...] = field(compare=False)
    depth: int = field(compare=False)

def fractional_bound(values: np.ndarray, visit_time: np.ndarray, T: np.ndarray,
                     node: Node, time_limit: float) -> float:
    """
    Limite superior fracionário tipo "mochila":
    - Estima tempo mínimo para inserir um POI i: cur->i + visit_i + i->0
    - Preenche o tempo restante com razões valor/tempo decrescentes.
    """
    n = len(values)
    remaining = [i for i in range(1, n) if i not in node.visited]
    cur = node.current
    remaining_time = time_limit - node.time_used

    if remaining_time <= 0:
        return node.value

    items = []
    for i in remaining:
        cost = T[cur, i] + visit_time[i] + T[i, 0]
        if cost <= 0:
            cost = 1e-9
        items.append((values[i] / cost, values[i], cost, i))
    items.sort(reverse=True, key=lambda x: x[0])

    ub = node.value
    cap = remaining_time
    for ratio, val, cost, i in items:
        if cap <= 0:
            break
        take = min(1.0, cap / cost)
        ub += val * take
        cap -= cost * take
    return ub

def branch_and_bound(values: np.ndarray, visit_time: np.ndarray, T: np.ndarray,
                     time_limit: float, max_nodes: int = 100000, policy: str = "best_first",
                     time_cap_seconds: float | None = None) -> Dict:
    """
    Branch and Bound para Orienteering (começa/termina em 0).
    Estados ramificam escolhendo o próximo POI.
    """
    t0 = time.time()
    start = Node(priority=0.0, bound=0.0, value=0.0, time_used=0.0, current=0, visited=(0,), depth=0)
    start.bound = fractional_bound(values, visit_time, T, start, time_limit)
    start.priority = -start.bound  # heapq é min-heap

    heap: List[Node] = [start]
    best = {"route": [0, 0], "total_value": 0.0, "total_time": 0.0}
    expanded = 0
    max_depth = 0

    while heap:
        if expanded >= max_nodes:
            break
        if time_cap_seconds is not None and (time.time() - t0) >= time_cap_seconds:
            break

        node = heapq.heappop(heap)
        # poda por bound
        if node.bound <= best["total_value"] + 1e-9:
            continue

        cur = node.current
        # tentar expandir cada candidato não visitado
        moved = False
        for j in range(1, len(values)):
            if j in node.visited:
                continue
            # tempo se visitar j e POSSÍVEL RETORNO ao depósito depois
            new_time = node.time_used + T[cur, j] + visit_time[j]
            feasible = new_time + T[j, 0] <= time_limit
            if not feasible:
                continue

            moved = True
            new_value = node.value + values[j]
            new_visited = tuple(list(node.visited) + [j])
            child = Node(priority=0.0, bound=0.0, value=new_value, time_used=new_time,
                         current=j, visited=new_visited, depth=node.depth + 1)
            child.bound = fractional_bound(values, visit_time, T, child, time_limit)
            child.priority = -child.bound

            # atualização de melhor solução factível (fechando o ciclo)
            total_time_candidate = new_time + T[j, 0]
            if new_value > best["total_value"] and total_time_candidate <= time_limit:
                best = {"route": list(new_visited) + [0], "total_value": float(new_value),
                        "total_time": float(total_time_candidate)}

            # enfileira se ainda promissor
            if child.bound > best["total_value"] + 1e-9:
                heapq.heappush(heap, child)

        expanded += 1
        max_depth = max(max_depth, node.depth)
        # também considerar não expandir mais deste nó (já coberto pelo best update)

    runtime = time.time() - t0
    return {
        "best_route": best["route"],
        "best_value": best["total_value"],
        "best_time": best["total_time"],
        "expanded_nodes": expanded,
        "max_depth": max_depth,
        "runtime_sec": runtime
    }
