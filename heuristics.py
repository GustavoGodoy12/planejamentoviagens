# heuristics.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple

def greedy_itinerary(values: np.ndarray, visit_time: np.ndarray,
                     T: np.ndarray, time_limit: float) -> Dict:
    """
    Heurística simples:
      - começa no 0, escolhe próximo POI de melhor razão valor / (t_travel + t_visita + t_retorno_min),
        enquanto houver tempo.
    Retorna dict com rota, valor_total e tempo_total.
    """
    n = len(values)
    remaining = set(range(1, n))
    route = [0]
    total_value = 0.0
    total_time = 0.0

    while remaining:
        cur = route[-1]
        best, best_ratio = None, -1.0
        best_incr_time, best_next = None, None

        for j in list(remaining):
            # tempo extra = ir cur->j + visita j + voltar j->0 (mínimo necessário p/ viabilidade)
            incr = T[cur, j] + visit_time[j] + T[j, 0]
            ratio = values[j] / (incr + 1e-9)
            if ratio > best_ratio:
                best_ratio = ratio
                best = j
                best_incr_time = incr

        if best is None:
            break

        # checa viabilidade
        if total_time + T[cur, best] + visit_time[best] + T[best, 0] <= time_limit:
            # aceita
            total_time += T[cur, best] + visit_time[best]
            total_value += values[best]
            route.append(best)
            remaining.remove(best)
        else:
            # não cabe mais ninguém
            break

    # retornar ao depósito
    total_time += T[route[-1], 0]
    route.append(0)
    return {"route": route, "total_value": float(total_value), "total_time": float(total_time)}
