from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from data_prep import filter_constraints, to_recommendation_table
from model import score_routes

def recommend(df: pd.DataFrame, model, constraints: Dict[str, Any], user_overrides: Dict[str, Any], top_k: int = 10) -> pd.DataFrame:
    pool = filter_constraints(df, constraints.get("max_duration"), constraints.get("max_cost"), constraints.get("budget"))
    if len(pool) == 0:
        return pd.DataFrame()
    preds = score_routes(model, pool, user_overrides)
    rec = to_recommendation_table(pool, preds)
    rec = rec.head(top_k)
    return rec

def pareto_frontier(df: pd.DataFrame, x_col: str = "Total_Cost", y_col: str = "Total_Duration", score_col: str = "Predicted_Satisfaction") -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = df[[x_col, y_col, score_col]].copy()
    g = g.sort_values([x_col, y_col])
    frontier = []
    best_score = -np.inf
    for _, row in g.iterrows():
        if row[score_col] > best_score:
            frontier.append(row)
            best_score = row[score_col]
    frontier_df = pd.DataFrame(frontier)
    dominated = pd.merge(df, frontier_df, on=[x_col, y_col, score_col], how="left", indicator=True)
    dominated = dominated[dominated["_merge"] == "left_only"].drop(columns=["_merge"])
    return frontier_df, dominated
