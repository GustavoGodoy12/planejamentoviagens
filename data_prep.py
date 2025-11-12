from __future__ import annotations
import pandas as pd
import numpy as np

CATS = ["Weather","Traffic_Level","Crowd_Density","Event_Impact","Optimal_Route_Preference","Gender","Nationality","Travel_Companions","Budget_Category","Preferred_Theme","Preferred_Transport"]
NUMS = ["Total_Duration","Total_Cost","Age","User_ID"]
TARGET = "Satisfaction_Score"
IDCOLS = ["Route_ID","Sequence"]

def load_dynamic_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Route_ID" not in df.columns:
        raise ValueError("CSV inválido para schema dynamic.csv")
    return df

def clean_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATS + IDCOLS + [TARGET]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["Total_Duration","Total_Cost"])
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
    df["Total_Duration"] = df["Total_Duration"].clip(lower=1)
    df["Total_Cost"] = df["Total_Cost"].clip(lower=0)
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(df["Age"].median())
    else:
        df["Age"] = 40
    if "User_ID" not in df.columns:
        df["User_ID"] = 0
    for c in CATS:
        if c not in df.columns:
            df[c] = "Unknown"
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
        df[c] = df[c].replace({"": "Unknown", "nan": "Unknown", "NaN": "Unknown", "None": "Unknown"})
    if "Sequence" not in df.columns:
        df["Sequence"] = ""
    return df

def split_features(df: pd.DataFrame):
    X = df[NUMS + CATS]
    y = df[TARGET]
    return X, y

def filter_constraints(df: pd.DataFrame, max_duration: float | None, max_cost: float | None, budget: str | None) -> pd.DataFrame:
    g = df.copy()
    if max_duration is not None:
        g = g[g["Total_Duration"] <= max_duration]
    if max_cost is not None:
        g = g[g["Total_Cost"] <= max_cost]
    if budget and budget != "Any" and "Budget_Category" in g.columns:
        g = g[g["Budget_Category"] == budget]
    return g

def to_recommendation_table(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["Predicted_Satisfaction"] = preds
    cols = ["Route_ID","User_ID","Sequence","Total_Duration","Total_Cost","Weather","Traffic_Level","Crowd_Density","Event_Impact","Preferred_Theme","Preferred_Transport","Budget_Category","Satisfaction_Score","Predicted_Satisfaction"]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values("Predicted_Satisfaction", ascending=False)
