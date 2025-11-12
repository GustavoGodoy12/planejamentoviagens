from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from data_prep import CATS, NUMS, split_features

def build_model(random_state: int = 42) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATS)
    ])
    rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("rf", rf)])
    return pipe

def train_and_eval(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    X, y = split_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = build_model(random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {"r2": float(r2_score(y_test, y_pred)), "mae": float(mean_absolute_error(y_test, y_pred))}
    return {"model": model, "X_test": X_test, "y_test": y_test, "y_pred": y_pred, "metrics": metrics}

def feature_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1, scoring="r2")
    imp = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean}).sort_values("importance", ascending=False)
    return imp

def score_routes(model: Pipeline, base_df: pd.DataFrame, user_inputs: Dict[str, Any]) -> np.ndarray:
    g = base_df.copy()
    for k, v in user_inputs.items():
        if k in g.columns and v is not None:
            g[k] = v
    Xg = g[NUMS + CATS]
    preds = model.predict(Xg)
    return preds
