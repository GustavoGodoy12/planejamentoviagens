# data_prep.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict

EARTH_R = 6371000.0  # meters

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distância Haversine em metros."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return EARTH_R * c

def clean_and_standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Limpeza e padronização mínima:
    - renomeia colunas para ['name','latitude','longitude','rating','price_level','est_time_min']
    - remove duplicatas e NaNs críticos
    - preenche defaults
    Retorna dataframe e dicionário de decisões.
    """
    decisions = {}

    # Renomear prováveis colunas
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            for k, orig in cols.items():
                if n == k:
                    return orig
        return None

    rename = {}
    candidates = {
    "name": ["name", "title", "poi_name", "nama"],
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "lng", "x"],
    "rating": ["rating", "score", "stars"],
    "price_level": ["price_level", "price", "cost_level"],
    "est_time_min": ["est_time_min", "duration_min", "visit_time_min"]
}
    for std, opts in candidates.items():
        col = pick(*opts)
        if col:
            rename[col] = std
    df = df.rename(columns=rename)

    # Garantir colunas mínimas
    required = ["name", "latitude", "longitude"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"CSV precisa conter coluna '{r}' (pode renomear no arquivo).")

    # Tipos
    df["name"] = df["name"].astype(str)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Remover inválidos/duplicatas
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).drop_duplicates(subset=["name","latitude","longitude"])
    decisions["rows_removed"] = int(before - len(df))

    # Defaults
    if "rating" not in df.columns:
        df["rating"] = 4.0
        decisions["rating_filled_default"] = 4.0
    else:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0)

    if "price_level" not in df.columns:
        df["price_level"] = 1.0
        decisions["price_level_default"] = 1.0
    else:
        df["price_level"] = pd.to_numeric(df["price_level"], errors="coerce").fillna(1.0)

    if "est_time_min" not in df.columns:
        df["est_time_min"] = 60.0
        decisions["est_time_min_default"] = 60.0
    else:
        df["est_time_min"] = pd.to_numeric(df["est_time_min"], errors="coerce").fillna(60.0)

    # Normalizações simples
    df["rating"] = df["rating"].clip(0, 5)
    df["price_level"] = df["price_level"].clip(lower=0)
    df["est_time_min"] = df["est_time_min"].clip(lower=5)

    # Reindex simples e id
    df = df.reset_index(drop=True)
    df.insert(0, "poi_id", np.arange(1, len(df)+1))  # reserva 0 para o hotel
    return df, decisions

def build_distance_time_matrices(df: pd.DataFrame,
                                 start_lat: float, start_lon: float,
                                 speed_kmh: float = 30.0) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Constrói matriz de distâncias (m) e tempos (min) incluindo o nó 0 = hotel/depot.
    """
    points = pd.concat([
        pd.DataFrame([{"poi_id": 0, "name": "Hotel/Depósito", "latitude": start_lat, "longitude": start_lon,
                       "rating": 0.0, "price_level": 0.0, "est_time_min": 0.0}]),
        df[["poi_id","name","latitude","longitude","rating","price_level","est_time_min"]]
    ], ignore_index=True)

    n = len(points)
    D = np.zeros((n, n), dtype=float)  # metros
    for i in range(n):
        D[i, :] = haversine_m(points.loc[i, "latitude"], points.loc[i, "longitude"],
                              points["latitude"].values, points["longitude"].values)

    speed_mps = (speed_kmh * 1000.0) / 3600.0
    T = (D / speed_mps) / 60.0  # minutos

    return D, T, points

def compute_value_column(df: pd.DataFrame, w_rating: float = 1.0, w_cost: float = 0.0) -> pd.Series:
    """
    Valor = w_rating * rating - w_cost * price_level
    """
    return w_rating * df["rating"].astype(float) - w_cost * df["price_level"].astype(float)
