# tests/test_prep.py
import pandas as pd
from data_prep import clean_and_standardize, build_distance_time_matrices

def test_clean_and_standardize_minimal():
    df = pd.DataFrame({
        "name": ["A","B"],
        "latitude": [0.0, 0.01],
        "longitude": [0.0, 0.01]
    })
    clean, dec = clean_and_standardize(df)
    assert {"rating","price_level","est_time_min"}.issubset(set(clean.columns))
    assert len(clean) == 2

def test_build_matrices_shapes():
    df = pd.DataFrame({
        "poi_id":[1,2],
        "name": ["A","B"],
        "latitude": [0.0, 0.01],
        "longitude": [0.0, 0.01],
        "rating":[4,4],
        "price_level":[1,1],
        "est_time_min":[60,60]
    })
    D, T, pts = build_distance_time_matrices(df, start_lat=0.0, start_lon=0.0, speed_kmh=30.0)
    assert D.shape == (3,3) and T.shape == (3,3)
    assert pts.iloc[0]["name"] == "Hotel/Depósito"
