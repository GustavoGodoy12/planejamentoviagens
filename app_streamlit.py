
import streamlit as st
import pandas as pd
import numpy as np
from data_prep import clean_and_standardize, build_distance_time_matrices, compute_value_column
from eda import eda_summary, eda_plots
from heuristics import greedy_itinerary
from bnb import branch_and_bound

st.set_page_config(page_title="Itinerário Econômico (Branch & Bound)", layout="wide")

st.title("Planejamento de Viagens — Itinerário mais econômico (Branch & Bound)")

st.sidebar.header("1) Dados")
uploaded = st.sidebar.file_uploader("CSV de pontos turísticos (Kaggle ou próprio)", type=["csv"])
lat0 = st.sidebar.number_input("Latitude do hotel/depot", value=-8.4095, format="%.6f")
lon0 = st.sidebar.number_input("Longitude do hotel/depot", value=115.1889, format="%.6f")
sample_n = st.sidebar.slider("Amostra de POIs (para demo/perf)", min_value=5, max_value=100, value=20, step=1)

st.sidebar.header("2) Parâmetros do Modelo")
speed = st.sidebar.slider("Velocidade média (km/h)", 5, 80, 30, 1)
time_limit = st.sidebar.slider("Tempo total (min)", 60, 720, 480, 10)
w_rating = st.sidebar.slider("Peso rating", 0.0, 2.0, 1.0, 0.1)
w_cost = st.sidebar.slider("Peso custo (penaliza)", 0.0, 2.0, 0.0, 0.1)

st.sidebar.header("3) Execução Branch & Bound")
max_nodes = st.sidebar.number_input("Máx. nós", value=50000, step=1000)
time_cap = st.sidebar.number_input("Limite de tempo (s) (0=sem)", value=0, step=1)
if time_cap <= 0:
    time_cap = None

run = st.sidebar.button("Rodar Pipeline")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    df, decisions = clean_and_standardize(raw)
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)

    st.subheader("Dados carregados (amostra)")
    st.dataframe(df.head(20))

    st.subheader("Mapa (POIs)")
    st.map(df.rename(columns={"latitude":"lat","longitude":"lon"})[["lat","lon"]])

    st.subheader("EDA Rápida")
    summary = eda_summary(df)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dimensões**:", summary["shape"])
        st.write("**Tipos**:", summary["dtypes"])
        st.write("**Ausentes (%)**:", {k: round(v*100,2) for k,v in summary["missing_ratio"].items()})
    with col2:
        p1, p2, p3, p4 = eda_plots(df)
        st.image([p1, p2, p3, p4], caption=["Hist rating", "Hist tempo visita", "Box price", "Scatter preço x rating"])

    df["value"] = compute_value_column(df, w_rating, w_cost)
    D, T, points = build_distance_time_matrices(df, lat0, lon0, speed)

    values = np.array([0.0] + df["value"].tolist())
    visit_time = np.array([0.0] + df["est_time_min"].tolist())

    st.subheader("Matriz — Info básica")
    st.write(f"Pontos (inclui 0=hotel): {len(points)}")
    st.write("Exemplo de tempos (min):")
    st.dataframe(pd.DataFrame(T[:5,:5]).round(1))

    st.subheader("Heurística Gulosa (baseline)")
    greedy = greedy_itinerary(values, visit_time, T, time_limit)
    st.write(greedy)
    st.write("Rota gulosa (nomes):")
    st.write(points.loc[greedy["route"], "name"].tolist())

    if run:
        st.subheader("Branch & Bound — Execução")
        with st.spinner("Executando..."):
            res = branch_and_bound(values, visit_time, T, time_limit,
                                   max_nodes=int(max_nodes), time_cap_seconds=time_cap)
        st.success("Concluído!")
        st.write(res)
        st.write("**Rota ótima (nomes):**")
        st.write(points.loc[res["best_route"], "name"].tolist())

        st.subheader("Comparação")
        st.write({
            "Greedy_value": greedy["total_value"],
            "Greedy_time": round(greedy["total_time"],2),
            "BnB_value": res["best_value"],
            "BnB_time": round(res["best_time"],2)
        })

        st.subheader("Sensibilidade (slider interativo já serve)")
        st.info("Ajuste *Tempo total*, *Velocidade*, *Pesos* no menu e execute novamente para observar a sensibilidade.")

        st.subheader("Visualização contextual")
        st.map(points.rename(columns={"latitude":"lat","longitude":"lon"})[["lat","lon"]])
else:
    st.info("Faça upload de um CSV para começar. Sugestão: dataset de atrações de Bali no Kaggle (tem latitude/longitude).")
