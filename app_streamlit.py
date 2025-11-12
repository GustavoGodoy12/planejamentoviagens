import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_prep import load_dynamic_csv, clean_dynamic
from eda import eda_summary, eda_plots
from model import train_and_eval, feature_importance
from recommender import recommend, pareto_frontier

st.set_page_config(page_title="Recomendador de Rotas Dinâmicas", layout="wide")
st.title("Recomendador de Rotas Dinâmicas")

with st.sidebar:
    st.header("Dados")
    uploaded = st.file_uploader("Envie o dynamic.csv", type=["csv"])
    st.header("Filtros")
    max_duration = st.number_input("Duração máxima (min)", value=0, min_value=0)
    max_cost = st.number_input("Custo máximo", value=0, min_value=0)
    budget = st.selectbox("Budget", ["Any","Low","Medium","High"])
    st.header("Cenário do Usuário")
    weather = st.text_input("Weather")
    traffic = st.text_input("Traffic_Level")
    crowd = st.text_input("Crowd_Density")
    event = st.text_input("Event_Impact")
    theme = st.text_input("Preferred_Theme")
    transport = st.text_input("Preferred_Transport")
    topk = st.slider("Top-K recomendações", 1, 30, 10)
    run = st.button("Treinar e Recomendar")

if uploaded is None:
    st.info("Aguardando CSV.")
else:
    raw = load_dynamic_csv(uploaded)
    df = clean_dynamic(raw)
    st.subheader("Amostra dos dados")
    st.dataframe(df.head(20))
    st.subheader("Resumo")
    st.json(eda_summary(df))
    p1, p2, p3, p4 = eda_plots(df)
    st.subheader("Gráficos")
    st.image([p1, p2, p3, p4], caption=["Duração","Custo","Satisfação por Clima","Duração por Tráfego"])
    if run:
        with st.spinner("Treinando modelo..."):
            tr = train_and_eval(df)
        st.success("Modelo treinado")
        st.write(tr["metrics"])
        imp = feature_importance(tr["model"], tr["X_test"], tr["y_test"])
        st.subheader("Importância dos atributos")
        st.dataframe(imp.head(20))
        constraints = {"max_duration": None if max_duration==0 else max_duration, "max_cost": None if max_cost==0 else max_cost, "budget": budget}
        overrides = {"Weather": weather if weather else None, "Traffic_Level": traffic if traffic else None, "Crowd_Density": crowd if crowd else None, "Event_Impact": event if event else None, "Preferred_Theme": theme if theme else None, "Preferred_Transport": transport if transport else None}
        rec = recommend(df, tr["model"], constraints, overrides, top_k=topk)
        st.subheader("Recomendações")
        if len(rec):
            st.dataframe(rec.reset_index(drop=True))
            st.download_button("Baixar recomendações CSV", data=rec.to_csv(index=False).encode("utf-8"), file_name="recommendations.csv", mime="text/csv")
            st.subheader("Trade-off Custo × Duração (fronteira de Pareto)")
            fr, dom = pareto_frontier(rec, x_col="Total_Cost", y_col="Total_Duration", score_col="Predicted_Satisfaction")
            fig = plt.figure()
            plt.scatter(dom["Total_Cost"], dom["Total_Duration"], alpha=0.5)
            if len(fr):
                plt.plot(fr["Total_Cost"], fr["Total_Duration"], marker="o")
            plt.xlabel("Total_Cost")
            plt.ylabel("Total_Duration")
            st.pyplot(fig)
        else:
            st.warning("Nenhuma rota atende aos filtros.")
