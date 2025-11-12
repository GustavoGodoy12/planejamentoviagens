import argparse
import pandas as pd
from data_prep import load_dynamic_csv, clean_dynamic
from eda import eda_summary, eda_plots
from model import train_and_eval, feature_importance
from recommender import recommend

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--max-duration", type=float, default=None)
    p.add_argument("--max-cost", type=float, default=None)
    p.add_argument("--budget", type=str, default="Any")
    p.add_argument("--weather", type=str, default=None)
    p.add_argument("--traffic", type=str, default=None)
    p.add_argument("--crowd", type=str, default=None)
    p.add_argument("--event", type=str, default=None)
    p.add_argument("--theme", type=str, default=None)
    p.add_argument("--transport", type=str, default=None)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()

def main():
    args = parse_args()
    raw = load_dynamic_csv(args.csv)
    df = clean_dynamic(raw)
    summary = eda_summary(df)
    print("=== EDA ===")
    print(summary)
    eda_plots(df)
    tr = train_and_eval(df)
    print("=== Métricas ===")
    print(tr["metrics"])
    imp = feature_importance(tr["model"], tr["X_test"], tr["y_test"])
    print("=== Importância de Atributos (permutation) ===")
    print(imp.head(15).to_string(index=False))
    constraints = {"max_duration": args.max_duration, "max_cost": args.max_cost, "budget": args.budget}
    overrides = {"Weather": args.weather, "Traffic_Level": args.traffic, "Crowd_Density": args.crowd, "Event_Impact": args.event, "Preferred_Theme": args.theme, "Preferred_Transport": args.transport}
    rec = recommend(df, tr["model"], constraints, overrides, top_k=args.top_k)
    if len(rec):
        rec.to_csv("recommendations.csv", index=False, encoding="utf-8")
        print("=== Recomendações ===")
        print(rec.to_string(index=False))
        print("Arquivo salvo: recommendations.csv")
    else:
        print("Nenhuma rota atende aos filtros.")

if __name__ == "__main__":
    main()
