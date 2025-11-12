
import argparse
import pandas as pd
from data_prep import clean_and_standardize, build_distance_time_matrices, compute_value_column
from eda import eda_summary, eda_plots
from heuristics import greedy_itinerary
from bnb import branch_and_bound

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Caminho do CSV (Kaggle ou próprio)")
    p.add_argument("--lat0", type=float, default=-8.4095, help="Latitude do hotel/depot (default: Bali)")
    p.add_argument("--lon0", type=float, default=115.1889, help="Longitude do hotel/depot (default: Bali)")
    p.add_argument("--speed-kmh", type=float, default=30.0, help="Velocidade média de deslocamento (km/h)")
    p.add_argument("--time-limit", type=float, default=480.0, help="Tempo total disponível (min)")
    p.add_argument("--w-rating", type=float, default=1.0, help="Peso do rating no valor")
    p.add_argument("--w-cost", type=float, default=0.0, help="Peso do custo no valor")
    p.add_argument("--max-nodes", type=int, default=100000, help="Limite de nós expandido (B&B)")
    p.add_argument("--time-cap", type=float, default=None, help="Limite de tempo em segundos (B&B)")
    p.add_argument("--sample-n", type=int, default=20, help="Usar apenas N POIs (para demo/performance)")
    return p.parse_args()

def main():
    args = parse_args()
    raw = pd.read_csv(args.csv)
    df, decisions = clean_and_standardize(raw)
    if args.sample_n and len(df) > args.sample_n:
        df = df.sample(args.sample_n, random_state=42).reset_index(drop=True)


    summary = eda_summary(df)
    print(" EDA: resumo ")
    print(summary)
    eda_plots(df)


    df["value"] = compute_value_column(df, args.w_rating, args.w_cost)


    D, T, points = build_distance_time_matrices(df, args.lat0, args.lon0, args.speed_kmh)


    values = [0.0] + df["value"].tolist()
    visit_time = [0.0] + df["est_time_min"].tolist()
    values = pd.Series(values).to_numpy()
    visit_time = pd.Series(visit_time).to_numpy()


    greedy = greedy_itinerary(values, visit_time, T, args.time_limit)
    print("\n Heurística Gulosa ")
    print(greedy)

    # B&B
    bnb_res = branch_and_bound(values, visit_time, T, args.time_limit,
                               max_nodes=args.max_nodes, time_cap_seconds=args.time_cap)
    print("\nBranch and Bound ")
    print(bnb_res)

    # Comparação
    print("\nComparação ")
    print({"Greedy_value": greedy["total_value"], "B&B_value": bnb_res["best_value"]})
    print("Rota B&B (ids em 'points'):", bnb_res["best_route"])
    print(points.loc[bnb_res["best_route"], ["name","latitude","longitude"]].reset_index(drop=True))

if __name__ == "__main__":
    main()
