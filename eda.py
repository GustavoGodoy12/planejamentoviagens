from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

def eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    num_df = df.select_dtypes(include=["number"])
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_ratio": (df.isna().mean()).to_dict(),
        "describe_num": num_df.describe().to_dict()
    }
    return summary

def eda_plots(df: pd.DataFrame, outdir: str = "eda_outputs") -> Tuple[str, str, str, str]:
    import os
    os.makedirs(outdir, exist_ok=True)
    p1 = f"{outdir}/hist_duration.png"
    plt.figure()
    df["Total_Duration"].plot(kind="hist", bins=20)
    plt.title("Distribuição da Duração Total")
    plt.xlabel("min")
    plt.ylabel("freq")
    plt.tight_layout(); plt.savefig(p1); plt.close()
    p2 = f"{outdir}/hist_cost.png"
    plt.figure()
    df["Total_Cost"].plot(kind="hist", bins=20)
    plt.title("Distribuição de Custo Total")
    plt.xlabel("custo")
    plt.ylabel("freq")
    plt.tight_layout(); plt.savefig(p2); plt.close()
    p3 = f"{outdir}/box_weather_satisfaction.png"
    plt.figure()
    sns.boxplot(x="Weather", y="Satisfaction_Score", data=df)
    plt.title("Satisfação por Clima")
    plt.tight_layout(); plt.savefig(p3); plt.close()
    p4 = f"{outdir}/box_traffic_duration.png"
    plt.figure()
    sns.boxplot(x="Traffic_Level", y="Total_Duration", data=df)
    plt.title("Duração por Tráfego")
    plt.tight_layout(); plt.savefig(p4); plt.close()
    return p1, p2, p3, p4
