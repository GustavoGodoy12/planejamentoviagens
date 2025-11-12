
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

def eda_summary(df: pd.DataFrame) -> Dict[str, Any]:

    num_df = df.select_dtypes(include=["number"])
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_ratio": (df.isna().mean()).to_dict(),
        "describe": num_df.describe().to_dict()  
    }
    return summary

def eda_plots(df: pd.DataFrame, outdir: str = "eda_outputs") -> Tuple[str, str, str, str]:
    import os
    os.makedirs(outdir, exist_ok=True)

    p1 = f"{outdir}/hist_rating.png"
    plt.figure()
    df["rating"].plot(kind="hist", bins=20)
    plt.title("Distribuição de Rating")
    plt.xlabel("rating")
    plt.ylabel("freq")
    plt.tight_layout(); plt.savefig(p1); plt.close()

    p2 = f"{outdir}/hist_est_time.png"
    plt.figure()
    df["est_time_min"].plot(kind="hist", bins=20)
    plt.title("Distribuição do tempo de visita (min)")
    plt.xlabel("min")
    plt.ylabel("freq")
    plt.tight_layout(); plt.savefig(p2); plt.close()

    p3 = f"{outdir}/box_price.png"
    plt.figure()
    sns.boxplot(x=df["price_level"])
    plt.title("Boxplot price_level")
    plt.tight_layout(); plt.savefig(p3); plt.close()

    p4 = f"{outdir}/scatter_rating_price.png"
    plt.figure()
    plt.scatter(df["price_level"], df["rating"])
    plt.xlabel("price_level")
    plt.ylabel("rating")
    plt.title("Relação preço x rating")
    plt.tight_layout(); plt.savefig(p4); plt.close()

    return p1, p2, p3, p4
