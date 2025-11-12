
import pandas as pd
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "bali_orig.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "sample_data/bali.csv"

df = pd.read_csv(src, encoding="utf-8-sig")


df = df.rename(columns={
    "nama": "name",
    "kategori": "category",
    "kabupaten_kota": "region",
    "link_lokasi": "maps_url",
    "link_gambar": "image_url"
})


assert {"name","latitude","longitude"}.issubset(df.columns), "Faltam colunas obrigatórias"

df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").round(6)
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").round(6)

df.to_csv(dst, index=False, encoding="utf-8")
print("CSV pronto em:", dst)
