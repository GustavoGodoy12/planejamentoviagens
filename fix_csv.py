# fix_csv.py
import pandas as pd
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "bali_orig.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "sample_data/bali.csv"

df = pd.read_csv(src, encoding="utf-8-sig")

# renomeia colunas para o padrão do projeto
df = df.rename(columns={
    "nama": "name",
    "kategori": "category",
    "kabupaten_kota": "region",
    "link_lokasi": "maps_url",
    "link_gambar": "image_url"
})

# deixa as colunas que o app usa; o resto pode ficar
# o app exige no mínimo: name, latitude, longitude
assert {"name","latitude","longitude"}.issubset(df.columns), "Faltam colunas obrigatórias"

# se quiser, arredonde lat/lon
df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").round(6)
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").round(6)

df.to_csv(dst, index=False, encoding="utf-8")
print("CSV pronto em:", dst)
