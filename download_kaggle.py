
import os, glob, shutil
import kagglehub

DATASET = "ziya07/dynamic-tourism-route-dataset-dtrd"
DEST_DIR = "sample_data"
DEST_FILE = "dynamic.csv"

os.makedirs(DEST_DIR, exist_ok=True)

print("Baixando dataset do Kaggle (isso pode levar alguns segundos)...")
path = kagglehub.dataset_download(DATASET)
print("Arquivos baixados em:", path)


csv_candidates = []
for pattern in ("*.csv", "*/*.csv", "*/*/*.csv"):
    csv_candidates.extend(glob.glob(os.path.join(path, pattern)))

if not csv_candidates:
    raise FileNotFoundError("Não encontrei nenhum CSV dentro do dataset baixado.")

src_csv = csv_candidates[0]
dst_csv = os.path.join(DEST_DIR, DEST_FILE)
shutil.copy2(src_csv, dst_csv)
print(f"CSV copiado para: {dst_csv}")
