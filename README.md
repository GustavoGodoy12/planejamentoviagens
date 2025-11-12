# Planejamento de Viagens com Branch and Bound (Itinerário mais econômico no prazo)

Este projeto resolve a montagem de um itinerário turístico que **maximiza a utilidade** (popularidade / rating ou “valor”) sob um **orçamento de tempo** (limite de horas/dia), decidindo **quais pontos visitar e em que ordem**.  
É um caso de **Orienteering Problem** (OP) / “prize-collecting TSP”, resolvido com **Branch and Bound** e comparado com uma **heurística gulosa**.

## 1) Dados (Aquisição e Preparo)

**Dataset preferido (Kaggle):**
- Bali Tourist Attractions Dataset (Google Maps). Contém `name, latitude, longitude, rating, ...`.  
  Link: página Kaggle.  
- Alternativa ampla: Points of Interest POI Database (lat/lon, categorias).  
  Link: página Kaggle.

> Você pode usar qualquer CSV próprio contanto que contenha, no mínimo:  
> `name, latitude, longitude`. Colunas opcionais: `rating` (valor/benefício), `price_level` (custo relativo), `est_time_min` (tempo médio no local).

### Variáveis relevantes
- `name` (str): nome do ponto.
- `latitude`, `longitude` (float): coordenadas para distâncias.
- `rating` (float, opcional): usado como “valor” do ponto (default = 4.0).
- `price_level` (int/float, opcional): custo relativo (0–4; default = 1).
- `est_time_min` (float, opcional): tempo de visita; default = 60 min.

### Contexto / Problema
Planejar uma rota diária que **maximiza o valor total** visitado, sem exceder:
- **Tempo total** = deslocamentos + tempos de visita,
- e respeitando início/fim no **hotel (depot)**.

O modelo usa **matriz de tempos** derivada por **distância Haversine / velocidade média**.

## 2) Modelagem

### 2.1 Variáveis de decisão
- Ordem e subconjunto de pontos a visitar (variável binária por ponto + sequência implícita).
- Estado no B&B: `(visited_set, current_node, elapsed_time, value)`.

### 2.2 Função objetivo
Maximizar `sum(value_i)` dos pontos visitados, onde `value_i = w_rating*rating_i - w_cost*price_level_i` (ajustável).

### 2.3 Restrições
- Tempo total ≤ `T_max` (min).
- Começa e termina no nó 0 (hotel).
- Cada ponto visitado no máx. uma vez.

### 2.4 Hipótese de relaxação (Bound)
Para o *bound* superior, relaxamos a integralidade do conjunto restante e calculamos um **upper bound fracionário no tempo remanescente**:
- Transformamos os POIs restantes em itens com “custo temporal mínimo de inserção” aproximado (ida do nó atual → POI → melhor retorno), e “valor” = utilidade do POI.
- Preenchemos o tempo restante **frac** ordenando por `valor / tempo_mín` (mochila fracionária).  
Isso dá um **limite superior** para poda.

### 2.5 Política de busca e poda
- **Best-first** por maior bound (priority queue).
- Poda quando `bound ≤ melhor_valor` ou inviabilidade de tempo.
- Parada: fila esvaziada ou tempo limite de execução.

## 3) Implementação

- `bnb.py`: Branch and Bound.
- `heuristics.py`: gulosa de referência (inserção por melhor razão valor/tempo).
- `data_prep.py`: limpeza, padronização, matriz de tempo/distância.
- `eda.py`: EDA (estatísticas e gráficos).
- `main.py`: executa pipeline completo no terminal.
- `app_streamlit.py`: UI com upload, parâmetros, EDA, execução e resultados.
- `tests/`: pytest de *bound*, geração/validação de estados e poda.

## 4) Como rodar

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# CLI
python main.py --csv sample_data/bali_attractions_sample.csv --time-limit 480 --speed-kmh 30

# Dashboard
streamlit run app_streamlit.py
python -m streamlit run app_streamlit.py
