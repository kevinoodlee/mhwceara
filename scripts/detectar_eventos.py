"""Detecta as ondas de calor marinhas em cada trecho do litoral do Ceará (1982 a 2025).

Calcula a TSM diária média de cada trecho, a climatologia e o limiar (percentil 90,
período-base 1982 a 2011) e detecta os eventos pela definição de Hobday et al. (2016).
Gera:
  - data/processed/tsm_por_trecho.csv: TSM diária média de cada trecho;
  - data/processed/climatologia_por_trecho.csv: TSM, climatologia e limiar diários;
  - outputs/tabelas/eventos_por_trecho.csv: um evento por linha;
  - outputs/tabelas/metricas_anuais_por_trecho.csv: um ano por linha, por trecho;
  - outputs/tabelas/resumo_por_trecho.csv: resumo de cada trecho;
  - outputs/tabelas/validacao_branqueamento.csv: dias em onda de calor nos anos com
    branqueamento de corais documentado no Ceará e a posição desses anos na série;
  - outputs/figuras/dias_onda_calor_por_ano.png e outputs/figuras/tsm_eventos_2020.png.
Uso:  python scripts/detectar_eventos.py
"""

import pandas as pd
from tqdm import tqdm

from mhwceara import dados, deteccao, figuras, trechos
from mhwceara.config import DESTAQUE_2020, PASTAS, PERIODO_BASE, REGISTROS_BRANQUEAMENTO

FONTE = (
    "Fonte: elaborado pelo autor com dados do NOAA OISST v2.1. "
    f"Período-base da climatologia: {PERIODO_BASE[0]} a {PERIODO_BASE[1]}."
)

print("lendo a TSM e calculando a média de cada trecho...")
series = trechos.serie_por_trecho(dados.ler_oisst(), trechos.ler_celulas())

resultados = {}
for nome in tqdm(series.columns, desc="detectando eventos", unit="trecho"):
    resultados.update(deteccao.detectar_por_trecho(series[[nome]], PERIODO_BASE))

for pasta in (PASTAS.processado, PASTAS.tabelas):
    pasta.mkdir(parents=True, exist_ok=True)
series.rename_axis("data").to_csv(PASTAS.processado / "tsm_por_trecho.csv")
deteccao.empilhar(resultados, "clim").to_csv(
    PASTAS.processado / "climatologia_por_trecho.csv", index=False
)
deteccao.empilhar(resultados, "eventos").to_csv(
    PASTAS.tabelas / "eventos_por_trecho.csv", index=False
)
deteccao.empilhar(resultados, "anual").to_csv(
    PASTAS.tabelas / "metricas_anuais_por_trecho.csv", index=False
)
resumo = pd.DataFrame({n: deteccao.resumo(r["eventos"], r["anual"]) for n, r in resultados.items()})
resumo.to_csv(PASTAS.tabelas / "resumo_por_trecho.csv")

print("\n== Resumo por trecho (1982 a 2025)")
print(resumo.to_string())
print("\n== Os 3 eventos de maior intensidade cumulativa em cada trecho")
colunas = ["inicio", "fim", "duracao", "intensidade_maxima", "intensidade_cumulativa"]
for nome, r in resultados.items():
    print(f"  {nome}:")
    maiores = r["eventos"].nlargest(3, "intensidade_cumulativa")[colunas]
    maiores = maiores.round({"intensidade_maxima": 2, "intensidade_cumulativa": 2})
    print("    " + maiores.to_string(index=False).replace("\n", "\n    "))

# validação: os anos com branqueamento documentado estão entre os de mais dias em onda de calor?
dias = deteccao.empilhar(resultados, "anual").pivot(
    index="ano", columns="trecho", values="dias_onda_calor"
)[list(resultados)]
posicao = dias.rank(ascending=False, method="min")  # 1 = ano com mais dias no trecho
validacao = (
    pd.DataFrame(REGISTROS_BRANQUEAMENTO)
    .set_index("ano")
    .join(dias.add_prefix("dias_"))
    .join(posicao.add_prefix("posicao_").astype(int))
)
validacao.to_csv(PASTAS.tabelas / "validacao_branqueamento.csv")
anos_registro = validacao.index
print("\n== Validação: anos com branqueamento de corais documentado no Ceará")
print(validacao.drop(columns=["registro", "fonte"]).to_string())
print(
    f"  média de dias em onda de calor: {dias.loc[anos_registro].mean().mean():.0f} nos anos com "
    f"registro | {dias.drop(index=anos_registro).mean().mean():.0f} nos demais anos "
    f"(posição 1 = ano com mais dias entre os {len(dias)})"
)

anual = {n: r["anual"] for n, r in resultados.items()}
fig1 = figuras.dias_por_ano(
    anual, "Dias em onda de calor marinha por ano, por trecho do litoral do Ceará", FONTE
)
fig2 = figuras.serie_com_eventos(
    resultados,
    2020,
    "TSM diária em 2020 e ondas de calor marinhas detectadas",
    FONTE + " Período de branqueamento: Lucas (2022).",
    destaque=DESTAQUE_2020,
)
print(f"\ngravado: {fig1}\ngravado: {fig2}")
print(f"tabelas em {PASTAS.tabelas}")
