"""Resposta da clorofila-a às ondas de calor marinhas, por trecho do litoral do Ceará.

Para cada trecho: média mensal de log10 da clorofila-a (OC-CCI, 4 km, pixels a mais de
cerca de 10 km da costa, meses com cobertura de pelo menos 50%), anomalia em relação à
climatologia mensal e comparação entre meses com onda de calor (15 dias ou mais em evento)
e sem, com defasagem de 0 a 2 meses (teste de permutação da diferença de medianas).
Cenário principal: eventos da linha de base 1982 a 2011. Sensibilidades: eventos da série
sem tendência, cobertura mínima de 30% e de 70% e todos os pixels da faixa costeira.
Gera:
  - data/processed/clorofila_por_trecho.csv: log10, cobertura e anomalia por mês e trecho;
  - outputs/tabelas/resposta_clorofila.csv: comparação por cenário, trecho e defasagem;
  - outputs/figuras/resposta_clorofila.png: cenário principal.
Uso:  python scripts/resposta_clorofila.py
"""

import pandas as pd
from tqdm import tqdm

from mhwceara import clorofila, dados, deteccao, figuras, trechos
from mhwceara.config import (
    DISTANCIA_MINIMA_COSTA,
    FRACAO_MINIMA,
    FRACOES_SENSIBILIDADE,
    PASTAS,
    PERIODO_BASE,
    TRECHOS,
)

PRINCIPAL = "principal"

print("lendo a clorofila-a e classificando os pixels em trechos (pode levar alguns segundos)...")
chl = dados.ler_clorofila()
limite = trechos.ler_limite(trechos.baixar_limite(progresso=False))
pixels = clorofila.classificar_pixels(chl, limite)

# cenários de série de clorofila: (distância mínima da costa, cobertura mínima)
series_cenario = {
    PRINCIPAL: (DISTANCIA_MINIMA_COSTA, FRACAO_MINIMA),
    "todos os pixels da faixa": (0.0, FRACAO_MINIMA),
}
for fracao in FRACOES_SENSIBILIDADE:
    if fracao != FRACAO_MINIMA:
        series_cenario[f"cobertura mínima {fracao:.0%}"] = (DISTANCIA_MINIMA_COSTA, fracao)
series_chl = {
    nome: clorofila.anomalia(clorofila.serie_por_trecho(chl, pixels, dist, frac))
    for nome, (dist, frac) in series_cenario.items()
}

print("detectando as ondas de calor (linha de base principal e série sem tendência)...")
sst = trechos.serie_por_trecho(dados.ler_oisst(), trechos.ler_celulas())
eventos = {
    PRINCIPAL: deteccao.detectar_por_trecho(sst, PERIODO_BASE),
    "eventos da série sem tendência": deteccao.detectar_por_trecho(
        sst, PERIODO_BASE, sem_tendencia=True
    ),
}

# cenários = combinação de uma série de clorofila com um conjunto de eventos
cenarios = {nome: (nome, PRINCIPAL) for nome in series_chl}
cenarios["eventos da série sem tendência"] = (PRINCIPAL, "eventos da série sem tendência")

partes = []
for cenario, (nome_serie, nome_eventos) in tqdm(cenarios.items(), desc="cenários", unit="cenário"):
    serie = series_chl[nome_serie]
    for trecho, r in eventos[nome_eventos].items():
        anom = serie[serie["trecho"] == trecho].set_index("data")["anomalia_log10"]
        dias = clorofila.dias_em_evento_por_mes(r["clim"], r["eventos"])
        resultado = clorofila.comparar(anom, dias)
        partes.append(resultado.assign(cenario=cenario, trecho=trecho))
tabela = pd.concat(partes, ignore_index=True)
tabela = tabela[
    ["cenario", "trecho"] + [c for c in tabela.columns if c not in ("cenario", "trecho")]
]

for pasta in (PASTAS.processado, PASTAS.tabelas):
    pasta.mkdir(parents=True, exist_ok=True)
series_chl[PRINCIPAL].round({"log10_chl": 4, "anomalia_log10": 4}).to_csv(
    PASTAS.processado / "clorofila_por_trecho.csv", index=False
)
tabela.to_csv(PASTAS.tabelas / "resposta_clorofila.csv", index=False)

principal = tabela[tabela["cenario"] == PRINCIPAL]
s = series_chl[PRINCIPAL]
print("\n== Pixels e meses válidos por trecho (cenário principal)")
print(
    s.groupby("trecho", sort=False)
    .agg(
        pixels=("pixels_total", "first"), meses_validos=("valido", "sum"), meses=("valido", "size")
    )
    .to_string()
)
colunas = [
    "trecho",
    "defasagem_meses",
    "meses_com_evento",
    "meses_sem_evento",
    "diferenca_log10",
    "variacao_percentual",
    "p_valor",
    "r_fracao_dias",
]
print("\n== Cenário principal: anomalia de clorofila-a em meses com e sem onda de calor")
print(principal[colunas].to_string(index=False))
print("\n== Sensibilidade (defasagem 0): variação percentual e valor-p")
d0 = tabela[tabela["defasagem_meses"] == 0]
ordem_trechos = [t for t in TRECHOS if t in set(d0["trecho"])]
for valor in ("variacao_percentual", "p_valor"):
    quadro = d0.pivot(index="cenario", columns="trecho", values=valor)
    print(quadro.loc[list(cenarios), ordem_trechos].to_string(), "\n")
print(
    "\n(variacao_percentual: diferença entre as medianas, em % da concentração de clorofila-a;"
    " negativa = menos clorofila nos meses com onda de calor)"
)

fonte = (
    "Fonte: elaborado pelo autor com dados do ESA OC-CCI v6.0 e do NOAA OISST v2.1. "
    "Ponto: mediana; barra: quartis (25% a 75%); p: teste de permutação."
)
arquivo = figuras.resposta_clorofila(
    principal, "Anomalia de clorofila-a em meses com e sem onda de calor marinha", fonte
)
print(f"\ngravado: {arquivo}")
print(f"tabelas em {PASTAS.tabelas}")
