"""Teste estatístico das tendências de 1982 a 2025, por trecho do litoral do Ceará.

Teste de Mann-Kendall com correção para autocorrelação (Hamed e Rao, 1998) e inclinação
de Sen, aplicados às séries anuais de:
  - TSM média anual (°C);
  - dias em onda de calor marinha por ano e número de eventos por ano, nas três variantes
    de linha de base (principal 1982 a 2011, base 1982 a 2020 e série sem tendência).
Gera: outputs/tabelas/tendencias.csv (uma linha por variante, trecho e indicador).
Uso:  python scripts/tendencias.py
"""

import pandas as pd
from tqdm import tqdm

from mhwceara import dados, deteccao, trechos
from mhwceara.config import PASTAS, PERIODO_BASE, PERIODO_BASE_SENSIBILIDADE

VARIANTES = {
    f"base {PERIODO_BASE[0]} a {PERIODO_BASE[1]} (principal)": {"periodo_base": PERIODO_BASE},
    f"base {PERIODO_BASE_SENSIBILIDADE[0]} a {PERIODO_BASE_SENSIBILIDADE[1]}": {
        "periodo_base": PERIODO_BASE_SENSIBILIDADE
    },
    "sem tendência linear": {"periodo_base": PERIODO_BASE, "sem_tendencia": True},
}
INDICADORES = {"dias_onda_calor": "dias em onda de calor por ano", "n_eventos": "eventos por ano"}

print("lendo a TSM e calculando a média de cada trecho...")
series = trechos.serie_por_trecho(dados.ler_oisst(), trechos.ler_celulas())

linhas = []
for trecho in series.columns:
    anual = series[trecho].groupby(series.index.year).mean()
    linhas.append(
        {
            "variante": "TSM observada",
            "trecho": trecho,
            "indicador": "TSM média anual (°C)",
            **deteccao.tendencia(anual),
        }
    )

for variante in tqdm(VARIANTES, desc="variantes", unit="variante"):
    resultados = deteccao.detectar_por_trecho(series, **VARIANTES[variante])
    for trecho, r in resultados.items():
        for coluna, descricao in INDICADORES.items():
            linhas.append(
                {
                    "variante": variante,
                    "trecho": trecho,
                    "indicador": descricao,
                    **deteccao.tendencia(r["anual"][coluna]),
                }
            )

tabela = pd.DataFrame(linhas)
PASTAS.tabelas.mkdir(parents=True, exist_ok=True)
tabela.to_csv(PASTAS.tabelas / "tendencias.csv", index=False)

pd.set_option("display.width", 200)
print(
    "\n== Tendências de 1982 a 2025 (Mann-Kendall com correção de Hamed e Rao; inclinação de Sen)"
)
for (variante, indicador), grupo in tabela.groupby(["variante", "indicador"], sort=False):
    print(f"\n  {indicador} | {variante}")
    colunas = [
        "trecho",
        "inclinacao_sen_por_decada",
        "tau",
        "p_valor",
        "p_valor_sem_correcao",
        "fator_correcao",
    ]
    print("    " + grupo[colunas].to_string(index=False).replace("\n", "\n    "))
print(
    "\n(inclinacao_sen_por_decada: variação por década na unidade do indicador; "
    "p_valor < 0,05: tendência estatisticamente significativa)"
)
print(f"\ngravado: {PASTAS.tabelas / 'tendencias.csv'}")
