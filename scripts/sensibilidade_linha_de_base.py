"""Sensibilidade da detecção à linha de base (período-base e tendência de longo prazo).

Compara três variantes, nos três trechos do litoral:
  1. principal: climatologia de 1982 a 2011;
  2. climatologia de 1982 a 2020 (período-base mais recente);
  3. série sem tendência linear (aquecimento de longo prazo removido), base 1982 a 2011.
Se o aumento de dias em onda de calor persistir na variante 3, os extremos ficaram mais
frequentes; se desaparecer, o aumento reflete o aquecimento médio visto por um limiar fixo.
Gera:
  - outputs/tabelas/sensibilidade_linha_de_base.csv: resumo por variante e trecho;
  - outputs/tabelas/dias_por_ano_variantes.csv: dias em onda de calor por ano, variante e trecho;
  - outputs/figuras/sensibilidade_linha_de_base.png: comparação das variantes por trecho.
Uso:  python scripts/sensibilidade_linha_de_base.py
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from mhwceara import dados, deteccao, figuras, trechos
from mhwceara.config import (
    PASTAS,
    PERIODO_BASE,
    PERIODO_BASE_SENSIBILIDADE,
    REGISTROS_BRANQUEAMENTO,
)

VARIANTES = {
    f"base {PERIODO_BASE[0]} a {PERIODO_BASE[1]} (principal)": {"periodo_base": PERIODO_BASE},
    f"base {PERIODO_BASE_SENSIBILIDADE[0]} a {PERIODO_BASE_SENSIBILIDADE[1]}": {
        "periodo_base": PERIODO_BASE_SENSIBILIDADE
    },
    "sem tendência linear": {"periodo_base": PERIODO_BASE, "sem_tendencia": True},
}
ANOS_REGISTRO = [r["ano"] for r in REGISTROS_BRANQUEAMENTO]

print("lendo a TSM e calculando a média de cada trecho...")
series = trechos.serie_por_trecho(dados.ler_oisst(), trechos.ler_celulas())

resultados = {}
for variante in tqdm(VARIANTES, desc="variantes", unit="variante"):
    resultados[variante] = deteccao.detectar_por_trecho(series, **VARIANTES[variante])

# dias em onda de calor por ano: {variante: tabela ano x trecho}
dias = {
    v: pd.DataFrame({t: r[t]["anual"]["dias_onda_calor"] for t in r}) for v, r in resultados.items()
}

linhas = []
for variante, por_trecho in resultados.items():
    for trecho, r in por_trecho.items():
        d = dias[variante][trecho]
        metade = int(d.index[len(d) // 2])
        antes, depois = d.loc[: metade - 1].mean(), d.loc[metade:].mean()
        posicao = d.rank(ascending=False, method="min")
        linha = {
            "variante": variante,
            "trecho": trecho,
            "eventos": len(r["eventos"]),
            f"dias_por_ano_{d.index[0]}_{metade - 1}": round(antes, 1),
            f"dias_por_ano_{metade}_{d.index[-1]}": round(depois, 1),
            "razao_segunda_primeira_metade": round(depois / antes, 1) if antes else np.nan,
            "tendencia_dias_por_decada": round(np.polyfit(d.index, d.values, 1)[0] * 10, 1),
        }
        linha.update({f"posicao_{ano}": int(posicao[ano]) for ano in ANOS_REGISTRO})
        if "tendencia_por_decada" in r:
            linha["tendencia_tsm_removida_C_por_decada"] = round(r["tendencia_por_decada"], 3)
        linhas.append(linha)
resumo = pd.DataFrame(linhas)

PASTAS.tabelas.mkdir(parents=True, exist_ok=True)
resumo.to_csv(PASTAS.tabelas / "sensibilidade_linha_de_base.csv", index=False)
longo = (
    pd.concat(
        {
            v: d.rename_axis("ano").reset_index().melt("ano", var_name="trecho")
            for v, d in dias.items()
        },
        names=["variante"],
    )
    .reset_index(level=0)
    .rename(columns={"value": "dias_onda_calor"})
)
longo.to_csv(PASTAS.tabelas / "dias_por_ano_variantes.csv", index=False)

pd.set_option("display.width", 200)
print("\n== Resumo por variante e trecho")
print(resumo.set_index(["variante", "trecho"]).T.to_string())
print(
    "\n(posicao_ANO: posição do ano entre os "
    f"{len(series.index.year.unique())}, pelo número de dias em onda de calor; 1 = maior)"
)

fonte = (
    "Fonte: elaborado pelo autor com dados do NOAA OISST v2.1. Na variante sem tendência, "
    "a tendência linear de 1982 a 2025 foi removida da TSM antes da detecção."
)
arquivo = figuras.comparar_variantes(
    dias, "Dias em onda de calor marinha por ano: sensibilidade à linha de base", fonte
)
print(f"\ngravado: {arquivo}")
print(f"tabelas em {PASTAS.tabelas}")
