"""
Detecção de ondas de calor marinhas pela definição de Hobday et al. (2016).

Implementação própria, usada como reserva caso o xmhw não sirva (licença ou
instalação) e como referência para validar o xmhw.

Parâmetros padrão (Hobday et al., 2016):
  - climatologia e limiar por dia do ano, em janela de ±5 dias;
  - limiar = percentil 90;
  - suavização circular de 31 dias da média e do limiar;
  - evento = 5 dias ou mais acima do limiar;
  - eventos separados por até 2 dias são unidos.

Entrada: série diária de TSM (pandas.Series com índice de datas), de um pixel
ou da média de um trecho do litoral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["dia_do_ano_366", "climatologia", "detectar_eventos", "metricas_anuais"]


def dia_do_ano_366(datas: pd.DatetimeIndex) -> np.ndarray:
    """Dia do ano numa escala fixa de 366 dias.

    29/02 é sempre o dia 60 e 01/03 é sempre o dia 61, em anos bissextos ou não.
    Assim o mesmo dia do calendário recebe o mesmo número em todos os anos.
    """
    datas = pd.DatetimeIndex(datas)
    doy = datas.day_of_year.to_numpy().copy()
    ajuste = (~datas.is_leap_year) & (datas.month > 2)
    doy[ajuste] += 1
    return doy


def _media_movel_circular(valores: np.ndarray, largura: int) -> np.ndarray:
    """Média móvel centrada de largura ímpar, tratando o ano como circular."""
    if largura <= 1:
        return valores.copy()
    meia = largura // 2
    estendido = np.concatenate([valores[-meia:], valores, valores[:meia]])
    nucleo = np.ones(largura) / largura
    return np.convolve(estendido, nucleo, mode="valid")


def _serie_diaria(temp: pd.Series) -> pd.Series:
    """Garante índice diário contínuo; dias ausentes viram NaN."""
    temp = temp.sort_index()
    indice = pd.date_range(temp.index.min(), temp.index.max(), freq="D")
    return temp.reindex(indice).astype(float)


def climatologia(
    temp: pd.Series,
    periodo_base: tuple[int, int] = (1982, 2011),
    janela: int = 5,
    percentil: float = 90.0,
    suavizacao: int = 31,
) -> pd.DataFrame:
    """Média climatológica e limiar para cada dia da série.

    Retorna DataFrame com as colunas 'temp', 'clim' (média) e 'limiar',
    no mesmo índice diário da série.
    """
    temp = _serie_diaria(temp)
    doy = dia_do_ano_366(temp.index)
    anos = temp.index.year
    na_base = (anos >= periodo_base[0]) & (anos <= periodo_base[1])
    if not na_base.any():
        raise ValueError(f"a série não cobre o período-base {periodo_base}")

    valores_base = temp.to_numpy()[na_base]
    doy_base = doy[na_base]

    media = np.full(366, np.nan)
    limiar = np.full(366, np.nan)
    for d in range(1, 367):
        distancia = np.abs(doy_base - d)
        distancia = np.minimum(distancia, 366 - distancia)  # ano circular
        amostra = valores_base[distancia <= janela]
        amostra = amostra[~np.isnan(amostra)]
        if amostra.size:
            media[d - 1] = amostra.mean()
            limiar[d - 1] = np.percentile(amostra, percentil)

    media = _media_movel_circular(media, suavizacao)
    limiar = _media_movel_circular(limiar, suavizacao)

    return pd.DataFrame(
        {"temp": temp.to_numpy(), "clim": media[doy - 1], "limiar": limiar[doy - 1]},
        index=temp.index,
    )


def _sequencias(mascara: np.ndarray) -> list[tuple[int, int]]:
    """Início e fim (inclusive) de cada sequência de True."""
    m = np.concatenate([[False], mascara, [False]]).astype(int)
    bordas = np.diff(m)
    inicios = np.where(bordas == 1)[0]
    fins = np.where(bordas == -1)[0] - 1
    return list(zip(inicios, fins, strict=True))


def detectar_eventos(
    clim: pd.DataFrame,
    duracao_minima: int = 5,
    intervalo_maximo: int = 2,
) -> pd.DataFrame:
    """Eventos de onda de calor marinha a partir da saída de `climatologia`.

    Dias sem dado (NaN) não contam como acima do limiar.
    Métricas por evento: início, fim, data do pico, duração (dias),
    intensidade média, máxima e cumulativa (°C e °C·dia, em relação à média
    climatológica). Dias de intervalo unidos ao evento entram nas métricas.
    """
    acima = (clim["temp"] > clim["limiar"]).to_numpy()  # NaN -> False

    candidatos = [(i, f) for i, f in _sequencias(acima) if f - i + 1 >= duracao_minima]

    unidos: list[list[int]] = []
    for i, f in candidatos:
        if unidos and i - unidos[-1][1] - 1 <= intervalo_maximo:
            unidos[-1][1] = f
        else:
            unidos.append([i, f])

    anomalia = (clim["temp"] - clim["clim"]).to_numpy()
    datas = clim.index
    linhas = []
    for i, f in unidos:
        trecho = anomalia[i : f + 1]
        pico = i + int(np.nanargmax(trecho))
        linhas.append(
            {
                "inicio": datas[i],
                "fim": datas[f],
                "data_pico": datas[pico],
                "duracao": f - i + 1,
                "intensidade_media": float(np.nanmean(trecho)),
                "intensidade_maxima": float(np.nanmax(trecho)),
                "intensidade_cumulativa": float(np.nansum(trecho)),
            }
        )
    colunas = [
        "inicio",
        "fim",
        "data_pico",
        "duracao",
        "intensidade_media",
        "intensidade_maxima",
        "intensidade_cumulativa",
    ]
    eventos = pd.DataFrame(linhas, columns=colunas)
    # tipos fixos, inclusive quando não há evento (tabela vazia)
    for c in ("inicio", "fim", "data_pico"):
        eventos[c] = pd.to_datetime(eventos[c])
    return eventos.astype({"duracao": int})


def metricas_anuais(eventos: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    """Métricas por ano civil.

    - n_eventos, duracao_media e intensidade_maxima: eventos atribuídos ao ano
      de início;
    - dias_onda_calor: dias do ano dentro de algum evento (um evento que cruza
      o ano conta dias nos dois anos).
    Anos sem evento aparecem com zero (intensidade_maxima e duracao_media NaN).
    """
    anos = np.arange(clim.index.year.min(), clim.index.year.max() + 1)
    em_evento = pd.Series(False, index=clim.index)
    for _, ev in eventos.iterrows():
        em_evento.loc[ev["inicio"] : ev["fim"]] = True

    saida = pd.DataFrame(index=pd.Index(anos, name="ano"))
    saida["dias_onda_calor"] = (
        em_evento.groupby(em_evento.index.year).sum().reindex(anos, fill_value=0)
    )

    if len(eventos):
        por_ano = eventos.assign(ano=eventos["inicio"].dt.year).groupby("ano")
        saida["n_eventos"] = por_ano.size().reindex(anos, fill_value=0)
        saida["duracao_media"] = por_ano["duracao"].mean().reindex(anos)
        saida["intensidade_maxima"] = por_ano["intensidade_maxima"].max().reindex(anos)
    else:
        saida["n_eventos"] = 0
        saida["duracao_media"] = np.nan
        saida["intensidade_maxima"] = np.nan

    return saida[["n_eventos", "dias_onda_calor", "duracao_media", "intensidade_maxima"]]


def remover_tendencia(temp: pd.Series) -> tuple[pd.Series, float]:
    """Remove a tendência linear de uma série diária, mantendo a média e o ciclo sazonal.

    A inclinação é estimada por mínimos quadrados sobre a anomalia em relação à média de
    cada dia do ano (sem o ciclo sazonal, que enviesaria a reta). A série devolvida é a
    original menos essa reta, centrada no meio do período, de modo que o nível médio e o
    ciclo sazonal permanecem. Devolve a série sem tendência e a tendência removida, em
    unidades por década.
    """
    temp = temp.astype(float)
    dias = (temp.index - temp.index[0]).days.to_numpy(dtype=float)
    doy = dia_do_ano_366(temp.index)
    anomalia = temp - temp.groupby(doy).transform("mean")
    validos = ~np.isnan(anomalia.to_numpy())
    inclinacao = np.polyfit(dias[validos], anomalia.to_numpy()[validos], 1)[0]  # por dia
    centro = dias[validos].mean()
    sem_tendencia = temp - inclinacao * (dias - centro)
    return sem_tendencia, float(inclinacao * 3652.5)


def detectar_por_trecho(
    series: pd.DataFrame,
    periodo_base: tuple[int, int] = (1982, 2011),
    sem_tendencia: bool = False,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Aplica climatologia, detecção e métricas anuais a cada coluna de ``series``.

    series: TSM diária, uma coluna por trecho, índice de datas.
    sem_tendencia: se verdadeiro, remove a tendência linear de cada série antes (variante de
    sensibilidade que separa os extremos do aquecimento de longo prazo).
    Devolve {trecho: {"clim": ..., "eventos": ..., "anual": ...}} e, com ``sem_tendencia``,
    também "tendencia_por_decada" (°C por década).
    """
    resultados = {}
    for nome in series.columns:
        serie = series[nome]
        extra = {}
        if sem_tendencia:
            serie, tendencia = remover_tendencia(serie)
            extra["tendencia_por_decada"] = tendencia
        clim = climatologia(serie, periodo_base=periodo_base)
        eventos = detectar_eventos(clim)
        resultados[nome] = {
            "clim": clim,
            "eventos": eventos,
            "anual": metricas_anuais(eventos, clim),
            **extra,
        }
    return resultados


def empilhar(
    resultados: dict[str, dict[str, pd.DataFrame]], chave: str, casas: int = 3
) -> pd.DataFrame:
    """Junta a tabela ``chave`` ("clim", "eventos" ou "anual") de todos os trechos numa só,
    com a coluna ``trecho`` na frente e os números arredondados em ``casas`` decimais."""
    partes = []
    for nome, r in resultados.items():
        tabela = r[chave].copy()
        if chave != "eventos":
            tabela = tabela.rename_axis(tabela.index.name or "data").reset_index()
        partes.append(tabela.assign(trecho=nome))
    junta = pd.concat(partes, ignore_index=True)
    numericas = junta.select_dtypes("number").columns
    junta[numericas] = junta[numericas].round(casas)
    return junta[["trecho"] + [c for c in junta.columns if c != "trecho"]]


def resumo(eventos: pd.DataFrame, anual: pd.DataFrame, corte: int | None = None) -> pd.Series:
    """Resumo de um trecho: número de eventos, dias, duração, intensidades e o maior evento.

    ``corte``: ano que divide a série em duas metades para comparar a média de dias em
    onda de calor por ano (padrão: o ano do meio).
    """
    anos = anual.index
    corte = corte or int(anos[len(anos) // 2])
    pico = eventos.loc[eventos["intensidade_maxima"].idxmax()] if len(eventos) else None
    maior = eventos.loc[eventos["intensidade_cumulativa"].idxmax()] if len(eventos) else None
    return pd.Series(
        {
            "eventos": len(eventos),
            "eventos_por_ano": round(len(eventos) / len(anos), 2),
            "dias_total": int(anual["dias_onda_calor"].sum()),
            "duracao_media_dias": round(float(eventos["duracao"].mean()), 1),
            "duracao_maxima_dias": int(eventos["duracao"].max()) if len(eventos) else 0,
            "intensidade_maxima_C": round(float(pico["intensidade_maxima"]), 2),
            "data_intensidade_maxima": pico["data_pico"].date() if pico is not None else None,
            "maior_evento_inicio": maior["inicio"].date() if maior is not None else None,
            "maior_evento_duracao_dias": int(maior["duracao"]) if maior is not None else 0,
            "maior_evento_intensidade_cumulativa_C_dia": (
                round(float(maior["intensidade_cumulativa"]), 1) if maior is not None else 0.0
            ),
            f"dias_por_ano_{anos[0]}_{corte - 1}": round(
                float(anual.loc[: corte - 1, "dias_onda_calor"].mean()), 1
            ),
            f"dias_por_ano_{corte}_{anos[-1]}": round(
                float(anual.loc[corte:, "dias_onda_calor"].mean()), 1
            ),
        }
    )


def inclinacao_sen(y: np.ndarray, x: np.ndarray | None = None) -> float:
    """Inclinação de Sen: mediana das inclinações entre todos os pares de pontos
    (estimativa robusta da taxa de variação, pouco sensível a valores extremos)."""
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    i, j = np.triu_indices(y.size, k=1)
    return float(np.median((y[j] - y[i]) / (x[j] - x[i])))


def tendencia(serie: pd.Series) -> dict:
    """Teste de Mann-Kendall com correção para autocorrelação (HAMED; RAO, 1998) e
    inclinação de Sen, para uma série anual (índice = ano).

    A autocorrelação positiva infla a significância do teste original; a correção aumenta a
    variância da estatística S usando as autocorrelações significativas dos postos da série
    sem tendência; o fator nunca é menor que 1, para que a correção só torne o teste mais
    conservador. Devolve n, tau de Kendall, inclinação de Sen por década, valor-p com e sem
    correção e o fator de correção da variância.
    """
    from math import erfc, sqrt

    serie = serie.dropna()
    y = serie.to_numpy(dtype=float)
    x = serie.index.to_numpy(dtype=float)
    n = y.size
    i, j = np.triu_indices(n, k=1)
    s = float(np.sum(np.sign(y[j] - y[i])))
    _, contagens = np.unique(y, return_counts=True)  # empates
    var_s = (
        n * (n - 1) * (2 * n + 5) - np.sum(contagens * (contagens - 1) * (2 * contagens + 5))
    ) / 18

    # correção de Hamed e Rao: autocorrelação dos postos da série sem a tendência de Sen
    inclinacao = inclinacao_sen(y, x)
    residuo = y - inclinacao * x
    postos = pd.Series(residuo).rank().to_numpy()
    postos = postos - postos.mean()
    limite = 1.96 / sqrt(n)  # autocorrelação significativa a 5%
    soma = 0.0
    for k in range(1, n - 2):
        rho = float(np.sum(postos[:-k] * postos[k:]) / np.sum(postos**2))
        if abs(rho) > limite:
            soma += (n - k) * (n - k - 1) * (n - k - 2) * rho
    # a correção só pode tornar o teste mais conservador (fator mínimo 1)
    fator = max(1 + 2 / (n * (n - 1) * (n - 2)) * soma, 1.0)

    def p_valor(variancia: float) -> float:
        z = (s - np.sign(s)) / sqrt(variancia) if s != 0 else 0.0
        return float(erfc(abs(z) / sqrt(2)))

    return {
        "n": n,
        "tau": round(s / (n * (n - 1) / 2), 3),
        "inclinacao_sen_por_decada": round(inclinacao * 10, 3),
        "p_valor": round(p_valor(var_s * fator), 5),
        "p_valor_sem_correcao": round(p_valor(var_s), 5),
        "fator_correcao": round(fator, 2),
    }
