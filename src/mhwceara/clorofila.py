"""Resposta da clorofila-a às ondas de calor marinhas, por trecho do litoral.

Etapas:
  1. classificar os pixels da grade da clorofila-a (4 km) em trechos, com a mesma faixa
     costeira e os mesmos limites da TSM, excluindo os pixels muito próximos da costa
     (água turva e foz de rios, onde o algoritmo do OC-CCI erra mais);
  2. calcular, para cada trecho e mês, a média de log10 da clorofila-a e a cobertura
     (fração de pixels com dado), descartando meses abaixo da cobertura mínima;
  3. calcular a anomalia em relação à climatologia mensal de cada trecho;
  4. contar os dias em onda de calor em cada mês e comparar a anomalia entre meses com e
     sem evento, com defasagem de 0 a 2 meses, por um teste de permutação.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DEFASAGENS,
    DIAS_MINIMOS_EVENTO,
    DISTANCIA_MINIMA_COSTA,
    FAIXA_COSTEIRA,
    FRACAO_MINIMA,
    LIMITES_INTERNOS,
    TRECHOS,
)


def classificar_pixels(
    chl,
    limite,
    faixa: float = FAIXA_COSTEIRA,
    trechos: tuple[str, ...] = TRECHOS,
    limites_internos: tuple[float, ...] | None = LIMITES_INTERNOS,
) -> pd.DataFrame:
    """Classifica os pixels da clorofila-a (DataArray time, lat, lon) em trechos.

    Considera oceano os pixels com dado em pelo menos um mês. Devolve uma tabela com uma
    linha por pixel oceânico (lat, lon, distância à costa em graus, na_faixa, trecho).
    """
    from .trechos import classificar_celulas  # usa geopandas/shapely só quando necessário

    dominio = ~np.isnan(chl.values).all(axis=0)
    celulas, _ = classificar_celulas(
        chl["lat"].values, chl["lon"].values, dominio, limite, faixa, trechos, limites_internos
    )
    return celulas


def serie_por_trecho(
    chl,
    pixels: pd.DataFrame,
    distancia_minima: float = DISTANCIA_MINIMA_COSTA,
    fracao_minima: float = FRACAO_MINIMA,
) -> pd.DataFrame:
    """Média mensal de log10 da clorofila-a em cada trecho.

    Usa os pixels da faixa costeira a pelo menos ``distancia_minima`` graus da costa.
    Devolve uma tabela longa (data, trecho, log10_chl, pixels_validos, pixels_total,
    cobertura, valido); ``log10_chl`` fica vazio nos meses com cobertura abaixo de
    ``fracao_minima``.
    """
    usar = pixels[pixels["na_faixa"] & (pixels["distancia_costa_graus"] >= distancia_minima)]
    i_lat = {v: k for k, v in enumerate(chl["lat"].values)}
    i_lon = {v: k for k, v in enumerate(chl["lon"].values)}
    valores = chl.values
    datas = pd.DatetimeIndex(chl["time"].values).to_period("M").to_timestamp()
    partes = []
    for trecho in [t for t in TRECHOS if t in set(usar["trecho"])]:
        grupo = usar[usar["trecho"] == trecho]
        ii = grupo["lat"].map(i_lat).to_numpy()
        jj = grupo["lon"].map(i_lon).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            log10 = np.log10(valores[:, ii, jj])
        n_validos = np.sum(~np.isnan(log10), axis=1)
        soma = np.nansum(log10, axis=1)
        media = np.where(n_validos > 0, soma / np.maximum(n_validos, 1), np.nan)
        cobertura = n_validos / len(grupo)
        partes.append(
            pd.DataFrame(
                {
                    "data": datas,
                    "trecho": trecho,
                    "log10_chl": np.where(cobertura >= fracao_minima, media, np.nan),
                    "pixels_validos": n_validos,
                    "pixels_total": len(grupo),
                    "cobertura": np.round(cobertura, 4),
                    "valido": cobertura >= fracao_minima,
                }
            )
        )
    return pd.concat(partes, ignore_index=True)


def anomalia(serie: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta a anomalia de log10 da clorofila-a em relação à média de cada mês do
    calendário (climatologia mensal) de cada trecho, calculada com os meses válidos."""
    serie = serie.copy()
    mes = serie["data"].dt.month
    climatologia = serie.groupby(["trecho", mes])["log10_chl"].transform("mean")
    serie["anomalia_log10"] = serie["log10_chl"] - climatologia
    return serie


def dias_em_evento_por_mes(clim: pd.DataFrame, eventos: pd.DataFrame) -> pd.Series:
    """Número de dias em onda de calor em cada mês (índice = primeiro dia do mês).

    clim: saída de ``deteccao.climatologia`` (índice diário); eventos: saída de
    ``deteccao.detectar_eventos``.
    """
    em_evento = pd.Series(0, index=clim.index)
    for ev in eventos.itertuples():
        em_evento.loc[ev.inicio : ev.fim] = 1
    return em_evento.resample("MS").sum()


def teste_permutacao(
    com: np.ndarray, sem: np.ndarray, n_permutacoes: int = 10000, semente: int = 0
) -> float:
    """Valor-p bilateral da diferença de medianas entre dois grupos, por permutação.

    Embaralha os rótulos (com ou sem evento) ``n_permutacoes`` vezes e conta em quantas a
    diferença de medianas é pelo menos tão grande quanto a observada.
    """
    rng = np.random.default_rng(semente)
    juntos = np.concatenate([com, sem])
    n = len(com)
    observada = abs(np.median(com) - np.median(sem))
    ordem = np.argsort(rng.random((n_permutacoes, len(juntos))), axis=1)
    embaralhado = juntos[ordem]
    diferencas = np.abs(
        np.median(embaralhado[:, :n], axis=1) - np.median(embaralhado[:, n:], axis=1)
    )
    return float((np.sum(diferencas >= observada - 1e-12) + 1) / (n_permutacoes + 1))


def comparar(
    anomalia_mensal: pd.Series,
    dias_evento: pd.Series,
    dias_minimos: int = DIAS_MINIMOS_EVENTO,
    defasagens: tuple[int, ...] = DEFASAGENS,
    n_permutacoes: int = 10000,
) -> pd.DataFrame:
    """Compara a anomalia de clorofila-a entre meses com e sem onda de calor.

    Para a defasagem k, a anomalia do mês t+k é associada à condição (com ou sem evento)
    do mês t. Devolve uma linha por defasagem com o tamanho dos grupos, medianas e
    quartis, a diferença de medianas (em log10 e em porcentagem da concentração), o
    valor-p do teste de permutação e a correlação entre a fração de dias em evento e a
    anomalia (Pearson).
    """
    dias_evento = dias_evento.reindex(anomalia_mensal.index)
    linhas = []
    for k in defasagens:
        resposta = anomalia_mensal.shift(-k)  # anomalia k meses depois
        par = pd.DataFrame({"dias": dias_evento, "anomalia": resposta}).dropna()
        com = par.loc[par["dias"] >= dias_minimos, "anomalia"].to_numpy()
        sem = par.loc[par["dias"] < dias_minimos, "anomalia"].to_numpy()
        if len(com) < 3 or len(sem) < 3:
            continue
        diferenca = float(np.median(com) - np.median(sem))
        fracao = par["dias"] / par.index.days_in_month
        linhas.append(
            {
                "defasagem_meses": k,
                "meses_com_evento": len(com),
                "meses_sem_evento": len(sem),
                "mediana_com": round(float(np.median(com)), 4),
                "q25_com": round(float(np.percentile(com, 25)), 4),
                "q75_com": round(float(np.percentile(com, 75)), 4),
                "mediana_sem": round(float(np.median(sem)), 4),
                "q25_sem": round(float(np.percentile(sem, 25)), 4),
                "q75_sem": round(float(np.percentile(sem, 75)), 4),
                "diferenca_log10": round(diferenca, 4),
                "variacao_percentual": round((10**diferenca - 1) * 100, 1),
                "p_valor": round(teste_permutacao(com, sem, n_permutacoes), 4),
                "r_fracao_dias": round(float(np.corrcoef(fracao, par["anomalia"])[0, 1]), 3),
            }
        )
    return pd.DataFrame(linhas)
