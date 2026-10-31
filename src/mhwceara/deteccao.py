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
    doy = datas.dayofyear.to_numpy().copy()
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
    return list(zip(inicios, fins))


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
    colunas = ["inicio", "fim", "data_pico", "duracao", "intensidade_media",
               "intensidade_maxima", "intensidade_cumulativa"]
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
    saida["dias_onda_calor"] = em_evento.groupby(em_evento.index.year).sum().reindex(anos, fill_value=0)

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