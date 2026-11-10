"""Testes da detecção com séries sintéticas de eventos conhecidos."""

import numpy as np
import pandas as pd

from mhwceara.deteccao import (
    climatologia,
    detectar_eventos,
    detectar_por_trecho,
    dia_do_ano_366,
    empilhar,
    inclinacao_sen,
    metricas_anuais,
    remover_tendencia,
    resumo,
    tendencia,
)

DATAS = pd.date_range("1982-01-01", "2015-12-31", freq="D")
RUIDO = 0.1  # ruído uniforme em [-0,1; 0,1] °C
SALTO = 1.0  # anomalia dos eventos inseridos (°C)


def serie_sintetica(eventos, sazonal=False, semente=0):
    """TSM sintética com eventos inseridos.

    eventos: lista de (data_inicial, n_dias). Os eventos ficam em 2013-2015,
    fora do período-base, para não alterar a climatologia. O ruído é zerado
    em ±7 dias de cada evento, para que as bordas do evento sejam exatas.
    """
    rng = np.random.default_rng(semente)
    base = np.full(len(DATAS), 26.0)
    if sazonal:
        base = base + 1.5 * np.sin(2 * np.pi * DATAS.day_of_year.to_numpy() / 365.25)
    ruido = rng.uniform(-RUIDO, RUIDO, len(DATAS))
    temp = pd.Series(base + ruido, index=DATAS)
    for inicio, n in eventos:
        ini = pd.Timestamp(inicio)
        margem = slice(ini - pd.Timedelta(days=7), ini + pd.Timedelta(days=n + 6))
        temp.loc[margem] = pd.Series(base, index=DATAS).loc[margem]
        temp.loc[ini : ini + pd.Timedelta(days=n - 1)] += SALTO
    return temp


def detectar(temp):
    clim = climatologia(temp, periodo_base=(1982, 2011))
    return clim, detectar_eventos(clim)


def eventos_em_2013_2015(ev):
    return ev[ev["inicio"].dt.year >= 2013].reset_index(drop=True)


def test_dia_do_ano_366():
    d = pd.DatetimeIndex(
        ["2001-02-28", "2001-03-01", "2004-02-29", "2004-03-01", "2001-12-31", "2004-12-31"]
    )
    assert list(dia_do_ano_366(d)) == [59, 61, 60, 61, 366, 366]


def test_climatologia_recupera_ciclo_sazonal():
    clim, _ = detectar(serie_sintetica([], sazonal=True))
    base = 26.0 + 1.5 * np.sin(2 * np.pi * clim.index.day_of_year / 365.25)
    assert np.nanmax(np.abs(clim["clim"] - base)) < 0.05


def test_limiar_e_o_percentil_90():
    # sem ciclo sazonal, o limiar é o percentil 90 do ruído uniforme em
    # [-0,1; 0,1] °C, isto é, 0,08 °C acima da média
    clim, _ = detectar(serie_sintetica([]))
    assert abs(np.nanmean(clim["limiar"] - clim["clim"]) - 0.08) < 0.005


def test_evento_de_10_dias():
    _, ev = detectar(serie_sintetica([("2013-06-10", 10)]))
    ev = eventos_em_2013_2015(ev)
    assert len(ev) == 1
    assert ev.loc[0, "inicio"] == pd.Timestamp("2013-06-10")
    assert ev.loc[0, "duracao"] == 10
    assert abs(ev.loc[0, "intensidade_media"] - SALTO) < 0.02
    assert abs(ev.loc[0, "intensidade_cumulativa"] - 10 * SALTO) < 0.2


def test_quatro_dias_nao_e_evento():
    _, ev = detectar(serie_sintetica([("2013-06-10", 4)]))
    assert len(eventos_em_2013_2015(ev)) == 0


def test_intervalo_de_2_dias_une_eventos():
    # 6 dias quentes, 2 dias normais, 6 dias quentes -> um evento de 14 dias
    temp = serie_sintetica([("2014-03-01", 14)])
    temp.loc["2014-03-07":"2014-03-08"] -= SALTO
    _, ev = detectar(temp)
    ev = eventos_em_2013_2015(ev)
    assert len(ev) == 1
    assert ev.loc[0, "duracao"] == 14


def test_intervalo_de_3_dias_separa_eventos():
    temp = serie_sintetica([("2014-03-01", 15)])
    temp.loc["2014-03-07":"2014-03-09"] -= SALTO
    _, ev = detectar(temp)
    ev = eventos_em_2013_2015(ev)
    assert len(ev) == 2
    assert list(ev["duracao"]) == [6, 6]


def test_evento_com_ciclo_sazonal():
    _, ev = detectar(serie_sintetica([("2015-04-01", 7)], sazonal=True))
    ev = eventos_em_2013_2015(ev)
    assert len(ev) == 1
    assert ev.loc[0, "duracao"] == 7


def test_dias_sem_dado_nao_criam_evento():
    temp = serie_sintetica([])
    temp.loc["2013-01-01":"2013-01-20"] = np.nan
    _, ev = detectar(temp)
    assert len(eventos_em_2013_2015(ev)) == 0


def test_metricas_anuais():
    clim, ev = detectar(serie_sintetica([("2013-06-10", 10)]))
    anual = metricas_anuais(ev, clim)
    assert anual.loc[2013, "n_eventos"] == 1
    assert anual.loc[2013, "dias_onda_calor"] == 10
    assert anual.loc[2014, "n_eventos"] == 0
    assert anual.loc[2014, "dias_onda_calor"] == 0


def test_detectar_por_trecho_e_empilhar():
    temp = serie_sintetica([("2013-06-10", 10)])
    series = pd.DataFrame({"oeste": temp, "leste": temp})
    resultados = detectar_por_trecho(series, periodo_base=(1982, 2011))
    assert list(resultados) == ["oeste", "leste"]
    eventos = empilhar(resultados, "eventos")
    assert list(eventos.columns[:2]) == ["trecho", "inicio"]
    assert (eventos["trecho"] == "oeste").sum() == (eventos["trecho"] == "leste").sum()
    clim = empilhar(resultados, "clim")
    assert list(clim.columns[:2]) == ["trecho", "data"]


def test_resumo():
    clim, ev = detectar(serie_sintetica([("2013-06-10", 10)]))
    r = resumo(eventos_em_2013_2015(ev), metricas_anuais(ev, clim), corte=2000)
    assert r["eventos"] == 1
    assert r["maior_evento_duracao_dias"] == 10
    assert str(r["maior_evento_inicio"]) == "2013-06-10"


def test_remover_tendencia():
    datas = pd.date_range("1982-01-01", "2025-12-31", freq="D")
    anos = (datas - datas[0]).days / 365.25
    sazonal = 1.5 * np.sin(2 * np.pi * datas.day_of_year / 365.25)
    temp = pd.Series(26.0 + sazonal + 0.02 * anos, index=datas)  # 0,2 °C por década
    sem, tendencia = remover_tendencia(temp)
    assert abs(tendencia - 0.2) < 0.01
    assert abs(sem.mean() - temp.mean()) < 1e-9  # a média é preservada
    _, resto = remover_tendencia(sem)
    assert abs(resto) < 1e-3  # a série resultante não tem tendência (só resíduo numérico)


def test_detectar_por_trecho_sem_tendencia():
    temp = serie_sintetica([("2013-06-10", 10)])
    resultados = detectar_por_trecho(pd.DataFrame({"oeste": temp}), sem_tendencia=True)
    assert "tendencia_por_decada" in resultados["oeste"]
    assert len(eventos_em_2013_2015(resultados["oeste"]["eventos"])) == 1


def test_inclinacao_sen_resiste_a_valor_extremo():
    y = 2.0 * np.arange(20)
    y[5] = 500.0
    assert inclinacao_sen(y) == 2.0


def test_tendencia_detecta_aumento_e_nao_acusa_ruido():
    gerador = np.random.default_rng(0)
    anos = np.arange(1982, 2026)
    com = tendencia(pd.Series(3.0 * (anos - 1982) + gerador.normal(0, 5, anos.size), index=anos))
    assert com["p_valor"] < 0.001 and abs(com["inclinacao_sen_por_decada"] - 30) < 5
    sem = tendencia(pd.Series(gerador.normal(0, 5, anos.size), index=anos))
    assert sem["p_valor"] > 0.05


def test_tendencia_correcao_so_torna_mais_conservador():
    gerador = np.random.default_rng(1)
    anos = np.arange(1982, 2026)
    ar1 = np.zeros(anos.size)
    for k in range(1, anos.size):
        ar1[k] = 0.8 * ar1[k - 1] + gerador.normal()
    r = tendencia(pd.Series(ar1, index=anos))
    assert r["fator_correcao"] >= 1 and r["p_valor"] >= r["p_valor_sem_correcao"]
