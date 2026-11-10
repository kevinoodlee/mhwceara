"""Testes da resposta da clorofila-a, com dados sintéticos."""

import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from mhwceara import clorofila

# "estado" quadrado: 3°O a 0° e 3°S a 1°S; oceano ao norte
ESTADO = box(-3.0, -3.0, 0.0, -1.0)
LAT = np.array([-0.95, -0.5])  # a 0,05° (perto da costa) e a 0,5° da costa
LON = np.array([-2.5, -1.5, -0.5])
MESES = pd.date_range("2000-01-01", "2009-12-01", freq="MS")


def grade_chl(valor=0.2):
    valores = np.full((MESES.size, LAT.size, LON.size), valor)
    return xr.DataArray(
        valores, dims=("time", "lat", "lon"), coords={"time": MESES, "lat": LAT, "lon": LON}
    )


def test_classificar_e_excluir_pixels_da_costa():
    chl = grade_chl()
    pixels = clorofila.classificar_pixels(chl, ESTADO, faixa=1.0, limites_internos=None)
    serie = clorofila.serie_por_trecho(chl, pixels, distancia_minima=0.09)
    # cada trecho tem 2 pixels na faixa, mas o de -0,95° está a menos de 0,09° da costa
    assert set(serie["pixels_total"]) == {1}
    serie_todos = clorofila.serie_por_trecho(chl, pixels, distancia_minima=0.0)
    assert set(serie_todos["pixels_total"]) == {2}


def test_cobertura_minima():
    chl = grade_chl()
    chl[0, 1, :] = np.nan  # primeiro mês sem dado nos pixels longe da costa
    pixels = clorofila.classificar_pixels(chl, ESTADO, faixa=1.0, limites_internos=None)
    serie = clorofila.serie_por_trecho(chl, pixels, distancia_minima=0.09, fracao_minima=0.5)
    primeiro = serie[serie["data"] == MESES[0]]
    assert (primeiro["cobertura"] == 0).all() and primeiro["log10_chl"].isna().all()


def test_anomalia_remove_a_climatologia_mensal():
    serie = pd.DataFrame(
        {
            "data": MESES,
            "trecho": "oeste",
            "log10_chl": np.log10(0.2) + 0.1 * np.sin(2 * np.pi * MESES.month / 12),
        }
    )
    anom = clorofila.anomalia(serie)["anomalia_log10"]
    assert np.allclose(anom, 0, atol=1e-12)


def test_dias_em_evento_por_mes():
    clim = pd.DataFrame(index=pd.date_range("2000-01-01", "2000-03-31", freq="D"))
    eventos = pd.DataFrame(
        {"inicio": [pd.Timestamp("2000-01-20")], "fim": [pd.Timestamp("2000-02-10")]}
    )
    dias = clorofila.dias_em_evento_por_mes(clim, eventos)
    assert list(dias) == [12, 10, 0]


def test_comparar_detecta_efeito_conhecido():
    rng = np.random.default_rng(1)
    indice = pd.date_range("1998-01-01", "2022-12-01", freq="MS")
    dias = pd.Series(np.where(rng.random(indice.size) < 0.3, 20, 0), index=indice)
    anom = pd.Series(rng.normal(0, 0.05, indice.size), index=indice)
    anom[dias >= 15] -= 0.1  # menos clorofila no mesmo mês do evento
    tabela = clorofila.comparar(anom, dias, n_permutacoes=2000).set_index("defasagem_meses")
    assert tabela.loc[0, "diferenca_log10"] < -0.05
    assert tabela.loc[0, "p_valor"] < 0.01
    assert tabela.loc[0, "variacao_percentual"] < 0
    assert tabela.loc[1, "p_valor"] > 0.01  # sem efeito defasado


def test_teste_permutacao_sem_efeito():
    rng = np.random.default_rng(2)
    p = clorofila.teste_permutacao(rng.normal(0, 1, 60), rng.normal(0, 1, 200), 2000)
    assert p > 0.05


def test_anomalia_relativa_e_escala_linear():
    serie = pd.DataFrame(
        {"data": MESES, "trecho": "oeste", "log10_chl": np.log10(0.2), "chl_media": 0.2}
    )
    serie.loc[serie["data"].dt.year == 2005, "chl_media"] = 0.18  # 10% abaixo
    anom = clorofila.anomalia(serie).set_index("data")["anomalia_relativa"]
    assert abs(anom.loc["2005-01-01"] - (0.18 / ((9 * 0.2 + 0.18) / 10) - 1) * 100) < 1e-9
    dias = pd.Series(np.where(anom.index.year == 2005, 20, 0), index=anom.index)
    tabela = clorofila.comparar(anom, dias, n_permutacoes=500, escala="relativa")
    linha = tabela.set_index("defasagem_meses").loc[0]
    assert np.isnan(linha["diferenca_log10"]) and linha["variacao_percentual"] < -9
