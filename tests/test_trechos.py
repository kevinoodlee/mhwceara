"""Testes da faixa costeira e da divisão em trechos, com um "estado" quadrado."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from mhwceara.trechos import classificar_celulas, limites_trechos, serie_por_trecho

# estado fictício: quadrado de 3°O a 0° de longitude e 3°S a 1°S de latitude; oceano ao norte
ESTADO = box(-3.0, -3.0, 0.0, -1.0)
LAT = np.array([-0.5, 0.5, 1.5])  # ao norte do estado: 0,5°, 1,5° e 2,5° de distância
LON = np.array([-4.5, -2.5, -1.5, -0.5, 1.5])
OCEANO = np.ones((LAT.size, LON.size), dtype=bool)


def test_limites_de_mesma_largura():
    np.testing.assert_allclose(limites_trechos(-3.0, 0.0, 3), [-3.0, -2.0, -1.0, 0.0])


def test_faixa_costeira_por_distancia_e_extensao():
    celulas, _ = classificar_celulas(LAT, LON, OCEANO, ESTADO, faixa=1.0, limites_internos=None)
    na_faixa = celulas[celulas["na_faixa"]]
    assert set(na_faixa["lat"]) == {-0.5}  # só a linha a 0,5° da costa
    assert set(na_faixa["lon"]) == {-2.5, -1.5, -0.5}  # fora da extensão do estado: -4,5 e 1,5


def test_um_trecho_por_terco():
    celulas, _ = classificar_celulas(LAT, LON, OCEANO, ESTADO, faixa=1.0, limites_internos=None)
    na_faixa = celulas[celulas["na_faixa"]].set_index("lon")["trecho"]
    assert na_faixa.to_dict() == {-2.5: "oeste", -1.5: "central", -0.5: "leste"}
    assert celulas.loc[~celulas["na_faixa"], "trecho"].isna().all()


def test_serie_por_trecho_e_a_media_das_celulas():
    celulas = pd.DataFrame(
        {
            "lat": [-0.5, -0.5, 0.5],
            "lon": [-2.5, -1.5, -1.5],
            "na_faixa": [True, True, True],
            "trecho": ["oeste", "central", "central"],
        }
    )
    tempo = pd.date_range("2020-01-01", periods=2)
    valores = np.zeros((2, LAT.size, LON.size))
    valores[:, 0, 1] = 10.0  # (-0,5; -2,5), oeste
    valores[:, 0, 2] = 20.0  # (-0,5; -1,5), central
    valores[:, 1, 2] = 30.0  # (0,5; -1,5), central
    da = xr.DataArray(
        valores, dims=("time", "lat", "lon"), coords={"time": tempo, "lat": LAT, "lon": LON}
    )
    serie = serie_por_trecho(da, celulas)
    assert list(serie.columns) == ["oeste", "central"]
    assert (serie["oeste"] == 10.0).all() and (serie["central"] == 25.0).all()


def test_limites_internos_definidos():
    np.testing.assert_allclose(
        limites_trechos(-3.0, 0.0, 3, (-2.5, -0.75)), [-3.0, -2.5, -0.75, 0.0]
    )


def test_limites_internos_fora_de_ordem():
    with pytest.raises(ValueError):
        limites_trechos(-3.0, 0.0, 3, (-0.75, -2.5))


def test_trechos_com_limites_internos():
    celulas, _ = classificar_celulas(
        LAT, LON, OCEANO, ESTADO, faixa=1.0, limites_internos=(-2.0, -0.75)
    )
    na_faixa = celulas[celulas["na_faixa"]].set_index("lon")["trecho"]
    assert na_faixa.to_dict() == {-2.5: "oeste", -1.5: "central", -0.5: "leste"}
