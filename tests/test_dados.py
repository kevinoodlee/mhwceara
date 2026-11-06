"""Testes das funções de dados que não dependem de internet."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mhwceara import dados


def serie_diaria(ano, sst=27.0):
    tempo = pd.date_range(f"{ano}-01-01", f"{ano}-12-31", freq="D")
    valores = np.full((tempo.size, 2, 3), sst, dtype="float32")
    valores[:, 0, 0] = np.nan  # uma célula de terra
    return xr.DataArray(
        valores,
        dims=("time", "lat", "lon"),
        coords={"time": tempo, "lat": [-1.0, -0.75], "lon": [320.0, 320.25, 320.5]},
    )


def test_e_netcdf():
    assert dados.e_netcdf(b"CDF\x01resto")
    assert dados.e_netcdf(b"\x89HDF\r\n\x1a\n")
    assert not dados.e_netcdf(b"<html>")


def test_causa_mostra_a_parte_depois_de_caused_by():
    erro = ConnectionError("Max retries exceeded (Caused by ConnectTimeoutError('lento'))")
    assert dados.causa(erro).startswith("ConnectionError: ConnectTimeoutError")


def test_com_tentativas_repete_ate_dar_certo():
    chamadas = []

    def instavel():
        chamadas.append(1)
        if len(chamadas) < 3:
            raise RuntimeError("falhou")
        return "ok"

    assert dados.com_tentativas(instavel, tentativas=3, espera=0, rotulo="teste") == "ok"
    assert len(chamadas) == 3


def test_com_tentativas_desiste_na_ultima():
    def sempre_falha():
        raise RuntimeError("falhou")

    with pytest.raises(RuntimeError, match="teste"):
        dados.com_tentativas(sempre_falha, tentativas=2, espera=0, rotulo="teste")


def test_meses_clorofila():
    meses = dados.meses_clorofila()
    assert len(meses) == 304
    assert meses[0] == (1997, 9) and meses[-1] == (2022, 12)


def test_conferir_oisst_aceita_ano_completo():
    dados.conferir_oisst(serie_diaria(2024).to_dataset(name="sst"), 2024)  # bissexto: 366 dias


def test_conferir_oisst_recusa_dia_faltando():
    sst = serie_diaria(2023).isel(time=slice(1, None))
    with pytest.raises(RuntimeError, match="364 dias"):
        dados.conferir_oisst(sst.to_dataset(name="sst"), 2023)


def test_conferir_oisst_recusa_valor_de_preenchimento():
    sst = serie_diaria(2023)
    sst[0, 1, 1] = -9.97e36
    with pytest.raises(RuntimeError, match="faixa física"):
        dados.conferir_oisst(sst.to_dataset(name="sst"), 2023)


def test_gravar_netcdf_ida_e_volta(tmp_path):
    sst = serie_diaria(2023)
    arq = tmp_path / "teste.nc"
    dados.gravar_netcdf(sst, arq, "sst", "degC", "teste", "teste")
    assert dados.arquivo_oisst_valido(arq, 2023)
    with xr.open_dataset(arq) as ds:
        np.testing.assert_array_equal(ds["sst"].values, sst.values)
    assert not (tmp_path / "teste.nc.part").exists()
