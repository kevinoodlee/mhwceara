"""
conferir_dados.py
Confere e consolida os dados brutos baixados por baixar_oisst.py e baixar_clorofila.py.

OISST (TSM diária):
  - confere dias por ano, continuidade das datas, grade, máscara de terra e faixa de valores;
  - compara tamanho dos arquivos e precisão dos valores por ano (arquivos menores de 2022 a 2025);
  - junta os 44 anos num único arquivo.
OC-CCI (clorofila-a mensal):
  - confere meses, continuidade e grade;
  - calcula a fração de pixels válidos em cada mês (cobertura espacial) e marca os meses
    abaixo de 30%, 50% e 70%;
  - junta os 304 meses num único arquivo.

Saídas em data/processed/.
Uso:  python scripts\\conferir_dados.py
"""
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
BRUTO_SST = RAIZ / "data" / "raw" / "sst"
BRUTO_CHL = RAIZ / "data" / "raw" / "clorofila"
PROCESSADO = RAIZ / "data" / "processed"

ANOS = range(1982, 2026)                 # 1982 a 2025
MESES_CHL = 304                          # set. 1997 a dez. 2022
LIMIARES = (0.30, 0.50, 0.70)            # fração mínima de pixels válidos (Metodologia, seção 7)


def mesma_grade(a, b):
    return np.array_equal(a["lat"].values, b["lat"].values) and np.array_equal(a["lon"].values, b["lon"].values)


def salvar(da, arq, nome, unidade, descricao, fonte):
    """Grava um DataArray limpo e comprimido."""
    ds = xr.Dataset(
        {nome: (da.dims, da.values.astype("float32"), {"units": unidade, "long_name": descricao})},
        coords={"time": da["time"].values,
                "lat": ("lat", da["lat"].values, {"units": "degrees_north"}),
                "lon": ("lon", da["lon"].values, {"units": "degrees_east"})},
        attrs={"fonte": fonte, "gerado_por": "mhwceara/scripts/conferir_dados.py"},
    )
    ds.to_netcdf(arq, engine="netcdf4", encoding={nome: {"zlib": True, "complevel": 4}})
    return arq.stat().st_size


# ------------------------------------------------------------------ OISST
def conferir_oisst():
    print("\n== OISST (TSM diária)")
    partes, linhas, problemas = [], [], []
    referencia = None
    for ano in tqdm(ANOS, desc="lendo OISST", unit="ano"):
        arq = BRUTO_SST / f"oisst_ceara_{ano}.nc"
        if not arq.exists():
            problemas.append(f"{ano}: arquivo ausente")
            continue
        with xr.open_dataset(arq, engine="netcdf4") as ds:
            sst = ds["sst"].transpose("time", "lat", "lon").load()
        if referencia is None:
            referencia = sst
        elif not mesma_grade(referencia, sst):
            problemas.append(f"{ano}: grade diferente da de {ANOS[0]}")
        esperado = 366 if calendar.isleap(ano) else 365
        if sst.sizes["time"] != esperado:
            problemas.append(f"{ano}: {sst.sizes['time']} dias (esperado {esperado})")
        v = sst.values
        validas = (~np.isnan(v)).sum(axis=(1, 2))      # células com dado em cada dia
        linhas.append({
            "ano": ano, "dias": sst.sizes["time"],
            "celulas_oceano_min": int(validas.min()), "celulas_oceano_max": int(validas.max()),
            "tsm_min": round(float(np.nanmin(v)), 2), "tsm_media": round(float(np.nanmean(v)), 2),
            "tsm_max": round(float(np.nanmax(v)), 2),
            "valores_distintos": int(np.unique(v[~np.isnan(v)]).size),
            "tamanho_kb": round(arq.stat().st_size / 1e3),
        })
        partes.append(sst)

    resumo = pd.DataFrame(linhas).set_index("ano")
    serie = xr.concat(partes, dim="time")
    datas = pd.DatetimeIndex(serie["time"].values)
    faltando = pd.date_range(datas.min(), datas.max(), freq="D").difference(datas)
    duplicadas = datas[datas.duplicated()]
    celulas = sorted(set(resumo["celulas_oceano_min"]) | set(resumo["celulas_oceano_max"]))

    print(f"  período: {datas.min():%Y-%m-%d} a {datas.max():%Y-%m-%d} | {len(datas)} dias")
    print(f"  grade: {serie.sizes['lat']} x {serie.sizes['lon']} células "
          f"({float(serie.lat.min()):.3f} a {float(serie.lat.max()):.3f}; "
          f"{float(serie.lon.min()):.3f} a {float(serie.lon.max()):.3f})")
    print(f"  dias faltando: {len(faltando)} | datas repetidas: {len(duplicadas)}")
    print(f"  células oceânicas por dia: {celulas} "
          f"({'máscara de terra constante' if len(celulas) == 1 else 'máscara VARIA entre dias'})")
    print(f"  TSM: mínimo {resumo.tsm_min.min():.2f} °C | média {serie.mean().item():.2f} °C | "
          f"máximo {resumo.tsm_max.max():.2f} °C")
    print("  tamanho e precisão dos arquivos (arquivos menores de 2022 a 2025):")
    print(resumo.loc[2018:2025, ["tamanho_kb", "valores_distintos", "tsm_media"]].to_string()
          .replace("\n", "\n    ").join(["    ", ""]))
    for p in problemas:
        print(f"  PROBLEMA: {p}")
    if not problemas and len(faltando) == 0 and len(duplicadas) == 0:
        print("  conferência: OK")

    PROCESSADO.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(PROCESSADO / "resumo_oisst_por_ano.csv")
    arq = PROCESSADO / "oisst_ceara_1982_2025.nc"
    n = salvar(serie, arq, "sst", "degC", "temperatura da superficie do mar (diaria)",
               "NOAA OISST v2.1, NOAA PSL (recorte do litoral do Ceara)")
    print(f"  gravado: {arq.name} ({n / 1e6:.1f} MB) e resumo_oisst_por_ano.csv")


# ------------------------------------------------------------------ OC-CCI
def conferir_clorofila():
    print("\n== OC-CCI (clorofila-a mensal)")
    arquivos = sorted(BRUTO_CHL.glob("occci_chla_ceara_*.nc"))
    partes, problemas = [], []
    referencia = None
    for arq in tqdm(arquivos, desc="lendo OC-CCI", unit="mês"):
        with xr.open_dataset(arq, engine="netcdf4") as ds:
            chl = ds["chlor_a"].transpose("time", "lat", "lon").load()
        if referencia is None:
            referencia = chl
        elif not mesma_grade(referencia, chl):
            problemas.append(f"{arq.name}: grade diferente")
        partes.append(chl)

    serie = xr.concat(partes, dim="time")
    meses = pd.DatetimeIndex(serie["time"].values).to_period("M")
    esperados = pd.period_range(meses.min(), meses.max(), freq="M")
    faltando = esperados.difference(meses)

    valido = serie.notnull()
    dominio = valido.any("time")                 # pixels com dado em pelo menos um mês (oceano)
    n_dominio = int(dominio.sum())
    validos = valido.where(dominio).sum(("lat", "lon")).values.astype(int)
    tabela = pd.DataFrame({"ano": meses.year, "mes": meses.month,
                           "pixels_validos": validos,
                           "fracao_valida": np.round(validos / n_dominio, 4)})
    for lim in LIMIARES:
        tabela[f"abaixo_{int(lim * 100)}"] = tabela["fracao_valida"] < lim

    print(f"  período: {meses.min()} a {meses.max()} | {len(meses)} meses "
          f"(esperado {MESES_CHL}) | meses faltando: {len(faltando)}")
    print(f"  grade: {serie.sizes['lat']} x {serie.sizes['lon']} pixels | "
          f"pixels oceânicos (com dado em algum mês): {n_dominio}")
    print(f"  clorofila-a: mediana {float(serie.median()):.3f} | "
          f"percentil 99 {float(serie.quantile(0.99)):.2f} | máximo {float(serie.max()):.2f} mg m-3")
    print(f"  fração de pixels válidos por mês: média {tabela.fracao_valida.mean():.0%} | "
          f"mínima {tabela.fracao_valida.min():.0%} "
          f"({tabela.loc[tabela.fracao_valida.idxmin(), 'ano']}-"
          f"{tabela.loc[tabela.fracao_valida.idxmin(), 'mes']:02d})")
    for lim in LIMIARES:
        print(f"  meses abaixo de {lim:.0%}: {int(tabela[f'abaixo_{int(lim * 100)}'].sum())}")
    print("  fração média de pixels válidos por mês do calendário:")
    por_mes = tabela.groupby("mes")["fracao_valida"].mean()
    print("    " + " | ".join(f"{m:02d}: {f:.0%}" for m, f in por_mes.items()))
    for p in problemas:
        print(f"  PROBLEMA: {p}")
    if not problemas and len(faltando) == 0 and len(meses) == MESES_CHL:
        print("  conferência: OK")

    PROCESSADO.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(PROCESSADO / "cobertura_clorofila.csv", index=False)
    arq = PROCESSADO / "occci_chla_ceara_199709_202212.nc"
    n = salvar(serie, arq, "chlor_a", "mg m-3", "concentracao de clorofila-a (media mensal)",
               "ESA OC-CCI v6.0, CEDA (recorte do litoral do Ceara)")
    print(f"  gravado: {arq.name} ({n / 1e6:.1f} MB) e cobertura_clorofila.csv")


if __name__ == "__main__":
    conferir_oisst()
    conferir_clorofila()
    print(f"\nsaídas em {PROCESSADO}")