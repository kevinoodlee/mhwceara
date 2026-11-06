"""Acesso aos dados: download, conferência, consolidação e leitura.

Fontes:
- TSM diária NOAA OISST v2.1 (0,25°), pelo NetCDF Subset Service (NCSS) da NOAA PSL;
- clorofila-a mensal ESA OC-CCI v6.0 (4 km), por OPeNDAP no CEDA (sem cadastro).

As funções baixam só o recorte da caixa de estudo e gravam arquivos NetCDF limpos e
comprimidos. Downloads interrompidos podem ser retomados: arquivos já baixados e válidos
são pulados.
"""

from __future__ import annotations

import calendar
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

from .config import (
    ANOS_OISST,
    CAIXA_CEARA,
    FIM_CLOROFILA,
    FRACOES_SENSIBILIDADE,
    INICIO_CLOROFILA,
    PASTAS,
    Caixa,
)

OISST_NCSS = (
    "https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.oisst.v2.highres/sst.day.mean.{ano}.nc"
)
OCCCI_OPENDAP = (
    "https://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/ocean_colour/data/v6.0-release/"
    "geographic/netcdf/chlor_a/monthly/v6.0/{ano}/"
    "ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_4km_GEO_PML_OCx-{ano}{mes:02d}-fv6.0.nc"
)
FONTE_OISST = "NOAA OISST v2.1, NOAA PSL"
FONTE_OCCCI = "ESA OC-CCI v6.0, CEDA"


# ------------------------------------------------------------------ utilidades
def causa(erro: Exception) -> str:
    """Resume um erro mostrando a causa real (a parte depois de 'Caused by', se houver)."""
    texto = str(erro)
    if "Caused by" in texto:
        texto = texto.split("Caused by", 1)[1].strip(" ():")
    return f"{type(erro).__name__}: {texto[:90]}"


def com_tentativas(
    funcao: Callable, *args, tentativas: int = 3, espera: float = 20, rotulo: str = "", **kwargs
):
    """Chama ``funcao`` até dar certo, com espera crescente entre tentativas (``espera``,
    o dobro, o quádruplo...). Na última falha, levanta RuntimeError com a causa."""
    for tentativa in range(1, tentativas + 1):
        try:
            return funcao(*args, **kwargs)
        except Exception as erro:
            if tentativa == tentativas:
                raise RuntimeError(f"{rotulo}: {causa(erro)}") from erro
            pausa = espera * 2 ** (tentativa - 1)
            tqdm.write(
                f"  {rotulo}: tentativa {tentativa} falhou ({causa(erro)}); "
                f"nova tentativa em {pausa:.0f} s"
            )
            time.sleep(pausa)


def com_tempo_limite(funcao: Callable, segundos: float, *args):
    """Roda ``funcao`` numa linha de execução separada e desiste se passar do tempo limite
    (o OPeNDAP pode ficar esperando o servidor indefinidamente)."""
    resultado: dict = {}

    def alvo():
        try:
            resultado["valor"] = funcao(*args)
        except Exception as erro:
            resultado["erro"] = erro

    linha = threading.Thread(target=alvo, daemon=True)
    linha.start()
    linha.join(segundos)
    if linha.is_alive():
        raise TimeoutError(f"sem resposta do servidor em {segundos:.0f} s")
    if "erro" in resultado:
        raise resultado["erro"]
    return resultado["valor"]


def e_netcdf(conteudo: bytes) -> bool:
    """Verdadeiro se os bytes começam como um arquivo NetCDF (clássico ou NetCDF-4/HDF5)."""
    return conteudo[:3] == b"CDF" or conteudo[:4] == b"\x89HDF"


def gravar_netcdf(
    da: xr.DataArray, arq: Path, nome: str, unidade: str, descricao: str, fonte: str
) -> int:
    """Grava ``da`` (dimensões time, lat, lon) num NetCDF limpo e comprimido.

    Monta um conjunto de dados novo, só com os valores e as coordenadas: os metadados
    herdados dos servidores causam conflitos de nome ao gravar. Grava primeiro num
    arquivo temporário e só então o renomeia. Devolve o tamanho do arquivo em bytes.
    """
    da = da.transpose("time", "lat", "lon")
    ds = xr.Dataset(
        {
            nome: (
                ("time", "lat", "lon"),
                da.values.astype("float32"),
                {"units": unidade, "long_name": descricao},
            )
        },
        coords={
            "time": da["time"].values,
            "lat": ("lat", da["lat"].values, {"units": "degrees_north"}),
            "lon": ("lon", da["lon"].values, {"units": "degrees_east"}),
        },
        attrs={"fonte": fonte, "gerado_por": "mhwceara"},
    )
    arq.parent.mkdir(parents=True, exist_ok=True)
    parcial = arq.with_suffix(arq.suffix + ".part")
    if parcial.exists():
        parcial.unlink()
    ds.to_netcdf(parcial, engine="netcdf4", encoding={nome: {"zlib": True, "complevel": 4}})
    parcial.replace(arq)
    return arq.stat().st_size


def dias_no_ano(ano: int) -> int:
    return 366 if calendar.isleap(ano) else 365


# ------------------------------------------------------------------ OISST
def conferir_oisst(ds: xr.Dataset, ano: int) -> None:
    """Levanta erro se faltar a variável sst, faltar dia ou houver valor fora da faixa física."""
    if "sst" not in ds:
        raise RuntimeError("variável sst ausente")
    if ds.sizes.get("time") != dias_no_ano(ano):
        raise RuntimeError(f"{ds.sizes.get('time')} dias, esperado {dias_no_ano(ano)}")
    v = ds["sst"].values
    if np.isnan(v).all() or np.nanmin(v) < -5 or np.nanmax(v) > 45:
        raise RuntimeError("valores fora da faixa física")


def arquivo_oisst_valido(arq: Path, ano: int) -> bool:
    try:
        with xr.open_dataset(arq, engine="netcdf4") as ds:
            conferir_oisst(ds, ano)
        return True
    except Exception:
        return False


def baixar_mes_oisst(
    ano: int,
    mes: int,
    caixa: Caixa = CAIXA_CEARA,
    sessao: requests.Session | None = None,
    pasta_tmp: Path | None = None,
) -> xr.DataArray:
    """Baixa um mês de TSM diária do OISST, recortado na caixa, e devolve um DataArray."""
    ultimo = calendar.monthrange(ano, mes)[1]
    oeste, leste = caixa.lon_0360
    params = {
        "var": "sst",
        "north": caixa.lat_norte,
        "south": caixa.lat_sul,
        "west": oeste,
        "east": leste,
        "time_start": f"{ano}-{mes:02d}-01T00:00:00Z",
        "time_end": f"{ano}-{mes:02d}-{ultimo:02d}T00:00:00Z",
        "accept": "netcdf",
    }
    r = (sessao or requests).get(OISST_NCSS.format(ano=ano), params=params, timeout=(30, 180))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    if not e_netcdf(r.content):
        raise RuntimeError("resposta não é NetCDF")
    pasta_tmp = pasta_tmp or PASTAS.sst
    pasta_tmp.mkdir(parents=True, exist_ok=True)
    tmp = pasta_tmp / f"oisst_{ano}_{mes:02d}.part"
    tmp.write_bytes(r.content)
    with xr.open_dataset(tmp, engine="netcdf4") as ds:
        sst = ds["sst"].load()
    tmp.unlink()
    if sst.sizes["time"] != ultimo:
        raise RuntimeError(f"{sst.sizes['time']} dias, esperado {ultimo}")
    return sst


def baixar_oisst(
    anos=ANOS_OISST,
    caixa: Caixa = CAIXA_CEARA,
    destino: Path | None = None,
    regiao: str = "ceara",
    tentativas: int = 5,
    espera: float = 30,
    pausa: float = 2,
    progresso: bool = True,
) -> list[int]:
    """Baixa a TSM diária do OISST, um arquivo por ano (``oisst_<regiao>_<ano>.nc``).

    Pede um mês por vez (pedidos de um ano inteiro são interrompidos pelo servidor com
    erro 502), reaproveita a conexão e faz uma pausa entre pedidos (o servidor recusa
    pedidos seguidos com erro 429). Devolve a lista de anos que falharam.
    """
    destino = destino or PASTAS.sst
    destino.mkdir(parents=True, exist_ok=True)
    anos = list(anos)
    pendentes = [a for a in anos if not arquivo_oisst_valido(destino / f"oisst_{regiao}_{a}.nc", a)]
    if progresso:
        print(f"OISST diário, {anos[0]} a {anos[-1]} | caixa {caixa}")
        print(f"destino: {destino}")
        print(
            f"{len(anos)} anos | {len(anos) - len(pendentes)} já baixados | "
            f"{len(pendentes)} a baixar\n"
        )

    sessao = requests.Session()
    falhas = []
    barra = tqdm(total=12 * len(pendentes), desc="OISST diário", unit="mês", disable=not progresso)
    for ano in pendentes:
        arq = destino / f"oisst_{regiao}_{ano}.nc"
        t0 = time.time()
        partes = []
        try:
            for mes in range(1, 13):
                barra.set_postfix_str(f"{ano}-{mes:02d}")
                partes.append(
                    com_tentativas(
                        baixar_mes_oisst,
                        ano,
                        mes,
                        caixa,
                        sessao,
                        destino,
                        tentativas=tentativas,
                        espera=espera,
                        rotulo=f"{ano}-{mes:02d}",
                    )
                )
                barra.update(1)
                time.sleep(pausa)
            sst = xr.concat(partes, dim="time")
            conferir_oisst(sst.to_dataset(name="sst"), ano)
            n = gravar_netcdf(
                sst, arq, "sst", "degC", "temperatura da superficie do mar (diaria)", FONTE_OISST
            )
            if progresso:
                tqdm.write(f"  {ano}: gravado ({n / 1e6:.1f} MB em {time.time() - t0:.0f} s)")
        except Exception as erro:
            barra.update(12 - len(partes))
            tqdm.write(f"  {ano}: FALHOU ({erro})")
            falhas.append(ano)
    barra.close()
    if progresso:
        print(
            f"\nanos com falha (rode de novo para tentar outra vez): {', '.join(map(str, falhas))}"
            if falhas
            else f"\ntodos os anos estão em {destino}"
        )
    return falhas


# ------------------------------------------------------------------ OC-CCI
def meses_clorofila(
    inicio: tuple[int, int] = INICIO_CLOROFILA, fim: tuple[int, int] = FIM_CLOROFILA
) -> list[tuple[int, int]]:
    """Lista de (ano, mês) de ``inicio`` a ``fim``, inclusive."""
    ano, mes = inicio
    meses = []
    while (ano, mes) <= fim:
        meses.append((ano, mes))
        ano, mes = (ano + 1, 1) if mes == 12 else (ano, mes + 1)
    return meses


def conferir_clorofila(ds: xr.Dataset) -> None:
    """Levanta erro se faltar a variável chlor_a ou houver valor fora da faixa física."""
    if "chlor_a" not in ds:
        raise RuntimeError("variável chlor_a ausente")
    v = ds["chlor_a"].values
    if np.isnan(v).all() or np.nanmin(v) < 0 or np.nanmax(v) > 1000:
        raise RuntimeError("valores fora da faixa física")


def arquivo_clorofila_valido(arq: Path) -> bool:
    try:
        with xr.open_dataset(arq, engine="netcdf4") as ds:
            conferir_clorofila(ds)
        return True
    except Exception:
        return False


def ler_recorte_clorofila(ano: int, mes: int, caixa: Caixa = CAIXA_CEARA) -> xr.DataArray:
    """Lê por OPeNDAP só a caixa de estudo do arquivo mensal global do OC-CCI."""
    with xr.open_dataset(OCCCI_OPENDAP.format(ano=ano, mes=mes)) as ds:
        lat = ds["lat"].values
        fatia_lat = (
            slice(caixa.lat_norte, caixa.lat_sul)
            if lat[0] > lat[-1]
            else slice(caixa.lat_sul, caixa.lat_norte)
        )
        return ds["chlor_a"].sel(lat=fatia_lat, lon=slice(caixa.lon_oeste, caixa.lon_leste)).load()


def baixar_clorofila(
    inicio: tuple[int, int] = INICIO_CLOROFILA,
    fim: tuple[int, int] = FIM_CLOROFILA,
    caixa: Caixa = CAIXA_CEARA,
    destino: Path | None = None,
    regiao: str = "ceara",
    tentativas: int = 3,
    espera: float = 20,
    pausa: float = 1,
    tempo_limite: float = 180,
    progresso: bool = True,
) -> list[str]:
    """Baixa a clorofila-a mensal do OC-CCI, um arquivo por mês
    (``occci_chla_<regiao>_AAAAMM.nc``). Devolve a lista de meses que falharam."""
    destino = destino or PASTAS.clorofila
    destino.mkdir(parents=True, exist_ok=True)
    todos = meses_clorofila(inicio, fim)

    def arquivo(a, m):
        return destino / f"occci_chla_{regiao}_{a}{m:02d}.nc"

    pendentes = [(a, m) for a, m in todos if not arquivo_clorofila_valido(arquivo(a, m))]
    if progresso:
        print(
            f"OC-CCI clorofila-a mensal, {inicio[0]}-{inicio[1]:02d} a {fim[0]}-{fim[1]:02d} "
            f"| caixa {caixa}"
        )
        print(f"destino: {destino}")
        print(
            f"{len(todos)} meses | {len(todos) - len(pendentes)} já baixados | "
            f"{len(pendentes)} a baixar\n"
        )

    def baixar_um(ano, mes):
        chl = com_tempo_limite(ler_recorte_clorofila, tempo_limite, ano, mes, caixa)
        conferir_clorofila(chl.to_dataset(name="chlor_a"))
        gravar_netcdf(
            chl,
            arquivo(ano, mes),
            "chlor_a",
            "mg m-3",
            "concentracao de clorofila-a (media mensal)",
            FONTE_OCCCI,
        )

    falhas = []
    barra = tqdm(pendentes, desc="OC-CCI mensal", unit="mês", disable=not progresso)
    for ano, mes in barra:
        barra.set_postfix_str(f"{ano}-{mes:02d}")
        try:
            com_tentativas(
                baixar_um, ano, mes, tentativas=tentativas, espera=espera, rotulo=f"{ano}-{mes:02d}"
            )
            time.sleep(pausa)
        except Exception as erro:
            tqdm.write(f"  {ano}-{mes:02d}: FALHOU ({erro})")
            falhas.append(f"{ano}-{mes:02d}")
    barra.close()
    if progresso:
        print(
            f"\nmeses com falha (rode de novo para tentar outra vez): {', '.join(falhas)}"
            if falhas
            else f"\ntodos os meses estão em {destino}"
        )
    return falhas


# ------------------------------------------------------------------ teste de acesso
def resumo_recorte(da: xr.DataArray) -> str:
    """Texto curto com dimensões e faixa de valores de um recorte."""
    v = da.values.astype(float)
    return (
        f"dimensões {dict(da.sizes)} | mínimo {np.nanmin(v):.3f} | máximo {np.nanmax(v):.3f} "
        f"| sem dado {100 * np.isnan(v).mean():.1f}%"
    )


def testar_acesso(caixa: Caixa = CAIXA_CEARA, ano: int = 2019, mes: int = 1) -> dict[str, bool]:
    """Testa o acesso às duas fontes baixando um único mês de cada (nada é gravado)."""
    testes = {
        "OISST diário (NCSS)": lambda: baixar_mes_oisst(ano, mes, caixa),
        "OC-CCI mensal (OPeNDAP)": lambda: com_tempo_limite(
            ler_recorte_clorofila, 180, ano, mes, caixa
        ),
    }
    resultado = {}
    for nome, teste in testes.items():
        print(f"\n== {nome}, {ano}-{mes:02d}")
        try:
            t0 = time.time()
            da = teste()
            print(f"  recebido em {time.time() - t0:.1f} s | {resumo_recorte(da)}")
            resultado[nome] = True
        except Exception as erro:
            print(f"  FALHOU: {causa(erro)}")
            resultado[nome] = False
    print("\n== Resumo")
    for nome, ok in resultado.items():
        print(f"  {'OK' if ok else 'FALHOU':6s} {nome}")
    return resultado


# ------------------------------------------------------------------ consolidação
def _mesma_grade(a: xr.DataArray, b: xr.DataArray) -> bool:
    return np.array_equal(a["lat"].values, b["lat"].values) and np.array_equal(
        a["lon"].values, b["lon"].values
    )


def _tabela_indentada(df: pd.DataFrame) -> str:
    return "    " + df.to_string().replace("\n", "\n    ")


def consolidar_oisst(
    anos=ANOS_OISST,
    origem: Path | None = None,
    destino: Path | None = None,
    regiao: str = "ceara",
    progresso: bool = True,
) -> pd.DataFrame:
    """Confere os arquivos anuais do OISST e os junta num único arquivo.

    Confere dias por ano, continuidade das datas, grade, máscara de terra e faixa de valores.
    Grava ``oisst_<regiao>_<inicio>_<fim>.nc`` e ``resumo_oisst_por_ano.csv`` e devolve o
    resumo por ano.
    """
    origem = origem or PASTAS.sst
    destino = destino or PASTAS.processado
    anos = list(anos)
    partes, linhas, problemas = [], [], []
    for ano in tqdm(anos, desc="lendo OISST", unit="ano", disable=not progresso):
        arq = origem / f"oisst_{regiao}_{ano}.nc"
        if not arq.exists():
            problemas.append(f"{ano}: arquivo ausente")
            continue
        with xr.open_dataset(arq, engine="netcdf4") as ds:
            sst = ds["sst"].transpose("time", "lat", "lon").load()
        if partes and not _mesma_grade(partes[0], sst):
            problemas.append(f"{ano}: grade diferente da de {anos[0]}")
        if sst.sizes["time"] != dias_no_ano(ano):
            problemas.append(f"{ano}: {sst.sizes['time']} dias (esperado {dias_no_ano(ano)})")
        v = sst.values
        validas = (~np.isnan(v)).sum(axis=(1, 2))  # células com dado em cada dia
        linhas.append(
            {
                "ano": ano,
                "dias": sst.sizes["time"],
                "celulas_oceano_min": int(validas.min()),
                "celulas_oceano_max": int(validas.max()),
                "tsm_min": round(float(np.nanmin(v)), 2),
                "tsm_media": round(float(np.nanmean(v)), 2),
                "tsm_max": round(float(np.nanmax(v)), 2),
                "valores_distintos": int(np.unique(v[~np.isnan(v)]).size),
                "tamanho_kb": round(arq.stat().st_size / 1e3),
            }
        )
        partes.append(sst)

    resumo = pd.DataFrame(linhas).set_index("ano")
    serie = xr.concat(partes, dim="time")
    datas = pd.DatetimeIndex(serie["time"].values)
    faltando = pd.date_range(datas.min(), datas.max(), freq="D").difference(datas)
    duplicadas = datas[datas.duplicated()]
    celulas = sorted(set(resumo["celulas_oceano_min"]) | set(resumo["celulas_oceano_max"]))

    destino.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(destino / "resumo_oisst_por_ano.csv")
    arq = destino / f"oisst_{regiao}_{anos[0]}_{anos[-1]}.nc"
    n = gravar_netcdf(
        serie, arq, "sst", "degC", "temperatura da superficie do mar (diaria)", FONTE_OISST
    )

    if progresso:
        print(f"  período: {datas.min():%Y-%m-%d} a {datas.max():%Y-%m-%d} | {len(datas)} dias")
        print(f"  grade: {serie.sizes['lat']} x {serie.sizes['lon']} células")
        print(f"  dias faltando: {len(faltando)} | datas repetidas: {len(duplicadas)}")
        print(
            f"  células oceânicas por dia: {celulas} "
            f"({'máscara de terra constante' if len(celulas) == 1 else 'máscara VARIA'})"
        )
        print(
            f"  TSM: mínimo {resumo.tsm_min.min():.2f} °C | média {float(serie.mean()):.2f} °C"
            f" | máximo {resumo.tsm_max.max():.2f} °C"
        )
        for p in problemas:
            print(f"  PROBLEMA: {p}")
        if not problemas and len(faltando) == 0 and len(duplicadas) == 0:
            print("  conferência: OK")
        print(f"  gravado: {arq.name} ({n / 1e6:.1f} MB) e resumo_oisst_por_ano.csv")
    return resumo


def consolidar_clorofila(
    inicio: tuple[int, int] = INICIO_CLOROFILA,
    fim: tuple[int, int] = FIM_CLOROFILA,
    origem: Path | None = None,
    destino: Path | None = None,
    regiao: str = "ceara",
    progresso: bool = True,
) -> pd.DataFrame:
    """Confere os arquivos mensais do OC-CCI, calcula a cobertura espacial de cada mês e
    junta tudo num único arquivo.

    Cobertura = fração dos pixels oceânicos (com dado em pelo menos um mês) que têm dado
    no mês. Grava ``occci_chla_<regiao>_<inicio>_<fim>.nc`` e ``cobertura_clorofila.csv``
    e devolve a tabela de cobertura.
    """
    origem = origem or PASTAS.clorofila
    destino = destino or PASTAS.processado
    esperados = meses_clorofila(inicio, fim)
    partes, problemas = [], []
    for ano, mes in tqdm(esperados, desc="lendo OC-CCI", unit="mês", disable=not progresso):
        arq = origem / f"occci_chla_{regiao}_{ano}{mes:02d}.nc"
        if not arq.exists():
            problemas.append(f"{ano}-{mes:02d}: arquivo ausente")
            continue
        with xr.open_dataset(arq, engine="netcdf4") as ds:
            chl = ds["chlor_a"].transpose("time", "lat", "lon").load()
        if partes and not _mesma_grade(partes[0], chl):
            problemas.append(f"{ano}-{mes:02d}: grade diferente")
        partes.append(chl)

    serie = xr.concat(partes, dim="time")
    meses = pd.DatetimeIndex(serie["time"].values).to_period("M")
    valido = serie.notnull()
    dominio = valido.any("time")  # pixels com dado em pelo menos um mês (oceano)
    n_dominio = int(dominio.sum())
    validos = valido.where(dominio).sum(("lat", "lon")).values.astype(int)
    tabela = pd.DataFrame(
        {
            "ano": meses.year,
            "mes": meses.month,
            "pixels_validos": validos,
            "fracao_valida": np.round(validos / n_dominio, 4),
        }
    )
    for lim in FRACOES_SENSIBILIDADE:
        tabela[f"abaixo_{int(lim * 100)}"] = tabela["fracao_valida"] < lim

    destino.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino / "cobertura_clorofila.csv", index=False)
    arq = destino / f"occci_chla_{regiao}_{inicio[0]}{inicio[1]:02d}_{fim[0]}{fim[1]:02d}.nc"
    n = gravar_netcdf(
        serie, arq, "chlor_a", "mg m-3", "concentracao de clorofila-a (media mensal)", FONTE_OCCCI
    )

    if progresso:
        pior = tabela.loc[tabela["fracao_valida"].idxmin()]
        print(
            f"  período: {meses.min()} a {meses.max()} | {len(meses)} meses "
            f"(esperado {len(esperados)})"
        )
        print(
            f"  grade: {serie.sizes['lat']} x {serie.sizes['lon']} pixels | "
            f"pixels oceânicos: {n_dominio}"
        )
        print(
            f"  clorofila-a: mediana {float(serie.median()):.3f} | "
            f"percentil 99 {float(serie.quantile(0.99)):.2f} | "
            f"máximo {float(serie.max()):.2f} mg m-3"
        )
        print(
            f"  cobertura por mês: média {tabela.fracao_valida.mean():.0%} | mínima "
            f"{pior.fracao_valida:.0%} ({int(pior.ano)}-{int(pior.mes):02d})"
        )
        for lim in FRACOES_SENSIBILIDADE:
            print(f"  meses abaixo de {lim:.0%}: {int(tabela[f'abaixo_{int(lim * 100)}'].sum())}")
        por_mes = tabela.groupby("mes")["fracao_valida"].mean()
        print("  cobertura média por mês do calendário:")
        print("    " + " | ".join(f"{m:02d}: {f:.0%}" for m, f in por_mes.items()))
        for p in problemas:
            print(f"  PROBLEMA: {p}")
        if not problemas and len(meses) == len(esperados):
            print("  conferência: OK")
        print(f"  gravado: {arq.name} ({n / 1e6:.1f} MB) e cobertura_clorofila.csv")
    return tabela


# ------------------------------------------------------------------ leitura
def ler_oisst(arquivo: Path | None = None) -> xr.DataArray:
    """Lê a série consolidada de TSM, com longitude convertida para -180 a 180."""
    arquivo = arquivo or PASTAS.processado / "oisst_ceara_1982_2025.nc"
    with xr.open_dataset(arquivo, engine="netcdf4") as ds:
        sst = ds["sst"].load()
    return sst.assign_coords(lon=((sst["lon"] + 180) % 360) - 180)


def ler_clorofila(arquivo: Path | None = None) -> xr.DataArray:
    """Lê a série consolidada de clorofila-a mensal."""
    arquivo = arquivo or PASTAS.processado / "occci_chla_ceara_199709_202212.nc"
    with xr.open_dataset(arquivo, engine="netcdf4") as ds:
        return ds["chlor_a"].load()
