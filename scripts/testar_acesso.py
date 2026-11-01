"""
01_testar_acesso.py
Testa, a partir da sua rede, o acesso às duas fontes de dados, baixando apenas recortes pequenos:
  A) NOAA OISST v2.1 mensal e diário (NOAA PSL), por NetCDF Subset Service e por OPeNDAP
  B) ESA OC-CCI clorofila-a v6.0 (CEDA): testa o download direto e o recorte remoto
     de um arquivo mensal (janeiro de 2019)

Não é um download definitivo: serve para saber o que funciona antes de montar o pipeline.
Uso:  python scripts\\01_testar_acesso.py
"""
import time
from pathlib import Path

import numpy as np
import requests
import xarray as xr

RAIZ = Path(__file__).resolve().parent.parent
TESTE = RAIZ / "data" / "raw" / "teste_acesso"

# Caixa de teste (graus): plataforma do Ceará e águas oceânicas adjacentes
LAT = (-8.0, 0.0)
LON_OESTE = (-44.0, -34.0)              # longitude de -180 a 180, usada no OC-CCI
LON_0360 = (360.0 + LON_OESTE[0], 360.0 + LON_OESTE[1])   # 316 a 326, usada no OISST

PSL = "https://psl.noaa.gov/thredds"
OISST_MENSAL = f"{PSL}/ncss/grid/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
OISST_DIARIO = f"{PSL}/ncss/grid/Datasets/noaa.oisst.v2.highres/sst.day.mean.2019.nc"
OISST_DODS = f"{PSL}/dodsC/Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"

# OC-CCI: o nome dos arquivos segue um padrão fixo (ano e mês no nome),
# então o endereço é montado direto, sem ler a listagem das pastas do CEDA
CCI_CAMINHO = ("/neodc/esacci/ocean_colour/data/v6.0-release/geographic/netcdf/chlor_a/monthly/v6.0/"
               "{ano}/ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_4km_GEO_PML_OCx-{ano}{mes:02d}-fv6.0.nc")
CCI_DOWNLOAD = [                        # candidatos a endereço de download direto
    "https://dap.ceda.ac.uk{caminho}?download=1",
    "https://data.ceda.ac.uk{caminho}?download=1",
    "https://dap.ceda.ac.uk{caminho}",
]
CCI_OPENDAP = "https://dap.ceda.ac.uk/thredds/dodsC{caminho}"


# ---------------------------------------------------------------- utilidades
def baixar_ncss(url, tempo_ini, tempo_fim, destino):
    """Pede ao servidor só o recorte (variável sst, caixa de teste, intervalo de tempo)."""
    params = {"var": "sst", "north": LAT[1], "south": LAT[0], "west": LON_0360[0], "east": LON_0360[1],
              "time_start": tempo_ini, "time_end": tempo_fim, "accept": "netcdf"}
    t0 = time.time()
    r = requests.get(url, params=params, timeout=(20, 300))
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]!r}")
    if r.content[:3] != b"CDF" and r.content[:4] != b"\x89HDF":
        raise RuntimeError(f"resposta não é NetCDF: {r.content[:120]!r}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(r.content)
    print(f"  recebido: {len(r.content) / 1e3:.0f} kB em {time.time() - t0:.1f} s")


def resumir(da, rotulo):
    lat = next(n for n in ("lat", "latitude") if n in da.coords)
    lon = next(n for n in ("lon", "longitude") if n in da.coords)
    if da.size == 0:
        raise RuntimeError("recorte vazio (nenhuma célula dentro da caixa de teste)")
    print(f"  {rotulo}: dimensões {dict(da.sizes)}")
    print(f"  {lat}: {float(da[lat].min()):.3f} a {float(da[lat].max()):.3f} | "
          f"{lon}: {float(da[lon].min()):.3f} a {float(da[lon].max()):.3f}")
    if "time" in da.coords and da["time"].ndim == 1:
        print(f"  tempo: {str(da['time'].values[0])[:10]} a {str(da['time'].values[-1])[:10]}")
    v = da.values.astype(float)
    if np.isnan(v).all() or np.nanmin(v) < -1e10:
        raise RuntimeError("valores inválidos (o servidor devolveu erro ou valor de preenchimento)")
    print(f"  valores: mínimo {np.nanmin(v):.3f} | máximo {np.nanmax(v):.3f} | "
          f"média {np.nanmean(v):.3f} | sem dado {100 * np.isnan(v).mean():.1f}% | unidade {da.attrs.get('units', '?')}")


def abrir(arq):
    try:
        return xr.open_dataset(arq)
    except ValueError:
        print("  (tempo não decodificado automaticamente)")
        return xr.open_dataset(arq, decode_times=False)


def sondar_download(url):
    """Lê só os 8 primeiros bytes do arquivo e verifica se é NetCDF, sem baixar o arquivo inteiro."""
    with requests.get(url, headers={"Range": "bytes=0-7"}, stream=True, timeout=(20, 60)) as r:
        inicio = r.raw.read(8)
        status = r.status_code
        tipo = r.headers.get("Content-Type", "?")
        faixa = r.headers.get("Content-Range", "")
        tamanho = faixa.rsplit("/", 1)[-1] if "/" in faixa else r.headers.get("Content-Length", "")
    netcdf = inicio[:4] == b"\x89HDF" or inicio[:3] == b"CDF"
    texto_tam = f"{int(tamanho) / 1e6:.0f} MB" if tamanho.isdigit() else "não informado"
    print(f"  {url}")
    print(f"    status {status} | tipo {tipo} | tamanho {texto_tam} | é NetCDF: {'sim' if netcdf else 'não'}")
    if not netcdf:
        print(f"    início da resposta: {inicio!r}")
        if status in (401, 403):
            print("    (acesso negado: este endereço pode exigir conta ou token do CEDA)")
    return netcdf


# ---------------------------------------------------------------- testes
def teste_oisst_mensal():
    arq = TESTE / "oisst_mensal_recorte.nc"
    baixar_ncss(OISST_MENSAL, "2019-01-01T00:00:00Z", "2019-03-01T00:00:00Z", arq)
    with abrir(arq) as ds:
        resumir(ds["sst"], "OISST mensal (NCSS)")


def teste_oisst_diario():
    arq = TESTE / "oisst_diario_recorte.nc"
    baixar_ncss(OISST_DIARIO, "2019-01-01T00:00:00Z", "2019-01-31T00:00:00Z", arq)
    with abrir(arq) as ds:
        resumir(ds["sst"], "OISST diário (NCSS)")
        print(f"  passos de tempo: {ds.sizes['time']} (esperado 31)")


def teste_oisst_opendap():
    t0 = time.time()
    with xr.open_dataset(OISST_DODS) as ds:
        sub = ds["sst"].sel(lat=slice(*LAT), lon=slice(*LON_0360)).isel(time=slice(-3, None)).load()
    print(f"  recebido em {time.time() - t0:.1f} s")
    resumir(sub, "OISST mensal (OPeNDAP), últimos 3 meses disponíveis")


def teste_occci():
    caminho = CCI_CAMINHO.format(ano=2019, mes=1)

    print("  -- download direto (lê só o início do arquivo)")
    ok_download = False
    for modelo in CCI_DOWNLOAD:
        try:
            ok_download = sondar_download(modelo.format(caminho=caminho)) or ok_download
        except Exception as erro:
            print(f"    FALHOU: {type(erro).__name__}: {str(erro)[:150]}")

    print("  -- recorte remoto (OPeNDAP)")
    ok_dap = False
    try:
        t0 = time.time()
        with xr.open_dataset(CCI_OPENDAP.format(caminho=caminho)) as ds:
            nome = "chlor_a" if "chlor_a" in ds else list(ds.data_vars)[0]
            lat = ds["lat"].values
            fatia_lat = slice(LAT[1], LAT[0]) if lat[0] > lat[-1] else slice(*LAT)
            sub = ds[nome].sel(lat=fatia_lat, lon=slice(*LON_OESTE)).load()
        print(f"  recebido em {time.time() - t0:.1f} s")
        resumir(sub.squeeze(), "OC-CCI clorofila-a (OPeNDAP)")
        ok_dap = True
    except Exception as erro:
        print(f"    FALHOU: {type(erro).__name__}: {str(erro)[:200]}")

    if not (ok_download or ok_dap):
        raise RuntimeError("nenhum caminho de acesso ao OC-CCI funcionou")


TESTES = [
    ("OISST mensal por NCSS", teste_oisst_mensal),
    ("OISST diário por NCSS", teste_oisst_diario),
    ("OISST mensal por OPeNDAP", teste_oisst_opendap),
    ("OC-CCI clorofila-a (CEDA)", teste_occci),
]


def main():
    status = {}
    for rotulo, funcao in TESTES:
        print(f"\n== {rotulo}")
        try:
            funcao()
            status[rotulo] = "OK"
        except Exception as erro:
            print(f"  FALHOU: {type(erro).__name__}: {str(erro)[:300]}")
            status[rotulo] = "FALHOU"
    print("\n== Resumo")
    for rotulo, s in status.items():
        print(f"  {s:6s} {rotulo}")


if __name__ == "__main__":
    main()
