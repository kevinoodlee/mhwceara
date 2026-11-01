"""
baixar_oisst.py
Baixa a TSM diária do NOAA OISST v2.1 (0,25°) recortada na caixa do litoral do Ceará,
de 1982 a 2025, pelo NetCDF Subset Service (NCSS) da NOAA PSL.

Pede um mês por vez (pedidos de um ano inteiro são interrompidos pelo servidor com erro 502),
reaproveitando a mesma conexão e com pausa entre pedidos, e junta os 12 meses num arquivo por ano.

Saída: data/raw/sst/oisst_ceara_AAAA.nc (um arquivo por ano)
Anos já baixados e válidos são pulados: se a conexão cair, basta rodar de novo.
Uso:  python scripts\\baixar_oisst.py
"""
import calendar
import time
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "raw" / "sst"

ANOS = range(1982, 2026)                 # 1982 a 2025
# Caixa de download (provisória, Metodologia, seção 7): 8°S–0°, 44°W–34°W
LAT = (-8.0, 0.0)
LON_0360 = (316.0, 326.0)                # 44°W a 34°W na convenção 0–360 do OISST

NCSS = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.oisst.v2.highres/sst.day.mean.{ano}.nc"
TENTATIVAS = 5                           # tentativas por mês
ESPERA = 30                              # segundos antes da 2ª tentativa (dobra a cada nova falha)
PAUSA = 2                                # segundos entre pedidos, para não sobrecarregar o servidor

SESSAO = requests.Session()              # reaproveita a conexão entre os pedidos


def dias_no_ano(ano):
    return 366 if calendar.isleap(ano) else 365


def causa(erro):
    """Resume o erro mostrando a causa real (a parte depois de 'Caused by', se houver)."""
    texto = str(erro)
    if "Caused by" in texto:
        texto = texto.split("Caused by", 1)[1].strip(" ():")
    return f"{type(erro).__name__}: {texto[:90]}"


def conferir(ds, ano):
    """Levanta erro se faltar dia ou se os valores estiverem fora da faixa física da TSM."""
    if "sst" not in ds:
        raise RuntimeError("variável sst ausente")
    if ds.sizes.get("time") != dias_no_ano(ano):
        raise RuntimeError(f"{ds.sizes.get('time')} dias, esperado {dias_no_ano(ano)}")
    v = ds["sst"].values
    if np.isnan(v).all() or np.nanmin(v) < -5 or np.nanmax(v) > 45:
        raise RuntimeError("valores fora da faixa física")


def arquivo_valido(arq, ano):
    try:
        with xr.open_dataset(arq, engine="netcdf4") as ds:
            conferir(ds, ano)
        return True
    except Exception:
        return False


def baixar_mes(ano, mes, pasta_tmp):
    ultimo = calendar.monthrange(ano, mes)[1]
    params = {"var": "sst", "north": LAT[1], "south": LAT[0], "west": LON_0360[0], "east": LON_0360[1],
              "time_start": f"{ano}-{mes:02d}-01T00:00:00Z",
              "time_end": f"{ano}-{mes:02d}-{ultimo:02d}T00:00:00Z", "accept": "netcdf"}
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            r = SESSAO.get(NCSS.format(ano=ano), params=params, timeout=(30, 180))
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            if r.content[:3] != b"CDF" and r.content[:4] != b"\x89HDF":
                raise RuntimeError("resposta não é NetCDF")
            tmp = pasta_tmp / f"oisst_{ano}_{mes:02d}.part"
            tmp.write_bytes(r.content)
            with xr.open_dataset(tmp, engine="netcdf4") as ds:
                parte = ds[["sst"]].load()
            tmp.unlink()
            if parte.sizes["time"] != ultimo:
                raise RuntimeError(f"{parte.sizes['time']} dias, esperado {ultimo}")
            time.sleep(PAUSA)
            return parte
        except Exception as erro:
            if tentativa == TENTATIVAS:
                raise RuntimeError(f"mês {mes:02d}: {causa(erro)}")
            espera = ESPERA * 2 ** (tentativa - 1)
            tqdm.write(f"  {ano}-{mes:02d}: tentativa {tentativa} falhou ({causa(erro)}); "
                       f"nova tentativa em {espera} s")
            time.sleep(espera)


def salvar(sub, arq, ano):
    """Confere, monta um conjunto de dados limpo, grava num temporário e só então renomeia para .nc."""
    conferir(sub, ano)
    # conjunto novo, só com dados e coordenadas: os metadados herdados do servidor
    # causavam o erro "NetCDF: String match to name in use"
    sst = sub["sst"].transpose("time", "lat", "lon")
    limpo = xr.Dataset(
        {"sst": (("time", "lat", "lon"), sst.values.astype("float32"),
                 {"units": "degC", "long_name": "temperatura da superficie do mar (diaria)"})},
        coords={"time": sst["time"].values,
                "lat": ("lat", sst["lat"].values, {"units": "degrees_north"}),
                "lon": ("lon", sst["lon"].values, {"units": "degrees_east"})},
        attrs={"fonte": "NOAA OISST v2.1, NOAA PSL (recorte do litoral do Ceara)",
               "gerado_por": "mhwceara/scripts/baixar_oisst.py"},
    )
    parcial = arq.with_suffix(".nc.part")
    if parcial.exists():
        parcial.unlink()
    limpo.to_netcdf(parcial, engine="netcdf4", encoding={"sst": {"zlib": True, "complevel": 4}})
    if not arquivo_valido(parcial, ano):
        parcial.unlink()
        raise RuntimeError("arquivo gravado não passou na conferência")
    parcial.replace(arq)
    return arq.stat().st_size


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    pendentes = [a for a in ANOS if not arquivo_valido(DESTINO / f"oisst_ceara_{a}.nc", a)]
    print(f"OISST diário, {ANOS[0]} a {ANOS[-1]} | caixa {LAT[0]}° a {LAT[1]}° de latitude, "
          f"{LON_0360[0] - 360:.0f}° a {LON_0360[1] - 360:.0f}° de longitude")
    print(f"destino: {DESTINO}")
    print(f"{len(ANOS)} anos no total | {len(ANOS) - len(pendentes)} já baixados | "
          f"{len(pendentes)} a baixar\n")

    falhas = []
    t_total = time.time()
    barra = tqdm(total=12 * len(pendentes), desc="OISST diário", unit="mês")
    for ano in pendentes:
        arq = DESTINO / f"oisst_ceara_{ano}.nc"
        t0 = time.time()
        partes = []
        try:
            for mes in range(1, 13):
                barra.set_postfix_str(f"{ano}-{mes:02d}")
                partes.append(baixar_mes(ano, mes, DESTINO))
                barra.update(1)
            n = salvar(xr.concat(partes, dim="time"), arq, ano)
            tqdm.write(f"  {ano}: gravado ({n / 1e6:.1f} MB em {time.time() - t0:.0f} s)")
        except Exception as erro:
            barra.update(12 - len(partes))       # mantém a barra coerente com o total
            tqdm.write(f"  {ano}: FALHOU ({erro})")
            falhas.append(ano)
    barra.close()

    print(f"\nconcluído em {(time.time() - t_total) / 60:.1f} min")
    if falhas:
        print("anos com falha (rode o script de novo para tentar outra vez):", ", ".join(map(str, falhas)))
    else:
        print(f"todos os {len(ANOS)} anos estão em {DESTINO}")


if __name__ == "__main__":
    main()