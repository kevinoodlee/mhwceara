"""
baixar_clorofila.py
Baixa a clorofila-a mensal do ESA OC-CCI v6.0 (4 km) recortada na caixa do litoral do Ceará,
de setembro de 1997 a dezembro de 2022, por OPeNDAP no servidor do CEDA (sem conta).

Lê só a caixa direto do arquivo mensal global (cada um tem cerca de 364 MB), então o download
total é de poucos MB.

Saída: data/raw/clorofila/occci_chla_ceara_AAAAMM.nc (um arquivo por mês)
Meses já baixados e válidos são pulados: se a conexão cair, basta rodar de novo.
Uso:  python scripts\\baixar_clorofila.py
"""
import threading
import time
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "raw" / "clorofila"

INICIO = (1997, 9)                       # primeiro mês do OC-CCI
FIM = (2022, 12)                         # último mês da versão 6.0
# Caixa de download (provisória, Metodologia, seção 7): 8°S–0°, 44°W–34°W
LAT = (-8.0, 0.0)
LON = (-44.0, -34.0)                     # o OC-CCI usa longitude de -180 a 180

OPENDAP = ("https://dap.ceda.ac.uk/thredds/dodsC/neodc/esacci/ocean_colour/data/v6.0-release/"
           "geographic/netcdf/chlor_a/monthly/v6.0/{ano}/"
           "ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_4km_GEO_PML_OCx-{ano}{mes:02d}-fv6.0.nc")
TENTATIVAS = 3
ESPERA = 20                              # segundos antes da 2ª tentativa (dobra na 3ª)
TEMPO_LIMITE = 180                       # segundos sem resposta antes de desistir do mês
PAUSA = 1                                # segundos entre pedidos, para não sobrecarregar o servidor


def meses():
    ano, mes = INICIO
    while (ano, mes) <= FIM:
        yield ano, mes
        ano, mes = (ano + 1, 1) if mes == 12 else (ano, mes + 1)


def conferir(ds):
    """Levanta erro se faltar a variável ou se os valores estiverem fora da faixa física."""
    if "chlor_a" not in ds:
        raise RuntimeError("variável chlor_a ausente")
    v = ds["chlor_a"].values
    if np.isnan(v).all() or np.nanmin(v) < 0 or np.nanmax(v) > 1000:
        raise RuntimeError("valores fora da faixa física")


def arquivo_valido(arq):
    try:
        with xr.open_dataset(arq, engine="netcdf4") as ds:
            conferir(ds)
        return True
    except Exception:
        return False


def com_tempo_limite(funcao, segundos, *args):
    """Roda a função numa linha de execução separada e desiste se passar do tempo limite
    (o OPeNDAP pode ficar esperando o servidor indefinidamente)."""
    resultado = {}

    def alvo():
        try:
            resultado["valor"] = funcao(*args)
        except Exception as erro:
            resultado["erro"] = erro

    linha = threading.Thread(target=alvo, daemon=True)
    linha.start()
    linha.join(segundos)
    if linha.is_alive():
        raise TimeoutError(f"sem resposta do servidor em {segundos} s")
    if "erro" in resultado:
        raise resultado["erro"]
    return resultado["valor"]


def ler_recorte(ano, mes):
    with xr.open_dataset(OPENDAP.format(ano=ano, mes=mes)) as ds:
        lat = ds["lat"].values
        fatia_lat = slice(LAT[1], LAT[0]) if lat[0] > lat[-1] else slice(*LAT)
        return ds["chlor_a"].sel(lat=fatia_lat, lon=slice(*LON)).load()


def salvar(chl, arq):
    """Monta um conjunto de dados limpo, confere, grava num temporário e só então renomeia."""
    chl = chl.transpose("time", "lat", "lon")
    limpo = xr.Dataset(
        {"chlor_a": (("time", "lat", "lon"), chl.values.astype("float32"),
                     {"units": "mg m-3", "long_name": "concentracao de clorofila-a (media mensal)"})},
        coords={"time": chl["time"].values,
                "lat": ("lat", chl["lat"].values, {"units": "degrees_north"}),
                "lon": ("lon", chl["lon"].values, {"units": "degrees_east"})},
        attrs={"fonte": "ESA OC-CCI v6.0, CEDA (recorte do litoral do Ceara)",
               "gerado_por": "mhwceara/scripts/baixar_clorofila.py"},
    )
    conferir(limpo)
    parcial = arq.with_suffix(".nc.part")
    if parcial.exists():
        parcial.unlink()
    limpo.to_netcdf(parcial, engine="netcdf4", encoding={"chlor_a": {"zlib": True, "complevel": 4}})
    if not arquivo_valido(parcial):
        parcial.unlink()
        raise RuntimeError("arquivo gravado não passou na conferência")
    parcial.replace(arq)


def baixar_mes(ano, mes, arq):
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            salvar(com_tempo_limite(ler_recorte, TEMPO_LIMITE, ano, mes), arq)
            time.sleep(PAUSA)
            return
        except Exception as erro:
            if tentativa == TENTATIVAS:
                raise
            espera = ESPERA * 2 ** (tentativa - 1)
            tqdm.write(f"  {ano}-{mes:02d}: tentativa {tentativa} falhou ({str(erro)[:80]}); "
                       f"nova tentativa em {espera} s")
            time.sleep(espera)


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    todos = list(meses())
    pendentes = [(a, m) for a, m in todos
                 if not arquivo_valido(DESTINO / f"occci_chla_ceara_{a}{m:02d}.nc")]
    print(f"OC-CCI clorofila-a mensal, {INICIO[0]}-{INICIO[1]:02d} a {FIM[0]}-{FIM[1]:02d} | "
          f"caixa {LAT[0]}° a {LAT[1]}° de latitude, {LON[0]:.0f}° a {LON[1]:.0f}° de longitude")
    print(f"destino: {DESTINO}")
    print(f"{len(todos)} meses no total | {len(todos) - len(pendentes)} já baixados | "
          f"{len(pendentes)} a baixar\n")

    falhas = []
    t_total = time.time()
    barra = tqdm(pendentes, desc="OC-CCI mensal", unit="mês")
    for ano, mes in barra:
        barra.set_postfix_str(f"{ano}-{mes:02d}")
        arq = DESTINO / f"occci_chla_ceara_{ano}{mes:02d}.nc"
        try:
            baixar_mes(ano, mes, arq)
        except Exception as erro:
            tqdm.write(f"  {ano}-{mes:02d}: FALHOU ({str(erro)[:120]})")
            falhas.append(f"{ano}-{mes:02d}")
    barra.close()

    print(f"\nconcluído em {(time.time() - t_total) / 60:.1f} min")
    if falhas:
        print("meses com falha (rode o script de novo para tentar outra vez):", ", ".join(falhas))
    else:
        print(f"todos os {len(todos)} meses estão em {DESTINO}")


if __name__ == "__main__":
    main()