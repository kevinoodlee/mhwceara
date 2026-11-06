"""Faixa costeira e divisão do litoral em trechos.

Uma célula da grade entra na faixa costeira se estiver a até ``faixa`` graus do limite
do estado e dentro da extensão do estado em longitude. A faixa é dividida em trechos de
mesma largura em longitude (padrão: oeste, central e leste).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import Point
from shapely.ops import unary_union
from tqdm import tqdm

from .config import FAIXA_COSTEIRA, LIMITES_INTERNOS, PASTAS, TRECHOS

IBGE_MALHAS = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais"
)
URLS_LIMITE_CEARA = [
    f"{IBGE_MALHAS}/municipio_2025/UFs/CE/CE_UF_2025.zip",
    f"{IBGE_MALHAS}/municipio_2024/UFs/CE/CE_UF_2024.zip",
]


def baixar_limite(
    urls: list[str] = URLS_LIMITE_CEARA, destino: Path | None = None, progresso: bool = True
) -> Path:
    """Baixa o limite estadual do IBGE (.zip com shapefile), tentando os endereços em ordem.
    Se o arquivo já existir, não baixa de novo."""
    destino = destino or PASTAS.limites
    destino.mkdir(parents=True, exist_ok=True)
    for url in urls:
        nome = url.rsplit("/", 1)[-1]
        arq = destino / nome
        if arq.exists() and zipfile.is_zipfile(arq):
            if progresso:
                print(f"limite estadual: {nome} já baixado")
            return arq
        parcial = arq.with_suffix(".part")
        try:
            with requests.get(url, stream=True, timeout=(20, 120)) as r:
                if r.status_code != 200:
                    print(f"  {nome}: HTTP {r.status_code}, tentando o próximo endereço")
                    continue
                total = int(r.headers.get("Content-Length", 0)) or None
                with (
                    open(parcial, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=f"baixando {nome}",
                        disable=not progresso,
                    ) as barra,
                ):
                    for bloco in r.iter_content(1 << 16):
                        f.write(bloco)
                        barra.update(len(bloco))
        except requests.RequestException as erro:
            print(f"  {nome}: falha de conexão ({type(erro).__name__})")
            continue
        if zipfile.is_zipfile(parcial):
            parcial.replace(arq)
            return arq
        parcial.unlink()
        print(f"  {nome}: o arquivo recebido não é um .zip")
    raise RuntimeError(
        "Não consegui baixar o limite estadual. Baixe o .zip na página de malhas territoriais "
        f"do IBGE e salve em {destino}; depois rode de novo."
    )


def ler_limite(arq: Path):
    """Extrai o .zip e devolve o limite estadual como uma única geometria em WGS 84."""
    pasta = arq.parent / arq.stem
    with zipfile.ZipFile(arq) as z:
        z.extractall(pasta)
    shp = next(pasta.rglob("*.shp"))
    estado = gpd.read_file(shp).to_crs(4326)  # SIRGAS 2000 -> WGS 84, referencial do OISST
    return unary_union(list(estado.geometry))


def limites_trechos(
    lon_oeste: float,
    lon_leste: float,
    n: int = len(TRECHOS),
    internos: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Limites em longitude de ``n`` trechos, do extremo oeste ao extremo leste.

    Com ``internos`` (os ``n - 1`` limites entre trechos, de oeste para leste), usa esses
    valores; sem eles, divide a extensão em trechos de mesma largura.
    """
    if internos is None:
        return np.linspace(lon_oeste, lon_leste, n + 1)
    limites = np.array([lon_oeste, *internos, lon_leste], dtype=float)
    if len(internos) != n - 1 or np.any(np.diff(limites) <= 0):
        raise ValueError(
            f"esperados {n - 1} limites internos em ordem crescente, entre "
            f"{lon_oeste:.2f} e {lon_leste:.2f}; recebidos {internos}"
        )
    return limites


def classificar_celulas(
    lat,
    lon,
    oceano,
    limite,
    faixa: float = FAIXA_COSTEIRA,
    trechos: tuple[str, ...] = TRECHOS,
    limites_internos: tuple[float, ...] | None = LIMITES_INTERNOS,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Classifica as células oceânicas da grade.

    lat, lon: coordenadas da grade (longitude de -180 a 180); oceano: matriz booleana
    (lat x lon) com as células que têm dado; limite: geometria do estado;
    limites_internos: longitudes dos limites entre trechos (None = mesma largura).
    Devolve uma tabela (uma linha por célula oceânica, com distância à costa em graus,
    ``na_faixa`` e ``trecho``) e os limites dos trechos em longitude.
    """
    lon_oeste, _, lon_leste, _ = limite.bounds
    limites = limites_trechos(lon_oeste, lon_leste, len(trechos), limites_internos)
    linhas = []
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            if oceano[i, j]:
                linhas.append(
                    {
                        "lat": float(la),
                        "lon": float(lo),
                        "distancia_costa_graus": round(limite.distance(Point(lo, la)), 3),
                    }
                )
    celulas = pd.DataFrame(linhas)
    celulas["na_faixa"] = (celulas["distancia_costa_graus"] <= faixa) & celulas["lon"].between(
        lon_oeste, lon_leste
    )
    trecho = pd.cut(celulas["lon"], bins=limites, labels=list(trechos), include_lowest=True)
    celulas["trecho"] = trecho.astype(object).where(celulas["na_faixa"], None)
    return celulas, limites


def definir_trechos(
    faixa: float = FAIXA_COSTEIRA,
    trechos: tuple[str, ...] = TRECHOS,
    limites_internos: tuple[float, ...] | None = LIMITES_INTERNOS,
    destino: Path | None = None,
    progresso: bool = True,
):
    """Baixa o limite do Ceará, classifica as células do OISST e grava ``celulas_trechos.csv``.
    Devolve (tabela de células, limites dos trechos, geometria do estado)."""
    from .dados import ler_oisst  # importado aqui para evitar dependência circular

    limite = ler_limite(baixar_limite(progresso=progresso))
    primeiro_dia = ler_oisst().isel(time=0)
    oceano = primeiro_dia.notnull().values
    celulas, limites = classificar_celulas(
        primeiro_dia["lat"].values,
        primeiro_dia["lon"].values,
        oceano,
        limite,
        faixa,
        trechos,
        limites_internos,
    )
    destino = destino or PASTAS.processado
    destino.mkdir(parents=True, exist_ok=True)
    celulas.to_csv(destino / "celulas_trechos.csv", index=False)
    if progresso:
        faixa_df = celulas[celulas["na_faixa"]]
        print(
            f"\ncélulas oceânicas na caixa: {len(celulas)} | "
            f"na faixa costeira (até {faixa}°): {len(faixa_df)}"
        )
        print(
            f"extensão do estado em longitude: {abs(limites[0]):.2f}°O a {abs(limites[-1]):.2f}°O"
        )
        for nome, x0, x1 in zip(trechos, limites[:-1], limites[1:], strict=True):
            n = int((faixa_df["trecho"] == nome).sum())
            print(f"  trecho {nome:7s}: {abs(x0):.2f}°O a {abs(x1):.2f}°O | {n} células")
        print(f"gravado: {destino / 'celulas_trechos.csv'}")
    return celulas, limites, limite


def ler_celulas(arquivo: Path | None = None) -> pd.DataFrame:
    """Lê a tabela de células por trecho gravada por ``definir_trechos``."""
    return pd.read_csv(arquivo or PASTAS.processado / "celulas_trechos.csv")


def serie_por_trecho(da: xr.DataArray, celulas: pd.DataFrame) -> pd.DataFrame:
    """Média espacial de ``da`` (dimensões time, lat, lon) nas células de cada trecho.

    ``da`` precisa estar na mesma grade da tabela de células (a do OISST). Devolve uma
    tabela com uma coluna por trecho e o tempo como índice. As células têm praticamente a
    mesma área nestas latitudes (0° a 8°S), então a média é simples.
    """
    faixa = celulas[celulas["na_faixa"]]
    nomes = [t for t in TRECHOS if t in set(faixa["trecho"])]
    nomes += [t for t in faixa["trecho"].unique() if t not in nomes]
    colunas = {}
    for nome in nomes:
        grupo = faixa[faixa["trecho"] == nome]
        pontos = da.sel(
            lat=xr.DataArray(grupo["lat"].values, dims="celula"),
            lon=xr.DataArray(grupo["lon"].values, dims="celula"),
        )
        colunas[nome] = pontos.mean("celula").to_pandas()
    return pd.DataFrame(colunas)
