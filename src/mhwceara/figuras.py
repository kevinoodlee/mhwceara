"""Estilo e figuras do projeto: cores, eixos geográficos, nota de fonte e mapas."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # grava os arquivos sem abrir janelas
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .config import PASTAS, TRECHOS  # noqa: E402

CORES_TRECHOS = {"oeste": "#1b9e77", "central": "#d95f02", "leste": "#7570b3"}
COR_OCEANO = "#d9e4ec"
COR_TERRA = "#efe9da"
FONTE_OISST_IBGE = "Fonte: elaborado pelo autor com dados do NOAA OISST v2.1 e do IBGE."


def rotulo_grau(valor: float, eixo: str) -> str:
    """Rótulo de coordenada em graus inteiros: '40°O', '3°S', '0°'."""
    if round(valor, 6) == 0:
        return "0°"
    if eixo == "lon":
        return f"{abs(valor):.0f}°{'O' if valor < 0 else 'L'}"
    return f"{abs(valor):.0f}°{'S' if valor < 0 else 'N'}"


def eixos_geograficos(ax, lon, lat, passo: float = 1.0, meia_celula: float = 0.125) -> None:
    """Ajusta limites, proporção e rótulos (°O, °S) de um mapa em graus."""
    lon, lat = np.asarray(lon), np.asarray(lat)
    ax.set_xlim(lon.min() - meia_celula, lon.max() + meia_celula)
    ax.set_ylim(lat.min() - meia_celula, lat.max() + meia_celula)
    ax.set_aspect("equal")
    xs = np.arange(np.ceil(lon.min() / passo) * passo, lon.max() + 1e-9, passo)
    ys = np.arange(np.ceil(lat.min() / passo) * passo, lat.max() + 1e-9, passo)
    ax.set_xticks(xs, [rotulo_grau(v, "lon") for v in xs])
    ax.set_yticks(ys, [rotulo_grau(v, "lat") for v in ys])


def nota_fonte(fig, texto: str = FONTE_OISST_IBGE) -> None:
    """Escreve a linha de fonte no rodapé da figura (padrão ABNT)."""
    fig.text(0.01, 0.01, texto, fontsize=8)


def salvar(fig, nome: str, pasta: Path | None = None, dpi: int = 200) -> Path:
    """Grava a figura em PNG em ``outputs/figuras`` e fecha a figura."""
    pasta = pasta or PASTAS.figuras
    pasta.mkdir(parents=True, exist_ok=True)
    arq = pasta / nome
    fig.savefig(arq, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return arq


def mapa_trechos(
    celulas: pd.DataFrame,
    limite,
    limites: np.ndarray,
    trechos: tuple[str, ...] = TRECHOS,
    nome: str = "mapa_area_estudo.png",
) -> Path:
    """Mapa das células da grade, da faixa costeira dividida em trechos e do estado."""
    import geopandas as gpd  # só esta figura precisa do geopandas

    lat = np.sort(celulas["lat"].unique())
    lon = np.sort(celulas["lon"].unique())
    codigo = {nome_t: k + 1 for k, nome_t in enumerate(trechos)}
    grade = np.full((lat.size, lon.size), np.nan)  # NaN = terra
    i_lat = {v: k for k, v in enumerate(lat)}
    i_lon = {v: k for k, v in enumerate(lon)}
    for c in celulas.itertuples():
        grade[i_lat[c.lat], i_lon[c.lon]] = codigo.get(c.trecho, 0) if c.na_faixa else 0

    fig, ax = plt.subplots(figsize=(10, 6.5))
    cores = [COR_OCEANO] + [CORES_TRECHOS.get(t, "gray") for t in trechos]
    ax.pcolormesh(
        lon,
        lat,
        grade,
        cmap=ListedColormap(cores),
        vmin=-0.5,
        vmax=len(trechos) + 0.5,
        shading="nearest",
        edgecolors="white",
        linewidth=0.3,
    )
    gpd.GeoSeries([limite], crs=4326).plot(ax=ax, color=COR_TERRA, edgecolor="black", linewidth=0.8)
    for x in limites[1:-1]:
        ax.axvline(x, color="black", linestyle="--", linewidth=0.8)
    for nome_t, x0, x1 in zip(trechos, limites[:-1], limites[1:], strict=True):
        ax.text(
            (x0 + x1) / 2,
            lat.max() - 0.15,
            nome_t,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=CORES_TRECHOS.get(nome_t, "gray"),
        )
    eixos_geograficos(ax, lon, lat)
    legenda = [Patch(color=CORES_TRECHOS.get(t, "gray"), label=f"trecho {t}") for t in trechos]
    legenda += [
        Patch(color=COR_OCEANO, label="oceano fora da faixa costeira"),
        Patch(facecolor=COR_TERRA, edgecolor="black", label="Ceará"),
        Patch(facecolor="white", edgecolor="gray", label="terra (fora do Ceará)"),
    ]
    ax.legend(handles=legenda, loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Células do OISST (0,25°) na faixa costeira do Ceará e trechos", fontsize=12)
    nota_fonte(fig)
    return salvar(fig, nome)


MESES = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")
COR_TEXTO = "#3d3d3d"
COR_GRADE = "#e6e6e6"


def virgula_decimal(eixo, casas: int = 1) -> None:
    """Rótulos do eixo com vírgula decimal (padrão brasileiro): 27,5 em vez de 27.5."""
    from matplotlib.ticker import FuncFormatter

    eixo.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{casas}f}".replace(".", ",")))


def _estilo_painel(ax) -> None:
    """Estilo discreto para painéis de séries: sem bordas de cima e da direita, grade leve."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", color=COR_GRADE, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COR_TEXTO, labelsize=9)


def dias_por_ano(
    anual: dict[str, pd.DataFrame],
    titulo: str,
    fonte: str,
    nome: str = "dias_onda_calor_por_ano.png",
) -> Path:
    """Pequenos múltiplos: dias em onda de calor marinha por ano, um painel por trecho."""
    nomes = list(anual)
    fig, eixos = plt.subplots(
        len(nomes), 1, figsize=(10, 2.2 * len(nomes) + 1), sharex=True, sharey=True
    )
    eixos = np.atleast_1d(eixos)
    for ax, trecho in zip(eixos, nomes, strict=True):
        a = anual[trecho]
        ax.bar(a.index, a["dias_onda_calor"], width=0.75, color=CORES_TRECHOS.get(trecho, "gray"))
        ax.set_title(f"Trecho {trecho}", loc="left", fontsize=10, color=COR_TEXTO)
        _estilo_painel(ax)
    eixos[-1].set_xlabel("Ano", color=COR_TEXTO)
    fig.supylabel("Dias em onda de calor marinha", color=COR_TEXTO, fontsize=10)
    fig.suptitle(titulo, fontsize=12, x=0.01, ha="left")
    nota_fonte(fig, fonte)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return salvar(fig, nome)


def serie_com_eventos(
    resultados: dict[str, dict[str, pd.DataFrame]],
    ano: int,
    titulo: str,
    fonte: str,
    nome: str | None = None,
) -> Path:
    """TSM diária de um ano, com climatologia, limiar e os eventos detectados sombreados,
    um painel por trecho (a figura clássica de Hobday et al., 2016)."""
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    nomes = list(resultados)
    fig, eixos = plt.subplots(len(nomes), 1, figsize=(10, 2.4 * len(nomes) + 1.2), sharex=True)
    eixos = np.atleast_1d(eixos)
    for ax, trecho in zip(eixos, nomes, strict=True):
        c = resultados[trecho]["clim"].loc[str(ano)]
        em_evento = pd.Series(False, index=c.index)
        for ev in resultados[trecho]["eventos"].itertuples():
            em_evento.loc[ev.inicio : ev.fim] = True
        cor = CORES_TRECHOS.get(trecho, "gray")
        ax.fill_between(
            c.index,
            c["limiar"],
            c["temp"],
            where=em_evento.values,
            color=cor,
            alpha=0.45,
            linewidth=0,
            interpolate=True,
        )
        ax.plot(c.index, c["temp"], color="#1f1f1f", linewidth=1.2)
        ax.plot(c.index, c["clim"], color="#8a8a8a", linewidth=1.0)
        ax.plot(c.index, c["limiar"], color="#8a8a8a", linewidth=1.0, linestyle="--")
        ax.set_title(f"Trecho {trecho}", loc="left", fontsize=10, color=COR_TEXTO)
        virgula_decimal(ax.yaxis)
        _estilo_painel(ax)
    eixos[-1].xaxis.set_major_locator(mdates.MonthLocator())
    eixos[-1].xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: MESES[mdates.num2date(x).month - 1])
    )
    eixos[-1].set_xlabel(str(ano), color=COR_TEXTO)
    fig.supylabel("TSM (°C)", color=COR_TEXTO, fontsize=10)
    legenda = [
        Line2D([], [], color="#1f1f1f", linewidth=1.2, label="TSM diária"),
        Line2D([], [], color="#8a8a8a", linewidth=1.0, label="climatologia (média)"),
        Line2D(
            [], [], color="#8a8a8a", linewidth=1.0, linestyle="--", label="limiar (percentil 90)"
        ),
        Patch(color="#9a9a9a", alpha=0.45, label="onda de calor marinha detectada"),
    ]
    eixos[0].legend(
        handles=legenda,
        loc="upper left",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0, 1.32),
    )
    fig.suptitle(titulo, fontsize=12, x=0.01, ha="left")
    nota_fonte(fig, fonte)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return salvar(fig, nome or f"tsm_eventos_{ano}.png")
