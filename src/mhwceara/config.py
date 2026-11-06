"""Configuração do estudo de caso: caixa de estudo, períodos, trechos e pastas do projeto.

Os valores aqui são os padrões usados pelas funções do pacote. Para aplicar o pacote a outra
região ou período, passe outros valores como argumento às funções (por exemplo,
``dados.baixar_oisst(caixa=Caixa(...), anos=range(2000, 2021))``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Caixa:
    """Retângulo geográfico em graus, com longitude de -180 a 180."""

    lat_sul: float
    lat_norte: float
    lon_oeste: float
    lon_leste: float

    @property
    def lon_0360(self) -> tuple[float, float]:
        """Longitudes oeste e leste na convenção de 0 a 360 (usada pelo OISST)."""
        return self.lon_oeste % 360, self.lon_leste % 360

    def __str__(self) -> str:
        def lat(v):
            return f"{abs(v):g}°{'S' if v < 0 else 'N' if v > 0 else ''}"

        def lon(v):
            return f"{abs(v):g}°{'O' if v < 0 else 'L' if v > 0 else ''}"

        return (
            f"{lat(self.lat_sul)} a {lat(self.lat_norte)}, "
            f"{lon(self.lon_oeste)} a {lon(self.lon_leste)}"
        )


# Caixa de download (Metodologia, seção 7): 8°S a 0°, 44°O a 34°O
CAIXA_CEARA = Caixa(lat_sul=-8.0, lat_norte=0.0, lon_oeste=-44.0, lon_leste=-34.0)

ANOS_OISST = range(1982, 2026)  # 1982 a 2025, só anos completos
INICIO_CLOROFILA = (1997, 9)  # primeiro mês do OC-CCI
FIM_CLOROFILA = (2022, 12)  # último mês da versão 6.0
PERIODO_BASE = (1982, 2011)  # climatologia principal (sensibilidade: 1982 a 2020 e base móvel)

TRECHOS = ("oeste", "central", "leste")
# Limites entre trechos (longitude, de oeste para leste), decididos em 06/11/2026:
# 39,75°O, onde a costa deixa de correr leste-oeste e passa a descer para sudeste;
# 38,5°O, logo a leste de Fortaleza. Os extremos são os do limite estadual (IBGE).
LIMITES_INTERNOS = (-39.75, -38.5)
FAIXA_COSTEIRA = 1.0  # distância máxima da costa, em graus (cerca de 110 km)
FRACAO_MINIMA = 0.5  # cobertura espacial mínima por trecho e mês (clorofila-a)
FRACOES_SENSIBILIDADE = (0.3, 0.5, 0.7)


def raiz_projeto(inicio: Path | None = None) -> Path:
    """Pasta do projeto: a primeira, a partir de ``inicio`` (padrão: pasta atual) e subindo,
    que contém um ``pyproject.toml``. Se nenhuma contiver, devolve a própria pasta inicial."""
    atual = (inicio or Path.cwd()).resolve()
    for pasta in (atual, *atual.parents):
        if (pasta / "pyproject.toml").exists():
            return pasta
    return atual


@dataclass(frozen=True)
class Pastas:
    """Pastas de dados e saídas, a partir da raiz do projeto."""

    raiz: Path

    @property
    def sst(self) -> Path:
        return self.raiz / "data" / "raw" / "sst"

    @property
    def clorofila(self) -> Path:
        return self.raiz / "data" / "raw" / "clorofila"

    @property
    def limites(self) -> Path:
        return self.raiz / "data" / "raw" / "boundaries"

    @property
    def processado(self) -> Path:
        return self.raiz / "data" / "processed"

    @property
    def figuras(self) -> Path:
        return self.raiz / "outputs" / "figuras"


PASTAS = Pastas(raiz_projeto())
