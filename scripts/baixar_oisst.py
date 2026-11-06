"""Baixa a TSM diária do NOAA OISST v2.1 na caixa do litoral do Ceará (1982 a 2025).

Gera: data/raw/sst/oisst_ceara_AAAA.nc (um arquivo por ano, cerca de 0,5 MB cada).
Anos já baixados e válidos são pulados: se a conexão cair, basta rodar de novo.
Uso:  python scripts/baixar_oisst.py
"""

from mhwceara import dados

dados.baixar_oisst()
