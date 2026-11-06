"""Baixa a clorofila-a mensal do ESA OC-CCI v6.0 na caixa do litoral do Ceará
(setembro de 1997 a dezembro de 2022).

Gera: data/raw/clorofila/occci_chla_ceara_AAAAMM.nc (um arquivo por mês, cerca de 90 kB cada).
Meses já baixados e válidos são pulados: se a conexão cair, basta rodar de novo.
Uso:  python scripts/baixar_clorofila.py
"""

from mhwceara import dados

dados.baixar_clorofila()
