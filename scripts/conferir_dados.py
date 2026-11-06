"""Confere e consolida os dados brutos de TSM e de clorofila-a.

Gera, em data/processed/:
  - oisst_ceara_1982_2025.nc: série diária completa de TSM;
  - resumo_oisst_por_ano.csv: dias, células oceânicas e faixa de TSM por ano;
  - occci_chla_ceara_199709_202212.nc: série mensal completa de clorofila-a;
  - cobertura_clorofila.csv: fração de pixels válidos em cada mês.
Uso:  python scripts/conferir_dados.py
"""

from mhwceara import dados

print("\n== OISST (TSM diária)")
dados.consolidar_oisst()
print("\n== OC-CCI (clorofila-a mensal)")
dados.consolidar_clorofila()
