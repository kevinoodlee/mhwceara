"""Mapa da área de estudo e proposta de divisão do litoral em trechos.

Baixa o limite do Ceará (IBGE), seleciona as células do OISST na faixa costeira e as divide
em três trechos de mesma largura em longitude (oeste, central e leste).
Gera:
  - data/raw/boundaries/CE_UF_AAAA.zip (limite estadual) e a pasta extraída;
  - data/processed/celulas_trechos.csv: uma linha por célula oceânica, com trecho;
  - outputs/figuras/mapa_area_estudo.png: mapa das células e dos trechos.
Uso:  python scripts/mapa_area_estudo.py
"""

from mhwceara import figuras, trechos

celulas, limites, limite = trechos.definir_trechos()
arquivo = figuras.mapa_trechos(celulas, limite, limites)
print(f"gravado: {arquivo}")
