"""Testa o acesso às fontes de dados baixando um único mês de cada (janeiro de 2019).

Não grava arquivos; mostra no terminal o tempo de resposta, as dimensões e a faixa de valores
de cada recorte e um resumo com OK ou FALHOU.
Uso:  python scripts/testar_acesso.py
"""

from mhwceara import dados

dados.testar_acesso()
