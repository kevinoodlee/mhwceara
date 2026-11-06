# mhwceara

Pacote em Python para detectar e caracterizar ondas de calor marinhas (MHW) e avaliar a
resposta da clorofila-a a esses eventos, a partir de dados públicos de satélite. O estudo
de caso é o litoral do Ceará, dividido em três trechos (oeste, central e leste).

- Detecção pela definição de Hobday et al. (2016): 5 dias ou mais com temperatura da
  superfície do mar (TSM) acima do percentil 90 de uma climatologia de 30 anos.
- TSM diária: NOAA OISST v2.1 (0,25°, 1982 a 2025).
- Clorofila-a: ESA OC-CCI v6.0 (4 km, mensal, set. 1997 a dez. 2022).

Projeto final do Space Tech Bootcamp 2026 (Future Tech Leaders Brasil).

## Situação

Em desenvolvimento.

- [x] Estrutura do pacote e testes automatizados
- [x] Módulo de detecção (`mhwceara.deteccao`), validado com séries sintéticas
- [x] Scripts de download dos dados
- [ ] Conferência e consolidação dos dados
- [ ] Divisão do litoral em trechos e métricas por trecho
- [ ] Resposta da clorofila-a
- [ ] Atlas e documentação no GitHub Pages

## Estrutura

```
src/mhwceara/   código do pacote
tests/          testes automatizados (séries sintéticas com eventos conhecidos)
scripts/        download e processamento dos dados
data/raw/       dados brutos (não versionados)
data/processed/ dados processados (não versionados)
notebooks/      análise exploratória
outputs/        figuras e tabelas
docs/           documentação e atlas
```

## Instalação

Requer Python 3.10 ou superior.

```
git clone https://github.com/kevinoodlee/mhwceara.git
cd mhwceara
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source .venv/bin/activate      # Linux ou macOS
python -m pip install -e ".[dev]"
python -m pytest
```

O último comando roda os testes; o resultado esperado é `10 passed`.

## Uso do pacote

```python
from mhwceara import climatologia, detectar_eventos, metricas_anuais

clim = climatologia(serie_tsm, periodo_base=(1982, 2011))  # serie_tsm: pandas.Series diária
eventos = detectar_eventos(clim)        # um evento por linha: início, fim, duração, intensidades
anual = metricas_anuais(eventos, clim)  # por ano: eventos, dias em onda de calor, duração, intensidade
```

## Reprodução dos dados

Os scripts baixam apenas o recorte da área de estudo (8°S a 0°, 44°O a 34°O). Não é
necessário cadastro em nenhuma das fontes.

```
python scripts/testar_acesso.py       # testa o acesso às fontes (cerca de 1 min)
python scripts/baixar_oisst.py        # TSM diária, 44 arquivos anuais (cerca de 1 h; ~22 MB)
python scripts/baixar_clorofila.py    # clorofila-a mensal, 304 arquivos (cerca de 25 min; ~27 MB)
```

Os scripts retomam de onde pararam: arquivos já baixados e válidos são pulados.

## Dados e citação

| Dado | Fonte | Citação |
|---|---|---|
| TSM diária OISST v2.1 | NOAA PSL | HUANG, B. et al. *Journal of Climate*, v. 34, n. 8, p. 2923-2939, 2021. |
| Clorofila-a OC-CCI v6.0 | ESA, via [CEDA](https://catalogue.ceda.ac.uk/uuid/b0ec72a28b6a4829a33ed9adc215d5bc) | Conforme os termos de uso da ESA indicados no catálogo do CEDA. |
| Limite do Ceará (CE_UF_2025) | IBGE | IBGE, Malha Municipal 2025. |

## Licença

O código é distribuído sob a licença MIT (ver `LICENSE`). Os dados não fazem parte do
repositório e seguem os termos de uso de cada fonte.