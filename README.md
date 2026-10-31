# mhwceara

Pacote em Python e atlas para detectar e caracterizar ondas de calor marinhas
(MHW) no litoral do Ceará, a partir da TSM diária do NOAA OISST v2.1
(definição de Hobday et al., 2016), e avaliar a resposta da clorofila-a
(ESA OC-CCI v6.0).

Projeto final do bootcamp FTL Tech Leaders (AEB/UNDP).

## Instalação (desenvolvimento)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"
    python -m pytest

## Estrutura

- `src/mhwceara/`: código do pacote
- `tests/`: testes automatizados
- `scripts/`: scripts de download e processamento
- `data/`: dados (não versionados)
- `outputs/`: figuras e tabelas

## Dados

Todos públicos: NOAA OISST v2.1, ESA OC-CCI v6.0 e limites do IBGE.

## Licença

A definir (prevista: MIT, sujeita à compatibilidade com o xmhw).