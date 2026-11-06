"""mhwceara: ondas de calor marinhas e resposta da clorofila-a no litoral do Ceará.

Módulos:
  - config: caixa de estudo, períodos, trechos e pastas do projeto;
  - dados: download, conferência, consolidação e leitura do OISST e do OC-CCI;
  - deteccao: climatologia, detecção de eventos e métricas (Hobday et al., 2016);
  - trechos: faixa costeira, divisão do litoral e séries médias por trecho;
  - figuras: estilo dos gráficos e mapas.
"""

from .deteccao import climatologia, detectar_eventos, dia_do_ano_366, metricas_anuais

__version__ = "0.0.1"
__all__ = ["climatologia", "detectar_eventos", "dia_do_ano_366", "metricas_anuais"]
