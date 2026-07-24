<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/logotipo-horizontal-escuro.svg">
  <img src=".github/assets/logotipo-horizontal-claro.svg" alt="campo-aberto" width="360">
</picture>

Probabilidades do Campeonato Brasileiro (Série A e Série B): título,
torneios continentais, acesso e rebaixamento, atualizadas a cada rodada por
um modelo bayesiano de força dos times.

[![Quality checks](https://github.com/IgorMichels/campo-aberto/actions/workflows/quality.yml/badge.svg)](https://github.com/IgorMichels/campo-aberto/actions/workflows/quality.yml)
[![Deploy site](https://github.com/IgorMichels/campo-aberto/actions/workflows/deploy-site.yml/badge.svg)](https://github.com/IgorMichels/campo-aberto/actions/workflows/deploy-site.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[igormichels.github.io/campo-aberto](https://igormichels.github.io/campo-aberto/)**

</div>

## O que faz

Os resultados oficiais da CBF são raspados e alimentam um modelo de Poisson
ajustado por Dixon-Coles (ajustado em Stan) que estima a força de ataque e
defesa de cada time; o restante da temporada é então simulado via Monte
Carlo milhares de vezes para reportar, por time, as probabilidades de
título, classificação para Libertadores/Sul-Americana, acesso e
rebaixamento. Detalhes técnicos de como cada estágio funciona:
[CODEBASE.md](CODEBASE.md).

## Funcionalidades do site

- **Classificação**
  - tabela de cada temporada (Série A e Série B, 2022-2026) com
    probabilidades de título, Libertadores, Sul-Americana e rebaixamento
    por time, numa data de referência à sua escolha
- **Evolução**
  - como essas probabilidades mudaram rodada a rodada ao longo da temporada
- **Jogos**
  - grade de probabilidades de placar para cada confronto: Próximos (ainda
    não disputados), Passados (comparados ao resultado real) e Simule
    (escolha dois times quaisquer e veja o placar mais provável)
- **Modelo**
  - documentação de como o modelo bayesiano funciona, e uma página de
    estatísticas com o desempenho dele (acerto de placar, direção, Brier,
    calibração) contra todos os jogos já disputados
- Tema claro/escuro

## Licença

[MIT](LICENSE)
