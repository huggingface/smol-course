# Preference Alignment (Alinhamento de Preferência)

Este módulo abrange técnicas para alinhar modelos de linguagem com preferências humanas. Enquanto o ajuste fino supervisionado ajuda os modelos a aprender tarefas, o alinhamento de preferência incentiva os resultados a corresponder às expectativas e valores humanos.

## Visão Geral

Métodos de alinhamento típicos envolvem vários estágios:
1. Ajuste fino supervisionado (SFT) para adaptar os modelos a domínios específicos
2. Alinhamento de preferência (como RLHF ou DPO) para melhorar a qualidade da resposta

Abordagens alternativas como ORPO combinam ajuste de instrução e alinhamento de preferência em um único processo. Aqui, vamos focar nos algoritmos DPO e ORPO.

Se você quer aprender mais sobre as diferentes técnicas de alinhamento, você pode ler mais sobre isso no [Argilla Blog](https://argilla.io/blog/mantisnlp-rlhf-part-8). 

### 1️⃣ Otimização de Preferência Direta (DPO - Direct Preference Optimization)

Otimização de preferência direta (DPO) simplifica o alinhamento de preferência otimizando diretamente os modelos usando dados de preferência. Essa abordagem elimina a necessidade de usar modelos de recompensa que não fazem parte do sistema e aprendizado de reforço complexo, tornando-o mais estável e eficiente do que o tradicional aprendizado de reforço com o feedback humano (RLHF). Para mais detalhes,você pode ler mais em [Documentação sobre otimização de preferência direta (DPO)](./dpo.md).


### 2️⃣ Otimização de Preferências de Razão de Chances (ORPO - Odds Ratio Preference Optimization)

ORPO introduz uma abordagem combinada para ajuste de instrução e alinhamento de preferência em um único processo. Ele modifica o objetivo padrão de modelagem de linguagem combinando a perda de log-verossimilhança negativa com um termo de razão de chances em um nível de token. A abordagem apresenta um processo de treinamento unificado de estágio único, arquitetura sem modelo de referência e eficiência computacional aprimorada. O ORPO apresentou resultados impressionantes em vários benchmarks, demonstrando melhor desempenho no AlpacaEval em comparação com os métodos tradicionais. Para obter mais detalhes, consulte a [Documentação sobre Otimização de Preferências de Razão de Chances (ORPO)](./orpo.md)

## Caderno de Exercícios

| Título | Descrição | Exercício | Link | Colab |
|-------|-------------|----------|------|-------|
| Treinamento em DPO | Aprenda a treinar modelos usando a Otimização Direta de Preferência | 🐢 Treine um modelo usando o conjunto de dados Anthropic HH-RLHF<br>🐕 Use seu próprio conjunto de dados de preferências<br>🦁 Experimente diferentes conjuntos de dados de preferências e tamanhos de modelos | [Exercício](./notebooks/dpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/dpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Treinamento em ORPO | Aprenda a treinar modelos usando a otimização de preferências de razão de chances | 🐢 Treine um modelo usando instruções e dados de preferências<br>🐕 Experimente com diferentes pesos de perda<br>🦁 Comparar os resultados de ORPO com DPO | [Exercício](./notebooks/orpo_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/2_preference_alignment/notebooks/orpo_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Referências

- [Documentação do TRL](https://huggingface.co/docs/trl/index) - Documentação do módulo Transformers Reinforcement Learning (TRL), que implementa várias técnicas de alinhamento, inclusive a DPO.
- [Artigo sobre DPO](https://arxiv.org/abs/2305.18290) - Documento de pesquisa original que apresenta a Otimização Direta de Preferência como uma alternativa mais simples ao RLHF que otimiza diretamente os modelos de linguagem usando dados de preferência.
- [Artigo sobre ORPO](https://arxiv.org/abs/2403.07691) - Apresenta a Otimização de preferências de razão de chances, uma nova abordagem que combina o ajuste de instruções e o alinhamento de preferências em um único estágio de treinamento.
- [Guia Argilla RLHF](https://argilla.io/blog/mantisnlp-rlhf-part-8/) - Um guia que explica diferentes técnicas de alinhamento, incluindo RLHF, DPO e suas implementações práticas.
- [Postagem de blog sobre DPO](https://huggingface.co/blog/dpo-trl) - Guia prático sobre a implementação de DPO usando a biblioteca TRL com exemplos de código e práticas recomendadas.
- [Exemplo de script TRL no DPO](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py) - Script de exemplo completo que demonstra como implementar o treinamento em DPO usando a biblioteca TRL.
- [Exemplo de script TRL no ORPO](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) - Implementação de referência do treinamento ORPO usando a biblioteca TRL com opções de configuração detalhadas.
- [Manual de alinhamento do Hugging Face](https://github.com/huggingface/alignment-handbook) - Guias de recursos e base de código para alinhamento de modelos de linguagem usando várias técnicas, incluindo SFT, DPO e RLHF.