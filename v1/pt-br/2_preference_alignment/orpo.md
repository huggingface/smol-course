# Otimização de Preferências de Razão de Chances (ORPO)

A ORPO (Odds Ratio Preference Optimization) é uma nova técnica de ajuste fino que combina o ajuste fino e o alinhamento de preferências em um único processo unificado. Essa abordagem combinada oferece vantagens em termos de eficiência e desempenho em comparação com métodos tradicionais como RLHF ou DPO.

## Entendendo sobre o ORPO

O alinhamento com métodos como o DPO normalmente envolve duas etapas separadas: ajuste fino supervisionado para adaptar o modelo a um domínio e formato, seguido pelo alinhamento de preferências para alinhar com as preferências humanas. Embora o SFT adapte efetivamente os modelos aos domínios-alvo, ele pode aumentar inadvertidamente a probabilidade de gerar respostas desejáveis e indesejáveis. O ORPO aborda essa limitação integrando as duas etapas em um único processo, conforme ilustrado na comparação abaixo:

![Comparação das Técnicas de Alinhamento](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-alignments.png)
*Comparação de diferentes técnicas de alinhamento de modelos*

## Como o ORPO Funciona

O processo de treinamento utiliza um conjunto de dados de preferências semelhante ao que usamos para o DPO, em que cada exemplo de treinamento contém um prompt de entrada com duas respostas: uma que é preferida e outra que é rejeitada. Diferentemente de outros métodos de alinhamento que exigem estágios e modelos de referência separados, o ORPO integra o alinhamento de preferências diretamente ao processo de ajuste fino supervisionado. Essa abordagem monolítica o torna livre de modelos de referência, computacionalmente mais eficiente e eficiente em termos de memória com menos FLOPs.

O ORPO cria um novo objetivo ao combinar dois componentes principais:

1. **Função de perda do SFT**: A perda padrão de log-verossimilhança negativa usada na modelagem de linguagem, que maximiza a probabilidade de gerar tokens de referência. Isso ajuda a manter as capacidades gerais de linguagem do modelo.

2. **Função de perda da Razão de Chances**: Um novo componente que penaliza respostas indesejáveis e recompensa as preferidas. Essa função de perda usa índices de probabilidade (razão de chances) para contrastar efetivamente entre respostas favorecidas e desfavorecidas no nível do token.

Juntos, esses componentes orientam o modelo a se adaptar às gerações desejadas para o domínio específico e, ao mesmo tempo, desencorajam ativamente as gerações do conjunto de respostas rejeitadas. O mecanismo de razão de chances oferece uma maneira natural de medir e otimizar a preferência do modelo entre os resultados escolhidos e rejeitados. Se quiser se aprofundar na matemática, você pode ler o [artigo sobre ORPO](https://arxiv.org/abs/2403.07691). Se quiser saber mais sobre o ORPO do ponto de vista da implementação, confira como a função de perda do ORPO é calculada no [módulo TRL](https://github.com/huggingface/trl/blob/b02189aaa538f3a95f6abb0ab46c0a971bfde57e/trl/trainer/orpo_trainer.py#L660).

## Desempenho e Resultados

O ORPO demonstrou resultados impressionantes em vários benchmarks. No MT-Bench, ele alcança pontuações competitivas em diferentes categorias:

![Resultados do MT-Bench](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-mtbench.png)
*Resultados do MT-Bench por categoria para os modelos Mistral-ORPO*

Quando comparado a outros métodos de alinhamento, o ORPO apresenta desempenho superior no AlpacaEval 2.0:

![Resultados do AlpacaEval](https://argilla.io/images/blog/mantisnlp-rlhf/part-8-winrate.png)
*Pontuações do AlpacaEval 2.0 em diferentes métodos de alinhamento*

Em comparação com p SFT+DPO, o ORPO reduz os requisitos de computação, eliminando a necessidade de um modelo de referência e reduzindo pela metade o número de passagens de avanço por lote. Além disso, o processo de treinamento é mais estável em diferentes tamanhos de modelos e conjuntos de dados, exigindo menos hiperparâmetros para ajustar. Em termos de desempenho, o ORPO se iguala aos modelos maiores e mostra melhor alinhamento com as preferências humanas.

## Implementação 

A implementação bem-sucedida do ORPO depende muito de dados de preferência de alta qualidade. Os dados de treinamento devem seguir diretrizes claras de anotação e fornecer uma representação equilibrada das respostas preferidas e rejeitadas em diversos cenários. 

### Implementação com TRL

O ORPO pode ser implementado usando o módulo Transformers Reinforcement Learning (TRL). Aqui está um exemplo básico:

```python
from trl import ORPOConfig, ORPOTrainer

# Configure ORPO training
orpo_config = ORPOConfig(
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_steps=1000,
    orpo_alpha=1.0,  # Controls strength of preference optimization
    orpo_beta=0.1,   # Temperature parameter for odds ratio
)

# Initialize trainer
trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```

Principais parâmetros a serem considerados:
- `orpo_alpha`: Controla a força da otimização de preferências
- `orpo_beta`: Parâmetro de temperatura para o cálculo da razão de chances
- `learning_rate`: Deve ser relativamente pequeno para evitar o esquecimento catastrófico
- `gradient_accumulation_steps`: Ajuda na estabilidade do treinamento

## Próximos Passos

⏩ Experimente o [Tutorial do ORPO](./notebooks/orpo_tutorial.ipynb) para implementar essa abordagem unificada ao alinhamento de preferências.

## Referências
- [Artigo sobre ORPO](https://arxiv.org/abs/2403.07691)
- [Documentação TRL](https://huggingface.co/docs/trl/index)
- [Guia Argilla RLHF](https://argilla.io/blog/mantisnlp-rlhf-part-8/) 