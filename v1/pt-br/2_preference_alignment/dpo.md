# Otimização de Preferência Direta (DPO)

Otimização de Preferência Direta (DPO) oferece uma abordagem simplificada para alinhar modelos de linguagem com preferências humanas. Ao contrário dos métodos RLHF tradicionais que requerem modelos de recompensa separados e aprendizado de reforço complexo, o DPO otimiza diretamente o modelo usando dados de preferência.

## Entendendo sobre o DPO

O DPO reformula o alinhamento da preferência como um problema de classificação nos dados de preferência humana. As abordagens tradicionais de RLHF exigem treinamento de um modelo de recompensa separado e usando algoritmos de aprendizado de reforço complexos como o PPO para alinhar os outputs do modelo. O DPO simplifica esse processo, definindo uma função de perda que otimiza diretamente a política do modelo com base em outputs preferidos versus não preferidos.

Esta abordagem se mostrou altamente eficaz quando colocada em prática, sendo usada para treinar modelos como o Llama. Ao eliminar a necessidade de usar modelos de recompensa que não fazem parte do sistema e aprendizado de reforço complexo, DPO faz o alinhamento de preferência muito mais acessível e estável.

## Como o DPO Funciona

O processo do DPO exige que o ajuste fino supervisionado (SFT) adapte o modelo ao domínio-alvo. Isso cria uma base para a aprendizagem de preferências por meio do treinamento em conjuntos de dados padrão que seguem instruções. O modelo aprende a concluir tarefas básicas enquanto mantém suas capacidades gerais.

Em seguida, vem o aprendizado de preferências, em que o modelo é treinado em pares de outputs - um preferido e outro não preferido. Os pares de preferências ajudam o modelo a entender quais respostas se alinham melhor com os valores e as expectativas humanas.

A principal inovação do DPO está em sua abordagem de otimização direta. Em vez de treinar um modelo de recompensa separado, o DPO usa uma perda de entropia cruzada binária para atualizar diretamente os pesos do modelo com base nos dados de preferência. Esse processo simplificado torna o treinamento mais estável e eficiente e, ao mesmo tempo, obtém resultados comparáveis ou melhores do que a RLHF tradicional.

## Conjuntos de Dados para DPO

Os conjuntos de dados para DPO são normalmente criados anotando pares de respostas como preferidas ou não preferidas. Isso pode ser feito manualmente ou usando técnicas de filtragem automatizadas. Abaixo está um exemplo de estrutura de conjunto de dados de preferência de turno único para DPO:

| Prompt | Escolhido | Rejeitado |
|--------|-----------|-----------|
| ...    | ...       | ...       |
| ...    | ...       | ...       |
| ...    | ...       | ...       |

A coluna `Prompt` contém o prompt usado para gerar as respostas `Escolhido` e `Rejeitado`. As colunas `Escolhido` e `Rejeitado` contêm as respostas preferidas e não preferidas, respectivamente. Há variações nessa estrutura, por exemplo, incluindo uma coluna de prompt do sistema ou uma coluna `Input` que contém material de referência. Os valores de `Escolhido` e `Rejeitado` podem ser representados como cadeias de caracteres para conversas de turno único ou como listas de conversas. 

Você pode encontrar uma coleção de conjuntos de dados DPO em Hugging Face [aqui](https://huggingface.co/collections/argilla/preference-datasets-for-dpo-656f0ce6a00ad2dc33069478).

## Implementação com TRL

O módulo Transformers Reinforcement Learning (TRL) facilita a implementação do DPO. As classes `DPOConfig` e `DPOTrainer` seguem a mesma API no estilo `transformers`.
Veja a seguir um exemplo básico de configuração de treinamento de DPO:

```python
from trl import DPOConfig, DPOTrainer

# Define arguments
training_args = DPOConfig(
    ...
)

# Initialize trainer
trainer = DPOTrainer(
    model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    ...
)

# Train model
trainer.train()
```

Abordaremos mais detalhes sobre como usar as classes `DPOConfig` e `DPOTrainer` no [Tutotial sobre DPO](./notebooks/dpo_finetuning_example.ipynb).

## Práticas Recomendadas

A qualidade dos dados é fundamental para a implementação bem sucedida do DPO. O conjunto de dados de preferências deve incluir diversos exemplos que abranjam diferentes aspectos do comportamento desejado. Diretrizes claras de anotação garantem a rotulagem consistente das respostas preferidas e não preferidas. Você pode aprimorar o desempenho do modelo melhorando a qualidade do conjunto de dados de preferências. Por exemplo, filtrando conjuntos de dados maiores para incluir apenas exemplos de alta qualidade ou exemplos relacionados ao seu caso de uso.

Durante o treinamento, monitore cuidadosamente a convergência da perda e valide o desempenho em dados retidos. O parâmetro beta pode precisar de ajustes para equilibrar o aprendizado de preferências com a manutenção das capacidades gerais do modelo. A avaliação regular em diversos prompts ajuda a garantir que o modelo esteja aprendendo as preferências pretendidas sem se ajustar demais.

Compare os resultados do modelo com o modelo de referência para verificar a melhoria no alinhamento de preferências. Testar em uma variedade de prompts, incluindo casos extremos, ajuda a garantir um aprendizado de preferências robusto em diferentes cenários.

## Próximos Passos

⏩ Para obter experiência prática com o DPO, experimente fazer o [Tutorial DPO](./notebooks/dpo_finetuning_example.ipynb). Esse guia prático o orientará na implementação do alinhamento de preferências com seu próprio modelo, desde a preparação dos dados até o treinamento e a avaliação. 

⏭️ Depois de concluir o tutorial, você pode explorar a página do [ORPO](./orpo.md) para conhecer outra técnica de alinhamento de preferências.