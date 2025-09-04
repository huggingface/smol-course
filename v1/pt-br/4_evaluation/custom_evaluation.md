# Avaliação de Domínio Personalizado

Embora benchmarks padrão ofereçam insights valiosos, muitas aplicações requerem abordagens de avaliação especializadas adaptadas a domínios ou casos de uso específicos. Este guia ajudará você a criar pipelines de avaliação personalizados que avaliem com precisão o desempenho do seu modelo no domínio-alvo.

## Planejando Sua Estratégia de Avaliação

Uma estratégia de avaliação personalizada bem-sucedida começa com objetivos claros. Considere quais capacidades específicas seu modelo precisa demonstrar no seu domínio. Isso pode incluir conhecimento técnico, padrões de raciocínio ou formatos específicos de domínio. Documente esses requisitos cuidadosamente, eles guiarão tanto o design da tarefa quanto a seleção de métricas.

Sua avaliação deve testar tanto casos de uso padrão quanto casos extremos. Por exemplo, em um domínio de medicina, você pode avaliar cenários com diagnósticos comuns e condições raras. Em aplicações financeiras, pode-se testar transações rotineiras e casos extremos envolvendo múltiplas moedas ou condições especiais.

## Implementação com LightEval

O LightEval fornece um framework flexível para implementar avaliações personalizadas. Veja como criar uma tarefa personalizada:

```python
from lighteval.tasks import Task, Doc
from lighteval.metrics import SampleLevelMetric, MetricCategory, MetricUseCase

class CustomEvalTask(Task):
    def __init__(self):
        super().__init__(
            name="custom_task",
            version="0.0.1",
            metrics=["accuracy", "f1"],  # Your chosen metrics
            description="Description of your custom evaluation task"
        )
    
    def get_prompt(self, sample):
        # Format your input into a prompt
        return f"Question: {sample['question']}\nAnswer:"
    
    def process_response(self, response, ref):
        # Process model output and compare to reference
        return response.strip() == ref.strip()
```

## Métricas Personalizadas

Tarefas específicas de domínio frequentemente exigem métricas especializadas. O LightEval fornece um framework flexível para criar métricas personalizadas que capturam aspectos relevantes do desempenho no domínio:

```python
from aenum import extend_enum
from lighteval.metrics import Metrics, SampleLevelMetric, SampleLevelMetricGrouping
import numpy as np

# Define a sample-level metric function
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    """Example metric that returns multiple scores per sample"""
    response = predictions[0]
    return {
        "accuracy": response == formatted_doc.choices[formatted_doc.gold_index],
        "length_match": len(response) == len(formatted_doc.reference)
    }

# Create a metric that returns multiple values per sample
custom_metric_group = SampleLevelMetricGrouping(
    metric_name=["accuracy", "length_match"],  # Names of sub-metrics
    higher_is_better={  # Whether higher values are better for each metric
        "accuracy": True,
        "length_match": True
    },
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=custom_metric,
    corpus_level_fn={  # How to aggregate each metric
        "accuracy": np.mean,
        "length_match": np.mean
    }
)

# Register the metric with LightEval
extend_enum(Metrics, "custom_metric_name", custom_metric_group)
```

Para casos mais simples onde você precisa de apenas um valor por amostra:

```python
def simple_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    """Example metric that returns a single score per sample"""
    response = predictions[0]
    return response == formatted_doc.choices[formatted_doc.gold_index]

simple_metric_obj = SampleLevelMetric(
    metric_name="simple_accuracy",
    higher_is_better=True,
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=simple_metric,
    corpus_level_fn=np.mean  # How to aggregate across samples
)

extend_enum(Metrics, "simple_metric", simple_metric_obj)
```

Você pode então usar suas métricas personalizadas em tarefas de avaliação referenciando-as na configuração da tarefa. As métricas serão automaticamente calculadas para todas as amostras e agregadas de acordo com as funções especificadas.

Para métricas mais complexas, considere:
- Usar metadados em seus documentos formatados para ponderar ou ajustar pontuações
- Implementar funções de agregação personalizadas para estatísticas em nível de corpus
- Adicionar verificações de validação para as entradas da métrica
- Documentar casos extremos e comportamentos esperados

Para um exemplo completo de métricas personalizadas em ação, veja nosso [projeto de avaliação de domínio](./project/README.md).

## Criação de Conjuntos de Dados

Uma avaliação de alta qualidade requer conjuntos de dados cuidadosamente selecionados. Considere estas abordagens para criação de conjuntos de dados:

1. Anotação Avançada: Trabalhe com especialistas no domínio para criar e validar exemplos de avaliação. Ferramentas como o [Argilla](https://github.com/argilla-io/argilla) tornam esse processo mais eficiente.

2. Dados do Mundo Real: Colete e anonimize dados de uso real, garantindo que representem cenários reais de implantação.

3. Geração Sintética: Use LLMs para gerar exemplos iniciais e, em seguida, peça para especialistas validarem e refinarem esses exemplos.

## Melhores Práticas

- Documente sua metodologia de avaliação detalhadamente, incluindo quaisquer suposições ou limitações
- Inclua casos de teste diversificados que cubram diferentes aspectos do seu domínio
- Considere métricas automatizadas e avaliação humana, quando apropriado
- Faça o controle de versão dos seus conjuntos de dados e código de avaliação
- Atualize regularmente seu conjunto de avaliação à medida que descobrir novos casos extremos ou requisitos

## Referências

- [Guia de Tarefas Personalizadas do LightEval](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Métricas Personalizadas do LightEval](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Documentação do Argilla](https://docs.argilla.io) para anotação de conjuntos de dados
- [Guia de Avaliação](https://github.com/huggingface/evaluation-guidebook) para princípios gerais de avaliação

# Próximos Passos

⏩ Para um exemplo completo de implementação desses conceitos, veja nosso [projeto de avaliação de domínio](./project/README.md).