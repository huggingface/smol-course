# Benchmarks Automáticos

Os benchmarks automáticos servem como ferramentas padronizadas para avaliar modelos de linguagem em diferentes tarefas e capacidades. Embora forneçam um ponto de partida útil para entender o desempenho do modelo, é importante reconhecer que eles representam apenas uma parte de uma estratégia abrangente de avaliação.

## Entendendo os Benchmarks Automáticos

Os benchmarks automáticos geralmente consistem em conjuntos de dados organizados com tarefas e métricas de avaliação predefinidas. Esses benchmarks têm como objetivo avaliar vários aspectos da capacidade do modelo, desde compreensão básica de linguagem até raciocínio complexo. A principal vantagem de usar benchmarks automáticos é a padronização, eles permitem comparações consistentes entre diferentes modelos e fornecem resultados reproduzíveis.

No entanto, é crucial entender que o desempenho em benchmarks nem sempre se traduz diretamente em eficácia no mundo real. Um modelo que se destaca em benchmarks acadêmicos pode ainda enfrentar dificuldades em aplicações específicas de domínio ou casos práticos.

## Benchmarks e Suas Limitações

### Benchmarks de Conhecimento Geral

O MMLU (Massive Multitask Language Understanding) testa conhecimentos em 57 disciplinas, de ciências a humanidades. Embora abrangente, este benchmark pode não refletir a profundidade de especialização necessária para domínios específicos. O TruthfulQA avalia a tendência de um modelo em reproduzir conceitos equivocados comuns, embora não capture todas as formas de desinformação.

### Benchmarks de Raciocínio

BBH (Big Bench Hard) e GSM8K focam em tarefas de raciocínio complexo. O BBH testa pensamento lógico e planejamento, enquanto o GSM8K visa especificamente a resolução de problemas matemáticos. Esses benchmarks ajudam a avaliar capacidades analíticas, mas podem não captar o raciocínio detalhado exigido em cenários do mundo real.

### Compreensão de Linguagem

HELM fornece uma estrutura de avaliação holística, enquanto o WinoGrande testa o senso comum através da desambiguação de pronomes. Esses benchmarks oferecem insights sobre capacidades de processamento de linguagem, mas podem não representar totalmente a complexidade da conversação natural ou terminologia específica de domínio.

## Abordagens Alternativas de Avaliação

Muitas organizações desenvolveram métodos alternativos de avaliação para lidar com as limitações dos benchmarks padrão:

### LLM como Juiz

Usar um modelo de linguagem para avaliar os outputs de outro tornou-se cada vez mais popular. Essa abordagem pode fornecer feedback mais detalhado do que métricas tradicionais, embora apresente seus próprios vieses e limitações.

### Arenas de Avaliação

Plataformas como a IA Arena Constitucional da Anthropic permitem que modelos interajam e avaliem uns aos outros em ambientes controlados. Isso pode revelar pontos fortes e fracos que podem não ser evidentes em benchmarks tradicionais.

### Conjuntos de Benchmarks Personalizados

As organizações frequentemente desenvolvem conjuntos de benchmarks internos adaptados às suas necessidades e casos de uso específicos. Esses conjuntos podem incluir testes de conhecimento de domínio ou cenários de avaliação que refletem condições reais de implantação.

## Criando Sua Própria Estratégia de Avaliação

Embora o LightEval facilite a execução de benchmarks padrão, você também deve investir tempo no desenvolvimento de métodos de avaliação específicos para o seu caso de uso.

Sabendo que benchmarks padrão fornecem uma linha de base útil, eles não devem ser seu único método de avaliação. Aqui está como desenvolver uma abordagem mais abrangente:

1. Comece com benchmarks padrão relevantes para estabelecer uma linha de base e permitir comparações com outros modelos.

2. Identifique os requisitos específicos e desafios do seu caso de uso. Quais tarefas seu modelo realmente executará? Que tipos de erros seriam mais problemáticos?

3. Desenvolva conjuntos de dados de avaliação personalizados que reflitam seu caso de uso real. Isso pode incluir:
    - Consultas reais de usuários do seu domínio
    - Casos extremos comuns que você encontrou
    - Exemplos de cenários particularmente desafiadores

4. Considere implementar uma estratégia de avaliação em camadas:
    - Métricas automatizadas para feedback rápido
    - Avaliação humana para entendimento mais detalhado
    - Revisão por especialistas no domínio para aplicações especializadas
    - Testes A/B em ambientes controlados

## Usando LightEval para Benchmarks

As tarefas do LightEval são definidas usando um formato específico:

```
{suite}|{task}|{num_few_shot}|{auto_reduce}
```

- **suite**: O conjunto de benchmarks (ex.: 'mmlu', 'truthfulqa')
- **task**: Tarefa específica dentro do conjunto (ex.: 'abstract_algebra')
- **num_few_shot**: Número de exemplos a incluir no prompt (0 para zero-shot)
- **auto_reduce**: Se deve reduzir automaticamente exemplos few-shot caso o prompt seja muito longo (0 ou 1)

Exemplo: `"mmlu|abstract_algebra|0|0"` avalia a tarefa de álgebra abstrata do MMLU com inferência zero-shot.

### Exemplo de Pipeline de Avaliação

Aqui está um exemplo completo de configuração e execução de uma avaliação em benchmarks automáticos relevantes para um domínio específico:

```python
from lighteval.tasks import Task, Pipeline
from transformers import AutoModelForCausalLM

# Define tasks to evaluate
domain_tasks = [
    "mmlu|anatomy|0|0",
    "mmlu|high_school_biology|0|0", 
    "mmlu|high_school_chemistry|0|0",
    "mmlu|professional_medicine|0|0"
]

# Configure pipeline parameters
pipeline_params = {
    "max_samples": 40,  # Number of samples to evaluate
    "batch_size": 1,    # Batch size for inference
    "num_workers": 4    # Number of worker processes
}

# Create evaluation tracker
evaluation_tracker = EvaluationTracker(
    output_path="./results",
    save_generations=True
)

# Load model and create pipeline
model = AutoModelForCausalLM.from_pretrained("your-model-name")
pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=model
)

# Run evaluation
pipeline.evaluate()

# Get and display results
results = pipeline.get_results()
pipeline.show_results()
```

Os resultados são exibidos em formato tabular, mostrando:
```
|                  Task                  |Version|Metric|Value |   |Stderr|
|----------------------------------------|------:|------|-----:|---|-----:|
|all                                     |       |acc   |0.3333|±  |0.1169|
|leaderboard:mmlu:_average:5             |       |acc   |0.3400|±  |0.1121|
|leaderboard:mmlu:anatomy:5              |      0|acc   |0.4500|±  |0.1141|
|leaderboard:mmlu:high_school_biology:5  |      0|acc   |0.1500|±  |0.0819|
```

Você também pode manipular os resultados em um DataFrame do pandas e visualizá-los conforme necessário.

# Próximos Passos

⏩ Veja a [Avaliação Personalizada de Domínio](./custom_evaluation.md) para aprender a criar pipelines de avaliação adaptados às suas necessidades específicas.
