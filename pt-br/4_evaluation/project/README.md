# Avaliação Específica de Domínio com Argilla, Distilabel e LightEval

Os benchmarks mais populares analisam capacidades muito gerais (raciocínio, matemática, código), mas você já precisou estudar sobre capacidades mais específicas? 

O que fazer se você precisar avaliar um modelo em um **domínio personalizado** relevante para seus casos de uso? (Por exemplo aplicações financeiras, jurídicas ou médicas)

Este tutorial mostra todo o pipeline que você pode seguir, desde a criação de dados relevantes e a anotação de suas amostras até a avaliação de seu modelo com elas, usando ferramentas como [Argilla](https://github.com/argilla-io/argilla), [distilabel](https://github.com/argilla-io/distilabel) e [lighteval](https://github.com/huggingface/lighteval). Para nosso exemplo, focaremos na geração de questões de exame a partir de múltiplos documentos.

## Estrutura do Projeto

Para o nosso processo, seguiremos 4 etapas, com um script para cada uma: gerar um conjunto de dados, anotar dados, extrair amostras relevantes para avaliação e, finalmente, avaliar os modelos.

| Nome do Script | Descrição |
|-------------|-------------|
| generate_dataset.py | Gera questões de exame a partir de múltiplos documentos de texto usando um modelo de linguagem especificado.. |
| annotate_dataset.py | Cria um conjunto de dados no Argilla para anotação manual das questões geradas. |
| create_dataset.py | Processa os dados anotados no Argilla e cria um conjunto de dados no Hugging Face. |
| evaluation_task.py | Define uma tarefa personalizada no LightEval para avaliar modelos de linguagem no conjunto de questões de exame. |

## Etapas

### 1. Gerar Conjunto de Dados

O script `generate_dataset.py` usa o módulo distilabel para gerar questões de exame com base em múltiplos documentos de texto. Ele utiliza o modelo especificado (padrão: Meta-Llama-3.1-8B-Instruct) para criar perguntas, respostas corretas e respostas incorretas (chamadas de distrações). Você deve adicionar seus próprios exemplos de dados e talvez até usar um modelo diferente.

Para executar a geração:

```sh
python generate_dataset.py --input_dir path/to/your/documents --model_id your_model_id --output_path output_directory
```

Isso criará um [Distiset](https://distilabel.argilla.io/dev/sections/how_to_guides/advanced/distiset/) contendo as questões de exame geradas para todos os documentos no diretório de entrada.

### 2. Anotar Conjunto de Dados

O script `annotate_dataset.py` utiliza as questões geradas e cria um conjunto de dados no Argilla para anotação. Ele configura a estrutura do conjunto de dados e o popula com as perguntas e respostas geradas, randomizando a ordem das respostas para evitar vieses (biases). No Argilla, você ou um especialista no domínio pode validar o conjunto de dados com as respostas corretas.

Você verá as respostas corretas sugeridas pelo LLM em ordem aleatória e poderá aprovar a resposta correta ou selecionar outra. A duração desse processo dependerá da escala do seu conjunto de avaliação, da complexidade dos dados do domínio e da qualidade do seu LLM. Por exemplo, fomos capazes de criar 150 amostras em 1 hora no domínio de transferência de aprendizado, usando o Llama-3.1-70B-Instruct, aprovando principalmente as respostas corretas e descartando as incorretas

Para executar o processo de anotação:

```sh
python annotate_dataset.py --dataset_path path/to/distiset --output_dataset_name argilla_dataset_name
```

Isso criará um conjunto de dados no Argilla que pode ser usado para revisão e anotação manual.

![argilla_dataset](./images/domain_eval_argilla_view.png)

Se você não estiver usando o Argilla, implante-o localmente ou no Spaces seguindo este [guia de início rápido](https://docs.argilla.io/latest/getting_started/quickstart/).

### 3. Criar Conjunto de Dados

O script `create_dataset.py` processa os dados anotados no Argilla e cria um conjunto de dados no Hugging Face. Ele manipula tanto as respostas sugeridas quanto as anotadas manualmente. O script criará um conjunto de dados contendo a pergunta, as possíveis respostas e o nome da coluna para a resposta correta. Para criar o conjunto de dados final:

```sh
huggingface_hub login
python create_dataset.py --dataset_path argilla_dataset_name --dataset_repo_id your_hf_repo_id
```

Isso enviará o conjunto de dados para o Hugging Face Hub sob o repositório especificado. Você pode visualizar o conjunto de dados de exemplo no hub [aqui](https://huggingface.co/datasets/burtenshaw/exam_questions/viewer/default/train), e uma pré-visualização do conjunto de dados se parece com isto:

![hf_dataset](./images/domain_eval_dataset_viewer.png)

### 4. Tarefa de Avaliação

O script `evaluation_task.py` define uma tarefa personalizada no LightEval para avaliar modelos de linguagem no conjunto de questões de exame. Ele inclui uma função de prompt, uma métrica de precisão personalizada e a configuração da tarefa. 

Para avaliar um modelo usando o lighteval com a tarefa personalizada de questões de exame:

```sh
lighteval accelerate \
    --model_args "pretrained=HuggingFaceH4/zephyr-7b-beta" \
    --tasks "community|exam_questions|0|0" \
    --custom_tasks domain-eval/evaluation_task.py \
    --output_dir "./evals"
```

Você pode encontrar guias detalhados no wiki do lighteval sobre cada uma dessas etapas: 

- [Criando uma Tarefa Personalizada](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Criando uma Métrica Personalizada](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Usando Métricas Existentes](https://github.com/huggingface/lighteval/wiki/Metric-List)


