# Evaluation (Avaliação)

A avaliação é uma etapa crítica no desenvolvimento e implantação de modelos de linguagem. Ela nos ajuda a entender quão bem nossos modelos desempenham em diferentes capacidades e a identificar áreas para melhorias. Este módulo aborda benchmarks padrão e abordagens de avaliação específicas de domínio para avaliar de forma abrangente o seu modelo smol (miudinho).

Usaremos o [`lighteval`](https://github.com/huggingface/lighteval), uma poderosa biblioteca de avaliação desenvolvida pelo Hugging Face que se integra perfeitamente ao ecossistema Hugging Face. Para um aprendizado mais profundo nos conceitos e práticas recomendadas de avaliação, confira o [guia](https://github.com/huggingface/evaluation-guidebook).

## Visão Geral do Módulo 

Uma estratégia de avaliação completa examina múltiplos aspectos do desempenho do modelo. Avaliamos capacidades específicas de tarefas, como responder a perguntas e sumarização, para entender como o modelo lida com diferentes tipos de problemas. Medimos a qualidade do output através de fatores como coerência e precisão factual. A avaliação de segurança ajuda a identificar outputs potencialmente prejudiciais ou biases. Por fim, os testes de especialização de domínio verificam o conhecimento especializado do modelo no campo-alvo.

## Conteúdo

### 1️⃣ [Benchmarks Automáticos](./automatic_benchmarks.md)

Aprenda a avaliar seu modelo usando benchmarks e métricas padronizados. Exploraremos benchmarks comuns, como MMLU e TruthfulQA, entenderemos as principais métricas e configurações de avaliação e abordaremos as melhores práticas para avaliações reproduzíveis.

### 2️⃣ [Avaliação de Domnínio Personalizado](./custom_evaluation.md)

Descubra como criar pipelines de avaliação adaptados ao seu caso de uso específico. Veremos o design de tarefas de avaliação personalizadas, implementação de métricas especializadas e construção de conjuntos de dados de avaliação que atendam às suas necessidades.

### 3️⃣ [Projeto de Avaliação de Domínio](./project/README.md)

Siga um exemplo completo de construção de um pipeline de avaliação de domínio específico. Você aprenderá a gerar conjuntos de dados de avaliação, usar o Argilla para anotação de dados, criar conjuntos de dados padronizados e avaliar modelos usando o LightEval.

### Cadernos de Exercício

| Título | Descrição | Exercício | Link | Colab |
|-------|-------------|----------|------|-------|
| Avalie e Analise Seu LLM | Aprenda a usar o LightEval para avaliar e comparar modelos em domínios específicos | 🐢 Use tarefas do domínio da medicina para avaliar um modelo <br> 🐕 Crie uma nova avaliação de domínio com diferentes tarefas do MMLU <br> 🦁 Crie uma tarefa de avaliação personalizada para o seu domínio | [Notebook](./notebooks/lighteval_evaluate_and_analyse_your_LLM.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/4_evaluation/notebooks/lighteval_evaluate_and_analyse_your_LLM.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Referências

- [Guia de Avaliação](https://github.com/huggingface/evaluation-guidebook) - Guia abrangente de avaliação de LLMs
- [Documentação do LightEval](https://github.com/huggingface/lighteval) - Documentação oficial do módulo LightEval
- [Documentação do Argilla](https://docs.argilla.io) - Saiba mais sobre a plataforma de anotação Argilla
- [Artigo do MMLU](https://arxiv.org/abs/2009.03300) - Artigo que descreve o benchmark MMLU
- [Criando uma Tarefa Personalizada](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Criando uma Métrica Personalizada](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Usando Métricas Existentes](https://github.com/huggingface/lighteval/wiki/Metric-List)