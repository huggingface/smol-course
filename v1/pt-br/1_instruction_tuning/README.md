# Instruction Tuning (Ajuste de Instrução)

Este módulo o guiará através de modelos de linguagem de ajuste de instrução. O ajuste de instrução envolve a adaptação de modelos pré-treinados a tarefas específicas, treinando-os ainda mais em conjuntos de dados específicos de tarefas. Esse processo ajuda os modelos a melhorar seu desempenho em certas tarefas específicas.

Neste módulo, exploraremos dois tópicos: 1) Modelos de bate-papo e 2) Ajuste fino supervisionado.

## 1️⃣ Modelos de Bate-Papo

Modelos de bate-papo estruturam interações entre usuários e modelos de IA, garantindo respostas consistentes e contextualmente apropriadas. Eles incluem componentes como avisos de sistema e mensagens baseadas em funções. Para informações mais detalhadas, Consulte a seção [Chat Templates (Modelos de Bate-Papo)](./chat_templates.md).

## 2️⃣ Ajuste Fino Supervisionado

Ajuste fino supervisionado (em inglês, SFT - Supervised Fine-Tuning) é um processo crítico para adaptar modelos de linguagem pré-treinados a tarefas específicas. O ajuste envolve treinar o modelo em um conjunto de dados de uma tarefa específica com exemplos rotulados. Para um guia detalhado sobre SFT, incluindo etapas importantes e práticas recomendadas, veja a página [Supervised Fine-Tuning (Ajuste Fino Supervisionado)](./supervised_fine_tuning.md).

## Cadernos de Exercícios

| Título | Descrição | Exercício | Link | Colab |
|-------|-------------|----------|------|-------|
| Modelos de Bate-Papo | Aprenda a usar modelos de bate-papo com SmolLM2 and a processar conjunto de dados para o formato chatml | 🐢 Converta o conjunto de dados `HuggingFaceTB/smoltalk` para o formato chatml<br> 🐕 Converta o conjunto de dados `openai/gsm8k` para o formato chatml | [Exercício](./notebooks/chat_templates_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/chat_templates_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Ajuste Fino Supervisionado | Aprenda como fazer o ajuste fino no modelo SmolLM2 usando o SFTTrainer | 🐢 Use o conjunto de dados `HuggingFaceTB/smoltalk`<br>🐕 Experimente o conjunto de dados `bigcode/the-stack-smol`<br>🦁 Selecione um conjunto de dados para um caso de uso do mundo real | [Exercício](./notebooks/sft_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Referências

- [Documentação dos transformadores em modelos de bate-papo](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script para ajuste fino supervisionado em TRL](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py)
- [`SFTTrainer` em TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Artigo de otimização de preferência direta](https://huggingface.co/papers/2305.18290)
- [Ajuste fino supervisionado com TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [Como ajustar o Google Gemma com ChatML e Hugging Face TRL](https://www.philschmid.de/fine-tune-google-gemma)
- [Fazendo o ajuste fino em uma LLM para gerar catálogos de produtos persas em formato JSON](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)
