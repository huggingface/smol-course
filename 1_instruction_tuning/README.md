# Instruction Tuning

Ce module vous guidera dans la mise au point des modèles linguistiques. L'ajustement des instructions consiste à adapter des modèles pré-entraînés à des tâches spécifiques en les entraînant de nouveau sur des ensembles de données spécifiques à ces tâches. Ce processus permet aux modèles d'améliorer leurs performances sur les tâches ciblées. 

Dans ce module, nous allons explorer deux sujets : 1) les modèles de conversation et 2) le réglage fin supervisé.

## 1️⃣ Modèles de conversation

Les modèles de conversation structurent les interactions entre les utilisateurs et les modèles d'IA, garantissant des réponses cohérentes et adaptées au contexte. Ils comprennent des éléments tels que les invites du système et les messages basés sur le rôle. Pour plus d'informations, consultez le site [Chat Templates](./chat_templates.md) section.

## 2️⃣ Fine-Tuning Supervisé

Le réglage fin supervisé (SFT) est un processus essentiel pour adapter les modèles linguistiques pré-entraînés à des tâches spécifiques. Il s'agit d'entraîner le modèle sur un ensemble de données spécifiques à une tâche, avec des exemples étiquetés. Pour un guide détaillé sur le SFT, y compris les étapes clés et les meilleures pratiques, voir le document [Supervised Fine-Tuning](./supervised_fine_tuning.md).

## Cahiers d'exercices

| Titre | Description | Exercice | Lien | Colab |
|-------|-------------|----------|------|-------|
| Modèles de conversation | Apprenez à utiliser les modèles de chat avec SmolLM2 et à traiter les ensembles de données au format chatml. | 🐢 Convertir le jeu de données(dataset) `HuggingFaceTB/smoltalk` dans un format chatml <br> 🐕 Convertir le dataset `openai/gsm8k` dans un format chatml | [Notebook](./notebooks/chat_templates_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/chat_templates_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Fine-Tuning Supervisé | Apprenez à affiner(fine-tune) SmolLM2 à l'aide du SFTTrainer | 🐢 Utilisez le dataset `HuggingFaceTB/smoltalk` <br>🐕 Essayez le dataset `bigcode/the-stack-smol` <br>🦁 Sélectionner un jeu de données(dataset) pour un cas d'utilisation réel | [Notebook](./notebooks/sft_finetuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/1_instruction_tuning/notebooks/sft_finetuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## References

- [Transformers documentation on chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script for Supervised Fine-Tuning in TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [`SFTTrainer` in TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Supervised Fine-Tuning with TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [How to fine-tune Google Gemma with ChatML and Hugging Face TRL](https://www.philschmid.de/fine-tune-google-gemma)
- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)
