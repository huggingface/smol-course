# Supervised Fine-Tuning

Supervised Fine-Tuning (SFT) is a critical process for adapting pre-trained language models to specific tasks or domains. While pre-trained models have impressive general capabilities, they often need to be customized to excel at particular use cases. SFT bridges this gap by further training the model on carefully curated datasets with human-validated examples.

## Understanding Supervised Fine-Tuning

At its core, supervised fine-tuning is about teaching a pre-trained model to perform specific tasks through examples of labeled tokens. The process involves showing the model many examples of the desired input-output behavior, allowing it to learn the patterns specific to your use case.

SFT is effective because it uses the foundational knowledge acquired during pre-training while adapting the model's behavior to match your specific needs.

## When to Use Supervised Fine-Tuning

The decision to use SFT often comes down to the gap between your model's current capabilities and your specific requirements. SFT becomes particularly valuable when you need precise control over the model's outputs or when working in specialized domains.

For example, if you're developing a customer service application, you might want your model to consistently follow company guidelines and handle technical queries in a standardized way. Similarly, in medical or legal applications, accuracy and adherence to domain-specific terminology becomes crucial. In these cases, SFT can help align the model's responses with professional standards and domain expertise.

## The Fine-Tuning Process

The supervised fine-tuning process involves adjusting a model's weights on a task-specific dataset. 

First, you'll need to prepare or select a dataset that represents your target task. This dataset should include diverse examples that cover the range of scenarios your model will encounter. The quality of this data is important - each example should demonstrate the kind of output you want your model to produce. Next comes the actual fine-tuning phase, where you'll use frameworks like Hugging Face's `transformers` and `trl` to train the model on your dataset. 

Throughout the process, continuous evaluation is essential. You'll want to monitor the model's performance on a validation set to ensure it's learning the desired behaviors without losing its general capabilities. In [module 4](../4_evaluation), we'll cover how to evaluate your model.

## The Role of SFT in Preference Alignment

SFT plays a fundamental role in aligning language models with human preferences. Techniques such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) rely on SFT to form a base level of task understanding before further aligning the model’s responses with desired outcomes. Pre-trained models, despite their general language proficiency, may not always generate outputs that match human preferences. SFT bridges this gap by introducing domain-specific data and guidance, which improves the model’s ability to generate responses that align more closely with human expectations.

## Supervised Fine-Tuning With Transformer Reinforcement Learning

A key software package for Supervised Fine-Tuning is Transformer Reinforcement Learning (TRL). TRL is a toolkit used to train transformer language models using reinforcement learning (RL).

Built on top of the Hugging Face Transformers library, TRL allows users to directly load pretrained language models and supports most decoder and encoder-decoder architectures. The library facilitates major processes of RL used in language modelling, including supervised fine-tuning (SFT), reward modeling (RM), proximal policy optimization (PPO), and Direct Preference Optimization (DPO). We will use TRL in a number of modules throughout this repo.

# Next Steps

Try out the following tutorials to get hands on experience with SFT using TRL:

⏭️ [Chat Templates Tutorial](./notebooks/chat_templates_example.ipynb)

⏭️ [Supervised Fine-Tuning Tutorial](./notebooks/sft_finetuning_example.ipynb)
