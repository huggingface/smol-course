# Parameter-Efficient Fine-Tuning (PEFT) (Ajuste Fino com Eficiência de Parâmetro)

À medida que os modelos de linguagem aumentam, o ajuste fino tradicional torna-se cada vez mais desafiador. O ajuste fino completo de um modelo com 1,7 bilhão de parâmetros requer uma quantidade considerável de memória da GPU, torna caro o armazenamento de cópias separadas do modelo e apresenta o risco de um esquecimento catastrófico das capacidades originais do modelo. Os métodos de ajuste fino com eficiência de parâmetros (PEFT) abordam esses desafios modificando apenas um pequeno subconjunto de parâmetros do modelo e mantendo a maior parte do modelo congelada.

O ajuste fino tradicional atualiza todos os parâmetros do modelo durante o treinamento, o que se torna impraticável para modelos grandes. Os métodos PEFT introduzem abordagens para adaptar modelos usando menos parâmetros treináveis, geralmente menos de 1% do tamanho do modelo original. Essa redução drástica nos parâmetros treináveis permite:

- Ajuste fino no hardware do consumidor com memória de GPU limitada
- Armazenamento eficiente de várias adaptações de tarefas específicas
- Melhor generalização em cenários com poucos dados
- Ciclos de treinamento e iteração mais rápidos

## Métodos Disponíveis

Neste módulo, abordaremos dois métodos populares de PEFT:

### 1️⃣ LoRA (Low-Rank Adaptation - Adaptação de Baixa Classificação)

O LoRA surgiu como o método PEFT mais amplamente adotado, oferecendo uma solução elegante para a adaptação eficiente do modelo. Em vez de modificar o modelo inteiro, o **LoRA injeta matrizes treináveis nas camadas de atenção do modelo.**  Essa abordagem normalmente reduz os parâmetros treináveis em cerca de 90%, mantendo um desempenho comparável ao ajuste fino completo. Exploraremos o LoRA na seção [LoRA (Adaptação de Baixa Classificação)](./lora_adapters.md).
 
### 2️⃣ Ajuste de Prompts

O ajuste de prompts oferece uma abordagem **ainda mais leve** ao **adicionar tokens treináveis à entrada** em vez de modificar os pesos do modelo. O ajuste de prompt é menos popular que o LoRA, mas pode ser uma técnica útil para adaptar rapidamente um modelo a novas tarefas ou domínios. Exploraremos o ajuste de prompt na seção [Ajuste de Prompt](./prompt_tuning.md).

## Cadernos de Exercícios

| Título | Descrição | Exercício | Link | Colab |
|-------|-------------|----------|------|-------|
| Ajuste fino do LoRA | Aprenda a fazer o ajuste fino de modelos usando adaptadores do LoRA | 🐢 Treine um modelo usando o LoRA< br>🐕 Experimente com diferentes valores de classificação<br>🦁 Compare o desempenho com o ajuste fino completo | [Exercício](./notebooks/finetune_sft_peft.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/finetune_sft_peft.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | |
| Carregue adaptadores LoRA | Aprenda como carregar e usar adaptadores LoRA treinados | 🐢 Carregar adaptadores pré-treinados< br>🐕 Mesclar adaptadores com o modelo de base<br>🦁 Alternar entre vários adaptadores | [Exercício](./notebooks/load_lora_adapter.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/load_lora_adapter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 
<!-- | Ajuste de prompts | Aprenda como implementar o ajuste de prompts | 🐢 Treine prompts flexíveis<br>🐕 Compare diferentes estratégias de inicialização< br>🦁 Avalie em várias tarefas | [Exercício](./notebooks/prompt_tuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/prompt_tuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | -->

## Referências

- [Documentação PEFT](https://huggingface.co/docs/peft)
- [Artigo sobre LoRA](https://arxiv.org/abs/2106.09685)
- [Artigo sobre QLoRA](https://arxiv.org/abs/2305.14314)
- [Artigo sobre Ajuste de Prompts](https://arxiv.org/abs/2104.08691)
- [Guia PEFT do Hugging Face](https://huggingface.co/blog/peft)
- [Como ajustar os LLMs em 2024 com o Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) 
- [TRL](https://huggingface.co/docs/trl/index)