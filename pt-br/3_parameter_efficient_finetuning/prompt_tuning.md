# Ajuste de Prompts

O ajuste de prompts (prompt tuning) é uma abordagem eficiente em termos de parâmetros que modifica as representações de entrada em vez dos pesos do modelo. Diferente do ajuste fino tradicional, que atualiza todos os parâmetros do modelo, o ajuste de prompts adiciona e otimiza um pequeno conjunto de tokens treináveis, mantendo o modelo base congelado.

## Entendendo o Ajuste de Prompts

O ajuste de prompts é uma alternativa eficiente ao ajuste fino de modelos que adiciona vetores contínuos treináveis (soft prompts) ao texto de entrada. Diferente dos prompts de texto discretos, esses soft prompts são aprendidos através de retropropagação enquanto o modelo de linguagem permanece congelado. O método foi introduzido em ["The Power of Scale for Parameter-Efficient Prompt Tuning"](https://arxiv.org/abs/2104.08691) (Lester et al., 2021), que demonstrou que o ajuste de prompts se torna mais competitivo em relação ao ajuste fino conforme o tamanho do modelo aumenta. No artigo, em torno de 10 bilhões de parâmetros, o ajuste de prompts iguala o desempenho do ajuste fino, modificando apenas algumas centenas de parâmetros por tarefa.

Esses soft prompts são vetores contínuos no espaço de embedding do modelo que são otimizados durante o treinamento. Diferente dos prompts discretos tradicionais que usam tokens de linguagem natural, os soft prompts não possuem significado inerente, mas aprendem a evocar o comportamento desejado do modelo congelado por meio de gradiente descendente. A técnica é particularmente eficaz em cenários multitarefa, pois cada tarefa exige apenas o armazenamento de um pequeno vetor de prompt (normalmente algumas centenas de parâmetros) em vez de uma cópia completa do modelo. Essa abordagem não apenas mantém um uso mínimo de memória, mas também possibilita a troca rápida de tarefas apenas trocando os vetores de prompt sem precisar recarregar o modelo.

## Processo de Treinamento

Os soft prompts geralmente têm entre 8 e 32 tokens e podem ser inicializados aleatoriamente ou a partir de texto existente. O método de inicialização desempenha um papel crucial no processo de treinamento, com inicializações baseadas em texto frequentemente apresentando melhor desempenho do que inicializações aleatórias.

Durante o treinamento, apenas os parâmetros do prompt são atualizados, enquanto o modelo base permanece congelado. Essa abordagem focada utiliza objetivos de treinamento padrão, mas exige atenção cuidadosa à taxa de aprendizado e ao comportamento do gradiente dos tokens do prompt.

## Implementação com PEFT

O módulo PEFT facilita a implementação do ajuste de prompts. Aqui está um exemplo básico:

```python
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model = AutoModelForCausalLM.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

# Configure prompt tuning
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,  # Number of trainable tokens
    prompt_tuning_init="TEXT",  # Initialize from text
    prompt_tuning_init_text="Classify if this text is positive or negative:",
    tokenizer_name_or_path="your-base-model",
)

# Create prompt-tunable model
model = get_peft_model(model, peft_config)
```

## Comparação com Outros Métodos

Quando comparado a outras abordagens PEFT, o ajuste de prompts se destaca por sua eficiência. Enquanto o LoRA oferece baixo uso de parâmetros e memória, mas exige o carregamento de adaptadores para troca de tarefas, o ajuste de prompts atinge um uso de recursos ainda menor e possibilita a troca imediata de tarefas. O ajuste fino completo, em contraste, demanda recursos significativos e requer cópias separadas do modelo para diferentes tarefas.

| Método | Parâmetros | Memória | Troca de Tarefas |
|--------|------------|---------|----------------|
| Ajuste de Prompts| Muito Baixo | Mínima | Fácil |
| LoRA | Baixo | Baixa | Requer Carregamento |
| Ajuste Fino | Alto | Alta | Nova Cópia do Modelo |

Ao implementar o ajuste de prompts, comece com um pequeno número de tokens virtuais (8-16) e aumente apenas se a complexidade da tarefa exigir. A inicialização baseada em texto geralmente apresenta melhores resultados do que a inicialização aleatória, especialmente ao usar texto relevante para a tarefa. A estratégia de inicialização deve refletir a complexidade da tarefa alvo.

O treinamento requer considerações ligeiramente diferentes do ajuste fino completo. Taxas de aprendizado mais altas geralmente funcionam bem, mas o monitoramento cuidadoso dos gradientes dos tokens do prompt é essencial. A validação regular com exemplos diversos ajuda a garantir um desempenho robusto em diferentes cenários.

## Aplicação

O ajuste de prompts se destaca em diversos cenários:

1. Implantação multitarefa
2. Ambientes com restrição de recursos
3. Adaptação rápida a tarefas
4. Aplicações sensíveis à privacidade

Conforme os modelos ficam menores, o ajuste de prompts se torna menos competitivo em comparação ao ajuste fino completo. Por exemplo, em modelos como SmolLM2, o ajuste de prompts é menos relevante do que o ajuste fino completo. 

## Próximos Passos

⏭️ Prossiga para o [Tutorial de Adaptadores LoRA](./notebooks/finetune_sft_peft.ipynb) para aprender como ajustar um modelo com adaptadores LoRA.

## Referências
- [Documentação PEFT](https://huggingface.co/docs/peft)
- [Artigo sobre Ajuste de Prompts](https://arxiv.org/abs/2104.08691)
- [Cookbook do Hugging Face](https://huggingface.co/learn/cookbook/prompt_tuning_peft)
