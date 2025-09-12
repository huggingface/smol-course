![smolcourse image](../banner.png)

# a smol course (um curso miudinho)

Este é um curso prático sobre alinhar modelos de linguagem para o seu caso de uso específico. É uma maneira útil de começar a alinhar modelos de linguagem, porque tudo funciona na maioria das máquinas locais. Existem requisitos mínimos de GPU e nenhum serviço pago. O curso é baseado na série de modelos de [SmolLM2](https://github.com/huggingface/smollm/tree/main), mas você pode transferir as habilidades que aprende aqui para modelos maiores ou outros pequenos modelos de linguagem.

<a href="http://hf.co/join/discord">
<img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
</a>

<div style="background: linear-gradient(to right, #e0f7fa, #e1bee7, orange); padding: 20px; border-radius: 5px; margin-bottom: 20px; color: purple;">
    <h2>A participação é aberta a todos, gratuita e já está disponível!</h2>
    <p>Este curso é aberto e avaliado por pares (peer reviewed). Para começar o curso, <strong>abra um pull request (PR)</strong> e envie seu trabalho para revisão. Aqui estão as etapas:</p>
    <ol>
        <li>Dê um fork no repositório <a href="https://github.com/huggingface/smol-course/fork">aqui</a></li>
        <li>Leia o material, faça alterações, faça os exercícios, adicione seus próprios exemplos</li>
        <li>Abra um PR no branch december_2024</li>
        <li>Tenha seu material revisado e mesclado no branch principal</li>
    </ol>
    <p>Isso deve te ajudar a aprender e a construir um curso feito pela comunidade, que está sempre melhorando.</p>
</div>

Podemos discutir o processo neste [tópico de discussão](https://github.com/huggingface/smol-course/discussions/2#discussion-7602932).

## Sumário do Curso

Este curso fornece uma abordagem prática para trabalhar com pequenos modelos de linguagem, desde o treinamento inicial até a implantação de produção.

| Módulo | Descrição | Status | Data de Lançamento |
|--------|-------------|---------|--------------|
| [Instruction Tuning (Ajuste de Instrução)](./1_instruction_tuning) | Aprenda sobre o ajuste fino supervisionado, modelos de bate-papo e a fazer o modelo seguir instruções básicas | ✅ Completo | 3 Dez, 2024 |
| [Preference Alignment (Alinhamento de Preferência)](./2_preference_alignment) | Explore técnicas DPO e ORPO para alinhar modelos com preferências humanas | ✅ Completo  | 6 Dez, 2024 |
| [Parameter-efficient Fine-tuning (Ajuste Fino com Eficiência de Parâmetro)](./3_parameter_efficient_finetuning) | Aprenda sobre LoRA, ajuste de prompt e métodos de adaptação eficientes | ✅ Completo | 9 Dez, 2024 |
| [Evaluation (Avaliação)](./4_evaluation) | Use benchmarks automáticos e crie avaliações de domínio personalizadas | ✅ Completo | 13 Dez, 2024 |
| [Vision-language Models (Modelos de Conjunto Visão-linguagem)](./5_vision_language_models) | Adapte modelos multimodais para tarefas visão-linguagem | ✅ Completo | 16 Dez, 2024 |
| [Synthetic Datasets (Conjuntos de Dados Sintéticos)](./6_synthetic_datasets) | Criar e validar conjuntos de dados sintéticos para treinamento | [🚧 Em Progresso](https://github.com/huggingface/smol-course/issues/83) | 20 Dez, 2024 |
| [Inference (Inferência)](./7_inference) | Infira modelos com eficiência | 📝 Planejado | 23 Dez, 2024 |
| Projeto Experimental | Use o que você aprendeu para ser o top 1 na tabela de classificação! | [🚧 Em Progresso](https://github.com/huggingface/smol-course/pull/97) | Dec 23, 2024 |

## Por Que Pequenos Modelos de Linguagem?

Embora os grandes modelos de linguagem tenham mostrado recursos e capacidades impressionantes, eles geralmente exigem recursos computacionais significativos e podem ser exagerados para aplicativos focados. Os pequenos modelos de linguagem oferecem várias vantagens para aplicativos de domínios específicos:

- **Eficiência**: Requer significativamente menos recursos computacionais para treinar e implantar
- **Personalização**: Mais fácil de ajustar e se adaptar a domínios específicos
- **Controle**: Melhor compreensão e controle do comportamento do modelo
- **Custo**: Menores custos operacionais para treinamento e inferência
- **Privacidade**: Pode ser executado localmente sem enviar dados para APIs externas
- **Tecnologia Verde**: Defende o uso eficiente de recursos com redução da pegada de carbono 
- **Desenvolvimento de Pesquisa Acadêmica Mais Fácil**: Oferece um ponto de partida fácil para a pesquisa acadêmica com LLMs de ponta com menos restrições logísticas

## Pré-requisitos

Antes de começar, verifique se você tem o seguinte:
- Entendimento básico de machine learning e natural language processing.
- Familiaridade com Python, PyTorch e o módulo `transformers`.
- Acesso a um modelo de linguagem pré-treinado e um conjunto de dados rotulado.

## Instalação

Mantemos o curso como um pacote para que você possa instalar dependências facilmente por meio de um gerenciador de pacotes. Recomendamos [uv](https://github.com/astral-sh/uv) para esse fim, mas você pode usar alternativas como `pip` ou` pdm`.

### Usando `uv`

Com o `uv` instalado, você pode instalar o curso deste modo:

```bash
uv venv --python 3.11.0
uv sync
```

### Usando `pip`

Todos os exemplos são executados no mesmo ambiente **python 3.11**, então você deve criar um ambiente e instalar dependências desta maneira:

```bash
# python -m venv .venv
# source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

**A partir do Google Colab** você precisará instalar dependências de maneira flexível com base no hardware que está usando. Pode fazer deste jeito:

```bash
pip install transformers trl datasets huggingface_hub
```

## Engajamento

Vamos compartilhar isso, desse jeito um monte de gente vai poder aprender a ajustar LLMs sem precisar de um computador super caro.

[![Gráfico de Histórico de Estrelas](https://api.star-history.com/svg?repos=huggingface/smol-course&type=Date)](https://star-history.com/#huggingface/smol-course&Date)
