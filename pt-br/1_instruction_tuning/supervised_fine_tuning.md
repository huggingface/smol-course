# Ajuste Fino Supervisionado

Ajuste fino supervisionado (em inglês, SFT - Supervised Fine-Tuning) é um processo crítico para adaptar modelos de linguagem pré-treinados a tarefas ou domínios específicos. Embora os modelos pré-treinados tenham recursos gerais impressionantes, eles geralmente precisam ser personalizados para se destacar em casos de usos específicos. O SFT preenche essa lacuna treinando ainda mais o modelo com conjuntos de dados cuidadosamente selecionados com exemplos validados por humanos.

## Entendendo o Ajuste Fino Supervisionado

Na sua essência, o ajuste fino supervisionado é sobre ensinar um modelo pré-treinado a executar tarefas específicas por meio de exemplos de tokens rotulados. O processo envolve mostrar muitos exemplos do comportamento desejado de input-output ao modelo, permitindo que ele aprenda os padrões específicos do seu caso de uso.

O SFT é eficaz porque usa o conhecimento fundamental adquirido durante o pré-treinamento, adaptando o comportamento do modelo para atender às suas necessidades específicas.

## Quando Usar o Ajuste Fino Supervisionado

A decisão de usar o SFT geralmente se resume à lacuna entre os recursos atuais do seu modelo e seus requisitos específicos. O SFT se torna particularmente valioso quando você precisa de controle preciso sobre os outputs do modelo ou ao trabalhar em domínios especializados.

Por exemplo, se você estiver desenvolvendo um aplicativo de atendimento ao cliente, você vai querer que seu modelo siga constantemente as diretrizes da empresa e lide com consultas técnicas de maneira padronizada. Da mesma forma, em aplicações médicas ou legais, a precisão e a adesão à terminologia específica do domínio se tornam cruciais. Nesses casos, o SFT pode ajudar a alinhar as respostas do modelo com padrões profissionais e experiência no campo de trabalho.

## O Processo de Ajuste Fino

O processo de ajuste fino supervisionado envolve o treinamento dos pesos do modelo em um conjunto de dados de tarefa específico. 

Primeiro, você precisará preparar ou selecionar um conjunto de dados que represente sua tarefa. Esse conjunto de dados deve incluir diversos exemplos que cobrem a gama de cenários que seu modelo encontrará. A qualidade desses dados é importante - cada exemplo deve demonstrar o tipo de output que você deseja que seu modelo produza. Em seguida, vem a fase real de ajuste fino, onde você usará estruturas como os módulos do Hugging Face, `transformers` e `trl`, para treinar o modelo no seu conjunto de dados. 

Ao longo do processo, a avaliação contínua é essencial. Você vai querer monitorar o desempenho do modelo em um conjunto de validação para garantir que ele esteja aprendendo os comportamentos desejados sem perder suas capacidades gerais. No [Módulo 4](../4_evaluation), abordaremos como avaliar seu modelo.

## O Papel do SFT no Alinhamento de Preferência

O SFT desempenha um papel fundamental no alinhamento de modelos de linguagem com preferências humanas. Técnicas como o aprendizado de reforço com o feedback humano (RLHF - Reinforcement Learning with Human Feedback) e a otimização de preferência direta (DPO - Direct Preference Optimization) dependem do SFT para formar um nível básico de entendimento da tarefa antes de alinhar ainda mais as respostas do modelo com os resultados desejados. Modelos pré-treinados, apesar de sua proficiência em linguagem geral, nem sempre podem gerar resultados que correspondam às preferências humanas. O SFT preenche essa lacuna introduzindo dados e orientações específicos de domínio, o que melhora a capacidade do modelo de gerar respostas que se alinham mais de perto com as expectativas humanas.

## Ajuste Fino Supervisionado com Aprendizado de Reforço de Transformadores

Um pacote de software importante para ajuste fino supervisionado é o aprendizado de reforço de transformadores (TRL - Transformer Reinforcement Learning). TRL é um kit de ferramentas usado para treinar modelos de linguagem de transformação usando o aprendizado de reforço.

Construído em cima do módulo de transformadores do Hugging Face, o TRL permite que os usuários carreguem diretamente modelos de linguagem pré-treinados e suporta a maioria das arquiteturas decodificadoras e codificador-decodificador. O módulo facilita os principais processos de RL usados ​​na modelagem de linguagem, incluindo ajuste fino supervisionado (SFT), modelagem de recompensa (RM - Reward Modeling), otimização de políticas proximais (PPO - Proximal Policy Optimization) e otimização de preferência direta (DPO). Usaremos o TRL em vários módulos ao longo deste repositório.

# Próximos passos

Experimente os seguintes tutoriais para obter experiência com o SFT usando TRL:

⏭️ [Tutorial de modelos de bate-papo](./notebooks/chat_templates_example.ipynb)

⏭️ [Tutorial de ajuste fino supervisionado](./notebooks/sft_finetuning_example.ipynb)