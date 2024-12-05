# Modèles de conversation

Les modèles de conversation sont essentiels pour structurer les interactions entre les modèles linguistiques et les utilisateurs. Ils fournissent un format cohérent pour les conversations, garantissant que les modèles comprennent le contexte et le rôle de chaque message tout en maintenant des modèles de réponse appropriés.

## Modèles de base et modèles d'apprentissage

Un modèle de base(base model) est entraîné sur des données textuelles brutes pour prédire le prochain élément, tandis qu'un modèle d'instruction(instruct model) est affiné spécifiquement pour suivre des instructions et engager des conversations. Par exemple, `SmolLM2-135M` est un modèle de base, tandis que `SmolLM2-135M-Instruct` est sa variante adaptée aux instructions.

Pour qu'un modèle de base se comporte comme un modèle d'instruction, nous devons formater nos invites d'une manière cohérente que le modèle peut comprendre. C'est là qu'interviennent les modèles de chat(chat templates). ChatML est un de ces modèles qui structure les conversations avec des indicateurs de rôle clairs (système, utilisateur, assistant).

Il est important de noter qu'un modèle de base peut être affiné sur différents modèles de conversation, de sorte que lorsque nous utilisons un modèle d'instruction, nous devons nous assurer que nous utilisons le bon modèle de conversation.

## Comprendre les modèles de conversation

À la base, les modèles de conversation définissent la manière dont les conversations doivent être formatées lors de la communication avec un modèle linguistique. Ils incluent les instructions au niveau du système, les messages de l'utilisateur et les réponses de l'assistant dans un format structuré que le modèle peut comprendre. Cette structure permet de maintenir la cohérence entre les interactions et de s'assurer que le modèle répond de manière appropriée aux différents types d'entrées. Vous trouverez ci-dessous un exemple de modèle de chat :

```sh
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

La bibliothèque `transformers` s'occupera des modèles de chat pour vous en relation avec le tokenizer du modèle. Pour en savoir plus sur la façon dont transformers construit les modèles de chat [ici](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates). Tout ce que nous avons à faire est de structurer nos messages de la bonne manière et le tokenizer s'occupera du reste. Voici un exemple basique de conversation :

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."}
]
```

Décomposons l'exemple ci-dessus et voyons comment il correspond au format du modèle de chat.

## Messages du système

Les messages système posent les bases du comportement du modèle. Ils agissent comme des instructions persistantes qui influencent toutes les interactions ultérieures. Par exemple :

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

## Conversations

Les modèles de conversation conservent le contexte grâce à l'historique des conversations, en stockant les échanges précédents entre les utilisateurs et l'assistant. Cela permet d'avoir des conversations plus cohérentes à plusieurs tours :

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

## Mise en œuvre avec les transformateurs

La bibliothèque de transformateurs fournit un support intégré pour les modèles de chat. Voici comment les utiliser :

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to sort a list"},
]

# Apply the chat template
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## Formatage personnalisé
Vous pouvez personnaliser le formatage des différents types de messages. Par exemple, en ajoutant des jetons spéciaux ou un formatage pour différents rôles :

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## Prise en charge des conversations à plusieurs tours

Les modèles peuvent gérer des conversations complexes à plusieurs tours tout en conservant le contexte :

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```

⏭️ [Suivant: Fine-Tuning Supervisé](./supervised_fine_tuning.md)

## Resources

- [Hugging Face Chat Templating Guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Chat Templates Examples Repository](https://github.com/chujiezheng/chat_templates) 
