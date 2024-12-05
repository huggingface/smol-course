![smolcourse image](./banner.png)

# a smol course

Il s'agit d'un cours pratique sur l'alignement des modèles linguistiques pour votre cas d'utilisation spécifique. C'est un moyen pratique de commencer à aligner des modèles de langage, car tout fonctionne sur la plupart des machines locales. Les besoins en GPU sont minimes et il n'y a pas de services payants. Le cours est basé sur la série de modèles [SmolLM2](https://github.com/huggingface/smollm/tree/main), mais vous pouvez transférer les compétences que vous apprenez ici à des modèles plus grands ou à d'autres petits modèles de langue.

<a href="http://hf.co/join/discord">
<img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
</a>

<div style="background: linear-gradient(to right, #e0f7fa, #e1bee7, orange); padding: 20px; border-radius: 5px; margin-bottom: 20px; color: purple;">
    <h2>La participation est ouverte, gratuite et immédiate!</h2>
    <p>Ce cours est ouvert et évalué par les pairs. Pour participer au cours <strong>ouvrir une demande de pull request</strong> t soumettez votre travail pour examen. Voici la marche à suivre :</p>
    <ol>
        <li>Fork le repo <a href="https://github.com/huggingface/smol-course/fork">ici</a></li>
        <li>Lisez le matériel, apportez des modifications, faites les exercices, ajoutez vos propres exemples.</li>
        <li>Ouvre un PR dans la branche december_2024</li>
        <li>Fais le réviser et fusionner (merged)</li>
    </ol>
    <p>Cela devrait t'aider à apprendre et à construire un cours axé sur la communauté qui s'améliore constamment.</p>
</div>

Nous pouvons discuter de ce processus dans ce document. [discussion thread](https://github.com/huggingface/smol-course/discussions/2#discussion-7602932).

## Plan du cours

Ce cours propose une approche pratique de l'utilisation de petits modèles linguistiques, depuis la formation initiale jusqu'au déploiement en production.

| Module | Description | Etat | Date de sortie |
|--------|-------------|---------|--------------|
| [Instruction Tuning](./1_instruction_tuning) | Apprendre les réglages fins supervisés (supervised fine-tuning), la création de modèles de chat (chat templating) et les instructions de base qui suivent. | ✅ Complet | Dec 3, 2024 |
| [Preference Alignment](./2_preference_alignment) | Explorer les techniques DPO et ORPO pour aligner les modèles sur les préférences humaines | 🚧 En Construction  | Dec 6, 2024 |
| [Parameter-efficient Fine-tuning](./3_parameter_efficient_finetuning) | Apprendre le LoRA, le réglage rapide et les méthodes d'adaptation efficaces | 🚧 En Construction | Dec 9, 2024 |
| [Evaluation](./4_evaluation) | Utiliser des critères de référence automatiques et créer des évaluations de domaine personnalisées | 🚧 En Construction | Dec 16, 2024 |
| [Vision-language Models](./5_vision_language_models) | Adapter les modèles multimodaux aux tâches vision-langage | 📝 Prévu | Dec 20, 2024 |
| [Synthetic Datasets](./6_synthetic_datasets) | Créer et valider des ensembles de données synthétiques pour le training | 📝 Prévu | Dec 23, 2024 |
| [Inference](./7_inference) | Inférer efficacement avec des modèles | 📝 Prévu | Dec 27, 2024 |
| [Deployment](./8_deplyment) | Déployer et servir des modèles à grande échelle | 📝 Prévu | Dec 30, 2024 |

## Pourquoi des petits modèles linguistiques ?

Bien que les modèles linguistiques de grande taille aient montré des capacités impressionnantes, ils nécessitent souvent des ressources informatiques importantes et peuvent être excessifs pour des applications ciblées. Les petits modèles linguistiques offrent plusieurs avantages pour les applications spécifiques à un domaine :

- **Efficacité** : La formation et le déploiement nécessitent beaucoup moins de ressources informatiques.
- **Personnalisation** : Plus facile à affiner(fine-tune) et à adapter à des domaines spécifiques
- **Contrôle** : Meilleure compréhension et contrôle du comportement du modèle
- **Coût** : Coûts opérationnels réduits pour le training et l'inférence
- **Confidentialité** : Peut être exécuté localement sans envoyer de données à des API externes
- **Technologie verte** : utilisation efficace des ressources et réduction de l'empreinte carbone
- **Développement plus facile de la recherche universitaire** : Facilite le démarrage de la recherche universitaire avec des LLM de pointe et des contraintes logistiques moindres.

## Conditions préalables

Avant de commencer, assurez-vous de disposer des éléments suivants
- Compréhension de base de l'apprentissage automatique et du traitement du langage naturel.
- Familiarité avec Python, PyTorch et la bibliothèque `transformers`.
- Accès à un modèle de langage pré-entraîné et à un ensemble de données étiquetées.

## Installation

Nous maintenons le cours sous forme de paquetage afin que vous puissiez installer facilement les dépendances via un gestionnaire de paquetage. Nous recommandons [uv](https://github.com/astral-sh/uv) à cette fin, mais vous pouvez utiliser des alternatives comme `pip` ou `pdm`.

### Utilisation de `uv`

Avec `uv` installé, vous pouvez installer le cours comme suit:

```bash
uv venv --python 3.11.0
uv sync
```

### Utilisation de `pip`

Tous les exemples fonctionnent dans le même environnement **python 3.11**, vous devez donc créer un environnement et installer les dépendances comme ceci :

```bash
# python -m venv .venv
# source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

**From Google Colab** vous devrez installer les dépendances de manière flexible en fonction du matériel que vous utilisez. Comme ceci :

```bash
pip install -r transformers trl datasets huggingface_hub
```

## Engagement

Partageons-le, afin que de nombreuses personnes puissent apprendre à affiner les LLM sans matériel coûteux.

[![Star History Chart](https://api.star-history.com/svg?repos=huggingface/smol-course&type=Date)](https://star-history.com/#huggingface/smol-course&Date)
