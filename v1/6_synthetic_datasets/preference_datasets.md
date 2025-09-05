# Generating Preference Datasets

Within [the chapter on preference alignment](../2_preference_alignment/README.md), we learned about Direct Preference Optimization. In this section, we will explore how to generate preference datasets for methods like DPO. We will build on top of the methods that were introduced in [generating instruction datasets](./instruction_datasets.md). Additionally, we will show how to add extra completions to the dataset using basic prompting or by using EvolQuality to improve the quality of responses. Lastly, we will show how UltraFeedback can be used to generate scores and critiques.

## Creating multiple completions

Preference data is a dataset with multiple `completions` for the same `instruction`. We can add more `completions` to a dataset by prompting a model to generate them. When doing this, we need to ensure that the second completion is not too similar to the first completion in terms of overall quality and phrasing. This is important because the model needs to be optimized for a clear preference. We want to know which completion is preferred over the other, normally referred to as `chosen` and `rejected`. We will go into more detail about determining chosen and rejected completions in the [section on creating scores](#creating-scores).

### Model pooling

You can use models from different model families to generate a second completion, which is called model pooling. To further improve the quality of the second completion, you can use different generation arguments, like tweaking the `temperature`. Lastly, you can use different prompt templates or system prompts to generate a second completion to ensure diversity based on specific characteristics defined in the template. In theory, we could take two models of varying quality and use the better one as the `chosen` completion.

Let's start with model pooling by loading the [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) models using the `transformers` integration of the `distilabel` library. Using these models, we will create two synthetic `responses` for a given `prompt`. We will create another pipeline with `LoadDataFromDicts`, `TextGeneration`, and `GroupColumns`. We will first load data, then use two generation steps, and then group the results. We connect the steps and flow the data through the pipeline using the `>>` operator and `[]`, which means that we want to use the output of the previous step as the input for both steps within the list.

```python
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    data = LoadDataFromDicts(data=[{"instruction": "What is synthetic data?"}])
    llm_a = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    gen_a = TextGeneration(llm=llm_a)
    llm_b = TransformersLLM(model="Qwen/Qwen2.5-1.5B-Instruct")
    gen_b = TextGeneration(llm=llm_b)
    group = GroupColumns(columns=["generation"])
    data >> [gen_a, gen_b] >> group

if __name__ == "__main__":
    distiset = pipeline.run()
    print(distiset["default"]["train"]["grouped_generation"][0])
# {[
#   'Synthetic data is artificially generated data that mimics real-world usage.',
#   'Synthetic data refers to data that has been generated artificially.'
# ]}
```

As you can see, we have two synthetic `completions` for the given `prompt`. We could have boosted diversity by initializing the `TextGeneration` steps with a specific `system_prompt` or by passing generation arguments to the `TransformersLLM`. Let's now see how we can improve the quality of the `completions` using EvolQuality.

### EvolQuality

EvolQuality is similar to [EvolInstruct](./instruction_datasets.md#evolinstruct) - it is a prompting technique but it evolves `completions` instead of the input `prompt`. The task takes both a `prompt` and `completion` and evolves the `completion` into a version that better responds to the `prompt` based on a set of criteria. This better version is defined according to criteria for improving helpfulness, relevance, deepening, creativity, or details. Because this automatically generates a second completion, we can use it to add more `completions` to a dataset. In theory, we could even assume the evolution is better than the original completion and use it as the `chosen` completion out of the box.

The prompt is [implemented in distilabel](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/evol_quality) and a simplified version is shown below:

```bash
I want you act as a Response Rewriter.
Given prompt a and a response, rewrite the response into a better version.
Complicate the prompt based on the following criteria:
{{ criteria }}

# Prompt
{{ input }}

# Response
{{ output }}

# Improved Response
```

Let's use the [EvolQuality class](https://distilabel.argilla.io/dev/components-gallery/tasks/evolquality/) to evolve the synthetic `prompt` and `completion` from [the Model Pooling section](#model-pooling) into a better version. For this example, we will only evolve for one generation.

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import EvolQuality

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
evol_quality = EvolQuality(llm=llm, num_evolutions=1)
evol_quality.load()

instruction = "What is synthetic data?"
completion = "Synthetic data is artificially generated data that mimics real-world usage."

next(evol_quality.process([{
    "instruction": instruction,
    "response": completion
}]))
# The process of generating synthetic data through manual prompting involves creating artificial data sets that mimic real-world usage patterns.
```

The `response` is now more complex and specific to the `instruction`. This is a good start, but as we have seen with EvolInstruct, evolved generations are not always better. Hence, it is important to use additional evaluation techniques to ensure the quality of the dataset. We will explore this in the next section.

## Creating Scores

Scores are a measure of how much one response is preferred over another. In general, these scores can be absolute, subjective, or relative. For this course, we will focus on the first two because they are most valuable for creating preference datasets. This scoring is a way of judging and evaluating using language models and therefore has some overlap with the evaluation techniques we have seen in [the chapter on evaluation](../3_evaluation/README.md). As with the other evaluation techniques, scores and evaluations normally require larger models to better align with human preferences.

### UltraFeedback

UltraFeedback is a technique that generates scores and critiques for a given `prompt` and its `completion`.

The scores are based on the quality of the `completion` according to a set of criteria. There are four fine-grained criteria: `helpfulness`, `relevance`, `deepening`, and `creativity`. These are useful but generally speaking, using the overall criteria is a good start, which allows us to simplify the process of generating scores. The scores can be used to determine which `completion` is the `chosen` and which is the `rejected` one. Because they are absolute, they can also be used as interesting filters for outliers in the dataset, either finding the worst completions or the pairs with more or less difference.

The critiques are added to provide reasoning for the score. They can be used as extra context to help us understand the differences between the scores. The language model generates extensive critiques which is very useful, but this also introduces extra cost and complexity to the process because generating critiques is more expensive than generating a single token to represent a score.

The prompt is [implemented in distilabel](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates/ultrafeedback) and a simplified version is shown below:

```bash
Evaluate the model's outputs based on various criteria: Helpfulness, Relevance, Deepening, Creativity
Your role is to provide a holistic assessment based on the above factors.
Score the output from 1 to 5 on overall quality.

Answer with the following format: score - rationale

# Input
{{ input }}

# Response
{{ output }}

# Score - Rationale
```

Let's use the [UltraFeedback class](https://distilabel.argilla.io/dev/components-gallery/tasks/ultrafeedback/) to evaluate the synthetic `prompt` and `completion` from [the Model Pooling section](#model-pooling).

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import UltraFeedback

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
ultrafeedback = UltraFeedback(llm=llm)
ultrafeedback.load()

instruction = "What is synthetic data?"
completion_a = "Synthetic data is artificially generated data that mimics real-world usage."
completion_b = "Synthetic data refers to data that has been generated artificially."

next(ultrafeedback.process([{
    "instruction": instruction,
    "generations": [completion_a, completion_b]
}]))
# [
#     {
#         'ratings': [4, 5],
#         'rationales': ['could have been more specific', 'good definition'],
#     }
# ]
```

## Best Practices

- Overall scores are cheaper and easier to generate than critiques and specific scores
- Use bigger models to generate scores and critiques
- Use a diverse set of models to generate scores and critiques
- Iterate on configuration of the `system_prompt` and models

## Next Steps

👨🏽‍💻 Code -[Exercise Notebook](./notebooks/preference_dpo_dataset.ipynb) to generate a dataset for instruction tuning

## References

- [Distilabel Documentation](https://distilabel.argilla.io/latest/)
- [Deita](https://arxiv.org/abs/2312.15685)
- [UltraFeedback](https://arxiv.org/abs/2310.01377)
