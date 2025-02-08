# 선호도 데이터셋 생성하기

[선호도 정렬 챕터](../2_preference_alignment/README.md)에서 DPO(Direct Preference Optimization)에 대해서 배웠습니다. 이번 섹션에서는 DPO와 같은 방법을 위한 선호도 데이터셋을 생성하는 방법을 살펴보겠습니다. [인스트럭션 튜닝 데이터셋 생성하기](./instruction_datasets.md)에서 소개된 방법을 기반으로 구축할 것입니다. 또한, 우리는 기본적인 프롬프팅을 사용하거나 응답 품질을 향상시키기 위해 EvolQuality를 사용하여 데이터셋에 추가적인 완성을 더하는 방법을 보여드릴 것입니다. 마지막으로, 점수와 비평을 생성하는 데 UltraFeedback이 어떻게 사용되는지를 보여 드리겠습니다.

## 다중 완성 생성하기

선호도 데이터는 동일한 `instruction`에 대하여 여러 개의 `completions`이 있는 데이터셋입니다. 모델에게 `competions`을 생성하도록 프롬프팅함으로써 하나의 데이터셋에 대하여 완성을 추가할 수 있습니다. 이때, 두 번재 완성이 전반적인 품질과 문구 측면에서 첫 번째 완성과 너무 유사하지 않도록 해야 할 필요가 있습니다. 이는 모델이 명확한 선호도에 맞게 최적화되야 하기 때문에 중요합니다. 일반적으로 `chosen`과 `rejected`라고 말하는, 어떤 완성이 다른 것보다 선호되는지 알고 싶습니다. [점수 생성하기 섹션](#점수-생성하기)에서 선택 및 거절되는 완성을 결정하는 것에 대해서 자세히 설명하겠습니다.

### 모델 풀링

여러분은 서로 다른 모델 군의 모델을 사용하여 두 번째 완성을 생성할 수 있는데, 이를 모델 풀링이라고 합니다. 두 번째 완성의 품질을 더욱 향상시키기 위해 `temperature`를 조정하는 것과 같이 다양한 생성 인자를 사용할 수 있습니다. 마지막으로, 템플릿에 정의된 구체적인 특성을 바탕으로 다양성을 보장하기 위해 다른 프롬프트 템플릿이나 시스템 프롬프트를 사용하여 두 번째 완성을 생성할 수 있습니다. 이론적으로 서로 다른 품질의 두 모델을 가지고 더 나은 모델을 `chosen` 완성으로 사용할 수 있습니다.

`distilabel` 라이브러리의 `transformers` 통합을 사용하여 [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)와 [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) 모델을 로드하여 모델 풀링을 시작해 보겠습니다. 이러한 모델을 사용하여 주어진 `prompt`에 대하여 두 개의 합성 `responses`를 생성할 것입니다. `LoadDataFromDicts`, `TextGeneration` 및 `GroupColumns`를 사용하여 또 다른 파이프라인을 만들겠습니다. 먼저 데이터를 로드한 다음에 다음 두 가지 생성 단계를 사용한 이후, 결과를 그룹화합니다. 단계들을 연결하고 `>>` 연산자와 `[]`를 사용하여 파이프라인을 통해 데이터가 흐르게 하는데, 이는 이전 단계의 출력을 리스트 내에 두 단계의 입력으로 사용하고자 함을 의미합니다.

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

보시다시피 주어진 `prompt`에 대해 두 가지 합성 `completions`가 있습니다. 특정 `system_prompt`로 `TextGeneration` 단계를 초기화하거나 생성 인자를 `TransformersLLM`에 전달하여 다양성을 높일 수 있었을 것입니다. 이제 EvolQuality를 사용하여 `competions`의 품질을 개선할 수 있는 방법을 살펴보겠습니다.

### EvolQuality

EvolQuality는 입력 `prompt` 대신 `completions`를 진화시키는 프롬프트 기법이라는 점에서 [EvolInstruct](./instruction_datasets.md#evolinstruct)와 유사합니다. `prompt`와 `completion`을 모두 받아 일련의 기준에 따라 `prompt`에 더 잘 응답하는 버전으로 `completion`을 진화시킵니다. 이 더 나은 버전은 유용성, 관련성, 심화, 창의성 또는 세부 사항을 개선하기 위한 기준에 따라 정의됩니다. 이것은 자동적으로 두 번째 완성을 생성하기 때문에, 하나의 데이터셋에 대해 `completions`을 더 많이 추가하는 것이 추가하는 데 사용할 수 있습니다. 이론적으로 진화한 버전은 원래의 완성보다 더 낫다고 가정하고 이를 즉시 `chosen` 완성으로 사용할 수 있습니다.

프롬프트는 [distilabel에 구현](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/evol_quality)되어 있으며 단순화된 버전은 아래와 같습니다:

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

[EvolQuality 클래스](https://distilabel.argilla.io/dev/components-gallery/tasks/evolquality/)를 사용하여 [모델 풀링 섹션](#모델-풀링)의 합성 `prompt`와 `competion`을 더 나은 버전으로 진화시켜 봅시다. 이 예제는 하나의 생성 동안만 진화시킵니다.

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

이제 `response`는 더 복잡하고 구체적인 `instruction`에 국한됩니다. 이것은 좋은 시작이지만, EvolInstrucdt에서 보았듯이 진화한 생성이 항상 더 나은 것은 아닙니다. 따라서 데이터셋의 품질을 보장하기 위한 추가적안 평가 기법을 사용하는 것이 중요합니다. 다음 섹션에서 이에 대해 살펴보도록 하겠습니다.

## 점수 생성하기

점수는 하나의 응답이 다른 응답보다 얼마나 선호되는지를 나타내는 척도입니다. 일반적으로 이러한 점수는 절대적이며, 주관적이거나 상대적일 수 있습니다. 이 과정에서는 선호도 데이터셋을 만드는 데 가장 유용하기 때문에 처음 두 가지에 초점을 맞출 것입니다. 이러한 점수는 언어 모델을 사용하여 판단하고 평가하는 방법이므로 [평가 챕터](../3_evaluation/README.md)에서 살펴본 평가 기법과 일부 겹치는 부분이 있습니다. 다른 평가 기법과 마찬가지로, 점수 및 평가는 일반적으로 사람의 선호도에 더 잘 맞추기 위해 더 큰 모델을 필요로 합니다.

### UltraFeedback

UltraFeedback은 주어진 `prompt`와 그에 대한 `completion`의 점수와 비평을 생성하는 기법입니다.

점수는 일련의 기준에 따라 `completion`의 품질을 바탕으로 합니다. `유용성(helpfulness)`, `관련성(relevance)`, `심화도(deepening)`, `창의성(creativity)` 등 네 가지 세분화된 기준이 있습니다. 이러한 기준은 유용하지만 일반적으로 전반적인 기준을 사용하는 것이 좋은 출발점이며, 이를 통해 점수 생성 과정을 간소화할 수 있습니다. 점수를 사용하여 어떤 `completion`이 `chosen`이며 어떤 것이 `rejected`인지를 결정할 수 있습니다. 절대적인 점수이기 때문에 데이터셋의 이상치에 대한 흥미로운 필터로 사용할 수 있으며, 최악의 완성이나 차이가 많거나 적은 쌍을 찾는 데 사용할 수도 있습니다.

비평은 점수에 대한 추론을 제공하기 위해 추가됩니다. 비평은 점수 간의 차이를 이해하는 데 도움이 되는 추가 컨텍스트로 사용될 수 있습니다. 언어 모델은 매우 유용한 광범위한 비평을 생성하지만, 점수를 나타내는 단일 토큰을 생성하는 것보다 더 비용이 많이 들기 때문에 과정에 대한 추가적인 비용과 복잡성을 초래하기도 합니다.

이 프롬프트는 [distilabel에 구현](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates/ultrafeedback)되어 있으며 단순화된 버전은 아래와 같습니다:

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

[UltraFeedback 클래스](https://distilabel.argilla.io/dev/components-gallery/tasks/ultrafeedback/)를 사용하여 [모델 풀링 섹션](#모델-풀링)의 합성 `prompt`와 `completion`을 평가해 봅시다.

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

## 모범 사례

- 전반적인 점수는 비평과 특정 점수보다 더 저렴하고 쉽게 생성할 수 있습니다
- 점수 및 비평을 생성하기 위해 더 큰 모델을 사용합니다
- 다양한 모델을 사용하여 점수와 비평을 생성합니다.
- `system_prompt`와 모델의 구성을 반복합니다.

## 다음 단계

👨🏽‍💻 Code - 선호도 정렬을 위한 데이터셋을 생성하는 [실습 노트북](./notebooks/preference_dpo_dataset.ipynb)

## 참고

- [Distilabel Documentation](https://distilabel.argilla.io/latest/)
- [Deita](https://arxiv.org/abs/2312.15685)
- [UltraFeedback](https://arxiv.org/abs/2310.01377)
