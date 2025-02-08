# 합성 데이터셋

합성 데이터는 현실 세계의 사용을 모방하는 인위적으로 생성된 데이터입니다. 이를 통해 데이터셋을 확장하거나 향상함으로써 데이터 한계를 극복할 수 있습니다. 비록 합성 데이터가 이미 일부 용례에 사용되고 있을지라도, 대규모 언어 모델은 합성 데이터셋이 사전(pre-) 및 사후(post) 학습과 언어 모델의 평가에 더 많이 사용되도록 하였습니다.

검증된 연구 논문을 바탕으로 빠르고 안정적이며 확장 가능한 파이프라인이 필요한 엔지니어를 위한 합성 데이터와 AI 피드백에 대한 프레임워크인 [`distilabel`](https://distilabel.argilla.io/latest/)을 사용할 것입니다. 패키지와 모범 사례에 대해서 자세히 살펴보고자 하시면 [문서](https://distilabel.argilla.io/latest/)를 확인해주세요.

## 개요

언어 모델에 대한 합성 데이터는 3가지 분류체계로 범주화될 수 있습니다: 인스트럭션(instructions), 선호도(preferences), 비평(critiques). 우리는 처음 두 가지 범주에 집중할 것이며, 즉 인스트럭션 튜닝과 선호도 조정을 위한 데이터셋 생성에 초점을 맞출 것입니다. 두 범주 모두에서 모델의 비평과 재작성을 통해 기존 데이터를 개선하는 데 중점을 두는 세 번째 범주의 측면을 다룰 것입니다.

![Synthetic Data Taxonomies](./images/taxonomy-synthetic-data.png)

## 내용

### 1. [인스트럭션 데이터셋(Instruction Datasets)](./instruction_datasets.md)

인스트럭션 튜닝(instruction tuning)을 위한 인스트럭션 데이터셋을 생성하는 방법을 배워보세요. 기본적인 프롬프팅을 통해 인스트럭션 튜닝 데이터셋을 만들고 논문을 통해 보다 프롬프트 기술을 사용하는 방법을 살펴봅니다. 인컨텍스트 러닝(in-context learning)을 위한 시드 데이터를 가지고 인스트럭션 튜닝 데이터셋은 SelfInstruct와 Magpie와 같은 방법을 통해 생성될 수 있습니다. 또한, EvalInstrct를 통한 인스트럭션의 진화를 살펴볼 것입니다. [학습 시작하기](./instruction_datasets.md).


### 2. [선호도 데이터셋(Preference Datasets)](./preference_datasets.md)

선호도 정렬(preference aligment)을 위한 선호도 데이터셋을 생성하는 방법을 배워보세요. 추가 응답을 생성하여 섹션 1에서 소개한 방법과 기법을 바탕으로 구축할 것입니다. 다음으로, EvolQuality 프롬프트를 사용하여 이러한 응답을 개선하는 방법을 배웁니다. 미지막으로, 점수와 비평을 생성하여 선호도 쌍을 생성할 수 있는 UltraFeedback 프롬프트를 사용하여 응답을 평가하는 방법을 살펴봅니다. [학습 시작하기](./preference_datasets.md).

### 실습 노트북


| 제목 | 설명 | 실습 | 링크 | Colab |
|-------|-------------|----------|------|-------|
| 인스트럭션 데이터셋 | 인스트럭션 튜닝을 위한 데이터셋 생성하기 | 🐢 인스트럭션 튜닝 데이터셋 생성하기 <br> 🐕 시드 데이터로 인스트럭션 튜닝을 위한 데이터셋 생성하기 <br> 🦁 시드 데이터와 인스트럭션 진화를 통해 인스트럭션 튜닝을 위한 데이터셋 생성하기 | [Link](./notebooks/instruction_sft_dataset.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/6_synthetic_datasets/notebooks/instruction_sft_dataset.ipynb) |
| 선호도 데이터셋 | 선호도 정렬을 위한 데이터셋 생성하기 | 🐢 선호도 정렬 데이터셋 생성하기 <br> 🐕 응답 진화를 통해 선호도 정렬 데이터셋 생성하기 <br> 🦁 응답 진화와 비평을 통해 선호도 정렬 데이터셋 생성하기  | [Link](./notebooks/preference_alignment_dataset.ipynb) | [Colab](https://githubtocolab.com/huggingface/smol-course/tree/main/6_synthetic_datasets/notebooks/preference_alignment_dataset.ipynb) |

## 자료

- [Distilabel Documentation](https://distilabel.argilla.io/latest/)
- [Synthetic Data Generator is UI app](https://huggingface.co/blog/synthetic-data-generator)
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)
- [Self-instruct](https://arxiv.org/abs/2212.10560)
- [Evol-Instruct](https://arxiv.org/abs/2304.12244)
- [Magpie](https://arxiv.org/abs/2406.08464)
- [UltraFeedback](https://arxiv.org/abs/2310.01377)
- [Deita](https://arxiv.org/abs/2312.15685)
