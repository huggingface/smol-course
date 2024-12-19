# Parameter-Efficient Fine-Tuning, PEFT

언어 모델이 커지면서 전통적인 미세 조정 방식을 적용하는 것이 점점 어려워지고 있습니다. 1.7B 모델조차도 전체 미세 조정을 수행하려면 상당한 GPU 메모리가 필요하며, 모델 사본을 별도로 저장하기 위한 비용이 많이 들고, 모델의 원래 능력을 상실하는 위험이 존재합니다. Parmeter-Efficient Fine-Tuning(PEFT) 방법은 대부분의 모델 파라미터가 고정된 상태에서 모델 파라미터의 일부만 수정하여 전체 미세 조정 과정에서 발생하는 문제를 해결헙니다.

학습 과정에서 모델의 모든 파라미터를 업데이트하는 전통적인 미세 조정 방법을 대형 언어 모델에 적용하는 것은 현실적으로 어렵습니다. PEFT는 원래 모델 크기의 1% 미만에 해당하는 파라미터만 학습시켜 모델을 조정하는 방법입니다. 학습 가능한 파라미터를 크게 줄이는 것은 다음과 같은 이점을 제공합니다:

- 제한된 GPU 메모리를 가진 하드웨어에서도 미세 조정 가능
- 효율적인 태스크별 적응 모델 저장
- 데이터가 적은 상황에서도 뛰어난 일반화 성능 제공
- 더 빠른 학습 및 반복 가능

## 사용 가능한 방법

이 모듈에서는 많이 사용되는 두 가지 PEFT 방법을 다룹니다:

### 1️⃣ LoRA (Low-Rank Adaptation)

LoRA는 효율적인 모델 적응을 위한 멋진 솔루션을 제공하면서 가장 많이 사용되는 PEFT 방법으로 자리 잡았습니다. LoRA는 전체 모델을 수정하는 대신 **학습 가능한 파라미터를 모델의 어텐션 레이어에 주입**합니다. 이 접근법은 전체 미세 조정과 비슷한 성능을 유지하면서 학습 가능한 파라미터를 약 90%까지 줄입니다. [LoRA (Low-Rank Adaptation)](./lora_adapters.md) 섹션에서 LoRA에 대해 자세히 알아보겠습니다.
 
### 2️⃣ 프롬프트 튜닝

프롬프트 튜닝은 모델 가중치를 수정하는 대신 **입력에 학습 가능한 토큰을 추가**하여 **더 경량화된** 접근법을 제공합니다. 프롬프트 튜닝은 LoRA만큼 유명하지는 않지만, 모델을 새로운 태스크나 도메인에 빠르게 적용할 때 유용하게 쓰일 수 있는 기술입니다. [프롬프트 튜닝](./prompt_tuning.md) 섹션에서 프롬프트 튜닝에 대해 탐구해볼 예정입니다.

## 실습 노트북

| 파일명 | 설명 | 실습 내용 | 링크 | Colab |
|-------|-------------|----------|------|-------|
| LoRA Fine-tuning | LoRA 어댑터를 사용해 모델을 미세 조정하는 방법 학습 | 🐢 LoRA를 사용해 모델 학습해보기<br>🐕 다양한 랭크 값으로 실험해보기<br>🦁 전체 미세 조정과 성능 비교해보기 | [Notebook](./notebooks/finetune_sft_peft.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/finetune_sft_peft.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Load LoRA Adapters | LoRA 어댑터를 불러오고 학습시키는 방법 배우기 | 🐢 사전 학습된 어댑터 불러오기<br>🐕 기본 모델과 어댑터 합치기<br>🦁 여러 어댑터 간 전환해보기 | [Notebook](./notebooks/load_lora_adapter_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/load_lora_adapter_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
<!-- | Prompt Tuning | Learn how to implement prompt tuning | 🐢 Train soft prompts<br>🐕 Compare different initialization strategies<br>🦁 Evaluate on multiple tasks | [Notebook](./notebooks/prompt_tuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/prompt_tuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | -->

## 참고
- [Hugging Face PEFT 문서](https://huggingface.co/docs/peft)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [QLoRA 논문](https://arxiv.org/abs/2305.14314)
- [프롬프트 튜닝 논문](https://arxiv.org/abs/2104.08691)
- [Hugging Face PEFT 가이드](https://huggingface.co/blog/peft)
- [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) 
- [TRL](https://huggingface.co/docs/trl/index)