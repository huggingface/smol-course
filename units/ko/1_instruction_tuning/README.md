# 지시 조정(Instruction Tuning)

이 모듈에서는 언어 모델의 지시 조정(instruction tuning) 방법을 설명합니다. 지시 조정이란, 사전 학습된 모델을 특정 태스크에 맞게 조정하기 위해 해당 태스크와 관련된 데이터셋으로 추가 학습시키는 과정을 의미합니다. 이를 통해 목표로 하는 작업에서 모델이 더 나은 성능을 발휘할 수 있습니다.

이 모듈에서는 다음 두 가지 주제를 다룰 예정입니다: 1) 대화 템플릿(Chat Templates) 2) 지도 학습 기반 미세 조정(Supervised Fine-Tuning)

## 1️⃣ 대화 템플릿

채팅 템플릿(Chat Templates)은 사용자와 AI 모델 간의 상호작용을 구조화하여 모델의 일관되고 적절한 맥락의 응답을 보장합니다. 템플릿에는 시스템 프롬프트와 역할 기반 메시지와 같은 구성 요소가 포함됩니다. 더 자세한 내용은 [대화 템플릿](./chat_templates.md) 섹션을 참고하세요.

## 2️⃣ 지도 학습 기반 미세 조정

지도 학습 기반 미세 조정(SFT)은 사전 학습된 언어 모델이 특정 작업에 적합하도록 조정하는 데 핵심적인 과정입니다. 이 과정에서는 레이블이 포함된 태스크별 데이터셋을 사용해 모델을 학습시킵니다. SFT의 주요 단계와 모범 사례를 포함한 자세한 가이드는 [지도 학습 기반 미세 조정](./supervised_fine_tuning.md) 섹션을 참고하세요.

## 참고

- [Transformers documentation on chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Script for Supervised Fine-Tuning in TRL](https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py)
- [`SFTTrainer` in TRL](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [Supervised Fine-Tuning with TRL](https://huggingface.co/docs/trl/main/en/tutorials/supervised_finetuning)
- [How to fine-tune Google Gemma with ChatML and Hugging Face TRL](https://www.philschmid.de/fine-tune-google-gemma)
- [Fine-tuning LLM to Generate Persian Product Catalogs in JSON Format](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_to_generate_persian_product_catalogs_in_json_format)