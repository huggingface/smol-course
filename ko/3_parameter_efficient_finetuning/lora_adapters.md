# LoRA (Low-Rank Adaptation)

LoRA는 널리 쓰이는 PEFT 방법으로 자리 잡았습니다. 어텐션 가중치에 작은 랭크 분해 행렬을 추가하는 방식으로 동작작하며 일반적으로 학습 가능한 파라미터를 약 90% 줄여줍니다.

## LoRA 이해하기

LoRA(Low-Rank Adaptation)는 사전 학습된 모델 가중치를 고정한 상태에서 학습 가능한 랭크 분해 행렬을 모델 레이어에 주입하는 파라미터 효율적인 미세 조정 기법입니다. 미세 조정 과정에서 모든 모델 파라미터를 학습시키는 대신, LoRA는 저랭크 분해를 통해 가중치 업데이트를 더 작은 행렬로 나눠 모델 성능은 유지하면서 학습 가능한 파라미터 수를 크게 줄입니다. 예를 들어, GPT-3 175B에 LoRA를 적용했을 때 전체 미세 조정 대비 학습 가능한 파라미터 수는 10,000배, GPU 메모리 요구 사항은 3배 감소했습니다. [LoRA 논문](https://arxiv.org/pdf/2106.09685)에서 LoRA에 관한 자세한 내용을 확인할 수 있습니다.

LoRA는 일반적으로 트랜스포머 레이어 중 어텐션 가중치에 랭크 분해 행렬 쌍을 추가하는 방식으로 동작합니다. 어댑터 가중치는 추론 과정에서 기본 모델과 병합될 수 있고 추가적인 지연 시간이 발생하지 않습니다. LoRA는 자원 요구 사항을 적절한 수준으로 유지하면서 대형 언어 모델을 특정 태스크나 도메인에 맞게 조정하는 데 특히 유용합니다.

## LoRA 어댑터 불러오기

load_adapter()를 사용하여 사전 학습된 모델에 어댑터를 불러올 수 있으며 가중치가 병합되지 않은 다른 어댑터를 사용해 볼 때 유용합니다. set_adapter() 함수로 활성 어댑터 가중치를 설정합니다. 기본 모델을 반환하려면 unload()를 사용하여 불러온 모든 LoRA 모듈을 내릴 수 있습니다. 이렇게 하면 태스크별 가중치를 쉽게 전환할 수 있습니다.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("<base_model_name>")
peft_model_id = "<peft_adapter_id>"
model = PeftModel.from_pretrained(base_model, peft_model_id)
```

![lora_load_adapter](./images/lora_adapter.png)

## LoRA 어댑터 병합

LoRA로 학습한 후에는 더 쉬운 배포를 위해 어댑터 가중치를 기본 모델에 다시 병합할 수 있습니다. 이를 통해 결합된 가중치를 가진 단일 모델을 생성할 수 있기 때문에 추론 과정에서 별도로 어댑터를 불러올 필요가 없습니다.

병합 과정에서는 메모리 관리와 정밀도에 주의해야 합니다. 기본 모델과 어댑터 가중치를 동시에 불러와야 하므로 GPU/CPU 메모리가 충분해야 합니다. `transformers`의 `device_map="auto"`를 사용하면 메모리를 자동으로 관리할 수 있습니다. 학습 중 사용한 정밀도(예: float16)를 병합 과정에서도 일관되게 유지하고, 병합된 모델을 같은 형식으로 저장하여 배포하세요. 배포 전에 반드시 병합된 모델의 출력 결과와 성능 지표를 어댑터 기반 버전과 비교하여 검증해야 합니다.

어댑터는 서로 다른 태스크나 도메인 간 전환도 간편하게 만듭니다. 기본 모델과 어댑터 가중치를 별도로 불러오면 빠르게 태스크별 가중치를 전환할 수 있습니다.

## 구현 가이드

`notebooks/` 디렉토리에는 다양한 PEFT 방법을 구현하기 위한 실용적인 튜토리얼과 예제가 포함되어 있습니다. `load_lora_adapter_example.ipynb`에서 기본 소개를 살펴본 다음, `lora_finetuning.ipynb`를 통해 LoRA와 SFT를 사용한 모델 미세 조정 과정을 더 자세히 탐구해 보세요.

PEFT 방법을 구현할 때는 LoRA의 랭크를 4~8 정도의 작은 값으로 설정하고 학습 손실을 지속적으로 모니터링하는 것이 좋습니다. 과적합을 방지하기 위해 검증 세트를 활용하고 가능하다면 전체 미세 조정 기준선과 결과를 비교하세요. 다양한 태스크에서 각 방법의 효과는 다를 수 있으므로 실험을 통해 최적의 방법을 찾는 것이 중요합니다.

## OLoRA

[OLoRA](https://arxiv.org/abs/2406.01775)는 LoRA 어댑터 초기화를 위해 QR 분해를 활용합니다. OLoRA는 모델의 기본 가중치를 QR 분해 계수에 따라 변환합니다. 즉, 모델 학습 전에 가중치를 변경합니다. 이 접근 방식은 안정성을 크게 향상시키고 수렴 속도를 빠르게 하여, 궁극적으로 더 우수한 성능을 달성합니다.

## PEFT와 함께 TRL 사용하기

효율적인 미세 조정을 위해 PEFT 방법을 TRL(Transformers Reinforcements Learning)과 결합할 수 있습니다. 이러한 통합은 메모리 요구사항을 줄여주기 때문에 RLHF (Reinforcement Learning from Human Feedback)에 특히 유용합니다.

```python
from peft import LoraConfig
from transformers import AutoModelForCausalLM

# PEFT configuration 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 특정 디바이스에서 모델 불러오기
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    load_in_8bit=True,  # 선택 사항: 8비트 정밀도 사용
    device_map="auto",
    peft_config=lora_config
)
```

위 코드에서 `device_map="auto"`를 사용해 모델을 적절한 디바이스에 자동으로 할당했습니다. `device_map={"": device_index}`를 써서 모델을 특정 디바이스에 직접 할당할 수도 있습니다. 또한, 메모리 사용량을 효율적으로 유지하면서 여러 GPU에 걸쳐 학습을 확장할 수도 있습니다.

## 기본적인 병합 구현

LoRA 어댑터 학습이 끝나면 어댑터 가중치를 기본 모델에 합칠 수 있습니다. 합치는 방법은 다음과 같습니다:

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. 기본 모델 불러오기
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 어댑터가 있는 PEFT 모델 불러오기
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. 어댑터 가중치를 기본 모델에 병합하기
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Merging failed: {e}")
    # fallback 전략 또는 메모리 최적화 구현

# 4. 병합된 모델 저장
merged_model.save_pretrained("path/to/save/merged_model")
```

저장된 모델의 크기가 일치하지 않으면 토크나이저도 함께 저장했는지 확인하세요:

```python
# 모델과 토크나이저를 모두 저장
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

## 다음 단계

⏩ [프롬프트 튜닝](prompt_tuning.md) 가이드로 이동해 프롬프트 튜닝으로 미세 조정하는 법을 배워보세요.
⏩ [LoRA 어댑터 튜토리얼](./notebooks/load_lora_adapter.ipynb)에서 LoRA 어댑터를 불러오는 방법을 배워보세요.

# 참고

- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)
- [Hugging Face PEFT 문서](https://huggingface.co/docs/peft)
- [Hugging Face blog post on PEFT](https://huggingface.co/blog/peft)
