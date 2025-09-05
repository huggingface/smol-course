![smolcourse image](../banner.png)

# 소형 언어 모델 과정

이 과정에서는 특정 사용 사례에 맞게 언어 모델을 정렬하는 법을 다룹니다. 모든 자료는 대부분의 로컬 컴퓨터에서 실행되므로 간편하게 언어 모델 정렬을 시작해볼 수 있습니다. 이 과정을 위해 필요한 최소한의 GPU 요구 사항이나 유료 서비스가 없습니다. [SmolLM2](https://github.com/huggingface/smollm/tree/main) 시리즈 모델을 기반으로 하는 과정이지만, 여기서 배운 기술을 더 큰 모델이나 다른 작은 언어 모델로 옮길 수 있습니다.

<a href="http://hf.co/join/discord">
<img src="https://img.shields.io/badge/Discord-7289DA?&logo=discord&logoColor=white"/>
</a>

<div style="background: linear-gradient(to right, #e0f7fa, #e1bee7, orange); padding: 20px; border-radius: 5px; margin-bottom: 20px; color: purple;">
    <h2>지금 바로 참여하세요!</h2>
    <p>이 과정은 열려 있으며 다른 사용자와의 상호 검토를 진행할 수 있습니다. 이 과정에 참여하려면 <strong>pull request(PR)</strong>를 열고 검토 받을 수 있도록 결과물을 제출하세요. 다음 단계를 따르면 됩니다:</p>
    <ol>
        <li><a href="https://github.com/huggingface/smol-course/fork">여기</a>에서 레포지토리를 fork하세요.</li>
        <li>자료를 읽고, 바꿔 보고, 실습해보고, 나만의 예제를 추가해보세요.</li>
        <li>december-2024 브랜치에 PR을 보내세요.</li>
        <li>검토가 끝나면 december-2024 브랜치에 병합됩니다.</li>
    </ol>
    <p>이 과정은 학습에 도움이 될 뿐만 아니라 지속적으로 발전하는 커뮤니티 기반 코스를 형성하는 데에도 기여할 것입니다.</p>
</div>

[discussion thread](https://github.com/huggingface/smol-course/discussions/2#discussion-7602932)에서 과정에 대해 토론할 수도 있습니다.

## 과정 개요

이 과정은 소형 언어 모델의 초기 학습부터 결과물 배포까지 실습할 수 있는 실용적인 내용을 제공합니다.

| 모듈 | 설명 | 상태 | 공개일 |
|--------|-------------|---------|--------------|
| [Instruction Tuning](./1_instruction_tuning) | 지도 학습 기반 미세 조정, 대화 템플릿 작성, 기본적인 지시를 따르게 하는 방법 학습 | ✅ 학습 가능 | 2024. 12. 3 |
| [Preference Alignment](./2_preference_alignment) | 모델을 인간 선호도에 맞게 정렬하기 위한 DPO와 ORPO 기법 학습 | ✅ 학습 가능 | 2024. 12. 6 |
| [Parameter-efficient Fine-tuning](./3_parameter_efficient_finetuning) | LoRA, 프롬프트 튜닝을 포함한 효율적인 적응 방법 학습 | ✅ 학습 가능 | 2024. 12. 9 |
| [Evaluation](./4_evaluation) | 자동 벤치마크 사용법 및 사용자 정의 도메인 평가 수행 방법 학습 | ✅ 학습 가능 | 2024. 12. 13 |
| [Vision-language Models](./5_vision_language_models) | 비전-언어 태스크를 위한 멀티모달 모델 적용 방법 학습 | [🚧 준비중](https://github.com/huggingface/smol-course/issues/49) | 2024. 12. 16 |
| [Synthetic Datasets](./6_synthetic_datasets) | 모델 학습을 위한 합성 데이터셋 생성 및 검증 | [🚧 준비중](https://github.com/huggingface/smol-course/issues/83) | 2024. 12. 20 |
| [Inference](./7_inference) | 모델의 효율적인 추론 방법 학습 | 📝 작성 예정 | 2024. 12. 23 |

## 왜 소형 언어 모델을 사용하나요?

대형 언어 모델은 뛰어난 능력을 보여주지만, 상당한 연산 자원을 필요로 하며 특정 기능에 초점을 맞춘 애플리케이션에 대해서는 대형 언어 모델이 과한 경우도 있습니다. 소형 언어 모델은 도메인 특화 애플리케이션에 있어서 몇 가지 이점을 제공합니다: 

- **효율성**: 대형 언어 모델보다 훨씬 적은 연산 자원으로 학습 및 배포 가능
- **맞춤화**: 특정 도메인에 대한 미세 조정 및 적응 용이
- **제어**: 모델 동작 과정을 잘 이해할 수 있고 모델의 동작을 쉽게 제어 가능
- **비용**: 학습과 추론 과정에서 필요한 비용 감소
- **프라이버시**: 데이터를 외부 API로 보내지 않고 로컬에서 실행 가능
- **친환경**: 탄소 발자국을 줄이는 효율적인 자원 사용
- **쉬운 학술 연구 및 개발**: 최신 LLM을 활용해 물리적 제약 없이 학술 연구를 쉽게 시작할 수 있도록 지원

## 사전 준비 사항

시작하기 전에 아래 사항이 준비되어 있는지 확인하세요:
- 머신러닝과 자연어처리에 대한 기본적인 이해가 필요합니다.
- Python, PyTorch 및 `transformers` 라이브러리에 익숙해야 합니다.
- 사전 학습된 언어 모델과 레이블이 있는 액세스할 수 있어야 합니다.

## 설치

이 과정은 패키지 형태로 관리되기 때문에 패키지 매니저를 이용해 의존성 설치를 쉽게 진행할 수 있습니다. 이를 위해 [uv](https://github.com/astral-sh/uv) 사용을 권장하지만 `pip`나 `pdm`을 사용할 수도 있습니다.

### `uv` 사용

`uv`를 설치한 후, 아래 명령어로 소형 언어 모델 과정을 설치할 수 있습니다:

```bash
uv venv --python 3.11.0
uv sync
```

### `pip` 사용

모든 예제는 동일한 **python 3.11** 환경에서 실행되기 때문에 아래처럼 환경을 생성하고 의존성을 설치해야 합니다:

```bash
# python -m venv .venv
# source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

**Google Colab**에서는 사용하는 하드웨어에 따라 유연하게 의존성을 설치해야 합니다. 아래와 같이 진행하세요:

```bash
pip install transformers trl datasets huggingface_hub
```

## 참여

많은 사람이 고가의 장비 없이 LLM을 미세 조정하는 법을 배울 수 있도록 이 자료를 공유해 봅시다!

[![Star History Chart](https://api.star-history.com/svg?repos=huggingface/smol-course&type=Date)](https://star-history.com/#huggingface/smol-course&Date)
