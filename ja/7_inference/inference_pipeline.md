# トランスフォーマーパイプラインを使用した基本的な推論

🤗 Transformersの`pipeline`抽象化は、Hugging Face Hubの任意のモデルで推論を実行するための簡単な方法を提供します。これにより、前処理と後処理のステップがすべて処理され、モデルのアーキテクチャや要件に関する深い知識がなくても簡単に使用できます。

## パイプラインの仕組み

Hugging Faceのパイプラインは、原始入力と人間が読める出力の間の3つの重要なステージを自動化することで、機械学習のワークフローを簡素化します：

**前処理ステージ**
パイプラインはまず、モデルのために原始入力を準備します。これは入力タイプによって異なります：
- テキスト入力は、単語をモデルに適したトークンIDに変換するためにトークナイズされます
- 画像はモデルの要件に合わせてリサイズおよび正規化されます
- 音声は特徴抽出を通じてスペクトログラムやその他の表現に変換されます

**モデル推論**
フォワードパス中に、パイプラインは次のことを行います：
- 効率的な処理のために入力のバッチ処理を自動的に処理します
- 最適なデバイス（CPU/GPU）で計算を行います
- サポートされている場合、半精度（FP16）推論などのパフォーマンス最適化を適用します

**後処理ステージ**
最後に、パイプラインは原始モデル出力を有用な結果に変換します：
- トークンIDを読みやすいテキストにデコードします
- ロジットを確率スコアに変換します
- 特定のタスクに応じて出力をフォーマットします（例：分類ラベル、生成されたテキスト）

この抽象化により、パイプラインがモデル推論の技術的な複雑さを処理する一方で、アプリケーションロジックに集中することができます。

## 基本的な使用方法

テキスト生成のためのパイプラインを使用する方法は次のとおりです：

```python
from transformers import pipeline

# 特定のモデルでパイプラインを作成
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# テキストを生成
response = generator(
    "コーディングについての短い詩を書いてください：",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)
print(response[0]['generated_text'])
```

## 主要な構成オプション

### モデルの読み込み
```python
# CPU推論
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device="cpu")

# GPU推論（デバイス0）
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", device=0)

# 自動デバイス配置
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device_map="auto",
    torch_dtype="auto"
)
```

### 生成パラメータ

```python
response = generator(
    "これをフランス語に翻訳してください：",
    max_new_tokens=100,     # 生成されるテキストの最大長
    do_sample=True,         # 貪欲なデコードの代わりにサンプリングを使用
    temperature=0.7,        # ランダム性を制御（高いほどランダム）
    top_k=50,               # 上位kトークンに制限
    top_p=0.95,             # ヌクレウスサンプリングの閾値
    num_return_sequences=1  # 異なる生成の数
)
```

## 複数の入力の処理

パイプラインはバッチ処理を通じて複数の入力を効率的に処理できます：

```python
# 複数のプロンプトを準備
prompts = [
    "プログラミングについての俳句を書いてください：",
    "APIとは何かを説明してください：",
    "ロボットについての短い物語を書いてください："
]

# すべてのプロンプトを効率的に処理
responses = generator(
    prompts,
    batch_size=4,              # 一度に処理するプロンプトの数
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

# 結果を表示
for prompt, response in zip(prompts, responses):
    print(f"プロンプト: {prompt}")
    print(f"応答: {response[0]['generated_text']}\n")
```

## Webサーバー統合

パイプラインをFastAPIアプリケーションに統合する方法は次のとおりです：

```python
from fastapi import FastAPI, HTTPException
from transformers import pipeline
import uvicorn

app = FastAPI()

# グローバルにパイプラインを初期化
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device_map="auto"
)

@app.post("/generate")
async def generate_text(prompt: str):
    try:
        if not prompt:
            raise HTTPException(status_code=400, detail="プロンプトが提供されていません")
            
        response = generator(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7
        )
        
        return {"generated_text": response[0]['generated_text']}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
```

## 制限事項

パイプラインはプロトタイピングや小規模なデプロイメントには最適ですが、いくつかの制限があります：

- 専用のサービングソリューションと比較して最適化オプションが限られている
- 動的バッチ処理などの高度な機能の組み込みサポートがない
- 高スループットのプロダクションワークロードには適していない可能性がある

高スループット要件を持つプロダクションデプロイメントには、Text Generation Inference（TGI）やその他の専門的なサービングソリューションの使用を検討してください。

## リソース

- [Hugging Face Pipeline Tutorial](https://huggingface.co/docs/transformers/en/pipeline_tutorial)
- [Pipeline API Reference](https://huggingface.co/docs/transformers/en/main_classes/pipelines)
- [Text Generation Parameters](https://huggingface.co/docs/transformers/en/main_classes/text_generation)
- [Model Quantization Guide](https://huggingface.co/docs/transformers/en/perf_infer_gpu_one)
