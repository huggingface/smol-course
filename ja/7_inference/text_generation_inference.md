# Text Generation Inference (TGI)

テキスト生成推論 (TGI) は、Hugging Face が開発した、大規模言語モデル (LLM) をデプロイおよびサービングするためのツールキットです。これは、人気のあるオープンソースのLLMの高性能なテキスト生成を可能にするように設計されています。TGIは、Hugging Chat - オープンアクセスモデルのためのオープンソースインターフェースでプロダクションで使用されています。

## テキスト生成推論を使用する理由

テキスト生成推論は、プロダクションで大規模言語モデルをデプロイする際の主要な課題に対処します。多くのフレームワークはモデル開発に優れていますが、TGIは特にプロダクションデプロイメントとスケーリングの最適化に焦点を当てています。主な機能には次のものがあります：

- **テンソル並列処理**：TGIはテンソル並列処理を通じてモデルを複数のGPUに分割でき、大規模なモデルを効率的にサービングするために不可欠です。
- **連続バッチ処理**：連続バッチ処理システムは、リクエストを動的に処理することでGPUの利用率を最大化し、Flash AttentionやPaged Attentionなどの最適化によりメモリ使用量を大幅に削減し、速度を向上させます。
- **トークンストリーミング**：リアルタイムアプリケーションは、サーバー送信イベントを介したトークンストリーミングの恩恵を受け、最小限の遅延で応答を提供します。

## テキスト生成推論の使用方法

### 基本的なPythonの使用法

TGIは、シンプルで強力なREST API統合を使用しており、アプリケーションとの統合が容易です。

### REST APIの使用

TGIはJSONペイロードを受け入れるRESTful APIを公開しており、HTTPリクエストを行うことができる任意のプログラミング言語やツールからアクセスできます。以下はcurlを使用した基本的な例です：

```bash
# 基本的な生成リクエスト
curl localhost:8080/v1/chat/completions \
    -X POST \
    -d '{
  "model": "tgi",
  "messages": [
    {
      "role": "system",
      "content": "あなたは役立つアシスタントです。"
    },
    {
      "role": "user",
      "content": "ディープラーニングとは何ですか？"
    }
  ],
  "stream": true,
  "max_tokens": 20
}' \
    -H 'Content-Type: application/json'
```

### `huggingface_hub` Pythonクライアントの使用

`huggingface_hub` Pythonクライアントは、接続管理、リクエストのフォーマット、およびレスポンスの解析を処理します。以下はその開始方法です。

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    base_url="http://localhost:8080/v1/",
)

output = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "あなたは役立つアシスタントです。"},
        {"role": "user", "content": "10まで数えてください"},
    ],
    stream=True,
    max_tokens=1024,
)

for chunk in output:
    print(chunk.choices[0].delta.content)
```


### OpenAI APIの使用

多くのライブラリがOpenAI APIをサポートしているため、同じクライアントを使用してTGIと対話できます。

```python
from openai import OpenAI

# クライアントを初期化し、TGIにポイントする
client = OpenAI(
    base_url="http://localhost:8080/v1/",
    api_key="-"
)

chat_completion = client.chat.completions.create(
    model="tgi",
    messages=[
        {"role": "system", "content": "あなたは役立つアシスタントです。" },
        {"role": "user", "content": "ディープラーニングとは何ですか？"}
    ],
    stream=True
)

# ストリームを反復して表示
for message in chat_completion:
    print(message)
```

## TGIのためのモデルの準備

TGIでモデルをサービングするには、次の要件を満たしていることを確認してください：

1. **サポートされているアーキテクチャ**：モデルアーキテクチャがサポートされていることを確認します（Llama、BLOOM、T5など）

2. **モデル形式**：重みをsafetensors形式に変換して高速な読み込みを実現します：

```python
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-model")
state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")
```

3. **量子化**（オプション）：メモリ使用量を削減するためにモデルを量子化します：

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)

model = AutoModelForCausalLM.from_pretrained(
    "your-model",
    quantization_config=quantization_config
)
```

## 参考文献

- [Text Generation Inference Documentation](https://huggingface.co/docs/text-generation-inference)
- [TGI GitHub Repository](https://github.com/huggingface/text-generation-inference)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [TGI API Reference](https://huggingface.co/docs/text-generation-inference/api_reference)
