# チャットテンプレート

チャットテンプレートは、言語モデルとユーザー間のインタラクションを構造化するために不可欠です。これらは会話の一貫した形式を提供し、モデルが各メッセージの文脈と役割を理解し、適切な応答パターンを維持することを保証します。

## ベースモデル vs インストラクションモデル

ベースモデルは次のトークンを予測するために生のテキストデータでトレーニングされる一方、インストラクションモデルは特定の指示に従い会話に参加するように微調整されたモデルです。例えば、`SmolLM2-135M`はベースモデルであり、`SmolLM2-135M-Instruct`はその指示に特化したバリアントです。

ベースモデルをインストラクションモデルのように動作させるためには、モデルが理解できるようにプロンプトを一貫してフォーマットする必要があります。ここでチャットテンプレートが役立ちます。ChatMLは、システム、ユーザー、アシスタントの役割を明確に示すテンプレート形式で会話を構造化します。

ベースモデルは異なるチャットテンプレートで微調整される可能性があるため、インストラクションモデルを使用する際には、正しいチャットテンプレートを使用していることを確認する必要があります。

## チャットテンプレートの理解

チャットテンプレートの核心は、言語モデルと通信する際に会話がどのようにフォーマットされるべきかを定義することです。これには、システムレベルの指示、ユーザーメッセージ、およびアシスタントの応答が含まれ、モデルが理解できる構造化された形式で提供されます。この構造は、インタラクションの一貫性を維持し、モデルがさまざまな種類の入力に適切に応答することを保証します。以下はチャットテンプレートの例です：

```sh
<|im_end|>ユーザー
こんにちは！<|im_end|>
<|im_end|>アシスタント
はじめまして！<|im_end|>
<|im_end|>ユーザー
質問してもいいですか？<|im_end|>
<|im_end|>アシスタント
```

`transformers`ライブラリは、モデルのトークナイザーに関連してチャットテンプレートを自動的に処理します。`transformers`でチャットテンプレートがどのように構築されるかについて詳しくは[こちら](https://huggingface.co/docs/transformers/en/chat_templating#how-do-i-use-chat-templates)を参照してください。私たちはメッセージを正しい形式で構造化するだけで、残りはトークナイザーが処理します。以下は基本的な会話の例です：

```python
messages = [
    {"role": "system", "content": "あなたは技術的なトピックに焦点を当てた役立つアシスタントです。"},
    {"role": "user", "content": "チャットテンプレートとは何か説明できますか？"},
    {"role": "assistant", "content": "チャットテンプレートは、ユーザーとAIモデル間の会話を構造化します..."}
]
```

上記の例を分解して、チャットテンプレート形式にどのようにマッピングされるかを見てみましょう。

## システムメッセージ

システムメッセージは、モデルの動作の基礎を設定します。これらは、以降のすべてのインタラクションに影響を与える持続的な指示として機能します。例えば：

```python
system_message = {
    "role": "system",
    "content": "あなたはプロフェッショナルなカスタマーサービスエージェントです。常に礼儀正しく、明確で、役立つようにしてください。"
}
```

## 会話

チャットテンプレートは、ユーザーとアシスタント間の以前のやり取りを保存し、会話の履歴を通じて文脈を維持します。これにより、複数ターンにわたる一貫した会話が可能になります：

```python
conversation = [
    {"role": "user", "content": "注文に関して助けが必要です"},
    {"role": "assistant", "content": "お手伝いします。注文番号を教えていただけますか？"},
    {"role": "user", "content": "注文番号はORDER-123です"},
]
```

## Transformersを使用した実装

`transformers`ライブラリは、チャットテンプレートのための組み込みサポートを提供します。使用方法は以下の通りです：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "あなたは役立つプログラミングアシスタントです。"},
    {"role": "user", "content": "リストをソートするPython関数を書いてください"},
]

# チャットテンプレートを適用
formatted_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

## カスタムフォーマット

異なる役割に対して特別なトークンやフォーマットを追加するなど、さまざまなメッセージタイプのフォーマットをカスタマイズできます。例えば：

```python
template = """
<|system|>{system_message}
<|user|>{user_message}
<|assistant|>{assistant_message}
""".lstrip()
```

## マルチターン会話のサポート

テンプレートは、文脈を維持しながら複雑なマルチターン会話を処理できます：

```python
messages = [
    {"role": "system", "content": "あなたは数学の家庭教師です。"},
    {"role": "user", "content": "微積分とは何ですか？"},
    {"role": "assistant", "content": "微積分は数学の一分野です..."},
    {"role": "user", "content": "例を教えてください。"},
]
```

⏭️ [次へ: Supervised Fine-Tuning](./supervised_fine_tuning.md)

## リソース

- [Hugging Face Chat Templating Guide](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Chat Templates Examples Repository](https://github.com/chujiezheng/chat_templates)
