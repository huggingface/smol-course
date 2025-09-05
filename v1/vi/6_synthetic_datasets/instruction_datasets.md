# Tạo tập dữ liệu hướng dẫn

Trong [chương về tinh chỉnh hướng dẫn (instruction tuning)](../1_instruction_tuning/README.md), chúng ta đã học về việc tinh chỉnh mô hình với Tinh chỉnh có giám sát (Supervised Fine-tuning). Trong phần này, chúng ta sẽ khám phá cách tạo tập dữ liệu hướng dẫn cho SFT. Chúng ta sẽ khám phá việc tạo các tập dữ liệu tinh chỉnh hướng dẫn thông qua việc nhắc nhở (prompting) cơ bản và sử dụng các kỹ thuật tinh tế hơn từ các bài báo. Các tập dữ liệu tinh chỉnh hướng dẫn với dữ liệu hạt giống (seed data) để học trong ngữ cảnh (in-context learning) có thể được tạo ra thông qua các phương pháp như SelfInstruct và Magpie. Ngoài ra, chúng ta sẽ khám phá sự tiến hóa hướng dẫn thông qua EvolInstruct. Cuối cùng, chúng ta sẽ khám phá cách tạo tập dữ liệu để tinh chỉnh hướng dẫn bằng cách sử dụng quy trình (pipeline) distilabel.

## Từ lời nhắc đến dữ liệu

Dữ liệu giả lập (Synthetic data) nghe có vẻ phức tạp, nhưng nó có thể được đơn giản hóa thành việc tạo dữ liệu thông qua việc nhắc nhở hiệu quả để trích xuất kiến thức từ mô hình. Đổi lại, bạn có thể coi đây là một cách để tạo dữ liệu cho một tác vụ cụ thể. Thách thức là nhắc nhở một cách hiệu quả trong khi đảm bảo dữ liệu đa dạng và mang tính đại diện. May mắn thay, nhiều bài báo đã khám phá vấn đề này và chúng ta sẽ khám phá một số bài báo hữu ích trong khóa học này. Trước hết, chúng ta sẽ khám phá cách tạo dữ liệu giả lập thông qua việc nhắc nhở thủ công.

### Nhắc nhở cơ bản (Basic Prompting)

Hãy bắt đầu với một ví dụ cơ bản và tải mô hình [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) bằng cách sử dụng tích hợp `transformers` của thư viện `distilabel`. Chúng ta sẽ sử dụng lớp `TextGeneration` để tạo ra một `lời nhắc` (prompt) tổng hợp và sử dụng nó để tạo ra một `phần hoàn thành` (completion).

Tiếp theo, chúng ta sẽ tải mô hình bằng thư viện `distilabel`.

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import TextGeneration

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
gen = TextGeneration(llm=llm)
gen.load()
```

> **Note:**  
> `Distilabel` tải `llm` vào bộ nhớ, vì vậy, khi làm việc trong notebook, chúng ta cần `gen.unload()` sau khi hoàn thành để tránh các vấn đề về bộ nhớ.

Bây giờ chúng ta sẽ sử dụng `llm` để tạo ra một `lời nhắc` để tinh chỉnh hướng dẫn.

```python
next(gen.process([{"instruction": "Tạo một câu hỏi về Khóa học Smol của Hugging Face về các mô hình AI nhỏ."}]))
# Ví dụ: Mục đích của Khóa học Smol là gì?
```

Cuối cùng, chúng ta có thể sử dụng cùng một `lời nhắc` đó làm đầu vào để tạo ra một `phần hoàn thành`.

```python
next(gen.process([{"instruction": "Mục đích của Khóa học Smol là gì?"}]))
# Ví dụ: Khóa học Smol là một nền tảng được thiết kế để học các khái niệm khoa học máy tính.
```

Tuyệt! Chúng ta có thể tạo ra một `lời nhắc` tổng hợp và một `phần hoàn thành` tương ứng. Việc sử dụng lại phương pháp đơn giản này trên quy mô lớn sẽ cho phép chúng ta tạo ra nhiều dữ liệu hơn, tuy nhiên, chất lượng của dữ liệu không tốt lắm và không tính đến các sắc thái của khóa học hoặc lĩnh vực của chúng ta. Ngoài ra, việc chạy lại mã hiện tại cho chúng ta thấy dữ liệu không đa dạng lắm. May mắn thay, có nhiều cách để giải quyết vấn đề này.

### SelfInstruct

`SelfInstruct` là một lời nhắc tạo ra các hướng dẫn mới dựa trên tập dữ liệu mẫu. Dữ liệu mẫu này có thể là một hướng dẫn đơn lẻ hoặc một đoạn ngữ cảnh. Quá trình bắt đầu với một nhóm dữ liệu mẫu ban đầu. Mô hình ngôn ngữ sau đó được nhắc để tạo ra các hướng dẫn mới dựa trên dữ liệu mẫu này bằng cách sử dụng phương pháp học trong ngữ cảnh (in-context learning). Lời nhắc được [triển khai trong distilabel](https://github.com/argilla-io/distilabel/blob/main/src/distilabel/steps/tasks/templates/self-instruct.jinja2) và một phiên bản đơn giản hóa được hiển thị bên dưới:

```
# Mô tả nhiệm vụ
Phát triển {{ num_instructions }} truy vấn của người dùng có thể được nhận bởi ứng dụng AI đã cho và áp dụng cho ngữ cảnh được cung cấp. Nhấn mạnh sự đa dạng trong động từ và cấu trúc ngôn ngữ trong khả năng văn bản của mô hình.

# Ngữ cảnh
{{ input }}

# Đầu ra
```

Để sử dụng nó, chúng ta cần truyền `llm` cho [lớp SelfInstruct](https://distilabel.argilla.io/dev/components-gallery/tasks/selfinstruct/). Hãy sử dụng văn bản từ [phần Từ lời nhắc đến dữ liệu](#prompt-to-data) làm ngữ cảnh và tạo ra một hướng dẫn mới.

```python
from distilabel.steps.tasks import SelfInstruct

self_instruct = SelfInstruct(llm=llm)
self_instruct.load()

context = "<prompt_to_data_section>" # Thay thế bằng nội dung của phần Từ lời nhắc đến dữ liệu

next(self_instruct.process([{"input": text}]))["instructions"][0]
# Quá trình tạo dữ liệu tổng hợp thông qua việc nhắc nhở thủ công là gì?
```

Hướng dẫn được tạo ra đã tốt hơn rất nhiều và nó phù hợp với nội dung và lĩnh vực thực tế của chúng ta. Tuy nhiên, chúng ta có thể làm tốt hơn nữa bằng cách cải thiện lời nhắc thông qua phương pháp tiến hóa (evolution).

### EvolInstruct

EvolInstruct là một kỹ thuật nhắc nhở lấy một hướng dẫn đầu vào và phát triển nó thành một phiên bản tốt hơn của cùng một hướng dẫn. Phiên bản tốt hơn này được định nghĩa theo một tập hợp các tiêu chí và bổ sung các ràng buộc, đào sâu, cụ thể hóa, lập luận hoặc phức tạp hóa cho hướng dẫn ban đầu. Quá trình này có thể được lặp lại nhiều lần để tạo ra các phiên bản tiến hóa khác nhau của cùng một hướng dẫn, lý tưởng nhất là dẫn đến một phiên bản tốt hơn của hướng dẫn ban đầu. Lời nhắc được [triển khai trong distilabel](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/evol_instruct) và một phiên bản đơn giản hóa được hiển thị bên dưới:

```
Tôi muốn bạn đóng vai trò là một Trình viết lại lời nhắc (Prompt Rewriter).
Cho một lời nhắc, hãy viết lại nó thành một phiên bản phức tạp hơn.
Phức tạp hóa lời nhắc dựa trên các tiêu chí sau:
{{ criteria }}

# Lời nhắc
{{ input }}

# Đầu ra
```

Để sử dụng nó, chúng ta cần truyền `llm` cho [lớp EvolInstruct](https://distilabel.argilla.io/dev/components-gallery/tasks/evolinstruct/). Hãy sử dụng lời nhắc tổng hợp từ [phần SelfInstruct](#selfinstruct) làm đầu vào và phát triển nó thành một phiên bản tốt hơn. Đối với ví dụ này, chúng ta sẽ chỉ tiến hóa trong một thế hệ.

```python
from distilabel.steps.tasks import EvolInstruct

evol_instruct = EvolInstruct(llm=llm, num_evolutions=1)
evol_instruct.load()

text = "Quá trình tạo dữ liệu tổng hợp thông qua việc nhắc nhở thủ công là gì"

next(evol_instruct.process([{"instruction": text}]))
# Quá trình tạo dữ liệu tổng hợp thông qua việc nhắc nhở thủ công là gì?
# Và, làm thế nào hệ thống trí tuệ nhân tạo, GPT4, sử dụng các thuật toán học máy để thao tác dữ liệu đầu vào thành dữ liệu tổng hợp?
```

Hướng dẫn bây giờ phức tạp hơn nhưng đã mất đi một số ý nghĩa ban đầu. Vì vậy, hãy lưu ý rằng việc tiến hóa có thể là một con dao hai lưỡi và chúng ta cần cẩn thận với chất lượng của dữ liệu chúng ta tạo ra.

### Magpie

Magpie là một kỹ thuật dựa vào các yếu tố tự suy luận (auto-regressive) của mô hình ngôn ngữ và [mẫu trò chuyện (chat-template)](../1_instruction_tuning/chat_templates.md) đã được sử dụng trong quá trình tinh chỉnh hướng dẫn. Như bạn có thể nhớ, mẫu trò chuyện là một định dạng cấu trúc các cuộc hội thoại với các chỉ số vai trò rõ ràng (hệ thống, người dùng, trợ lý). Trong giai đoạn tinh chỉnh hướng dẫn, mô hình ngôn ngữ đã được tối ưu hóa để tái tạo định dạng này và đó chính xác là những gì `Magpie` tận dụng. Nó bắt đầu với một lời nhắc trước truy vấn (pre-query-prompt) dựa trên mẫu trò chuyện nhưng nó dừng lại trước chỉ báo tin nhắn của người dùng, ví dụ: `<|im_start|>user\n`, và sau đó nó sử dụng mô hình ngôn ngữ để tạo ra lời nhắc của người dùng cho đến khi kết thúc chỉ báo trợ lý, ví dụ: `<|im_end|>`. Cách tiếp cận này cho phép chúng ta tạo ra rất nhiều dữ liệu một cách rất hiệu quả và thậm chí có thể mở rộng quy mô lên các cuộc hội thoại nhiều lượt. Người ta giả thuyết rằng dữ liệu được tạo ra này tái tạo dữ liệu huấn luyện từ giai đoạn tinh chỉnh hướng dẫn của mô hình được sử dụng.

Trong trường hợp này, các mẫu lời nhắc khác nhau cho mỗi mô hình vì chúng dựa trên định dạng mẫu trò chuyện. Nhưng chúng ta có thể đi qua một phiên bản đơn giản hóa của quá trình từng bước.

```bash
# Bước 1: cung cấp lời nhắc trước truy vấn
<|im_start|>user\n

# Bước 2: mô hình ngôn ngữ tạo ra lời nhắc của người dùng
<|im_start|>user\n
Mục đích của Khóa học Smol là gì?

# Bước 3: dừng quá trình tạo
<|im_end|>
```

Để sử dụng nó trong distilabel, chúng ta cần truyền `llm` cho [lớp Magpie](https://distilabel.argilla.io/dev/components-gallery/tasks/magpie/).

```python
from distilabel.steps.tasks import Magpie

magpie = Magpie(llm=llm)
magpie.load()

next(magpie.process([{"system_prompt": "Bạn là một trợ lý hữu ích."}]))
# [{
#   "role": "user",
#   "content": "Bạn có thể cung cấp cho tôi danh sách 3 trường đại học hàng đầu không?"
# },
# {
#   "role": "assistant",
#   "content": "3 trường đại học hàng đầu là: MIT, Yale, Stanford."
# }]
```

Chúng ta ngay lập tức nhận được một tập dữ liệu với một `lời nhắc` và `phần hoàn thành`. Để cải thiện hiệu suất trên lĩnh vực của riêng mình, chúng ta có thể đưa thêm ngữ cảnh vào `system_prompt`. Để LLM tạo ra dữ liệu lĩnh vực cụ thể kết hợp với Magpie, nó giúp mô tả trong lời nhắc hệ thống (system prompt) các truy vấn của người dùng sẽ là gì. Điều này sau đó được sử dụng trong lời nhắc trước truy vấn trước khi chúng ta bắt đầu tạo lời nhắc của người dùng và thiên về LLM để tạo ra các truy vấn của người dùng trong lĩnh vực đó.

```
Bạn là một trợ lý AI sẽ giúp người dùng giải các bài toán.
```

Điều quan trọng là phải viết lời nhắc hệ thống như được hiển thị ở trên thay vì một cái gì đó như:

```
Bạn là một trợ lý AI tạo ra các bài toán
```

Nói chung, các mô hình ngôn ngữ ít được tối ưu hóa hơn để truyền ngữ cảnh bổ sung cho `system_prompt` vì vậy điều này không phải lúc nào cũng hoạt động tốt cho việc tùy chỉnh như các kỹ thuật khác.

### Từ lời nhắc đến quy trình (pipeline)

Các lớp chúng ta đã thấy cho đến nay đều là các lớp độc lập có thể được sử dụng trong một quy trình. Đây là một khởi đầu tốt, nhưng chúng ta có thể làm tốt hơn nữa bằng cách sử dụng lớp `Pipeline` để tạo tập dữ liệu. Chúng ta sẽ sử dụng bước `TextGeneration` để tạo tập dữ liệu tổng hợp để tinh chỉnh hướng dẫn. Quy trình sẽ bao gồm bước `LoadDataFromDicts` để tải dữ liệu, bước `TextGeneration` để tạo `lời nhắc` và `phần hoàn thành` cho lời nhắc đó. Chúng ta sẽ kết nối các bước và luồng dữ liệu thông qua quy trình bằng toán tử `>>`. Trong [tài liệu của distilabel](https://distilabel.argilla.io/dev/components-gallery/tasks/textgeneration/#input-output-columns), chúng ta có thể thấy các cột đầu vào và đầu ra của bước. Để đảm bảo rằng dữ liệu chảy chính xác qua quy trình, chúng ta sẽ sử dụng tham số `output_mappings` để ánh xạ các cột đầu ra với các cột đầu vào của bước tiếp theo.

```python
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    data = LoadDataFromDicts(data=[{"instruction": "Tạo một câu hỏi ngắn về Khóa học Smol của Hugging Face."}])
    llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    gen_a = TextGeneration(llm=llm, output_mappings={"generation": "instruction"})
    gen_b = TextGeneration(llm=llm, output_mappings={"generation": "response"})
    data >> gen_a >> gen_b

if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    print(distiset["default"]["train"][0])
# [{
#   "instruction": "Mục đích của Khóa học Smol là gì?",
#   "response": "Khóa học Smol là một nền tảng được thiết kế để học các khái niệm khoa học máy tính."
# }]
```

Bên dưới, quy trình này có rất nhiều tính năng hay. Nó tự động lưu trữ các kết quả tạo, vì vậy chúng ta không phải chạy lại các bước tạo. Thư viện có tích hợp khả năng xử lý lỗi (fault-tolerance), vì vậy nếu các bước tạo thất bại, quy trình vẫn sẽ tiếp tục chạy. Và quy trình thực hiện tất cả các bước tạo song song, vì vậy việc tạo nhanh hơn. Chúng ta thậm chí có thể trực quan hóa quy trình bằng phương thức `draw`. Ở đây bạn có thể thấy cách dữ liệu chảy qua quy trình và cách `output_mappings` được sử dụng để ánh xạ các cột đầu ra với các cột đầu vào của bước tiếp theo.

![Pipeline](./images/pipeline.png)

## Các phương pháp hay nhất

- Đảm bảo bạn có dữ liệu hạt giống đa dạng để bao quát nhiều tình huống
- Thường xuyên đánh giá tập dữ liệu để đảm bảo dữ liệu được tạo ra đa dạng và có chất lượng cao
- Lặp lại trên (system)prompt để cải thiện chất lượng của dữ liệu

## Các bước tiếp theo

👨🏽‍💻 Lập trình -[Notebook bài tập](./notebooks/instruction_sft_dataset.ipynb) để tạo tập dữ liệu để tinh chỉnh hướng dẫn
🧑‍🏫 Tìm hiểu - Về [tạo tập dữ liệu sở thích](./preference_datasets.md)

## Tài liệu tham khảo

- [Tài liệu Distilabel](https://distilabel.argilla.io/latest/)
- [Self-instruct](https://arxiv.org/abs/2212.10560)
- [Evol-Instruct](https://arxiv.org/abs/2304.12244)
- [Magpie](https://arxiv.org/abs/2406.08464)
