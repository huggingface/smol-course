# Tạo tập dữ liệu ưu tiên (Preference Datasets)

Trong [chương về điều chỉnh ưu tiên (preference alignment)](../2_preference_alignment/README.md), chúng ta đã học về Tối ưu hóa ưu tiên trực tiếp (Direct Preference Optimization). Trong phần này, chúng ta sẽ khám phá cách tạo tập dữ liệu ưu tiên cho các phương pháp như DPO. Chúng ta sẽ xây dựng dựa trên các phương pháp đã được giới thiệu trong phần [tạo tập dữ liệu hướng dẫn](./instruction_datasets.md). Ngoài ra, chúng ta sẽ chỉ ra cách thêm các phần hoàn thành (completions) bổ sung vào tập dữ liệu bằng cách sử dụng kỹ thuật nhắc nhở (prompting) cơ bản hoặc bằng cách sử dụng EvolQuality để cải thiện chất lượng của các phản hồi. Cuối cùng, chúng ta sẽ chỉ ra cách `UltraFeedback` có thể được sử dụng để tạo điểm số và phê bình.

## Tạo nhiều phần hoàn thành (completions)

Dữ liệu ưu tiên là tập dữ liệu có nhiều `phần hoàn thành` cho cùng một `hướng dẫn`. Chúng ta có thể thêm nhiều `phần hoàn thành` hơn vào tập dữ liệu bằng cách nhắc nhở (prompt) một mô hình tạo ra chúng. Khi làm điều này, chúng ta cần đảm bảo rằng phần hoàn thành thứ hai không quá giống với phần hoàn thành đầu tiên về chất lượng tổng thể và cách diễn đạt. Điều này rất quan trọng vì mô hình cần được tối ưu hóa cho một ưu tiên rõ ràng. Chúng ta muốn biết phần hoàn thành nào được ưa thích hơn phần kia, thường được gọi là `chosen` (được chọn) và `rejected` (bị từ chối). Chúng ta sẽ đi vào chi tiết hơn về việc xác định các phần hoàn thành được chọn và bị từ chối trong [phần tạo điểm số](#creating-scores).

### Tổng hợp mô hình (Model pooling)

Bạn có thể sử dụng các mô hình từ các họ mô hình khác nhau để tạo phần hoàn thành thứ hai, được gọi là tổng hợp mô hình. Để cải thiện hơn nữa chất lượng của phần hoàn thành thứ hai, bạn có thể sử dụng các đối số tạo khác nhau, như điều chỉnh `temperature`. Cuối cùng, bạn có thể sử dụng các mẫu lời nhắc (prompt templates) hoặc lời nhắc hệ thống (system prompts) khác nhau để tạo phần hoàn thành thứ hai nhằm đảm bảo sự đa dạng dựa trên các đặc điểm cụ thể được xác định trong mẫu. Về lý thuyết, chúng ta có thể lấy hai mô hình có chất lượng khác nhau và sử dụng mô hình tốt hơn làm phần hoàn thành `chosen`.

Hãy bắt đầu với việc tổng hợp mô hình bằng cách tải các mô hình [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) và [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) bằng cách sử dụng tích hợp `transformers` của thư viện `distilabel`. Sử dụng các mô hình này, chúng ta sẽ tạo ra hai `phản hồi` tổng hợp cho một `lời nhắc` nhất định. Chúng ta sẽ tạo một quy trình (pipeline) khác với `LoadDataFromDicts`, `TextGeneration` và `GroupColumns`. Trước tiên, chúng ta sẽ tải dữ liệu, sau đó sử dụng hai bước tạo và sau đó nhóm các kết quả lại. Chúng ta kết nối các bước và luồng dữ liệu thông qua quy trình bằng toán tử `>>` và `[]`, có nghĩa là chúng ta muốn sử dụng đầu ra của bước trước làm đầu vào cho cả hai bước trong danh sách.

```python
from distilabel.llms import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import GroupColumns, LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with Pipeline() as pipeline:
    data = LoadDataFromDicts(data=[{"instruction": "Dữ liệu giả lập (synthetic data) là gì?"}])
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
#   'Dữ liệu giả lập là dữ liệu được tạo ra nhân tạo, bắt chước cách sử dụng trong thế giới thực.',
#   'Dữ liệu giả lập đề cập đến dữ liệu đã được tạo ra một cách nhân tạo.'
# ]}
```

Như bạn có thể thấy, chúng ta có hai `phần hoàn thành` tổng hợp cho `lời nhắc` đã cho. Chúng ta có thể tăng cường sự đa dạng bằng cách khởi tạo các bước `TextGeneration` với một `system_prompt` cụ thể hoặc bằng cách truyền các đối số tạo cho `TransformersLLM`. Bây giờ hãy xem cách chúng ta có thể cải thiện chất lượng của các `phần hoàn thành` bằng EvolQuality.

### EvolQuality

EvolQuality tương tự như [EvolInstruct](./instruction_datasets.md#evolinstruct) - đó là một kỹ thuật nhắc nhở nhưng nó phát triển `các phần hoàn thành` thay vì `lời nhắc` đầu vào. Tác vụ lấy cả `lời nhắc` và `phần hoàn thành` và phát triển `phần hoàn thành` thành một phiên bản phản hồi tốt hơn cho `lời nhắc` dựa trên một tập hợp các tiêu chí. Phiên bản tốt hơn này được định nghĩa theo các tiêu chí để cải thiện tính hữu ích, mức độ liên quan, đào sâu, sáng tạo hoặc chi tiết. Bởi vì điều này tự động tạo ra phần hoàn thành thứ hai, chúng ta có thể sử dụng nó để thêm nhiều `phần hoàn thành` hơn vào tập dữ liệu. Về lý thuyết, chúng ta thậm chí có thể giả định rằng sự tiến hóa tốt hơn phần hoàn thành ban đầu và sử dụng nó làm phần hoàn thành `chosen` ngay lập tức.

Lời nhắc được [triển khai trong distilabel](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/evol_quality) và phiên bản đơn giản hóa được hiển thị bên dưới:

```bash
Tôi muốn bạn đóng vai trò là một Trình viết lại phản hồi (Response Rewriter).
Cho một lời nhắc và một phản hồi, hãy viết lại phản hồi thành một phiên bản tốt hơn.
Phức tạp hóa lời nhắc dựa trên các tiêu chí sau:
{{ criteria }}

# Lời nhắc
{{ input }}

# Phản hồi
{{ output }}

# Phản hồi được cải thiện
```

Hãy sử dụng [lớp EvolQuality](https://distilabel.argilla.io/dev/components-gallery/tasks/evolquality/) để phát triển `lời nhắc` và `phần hoàn thành` tổng hợp từ [phần Tổng hợp mô hình](#model-pooling) thành một phiên bản tốt hơn. Đối với ví dụ này, chúng ta sẽ chỉ tiến hóa trong một thế hệ.

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import EvolQuality

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
evol_quality = EvolQuality(llm=llm, num_evolutions=1)
evol_quality.load()

instruction = "Dữ liệu giả lập (synthetic data) là gì?"
completion = "Dữ liệu giả lập là dữ liệu được tạo ra nhân tạo, bắt chước cách sử dụng trong thế giới thực."

next(evol_quality.process([{
    "instruction": instruction,
    "response": completion
}]))
# Quá trình tạo dữ liệu giả lập thông qua việc nhắc nhở thủ công bao gồm việc tạo ra các tập dữ liệu nhân tạo bắt chước các kiểu sử dụng trong thế giới thực.
```

`Phản hồi` bây giờ phức tạp hơn và cụ thể hơn cho `hướng dẫn`. Đây là một khởi đầu tốt, nhưng như chúng ta đã thấy với EvolInstruct, các thế hệ tiến hóa không phải lúc nào cũng tốt hơn. Do đó, điều quan trọng là phải sử dụng các kỹ thuật đánh giá bổ sung để đảm bảo chất lượng của tập dữ liệu. Chúng ta sẽ khám phá điều này trong phần tiếp theo.

## Tạo điểm số

Điểm số là thước đo mức độ phản hồi này được ưa thích hơn phản hồi khác. Nhìn chung, những điểm số này có thể là tuyệt đối, chủ quan hoặc tương đối. Đối với khóa học này, chúng ta sẽ tập trung vào hai loại đầu tiên vì chúng có giá trị nhất để tạo các tập dữ liệu ưu tiên. Việc chấm điểm này là một cách đánh giá và nhận xét bằng cách sử dụng các mô hình ngôn ngữ và do đó có một số điểm tương đồng với các kỹ thuật đánh giá mà chúng ta đã thấy trong [chương về đánh giá](../3_evaluation/README.md). Cũng như các kỹ thuật đánh giá khác, điểm số và đánh giá thường yêu cầu các mô hình lớn hơn để phù hợp hơn với ưu tiên của con người.

### UltraFeedback

UltraFeedback là một kỹ thuật tạo ra điểm số và phê bình cho một `lời nhắc` nhất định và `phần hoàn thành` của nó.

Điểm số dựa trên chất lượng của `phần hoàn thành` theo một tập hợp các tiêu chí. Có bốn tiêu chí chi tiết: `helpfulness` (tính hữu ích), `relevance` (mức độ liên quan), `deepening` (đào sâu) và `creativity` (sáng tạo). Chúng rất hữu ích nhưng nói chung, sử dụng các tiêu chí tổng thể là một khởi đầu tốt, cho phép chúng ta đơn giản hóa quá trình tạo điểm số. Điểm số có thể được sử dụng để xác định `phần hoàn thành` nào là `chosen` và phần nào là `rejected`. Bởi vì chúng là tuyệt đối, chúng cũng có thể được sử dụng làm bộ lọc thú vị cho các giá trị ngoại lệ trong tập dữ liệu, tìm các phần hoàn thành tệ nhất hoặc các cặp có ít nhiều sự khác biệt.

Các phê bình được thêm vào để cung cấp lý do cho điểm số. Chúng có thể được sử dụng làm ngữ cảnh bổ sung để giúp chúng ta hiểu sự khác biệt giữa các điểm số. Mô hình ngôn ngữ tạo ra các phê bình sâu rộng rất hữu ích, nhưng điều này cũng làm tăng thêm chi phí và độ phức tạp cho quá trình vì việc tạo ra các phê bình tốn kém hơn so với việc tạo ra một token duy nhất để đại diện cho điểm số.

Lời nhắc được [triển khai trong distilabel](https://github.com/argilla-io/distilabel/tree/main/src/distilabel/steps/tasks/templates/ultrafeedback) và phiên bản đơn giản hóa được hiển thị bên dưới:

```bash
Đánh giá đầu ra của mô hình dựa trên các tiêu chí khác nhau: Tính hữu ích, Mức độ liên quan, Đào sâu, Sáng tạo
Vai trò của bạn là cung cấp một đánh giá tổng thể dựa trên các yếu tố trên.
Chấm điểm đầu ra từ 1 đến 5 về chất lượng tổng thể.

Trả lời theo định dạng sau: điểm số - lý do

# Đầu vào
{{ input }}

# Phản hồi
{{ output }}

# Điểm số - Lý do
```

Hãy sử dụng [lớp UltraFeedback](https://distilabel.argilla.io/dev/components-gallery/tasks/ultrafeedback/) để đánh giá `lời nhắc` và `phần hoàn thành` tổng hợp từ [phần Tổng hợp mô hình](#model-pooling).

```python
from distilabel.llms import TransformersLLM
from distilabel.steps.tasks import UltraFeedback

llm = TransformersLLM(model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
ultrafeedback = UltraFeedback(llm=llm)
ultrafeedback.load()

instruction = "Dữ liệu giả lập (synthetic data) là gì?"
completion_a = "Dữ liệu giả lập là dữ liệu được tạo ra nhân tạo, bắt chước cách sử dụng trong thế giới thực."
completion_b = "Dữ liệu giả lập đề cập đến dữ liệu đã được tạo ra một cách nhân tạo."

next(ultrafeedback.process([{
    "instruction": instruction,
    "generations": [completion_a, completion_b]
}]))
# [
#     {
#         'ratings': [4, 5],
#         'rationales': ['có thể cụ thể hơn', 'định nghĩa tốt'],
#     }
# ]
```

## Các phương pháp hay nhất

- Phương pháp đánh giá điểm số tổng thể thường rẻ hơn và dễ tạo hơn so với phê bình và điểm số cụ thể
- Sử dụng các mô hình lớn hơn để tạo điểm số và phê bình
- Sử dụng một tập hợp đa dạng các mô hình để tạo điểm số và phê bình
- Lặp lại cấu hình của `system_prompt` và các mô hình

## Các bước tiếp theo

👨🏽‍💻 Lập trình -[Notebook bài tập](./notebooks/preference_dpo_dataset.ipynb) để tạo tập dữ liệu để tinh chỉnh hướng dẫn

## Tài liệu tham khảo

- [Tài liệu Distilabel](https://distilabel.argilla.io/latest/)
- [Deita](https://arxiv.org/abs/2312.15685)
- [UltraFeedback](https://arxiv.org/abs/2310.01377)
