# Đánh giá tự động

Đánh giá tự động là các công cụ chuẩn hoá để đánh giá các mô hình ngôn ngữ qua các tác vụ và khả năng khác nhau. Mặc dù chúng cung cấp điểm khởi đầu hữu ích để hiểu hiệu năng của mô hình, điều quan trọng là phải nhận ra rằng chúng chỉ là một phần trong toàn bộ khả năng của mô hình.

## Hiểu về đánh giá tự động

Đánh giá tự động thường bao gồm các tập dữ liệu được tuyển chọn với các tác vụ và tiêu chí đánh giá được định nghĩa trước. Những đánh giá này nhằm đánh giá nhiều khía cạnh về khả năng của mô hình, từ hiểu ngôn ngữ cơ bản đến suy luận phức tạp. Lợi thế chính của việc sử dụng đánh giá tự động là tính chuẩn hoá - chúng cho phép so sánh nhất quán giữa các mô hình khác nhau và cung cấp kết quả có thể tái tạo lại.

Tuy nhiên, điều quan trọng là phải hiểu rằng điểm số trên các bài đánh giá không phải lúc nào cũng thể hiện chính xác hiệu quả của mô hình trong thực tế. Một mô hình xuất sắc trong các bài đánh giá học thuật (academic benchmark) vẫn có thể gặp khó khăn với các ứng dụng vào lĩnh vực cụ thể hoặc các trường hợp sử dụng thực tế.

## Các bài đánh giá và hạn chế của chúng

### Đánh giá kiến thức tổng quát

`MMLU` (Massive Multitask Language Understanding) là bài đánh giá dùng để kiểm tra kiến thức trên *57 môn học*, từ khoa học tự nhiên đến khoa học xã hội. Mặc dù toàn diện, nó có thể không phản ánh độ sâu của chuyên môn cần thiết cho các lĩnh vực cụ thể. `TruthfulQA` đánh giá xu hướng mô hình tái tạo các quan niệm sai lầm phổ biến, mặc dù nó không thể nắm bắt tất cả các hình thức thông tin sai lệch.

### Đánh giá khả năng suy luận
`BBH` (Big Bench Hard) và `GSM8K` tập trung vào các tác vụ suy luận phức tạp. `BBH` kiểm tra tư duy logic và lập kế hoạch, trong khi `GSM8K` nhắm vào giải quyết vấn đề toán học. Những bài đánh giá này giúp kiểm tra khả năng phân tích nhưng có thể không nắm bắt được khả năng suy luận tinh tế cần thiết trong các tình huống thực tế.

### Đánh giá khả năng ngôn ngữ
`HELM` cung cấp một đánh giá toàn diện, trong khi `WinoGrande` kiểm tra hiểu biết thông thường thông qua việc giải nghĩa các đại từ. Những bài đánh giá này cung cấp cái nhìn sâu sắc về khả năng xử lý ngôn ngữ nhưng có thể không hoàn toàn đại diện cho độ phức tạp của giao tiếp tự nhiên hoặc thuật ngữ theo lĩnh vực cụ thể.

## Các phương pháp đánh giá thay thế

Nhiều tổ chức đã phát triển các phương pháp đánh giá thay thế để khắc phục những hạn chế của bài đánh giá tiêu chuẩn:

### LLM trong vài trò người đánh giá 
Sử dụng một mô hình ngôn ngữ để đánh giá đầu ra của mô hình khác ngày càng phổ biến. Phương pháp này có thể cung cấp phản hồi chi tiết hơn các bài đánh giá truyền thống, mặc dù nó đi kèm với những định kiến (bias) của mô hình đánh giá và hạn chế riêng.

### Môi trường đánh giá
Các nền tảng như Constitutional AI Arena của Anthropic cho phép các mô hình tương tác và đánh giá lẫn nhau trong môi trường được kiểm soát. Điều này có thể cho thấy những điểm mạnh và điểm yếu có thể không rõ ràng trong các benchmark truyền thống.

### Bộ đánh giá tuỳ chỉnh
Các tổ chức thường phát triển bộ benchmark nội bộ được điều chỉnh cho nhu cầu và trường hợp sử dụng cụ thể của họ. Những bộ này có thể bao gồm kiểm tra kiến thức theo lĩnh vực cụ thể hoặc các kịch bản đánh giá phản ánh điều kiện triển khai thực tế.

## Tạo chiến lược đánh giá riêng

Hãy nhớ rằng mặc dù LightEval giúp dễ dàng chạy các benchmark tiêu chuẩn, bạn cũng nên dành thời gian phát triển phương pháp đánh giá phù hợp với trường hợp sử dụng của mình.

Mặc dù benchmark tiêu chuẩn cung cấp một đường cơ sở hữu ích, chúng không nên là phương pháp đánh giá duy nhất của bạn. Dưới đây là cách phát triển một phương pháp toàn diện hơn:

1. Bắt đầu với các benchmark tiêu chuẩn liên quan để thiết lập đường cơ sở và cho phép so sánh với các mô hình khác.

2. Xác định các yêu cầu và thách thức cụ thể của trường hợp sử dụng của bạn. Mô hình của bạn sẽ thực hiện những tác vụ gì? Những loại lỗi nào sẽ gây rắc rối nhất?

3. Phát triển tập dữ liệu đánh giá tuỳ chỉnh phản ánh trường hợp sử dụng thực tế của bạn. Điều này có thể bao gồm:
   - Các câu hỏi thực tế từ người dùng trong lĩnh vực của bạn
   - Các trường hợp ngoại lệ phổ biến bạn đã gặp
   - Các ví dụ về tình huống đặc biệt khó khăn

4. Xem xét triển khai chiến lược đánh giá nhiều lớp:
   - Số liệu tự động để nhận phản hồi nhanh
   - Đánh giá bởi con người để hiểu sâu sắc hơn
   - Đánh giá của chuyên gia lĩnh vực cho các ứng dụng chuyên biệt
   - Kiểm tra A/B trong môi trường được kiểm soát

## Sử dụng LightEval để đánh giá

Các tác vụ LightEval được định nghĩa theo định dạng cụ thể:
```
{suite}|{task}|{num_few_shot}|{auto_reduce}
```

- **suite**: Bộ benchmark (ví dụ: 'mmlu', 'truthfulqa')  
- **task**: Tác vụ cụ thể trong bộ (ví dụ: 'abstract_algebra')
- **num_few_shot**: Số lượng ví dụ để đưa vào prompt (0 cho zero-shot)
- **auto_reduce**: Có tự động giảm ví dụ few-shot nếu prompt quá dài hay không (0 hoặc 1)

Ví dụ: `"mmlu|abstract_algebra|0|0"` đánh giá tác vụ đại số trừu tượng của MMLU với suy luận zero-shot.

### Ví dụ về Pipeline đánh giá 

Đây là một ví dụ hoàn chỉnh về việc thiết lập và chạy đánh giá trên các benchmark tự động liên quan đến một lĩnh vực cụ thể:

```python
from lighteval.tasks import Task, Pipeline
from transformers import AutoModelForCausalLM

# Định nghĩa các tác vụ để đánh giá
domain_tasks = [
    "mmlu|anatomy|0|0",
    "mmlu|high_school_biology|0|0", 
    "mmlu|high_school_chemistry|0|0",
    "mmlu|professional_medicine|0|0"
]

# Cấu hình tham số pipeline
pipeline_params = {
    "max_samples": 40,  # Số lượng mẫu để đánh giá
    "batch_size": 1,    # Kích thước batch cho inference 
    "num_workers": 4    # Số lượng worker process
}

# Tạo evaluation tracker
evaluation_tracker = EvaluationTracker(
    output_path="./results",
    save_generations=True
)

# Tải mô hình và tạo pipeline
model = AutoModelForCausalLM.from_pretrained("your-model-name")
pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=model
)

# Chạy đánh giá
pipeline.evaluate()

# Lấy và hiển thị kết quả 
results = pipeline.get_results()
pipeline.show_results()
```

Kết quả được hiển thị dưới dạng bảng thể hiện:
```
|                  Task                  |Version|Metric|Value |   |Stderr|
|----------------------------------------|------:|------|-----:|---|-----:|
|all                                     |       |acc   |0.3333|±  |0.1169|
|leaderboard:mmlu:_average:5             |       |acc   |0.3400|±  |0.1121|
|leaderboard:mmlu:anatomy:5              |      0|acc   |0.4500|±  |0.1141|
|leaderboard:mmlu:high_school_biology:5  |      0|acc   |0.1500|±  |0.0819|
```

Bạn cũng có thể xử lý kết quả trong pandas DataFrame và trực quan hoá hoặc biểu diễn chúng theo cách bạn muốn.

# Bước tiếp theo

⏩ Khám phá [Đánh giá theo lĩnh vực tuỳ chỉnh](./custom_evaluation.md) để học cách tạo pipeline đánh giá phù hợp với nhu cầu cụ thể của bạn