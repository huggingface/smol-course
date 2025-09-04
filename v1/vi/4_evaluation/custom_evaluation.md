# Đánh Giá Theo Lĩnh Vực Cụ Thể

Mặc dù các phương pháp đánh giá tiêu chuẩn (`benchmark`) cung cấp những thông tin chuyên sâu có giá trị, nhiều ứng dụng đòi hỏi các phương pháp đánh giá chuyên biệt phù hợp với các lĩnh vực hoặc trường hợp sử dụng cụ thể. Bài học này sẽ giúp bạn tạo các quy trình (`pipeline`) tùy chỉnh các phương pháp đánh giá để có cái nhìn chính xác hơn về hiệu suất của mô hình trong lĩnh vực cụ thể của bạn.

## Thiết kế chiến lược đánh giá của bạn

Một chiến lược đánh giá tùy chỉnh đầy đủ bắt đầu với các mục tiêu rõ ràng. Hãy xem xét những khả năng cụ thể mà mô hình của bạn cần thể hiện trong lĩnh vực của bạn. Điều này có thể bao gồm kiến thức kỹ thuật, các mẫu suy luận hoặc các định dạng đặc thù cho lĩnh vực đó. Ghi lại cẩn thận các yêu cầu này - chúng sẽ hướng dẫn cả việc thiết kế tác vụ (`task`) và lựa chọn các chỉ số (`metric`).

Việc đánh giá của bạn nên kiểm tra cả các trường hợp sử dụng tiêu chuẩn và các trường hợp biên (`edge case`). Ví dụ, trong lĩnh vực y tế, bạn có thể đánh giá cả các kịch bản chẩn đoán phổ biến và các tình trạng hiếm gặp. Trong các ứng dụng tài chính, bạn có thể kiểm tra cả các giao dịch thông thường và các trường hợp biên phức tạp liên quan đến nhiều loại tiền tệ hoặc các điều kiện đặc biệt.

## Triển khai với LightEval

`LightEval` cung cấp một khung (`framework`) linh hoạt để triển khai các đánh giá tùy chỉnh. Đây là cách để tạo một tác vụ (`task`) tùy chỉnh:

```python
from lighteval.tasks import Task, Doc
from lighteval.metrics import SampleLevelMetric, MetricCategory, MetricUseCase

class CustomEvalTask(Task):
    def __init__(self):
        super().__init__(
            name="custom_task",
            version="0.0.1",
            metrics=["accuracy", "f1"],  # Các chỉ số bạn chọn
            description="Mô tả tác vụ đánh giá tùy chỉnh của bạn"
        )
    
    def get_prompt(self, sample):
        # Định dạng đầu vào của bạn thành một lời nhắc (prompt)
        return f"Question: {sample['question']}\nAnswer:"
    
    def process_response(self, response, ref):
        # Xử lý đầu ra của mô hình và so sánh với tham chiếu
        return response.strip() == ref.strip()
```


## Các chỉ số (`metric`) tùy chỉnh

Các tác vụ theo lĩnh vực cụ thể thường yêu cầu các chỉ số chuyên biệt. `LightEval` cung cấp một khung linh hoạt để tạo các chỉ số tùy chỉnh nắm bắt các khía cạnh liên quan đến lĩnh vực của hiệu suất:

```python
from aenum import extend_enum
from lighteval.metrics import Metrics, SampleLevelMetric, SampleLevelMetricGrouping
import numpy as np

# Định nghĩa một hàm chỉ số mức mẫu (sample-level metric)
def custom_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> dict:
    """Chỉ số ví dụ trả về nhiều điểm số cho mỗi mẫu"""
    response = predictions[0]
    return {
        "accuracy": response == formatted_doc.choices[formatted_doc.gold_index],
        "length_match": len(response) == len(formatted_doc.reference)
    }

# Tạo một chỉ số trả về nhiều giá trị cho mỗi mẫu
custom_metric_group = SampleLevelMetricGrouping(
    metric_name=["accuracy", "length_match"],  # Tên của các chỉ số
    higher_is_better={  # Liệu giá trị cao hơn có tốt hơn cho mỗi chỉ số không
        "accuracy": True,
        "length_match": True
    },
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=custom_metric,
    corpus_level_fn={  # Cách tổng hợp từng chỉ số
        "accuracy": np.mean,
        "length_match": np.mean
    }
)

# Đăng ký chỉ số vào LightEval
extend_enum(Metrics, "custom_metric_name", custom_metric_group)
```

Đối với các trường hợp đơn giản hơn, nơi bạn chỉ cần một giá trị chỉ số cho mỗi mẫu:

```python
def simple_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> bool:
    """Chỉ số ví dụ trả về một điểm duy nhất cho mỗi mẫu"""
    response = predictions[0]
    return response == formatted_doc.choices[formatted_doc.gold_index]

simple_metric_obj = SampleLevelMetric(
    metric_name="simple_accuracy",
    higher_is_better=True,
    category=MetricCategory.CUSTOM,
    use_case=MetricUseCase.SCORING,
    sample_level_fn=simple_metric,
    corpus_level_fn=np.mean  # Cách tổng hợp trên các mẫu
)

extend_enum(Metrics, "simple_metric", simple_metric_obj)
```

Sau đó, bạn có thể sử dụng các chỉ số tùy chỉnh của mình trong các tác vụ đánh giá bằng cách tham chiếu chúng trong cấu hình tác vụ. Các chỉ số sẽ được tự động tính toán trên tất cả các mẫu và tổng hợp theo các hàm bạn đã chỉ định.

Đối với các chỉ số phức tạp hơn, hãy xem xét:
- Sử dụng siêu dữ liệu (`metadata`) trong các tài liệu được định dạng của bạn để đánh trọng số hoặc điều chỉnh điểm số
- Triển khai các hàm tổng hợp tùy chỉnh cho các thống kê cấp độ kho ngữ liệu (`corpus-level`)
- Thêm kiểm tra xác thực cho đầu vào chỉ số của bạn
- Ghi lại các trường hợp biên và hành vi mong đợi

Để có một ví dụ đầy đủ về các chỉ số tùy chỉnh trong thực tế, hãy xem [dự án đánh giá theo lĩnh vực](./project/README.md) của chúng tôi.

## Tạo tập dữ liệu

Đánh giá chất lượng cao đòi hỏi các tập dữ liệu được sắp xếp cẩn thận. Hãy xem xét các phương pháp sau để tạo tập dữ liệu:

1. Chú thích của chuyên gia: Làm việc với các chuyên gia trong lĩnh vực để tạo và xác thực các ví dụ đánh giá. Các công cụ như [Argilla](https://github.com/argilla-io/argilla) làm cho quá trình này hiệu quả hơn.

2. Dữ liệu thế giới thực: Thu thập và ẩn danh hóa dữ liệu sử dụng thực tế, đảm bảo nó đại diện cho các kịch bản triển khai thực tế.

3. Tạo tổng hợp: Sử dụng `LLM` để tạo các ví dụ ban đầu, sau đó để các chuyên gia xác thực và tinh chỉnh chúng.

## Các phương pháp tốt nhất

- Ghi lại phương pháp đánh giá của bạn một cách kỹ lưỡng, bao gồm mọi giả định hoặc hạn chế
- Bao gồm các trường hợp kiểm thử đa dạng bao gồm các khía cạnh khác nhau trong lĩnh vực của bạn
- Xem xét cả các chỉ số tự động và đánh giá thủ công khi thích hợp
- Kiểm soát phiên bản (`Version control`) các tập dữ liệu và mã đánh giá của bạn
- Thường xuyên cập nhật bộ đánh giá của bạn khi bạn phát hiện ra các trường hợp biên hoặc yêu cầu mới

## Tài liệu tham khảo

- [Hướng dẫn tác vụ tùy chỉnh LightEval](https://github.com/huggingface/lighteval/wiki/Adding-a-Custom-Task)
- [Chỉ số tùy chỉnh LightEval](https://github.com/huggingface/lighteval/wiki/Adding-a-New-Metric)
- [Tài liệu Argilla](https://docs.argilla.io) để chú thích tập dữ liệu
- [Hướng dẫn đánh giá](https://github.com/huggingface/evaluation-guidebook) cho các nguyên tắc đánh giá chung

# Các bước tiếp theo

⏩ Để có một ví dụ đầy đủ về việc triển khai các khái niệm này, hãy xem [dự án đánh giá theo lĩnh vực](./project/README.md) của chúng tôi.