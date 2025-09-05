# Phương Pháp LoRA (Low-Rank Adaptation)

LoRA đã trở thành phương pháp PEFT được sử dụng rộng rãi nhất. Nó hoạt động bằng cách thêm các ma trận phân rã hạng thấp (small rank decomposition matrices) vào các trọng số attention, điều này dẫn đến việc giảm khoảng 90% số lượng tham số có thể huấn luyện.

## Tìm Hiểu Về LoRA

LoRA (Low-Rank Adaptation) là một kỹ thuật tinh chỉnh hiệu quả tham số, đóng băng các trọng số của mô hình đã huấn luyện trước và đưa thêm các ma trận phân rã hạng (rank decomposition matrices) có thể huấn luyện vào các lớp của mô hình. Thay vì huấn luyện tất cả các tham số mô hình trong quá trình tinh chỉnh, LoRA phân rã việc cập nhật trọng số thành các ma trận nhỏ hơn thông qua phân rã hạng thấp (low-rank decomposition), giảm đáng kể số lượng tham số có thể huấn luyện trong khi vẫn duy trì hiệu suất mô hình. Ví dụ, khi áp dụng cho GPT-3 175B, LoRA giảm số lượng tham số có thể huấn luyện xuống 10.000 lần và yêu cầu bộ nhớ GPU giảm 3 lần so với tinh chỉnh đầy đủ. Bạn có thể đọc thêm về LoRA trong [bài báo nghiên cứu LoRA](https://arxiv.org/pdf/2106.09685).

LoRA hoạt động bằng cách thêm các cặp ma trận phân rã hạng vào các lớp transformer, thường tập trung vào các trọng số attention. Trong quá trình suy luận, các trọng số adapter có thể được gộp với mô hình cơ sở, không gây thêm độ trễ. LoRA đặc biệt hữu ích cho việc điều chỉnh các mô hình ngôn ngữ lớn cho các tác vụ hoặc lĩnh vực cụ thể trong khi không yêu cầu nhiều tài nguyên tính toán.

## Lắp Các LoRA Adapters Vào Mô Hình Ngôn Ngữ

Adapters có thể được lắp vào một mô hình đã huấn luyện trước với hàm `load_adapter()`, điều này hữu ích khi thử nghiệm các adapters khác nhau mà trọng số không được gộp. Đặt trọng số adapter đang hoạt động bằng hàm `set_adapter()`. Để trở về mô hình cơ sở, bạn có thể sử dụng `unload()` để gỡ tất cả các module LoRA. Điều này giúp dễ dàng chuyển đổi giữa các trọng số cho từng tác vụ cụ thể.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("<base_model_name>")
peft_model_id = "<peft_adapter_id>"
model = PeftModel.from_pretrained(base_model, peft_model_id)
```

![lora_load_adapter](./images/lora_adapter.png)

## Gộp LoRA Adapters

Sau khi huấn luyện với LoRA, bạn có thể muốn gộp các trọng số adapter trở lại mô hình cơ sở để triển khai dễ dàng hơn. Điều này tạo ra một mô hình duy nhất với các trọng số đã kết hợp, loại bỏ nhu cầu tải adapters riêng biệt trong quá trình suy luận.

Quá trình gộp đòi hỏi chú ý đến quản lý bộ nhớ và độ chính xác. Vì bạn sẽ cần tải cả mô hình cơ sở và trọng số adapter cùng lúc, hãy đảm bảo có đủ bộ nhớ GPU/CPU. Sử dụng `device_map="auto"` trong `transformers` sẽ giúp tự động quản lý bộ nhớ. Duy trì độ chính xác nhất quán (ví dụ: float16) trong suốt quá trình, phù hợp với độ chính xác được sử dụng trong quá trình huấn luyện và lưu mô hình đã gộp trong cùng một định dạng để triển khai. Trước khi triển khai, luôn xác thực mô hình đã gộp bằng cách so sánh đầu ra và các chỉ số hiệu suất với phiên bản dựa trên adapter.

Adapters cũng thuận tiện cho việc chuyển đổi giữa các tác vụ hoặc lĩnh vực khác nhau. Bạn có thể tải mô hình cơ sở và trọng số adapter riêng biệt. Điều này cho phép chuyển đổi nhanh chóng giữa các trọng số cho từng tác vụ cụ thể.

## Hướng dẫn triển khai

Thư mục `notebooks/` chứa các hướng dẫn thực hành và bài tập để triển khai các phương pháp PEFT khác nhau. Bắt đầu với `load_lora_adapter_example.ipynb` để có giới thiệu cơ bản, sau đó khám phá `lora_finetuning.ipynb` để xem xét chi tiết hơn về cách tinh chỉnh một mô hình với LoRA và SFT.

Khi triển khai các phương pháp PEFT, hãy bắt đầu với các giá trị hạng nhỏ (4-8) cho LoRA và theo dõi mất mát trong quá trình huấn luyện. Sử dụng tập validation để tránh overfitting và so sánh kết quả với các baseline tinh chỉnh đầy đủ khi có thể. Hiệu quả của các phương pháp khác nhau có thể thay đổi theo tác vụ, vì vậy thử nghiệm là chìa khóa.

## OLoRA

[OLoRA](https://arxiv.org/abs/2406.01775) sử dụng phân rã QR để khởi tạo các adapter LoRA. OLoRA dịch chuyển các trọng số cơ sở của mô hình bằng một hệ số của phân rã QR của chúng, tức là nó thay đổi trọng số trước khi thực hiện bất kỳ huấn luyện nào trên chúng. Cách tiếp cận này cải thiện đáng kể tính ổn định, tăng tốc độ hội tụ và cuối cùng đạt được hiệu suất vượt trội.

## Sử dụng TRL với PEFT

Các phương pháp PEFT có thể được kết hợp với thư viện TRL để tinh chỉnh hiệu quả. Sự tích hợp này đặc biệt hữu ích cho RLHF (Reinforcement Learning from Human Feedback) vì nó giảm yêu cầu bộ nhớ.

```python
from peft import LoraConfig
from transformers import AutoModelForCausalLM

# Tải mô hình với cấu hình PEFT
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Tải mô hình trên thiết bị cụ thể
model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    load_in_8bit=True,  # Tùy chọn: sử dụng độ chính xác 8-bit
    device_map="auto",
    peft_config=lora_config
)
```

Ở trên, chúng ta đã sử dụng `device_map="auto"` để tự động gán mô hình cho thiết bị phù hợp. Bạn cũng có thể gán thủ công mô hình cho một thiết bị cụ thể bằng cách sử dụng `device_map={"": device_index}`. Bạn cũng có thể mở rộng việc huấn luyện trên nhiều GPU trong khi vẫn giữ việc sử dụng bộ nhớ hiệu quả.

## Triển khai gộp cơ bản

Sau khi huấn luyện một adapter LoRA, bạn có thể gộp trọng số adapter trở lại mô hình cơ sở. Đây là cách thực hiện:

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Tải mô hình cơ sở
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Tải LoRA adapter
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. Gộp trọng số adapter với mô hình cơ sở
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Gộp thất bại: {e}")
    # Triển khai chiến lược dự phòng hoặc tối ưu hóa bộ nhớ

# 4. Lưu mô hình đã gộp
merged_model.save_pretrained("path/to/save/merged_model")
```

Nếu bạn gặp sự khác biệt về kích thước trong mô hình đã lưu, hãy đảm bảo bạn cũng lưu tokenizer:

```python
# Lưu cả mô hình và tokenizer
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

## Các bước tiếp theo

⏩ Chuyển sang [Hướng dẫn Prompt Tuning](prompt_tuning.md) để tìm hiểu cách tinh chỉnh một mô hình bằng prompt tuning.
⏩ Chuyển sang [Hướng dẫn lắp LoRA Adapters](./notebooks/load_lora_adapter.ipynb) để tìm hiểu cách lắp LoRA adapters.

# Tài liệu tham khảo

- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)
- [Tài liệu PEFT](https://huggingface.co/docs/peft)
- [Bài viết blog của Hugging Face về PEFT](https://huggingface.co/blog/peft)