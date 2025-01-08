# LoRA

LoRA đã trở thành phương pháp PEFT được áp dụng rộng rãi nhất. Nó hoạt động bằng cách thêm các ma trận phân rã hạng nhỏ vào trọng số attention, thường giảm khoảng 90% tham số cần huấn luyện.

## Hiểu về LoRA

LoRA là một kỹ thuật tinh chỉnh hiệu quả về tham số, đóng băng các trọng số của mô hình đã huấn luyện trước và đưa các ma trận phân rã hạng có thể huấn luyện vào các lớp của mô hình. Thay vì huấn luyện tất cả tham số mô hình trong quá trình tinh chỉnh, LoRA phân rã các cập nhật trọng số thành các ma trận nhỏ hơn thông qua phân rã hạng thấp, giảm đáng kể số lượng tham số cần huấn luyện trong khi vẫn duy trì hiệu suất mô hình. Ví dụ, khi áp dụng cho GPT-3 175B, LoRA đã giảm số tham số cần huấn luyện xuống 10.000 lần và yêu cầu bộ nhớ GPU xuống 3 lần so với tinh chỉnh toàn bộ. Bạn có thể đọc thêm về LoRA trong [bài báo LoRA](https://arxiv.org/pdf/2106.09685).

LoRA hoạt động bằng cách thêm các cặp ma trận phân rã hạng vào các lớp transformer, thường tập trung vào trọng số attention. Trong quá trình suy luận, các trọng số bộ điều hợp này có thể được kết hợp với mô hình cơ sở, không gây thêm độ trễ. LoRA đặc biệt hữu ích để điều chỉnh các mô hình ngôn ngữ lớn cho các nhiệm vụ hoặc lĩnh vực cụ thể trong khi vẫn giữ các yêu cầu tài nguyên ở mức quản lý được.

## Tải LoRA Adapters

Các bộ điều hợp có thể được tải vào mô hình đã huấn luyện trước với load_adapter(), điều này hữu ích khi thử nghiệm các bộ điều hợp khác nhau mà trọng số chưa được kết hợp. Thiết lập trọng số bộ điều hợp đang hoạt động với hàm set_adapter(). Để quay lại mô hình cơ sở, bạn có thể sử dụng unload() để dỡ tất cả các module LoRA. Điều này giúp dễ dàng chuyển đổi giữa các trọng số cho các nhiệm vụ cụ thể khác nhau.

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("<base_model_name>")
peft_model_id = "<peft_adapter_id>"
model = PeftModel.from_pretrained(base_model, peft_model_id)
```

![lora_load_adapter](./images/lora_adapter.png)

## Kết hợp LoRA Adapters

Sau khi huấn luyện với LoRA, bạn có thể muốn kết hợp trọng số bộ điều hợp trở lại mô hình cơ sở để triển khai dễ dàng hơn. Điều này tạo ra một mô hình duy nhất với trọng số kết hợp, loại bỏ nhu cầu tải bộ điều hợp riêng biệt trong quá trình suy luận.

Quá trình kết hợp đòi hỏi chú ý đến quản lý bộ nhớ và độ chính xác. Vì bạn sẽ cần tải cả mô hình cơ sở và trọng số bộ điều hợp cùng lúc, hãy đảm bảo có đủ bộ nhớ GPU/CPU. Sử dụng `device_map="auto"` trong `transformers` sẽ giúp quản lý bộ nhớ tự động. Duy trì độ chính xác nhất quán (ví dụ: float16) trong suốt quá trình, khớp với độ chính xác được sử dụng trong quá trình huấn luyện và lưu mô hình đã kết hợp ở cùng định dạng để triển khai. Trước khi triển khai, luôn xác nhận mô hình đã kết hợp bằng cách so sánh đầu ra và các chỉ số hiệu suất với phiên bản dựa trên bộ điều hợp.

Các bộ điều hợp cũng thuận tiện để chuyển đổi giữa các nhiệm vụ hoặc lĩnh vực khác nhau. Bạn có thể tải mô hình cơ sở và trọng số bộ điều hợp riêng biệt. Điều này cho phép chuyển đổi nhanh chóng giữa các trọng số cho nhiệm vụ cụ thể khác nhau.

## Hướng dẫn triển khai

Thư mục `notebooks/` chứa các hướng dẫn thực hành và bài tập để triển khai các phương pháp PEFT khác nhau. Bắt đầu với `load_lora_adapter_example.ipynb` để có giới thiệu cơ bản, sau đó khám phá `lora_finetuning.ipynb` để xem xét chi tiết hơn về cách tinh chỉnh mô hình với LoRA và SFT.

Khi triển khai các phương pháp PEFT, bắt đầu với các giá trị hạng nhỏ (4-8) cho LoRA và theo dõi độ mất trong huấn luyện. Sử dụng tập kiểm định để ngăn chặn overfitting và so sánh kết quả với các đường cơ sở tinh chỉnh toàn bộ khi có thể. Hiệu quả của các phương pháp khác nhau có thể thay đổi theo nhiệm vụ, vì vậy thực nghiệm là chìa khóa.

## OLoRA

[OLoRA](https://arxiv.org/abs/2406.01775) sử dụng phân rã QR để khởi tạo bộ điều hợp LoRA. OLoRA dịch chuyển trọng số cơ sở của mô hình theo hệ số phân rã QR của chúng, nghĩa là nó thay đổi trọng số trước khi thực hiện bất kỳ huấn luyện nào trên chúng. Cách tiếp cận này cải thiện đáng kể tính ổn định, tăng tốc độ hội tụ và cuối cùng đạt được hiệu suất vượt trội.

## Sử dụng TRL với PEFT

Các phương pháp PEFT có thể được kết hợp với TRL (Transformers Reinforcement Learning) để tinh chỉnh hiệu quả. Sự tích hợp này đặc biệt hữu ích cho RLHF (Reinforcement Learning from Human Feedback) vì nó giảm yêu cầu bộ nhớ.

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
    load_in_8bit=True,  # Tùy chọn: sử dụng 8-bit
    device_map="auto",
    peft_config=lora_config
)
```

Ở trên, chúng ta đã sử dụng `device_map="auto"` để tự động gán mô hình cho thiết bị phù hợp. Bạn cũng có thể gán thủ công mô hình cho một thiết bị cụ thể bằng cách sử dụng `device_map={"": device_index}`. Bạn cũng có thể mở rộng việc huấn luyện trên nhiều GPU trong khi vẫn giữ việc sử dụng bộ nhớ hiệu quả.

## Triển khai kết hợp cơ bản

Sau khi huấn luyện bộ điều hợp LoRA, bạn có thể kết hợp trọng số bộ điều hợp trở lại mô hình cơ sở. Đây là cách thực hiện:

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

# 2. Tải mô hình PEFT với bộ điều hợp
peft_model = PeftModel.from_pretrained(
    base_model,
    "path/to/adapter",
    torch_dtype=torch.float16
)

# 3. Kết hợp trọng số bộ điều hợp với mô hình cơ sở
try:
    merged_model = peft_model.merge_and_unload()
except RuntimeError as e:
    print(f"Kết hợp thất bại: {e}")
    # Triển khai dự phòng hoặc tối ưu hóa bộ nhớ

# 4. Lưu mô hình đã kết hợp
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

⏩ Chuyển đến [Hướng dẫn Prompt Tuning](prompt_tuning.md) để tìm hiểu cách tinh chỉnh mô hình với prompt tuning.
⏩ Chuyển đến [Hướng dẫn tải bộ điều hợp LoRA](./notebooks/load_lora_adapter.ipynb) để tìm hiểu cách tải các bộ điều hợp LoRA.

# Tài nguyên

- [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)
- [Tài liệu PEFT](https://huggingface.co/docs/peft)
- [Bài viết blog Hugging Face về PEFT](https://huggingface.co/blog/peft)


