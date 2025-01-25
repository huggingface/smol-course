# Tinh chỉnh Mô hình Ngôn ngữ Thị giác (VLM)

## Tinh chỉnh Hiệu quả

### Lượng tử hóa (Quantization)

Lượng tử hóa làm giảm độ chính xác của trọng số mô hình (model weights) và hàm kích hoạt (activations), giúp giảm đáng kể mức sử dụng bộ nhớ và tăng tốc độ tính toán. Ví dụ: chuyển từ `float32` sang `bfloat16` giúp giảm một nửa yêu cầu bộ nhớ cho mỗi tham số trong khi vẫn duy trì hiệu suất. Để nén mạnh hơn, có thể sử dụng lượng tử hóa `8 bit` và `4 bit`, giúp giảm mức sử dụng bộ nhớ hơn nữa, mặc dù phải đánh đổi bằng việc giảm độ chính xác. Những kỹ thuật này có thể được áp dụng cho cả trọng số mô hình và trình tối ưu hóa (optimizer), cho phép huấn luyện hiệu quả trên phần cứng có tài nguyên hạn chế.

### PEFT & LoRA

Như đã giới thiệu trong Bài 3, LoRA (Low-Rank Adaptation) tập trung vào việc học các ma trận phân rã hạng thấp (rank-decomposition matrices) nhỏ gọn trong khi vẫn giữ nguyên trọng số của mô hình gốc. Điều này làm giảm đáng kể số lượng tham số có thể huấn luyện, giảm đáng kể yêu cầu tài nguyên. LoRA, khi được tích hợp với PEFT (Parameter-Efficient Fine-Tuning), cho phép tinh chỉnh các mô hình lớn bằng cách chỉ điều chỉnh một tập hợp con nhỏ các tham số có thể huấn luyện. Phương pháp này đặc biệt hiệu quả cho các thích ứng theo nhiệm vụ cụ thể, giảm hàng tỷ tham số có thể huấn luyện xuống chỉ còn hàng triệu trong khi vẫn duy trì hiệu suất.

### Tối ưu hóa Kích thước Batch (Batch Size)

Để tối ưu hóa kích thước batch cho quá trình tinh chỉnh, hãy bắt đầu với một giá trị lớn và giảm nếu xảy ra lỗi out-of-memory (OOM). Bù lại bằng cách tăng `gradient_accumulation_steps`, duy trì hiệu quả tổng kích thước batch trên nhiều lần cập nhật. Ngoài ra, hãy bật `gradient_checkpointing` để giảm mức sử dụng bộ nhớ bằng cách tính toán lại các trạng thái trung gian trong quá trình lan truyền ngược (backward pass), đánh đổi thời gian tính toán để giảm yêu cầu bộ nhớ kích hoạt. Những chiến lược này tối đa hóa việc sử dụng phần cứng và giúp khắc phục các hạn chế về bộ nhớ.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",  # Thư mục cho các checkpoint của mô hình
    per_device_train_batch_size=4,   # Kích thước batch trên mỗi thiết bị (GPU/TPU)
    num_train_epochs=3,              # Tổng số epoch huấn luyện
    learning_rate=5e-5,              # Tốc độ học
    save_steps=1000,                 # Lưu checkpoint sau mỗi 1000 bước
    bf16=True,                       # Sử dụng mixed precision để huấn luyện
    gradient_checkpointing=True,     # Bật để giảm mức sử dụng bộ nhớ kích hoạt
    gradient_accumulation_steps=16,  # Tích lũy gradient qua 16 bước
    logging_steps=50                 # Ghi nhật ký các số liệu sau mỗi 50 bước
)
```

## **Tinh chỉnh có Giám sát (Supervised Fine-Tuning - SFT)**

Tinh chỉnh có Giám sát (SFT) điều chỉnh Mô hình Ngôn ngữ Thị giác (VLM) đã được huấn luyện trước cho các nhiệm vụ cụ thể bằng cách tận dụng các tập dữ liệu được gán nhãn có chứa các đầu vào được ghép nối, chẳng hạn như hình ảnh và văn bản tương ứng. Phương pháp này nâng cao khả năng của mô hình để thực hiện các chức năng cụ thể theo miền (domain-specific) hoặc theo nhiệm vụ (task-specific), chẳng hạn như trả lời câu hỏi bằng hình ảnh, chú thích hình ảnh hoặc diễn giải biểu đồ.

### **Tổng quan**

SFT là cần thiết khi bạn cần một VLM chuyên về một lĩnh vực cụ thể hoặc giải quyết các vấn đề cụ thể mà khả năng chung của mô hình cơ sở có thể không đáp ứng được. Ví dụ: nếu mô hình gặp khó khăn với các đặc điểm hình ảnh độc đáo hoặc thuật ngữ chuyên ngành, SFT cho phép mô hình tập trung vào các lĩnh vực này bằng cách học từ dữ liệu được gán nhãn.

Mặc dù SFT rất hiệu quả, nó có những hạn chế đáng chú ý:

- **Phụ thuộc vào dữ liệu**: Cần có các tập dữ liệu được gán nhãn chất lượng cao phù hợp với nhiệm vụ.
- **Tài nguyên tính toán**: Tinh chỉnh các VLM lớn đòi hỏi nhiều tài nguyên.
- **Nguy cơ quá khớp (Overfitting)**: Các mô hình có thể mất khả năng khái quát hóa nếu được tinh chỉnh quá hẹp.

Tuy vậy, SFT vẫn là một kỹ thuật mạnh mẽ để nâng cao hiệu suất của mô hình trong các bối cảnh cụ thể.

### **Cách sử dụng**

1. **Chuẩn bị dữ liệu**: Bắt đầu với tập dữ liệu được gán nhãn ghép nối hình ảnh với văn bản, chẳng hạn như câu hỏi và câu trả lời. Ví dụ: trong các tác vụ như phân tích biểu đồ, tập dữ liệu `HuggingFaceM4/ChartQA` bao gồm hình ảnh biểu đồ, truy vấn và câu trả lời ngắn gọn.

2. **Thiết lập mô hình**: Tải VLM đã được huấn luyện trước phù hợp với nhiệm vụ, chẳng hạn như `HuggingFaceTB/SmolVLM-Instruct`, và một bộ xử lý (processor) để chuẩn bị đầu vào văn bản và hình ảnh. Điều chỉnh cấu hình của mô hình để phù hợp với phần cứng của bạn.

3. **Quá trình tinh chỉnh**:
   - **Định dạng dữ liệu**: Cấu trúc tập dữ liệu thành định dạng giống như chatbot, ghép nối các câu lệnh hệ thống (system messages), các truy vấn của người dùng và các câu trả lời tương ứng.
   - **Cấu hình huấn luyện**: Sử dụng các công cụ như `TrainingArguments` của Hugging Face hoặc `SFTConfig` của TRL để thiết lập các tham số huấn luyện. Chúng bao gồm kích thước batch, tốc độ học và các bước tích lũy gradient để tối ưu hóa việc sử dụng tài nguyên.
   - **Kỹ thuật tối ưu hóa**: Sử dụng **gradient checkpointing** để tiết kiệm bộ nhớ trong quá trình huấn luyện. Sử dụng mô hình đã lượng tử hóa để giảm yêu cầu bộ nhớ và tăng tốc độ tính toán.
   - Sử dụng `SFTTrainer` từ thư viện TRL, để hợp lý hóa quá trình huấn luyện.

## Tối ưu hóa theo Sở thích (Preference Optimization)

Tối ưu hóa theo Sở thích, đặc biệt là Tối ưu hóa Sở thích Trực tiếp (Direct Preference Optimization - DPO), huấn luyện Mô hình Ngôn ngữ Thị giác (VLM) để phù hợp với sở thích của con người. Thay vì tuân theo các hướng dẫn được xác định trước một cách nghiêm ngặt, mô hình học cách ưu tiên các đầu ra mà con người chủ quan thích hơn. Phương pháp này đặc biệt hữu ích cho các tác vụ liên quan đến phán đoán sáng tạo, lý luận sắc thái hoặc các câu trả lời có thể chấp nhận được khác nhau.

### **Tổng quan**

Tối ưu hóa theo Sở thích giải quyết các tình huống trong đó sở thích chủ quan của con người là trung tâm của sự thành công của nhiệm vụ. Bằng cách tinh chỉnh trên các tập dữ liệu mã hóa sở thích của con người, DPO nâng cao khả năng của mô hình trong việc tạo ra các phản hồi phù hợp với ngữ cảnh và phong cách với mong đợi của người dùng. Phương pháp này đặc biệt hiệu quả cho các tác vụ như viết sáng tạo, tương tác với khách hàng hoặc các tình huống có nhiều lựa chọn.

Mặc dù có những lợi ích, Tối ưu hóa theo Sở thích có những thách thức:

- **Chất lượng dữ liệu**: Cần có các tập dữ liệu được chú thích theo sở thích chất lượng cao, thường làm cho việc thu thập dữ liệu trở thành một nút thắt cổ chai.
- **Độ phức tạp**: Việc huấn luyện có thể liên quan đến các quy trình phức tạp như lấy mẫu theo cặp các sở thích và cân bằng tài nguyên tính toán.

Các tập dữ liệu sở thích phải nắm bắt được các sở thích rõ ràng giữa các đầu ra ứng viên. Ví dụ: một tập dữ liệu có thể ghép nối một câu hỏi với hai câu trả lời—một câu trả lời được ưu tiên và câu trả lời kia ít được chấp nhận hơn. Mô hình học cách dự đoán câu trả lời được ưu tiên, ngay cả khi nó không hoàn toàn chính xác, miễn là nó phù hợp hơn với đánh giá của con người.

### **Cách sử dụng**

1. **Chuẩn bị tập dữ liệu**
   Một tập dữ liệu được gán nhãn sở thích là rất quan trọng để huấn luyện. Mỗi ví dụ thường bao gồm một lời nhắc (prompt) (ví dụ: một hình ảnh và câu hỏi) và hai câu trả lời ứng viên: một câu được chọn (ưa thích) và một câu bị từ chối. Ví dụ:

   - **Câu hỏi**: Có bao nhiêu gia đình?
     - **Bị từ chối**: Hình ảnh không cung cấp bất kỳ thông tin nào về các gia đình.
     - **Được chọn**: Hình ảnh cho thấy một bảng gồm 18.000 gia đình.

   Tập dữ liệu dạy cho mô hình ưu tiên các câu trả lời phù hợp hơn, ngay cả khi chúng không hoàn hảo.

2. **Thiết lập mô hình**
   Tải VLM đã được huấn luyện trước và tích hợp nó với thư viện TRL của Hugging Face, hỗ trợ DPO và bộ xử lý để chuẩn bị đầu vào văn bản và hình ảnh. Định cấu hình mô hình để học có giám sát và phù hợp với phần cứng của bạn.

3. **Quy trình huấn luyện**
   Việc huấn luyện bao gồm việc định cấu hình các tham số cụ thể cho DPO. Dưới đây là bản tóm tắt về quy trình:

   - **Định dạng tập dữ liệu**: Cấu trúc từng mẫu với lời nhắc, hình ảnh và câu trả lời ứng viên.
   - **Hàm mất mát (Loss Function)**: Sử dụng hàm mất mát dựa trên sở thích để tối ưu hóa mô hình để chọn đầu ra được ưu tiên.
   - **Huấn luyện hiệu quả**: Kết hợp các kỹ thuật như lượng tử hóa, tích lũy gradient và bộ điều hợp LoRA (LoRA adapters) để tối ưu hóa bộ nhớ và tính toán.

## Tài liệu tham khảo

- [Hugging Face Learn: Tinh chỉnh có giám sát VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)
- [Hugging Face Learn: Tinh chỉnh có giám sát SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)
- [Hugging Face Learn: Tinh chỉnh tối ưu hóa tùy chọn SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)
- [Hugging Face Blog: Tối ưu hóa tùy chọn cho VLMs](https://huggingface.co/blog/dpo_vlm)

## Các bước tiếp theo

⏩ Thử [vlm_finetune_sample.ipynb](./notebooks/vlm_finetune_sample.ipynb) để triển khai phương pháp thống nhất này để căn chỉnh tùy chọn.
