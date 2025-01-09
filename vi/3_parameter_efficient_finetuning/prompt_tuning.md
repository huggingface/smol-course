# Prompt Tuning

Kỹ thuật tinh chỉnh prompt là một cách tiếp cận hiệu quả về mặt tham số, tập trung vào việc điều chỉnh cách biểu diễn dữ liệu đầu vào thay vì thay đổi trọng số của mô hình. Thay vì cập nhật toàn bộ tham số như cách tinh chỉnh truyền thống, tinh chỉnh prompt chỉ thêm vào và tối ưu một số ít token có thể huấn luyện trong khi vẫn giữ nguyên mô hình gốc.

## Understanding Prompt Tuning

Điều chỉnh prompt (prompt tuning) là một phương pháp thay thế hiệu quả về thông số so với việc tinh chỉnh toàn bộ mô hình, bằng cách thêm các vector liên tục có thể huấn luyện (prompt mềm) vào phần đầu của văn bản đầu vào. Khác với các prompt văn bản rời rạc, những prompt mềm này được học thông qua lan truyền ngược (backpropagation) trong khi giữ nguyên mô hình ngôn ngữ. Phương pháp này được giới thiệu trong nghiên cứu [Sức mạnh của quy mô trong việc điều chỉnh prompt hiệu quả về thông số](https://arxiv.org/abs/2104.08691) (Lester và cộng sự, 2021), chứng minh rằng việc điều chỉnh prompt trở nên cạnh tranh hơn với phương pháp tinh chỉnh mô hình khi kích thước mô hình tăng lên. Trong bài báo, khi mô hình đạt khoảng 10 tỷ tham số, hiệu suất của việc điều chỉnh prompt ngang bằng với việc tinh chỉnh mô hình, trong khi chỉ cần điều chỉnh vài trăm thông số cho mỗi tác vụ.

Những prompt mềm này là các vector liên tục trong không gian nhúng của mô hình và được tối ưu hóa trong quá trình huấn luyện. Khác với các prompt rời rạc truyền thống sử dụng các token ngôn ngữ tự nhiên, các prompt mềm không mang ý nghĩa vốn có nhưng học cách kích hoạt hành vi mong muốn từ mô hình đã đóng băng thông qua phương pháp gradient descent. Kỹ thuật này đặc biệt hiệu quả cho các tình huống đa nhiệm vụ, bởi mỗi nhiệm vụ chỉ cần lưu trữ một vector prompt nhỏ (thường là vài trăm thông số) thay vì một bản sao đầy đủ của mô hình. Cách tiếp cận này không chỉ duy trì được dung lượng bộ nhớ tối thiểu mà còn cho phép chuyển đổi nhiệm vụ nhanh chóng bằng cách đơn giản là hoán đổi các vector prompt mà không cần tải lại mô hình.

## Training Process

Các prompt mềm thường có số lượng từ 8 đến 32 token và có thể được khởi tạo ngẫu nhiên hoặc từ văn bản có sẵn. Phương pháp khởi tạo đóng vai trò quan trọng trong quá trình huấn luyện, trong đó việc khởi tạo dựa trên văn bản thường mang lại hiệu quả tốt hơn so với khởi tạo ngẫu nhiên.

Trong quá trình huấn luyện, chỉ các thông số của prompt được cập nhật trong khi mô hình cơ sở vẫn giữ nguyên không đổi. Cách tiếp cận tập trung này sử dụng các mục tiêu huấn luyện tiêu chuẩn nhưng đòi hỏi phải chú ý kỹ lưỡng đến tốc độ học và hành vi gradient của các token prompt.

## Implementation with PEFT

Thư viện PEFT giúp việc triển khai tinh chỉnh prompt trở nên đơn giản. Đây là một ví dụ cơ bản:

```python
from peft import PromptTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Tải model ban đầu
model = AutoModelForCausalLM.from_pretrained("your-base-model")
tokenizer = AutoTokenizer.from_pretrained("your-base-model")

# Cài đặt tinh chỉnh prompt
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=8,  # Số lượng token huấn luyện
    prompt_tuning_init="TEXT",  # Khởi tạo từ dạng text
    prompt_tuning_init_text="Classify if this text is positive or negative:",
    tokenizer_name_or_path="your-base-model",
)

# Tạo mô hình đã tinh chỉnh
model = get_peft_model(model, peft_config)
```

## Comparison to Other Methods

Khi so sánh với các phương pháp PEFT khác, prompt tuning nổi bật về mặt hiệu quả. Trong khi LoRA cung cấp số lượng tham số và mức sử dụng bộ nhớ thấp nhưng yêu cầu tải bộ điều hợp khi chuyển đổi nhiệm vụ, prompt tuning đạt được mức sử dụng tài nguyên thậm chí còn thấp hơn và cho phép chuyển đổi nhiệm vụ ngay lập tức. Ngược lại, tinh chỉnh toàn bộ đòi hỏi tài nguyên đáng kể và cần các bản sao mô hình riêng biệt cho các nhiệm vụ khác nhau.

| Phương pháp | Tham số | Bộ nhớ | Chuyển đổi nhiệm vụ |
|--------|------------|---------|----------------|
| Prompt Tuning | Rất thấp | Tối thiểu | Dễ dàng |
| LoRA | Thấp | Thấp | Cần tải bộ điều hợp |
| Tinh chỉnh toàn bộ | Cao | Cao | Cần bản sao mới |

Khi triển khai prompt tuning, hãy bắt đầu với một số lượng nhỏ token ảo (8-16) và chỉ tăng lên nếu độ phức tạp của nhiệm vụ yêu cầu. Khởi tạo bằng văn bản thường mang lại kết quả tốt hơn so với khởi tạo ngẫu nhiên, đặc biệt khi sử dụng văn bản liên quan đến nhiệm vụ. Chiến lược khởi tạo nên phản ánh độ phức tạp của nhiệm vụ mục tiêu của bạn.

Việc huấn luyện đòi hỏi những cân nhắc hơi khác so với tinh chỉnh toàn bộ. Tốc độ huấn luyện cao hơn thường hoạt động tốt, nhưng việc theo dõi cẩn thận các gradient của token prompt là điều cần thiết. Kiểm định thường xuyên trên các ví dụ đa dạng giúp đảm bảo hiệu suất mạnh mẽ trong các tình huống khác nhau.

## Application

Tinh chỉnh prompt vượt trội trong một số tình huống:

1. Triển khai đa tác vụ
2. Môi trường hạn chế về tài nguyên
3. Thích ứng nhiệm vụ nhanh chóng
4. Các ứng dụng nhạy cảm về quyền riêng tư

Khi các mô hình nhỏ hơn, điều chỉnh nhanh trở nên kém cạnh tranh hơn so với tinh chỉnh hoàn toàn. Ví dụ, trên các mô hình như SmolLM2, điều chỉnh prompt ít hiệu quả hơn so với tinh chỉnh đầy đủ. 

## Next Steps

⏭️ Hãy đọc phần [LoRA Adapters Tutorial](./notebooks/finetune_sft_peft.ipynb) để tìm hiểu cách fine-tune mô hình với LoRA adapters

## Resources
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [Hugging Face Cookbook](https://huggingface.co/learn/cookbook/prompt_tuning_peft)
