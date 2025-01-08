# Parameter-Efficient Fine-Tuning (PEFT)

Khi các mô hình ngôn ngữ ngày càng lớn, việc tinh chỉnh theo cách truyền thống trở nên ngày càng khó khăn. Việc cập nhật toàn bộ một mô hình với hàng tỷ tham số đòi hỏi một lượng lớn bộ nhớ GPU, tốn kém chi phí lưu trữ các phiên bản tinh chỉnh, và thậm chí có thể làm giảm khả năng khái quát hóa ban đầu của mô hình. Phương pháp Tinh chỉnh Hiệu quả Tham số (Parameter-Efficient Fine-Tuning - PEFT) xuất hiện như một giải pháp nhằm giải quyết các thách thức này, bằng cách chỉ điều chỉnh một tập hợp nhỏ tham số của mô hình trong khi giữ nguyên phần lớn tham số ban đầu.

Trong khi tinh chỉnh truyền thống đòi hỏi phải cập nhật toàn bộ tham số của mô hình trong quá trình huấn luyện, các phương pháp PEFT chỉ yêu cầu điều chỉnh dưới 1% số tham số của mô hình gốc. Cách tiếp cận này mang lại những lợi ích vượt trội:

- Cho phép fine-tuning trên phần cứng phổ thông với bộ nhớ GPU hạn chế.
- Tối ưu hóa việc lưu trữ, dễ dàng quản lý nhiều mô hình thích ứng cho từng tác vụ cụ thể.
- Cải thiện khả năng tổng quát hóa trong các tình huống dữ liệu hạn chế.
- Rút ngắn đáng kể thời gian huấn luyện và đánh giá.

## Các phương pháp phổ biến

In this module, we will cover two popular PEFT methods:

### 1️⃣ LoRA (Low-Rank Adaptation)

LoRA đã trở thành phương pháp PEFT được áp dụng rộng rãi nhất, mang lại một giải pháp tinh tế cho việc thích nghi mô hình hiệu quả. Thay vì chỉnh sửa toàn bộ mô hình, **LoRA chèn thêm các ma trận có thể huấn luyện vào các lớp attention của mô hình**. Phương pháp này thường giúp giảm khoảng 90% số lượng tham số cần huấn luyện, đồng thời vẫn duy trì hiệu suất tương đương với việc fine-tuning toàn bộ mô hình. Chúng ta sẽ tìm hiểu chi tiết về LoRA trong mục [LoRA (Low-Rank Adaptation)](./lora_adapters.md).
 
### 2️⃣ Tinh chỉnh Prompt

Prompt tuning cung cấp một cách tiếp cận **nhẹ nhàng** hơn nữa bằng cách **thêm các token có thể huấn luyện** vào đầu vào thay vì chỉnh sửa trọng số của mô hình. Mặc dù prompt tuning ít phổ biến hơn so với LoRA, nhưng đây vẫn là một kỹ thuật hữu ích để nhanh chóng thích nghi mô hình với các nhiệm vụ hoặc lĩnh vực mới. Chúng ta sẽ khám phá prompt tuning trong mục [Prompt Tuning](./prompt_tuning.md) 

## Bài tập

| Tiêu đề | Mô tả | Bài tập | Đường dẫn | Colab |
|-------|-------------|----------|------|-------|
| LoRA Fine-tuning | Tìm hiểu cách Fine-tune mô hình sử dụng LoRA | 🐢 Huấn luyện mô hình sử dụng LoRA<br>🐕 Thử nghiệm với nhiều giá trị hạng khác nhau<br>🦁 So sánh hiệu quả với fine-tune toàn bộ | [Notebook](./notebooks/finetune_sft_peft.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/finetune_sft_peft.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |  
| Load LoRA Adapters | Tìm hiểu cách tải và sử dụng các LoRA adapter đã được huấn luyện | 🐢 Tải adapter đã được huấn luyện trước<br>🐕 Gộp adapter với mô hình gốc<br>🦁 Chuyển đổi giữa nhiều adapter | [Notebook](./notebooks/load_lora_adapter.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/load_lora_adapter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |  
<!-- | Prompt Tuning | Learn how to implement prompt tuning | 🐢 Train soft prompts<br>🐕 Compare different initialization strategies<br>🦁 Evaluate on multiple tasks | [Notebook](./notebooks/prompt_tuning_example.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/3_parameter_efficient_finetuning/notebooks/prompt_tuning_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | -->

## Resources
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [Hugging Face PEFT Guide](https://huggingface.co/blog/peft)
- [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl) 
- [TRL](https://huggingface.co/docs/trl/index)
